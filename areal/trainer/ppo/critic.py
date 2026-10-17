# [MODULE DOC]
# PPO critic (value model) wrappers. PPOCritic runs on a training worker and wraps a
# TrainEngine to compute values and perform clipped value-loss PPO updates;
# PPOCriticController (TrainController RPC) and PPOCriticControllerV2 (HTTP gateway)
# forward the same calls from a single controller process to the workers. ppo_loss_fn is
# the per-micro-batch critic loss with stats_tracker logging.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOCriticConfig
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker
from areal.utils.data import (
    batched_call,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import ppo_critic_loss_fn
from areal.utils.perf_tracer import trace_perf
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)


class PPOCritic:
    # [CLASS DOC]
    # Worker-side critic wrapper holding a PPOCriticConfig (eps_clip, ppo_n_minibatches,
    # mask_no_eos_with_zero, ...) and a TrainEngine that hosts the value model.
    def __init__(self, config: PPOCriticConfig, engine: TrainEngine):
        # [FUNCTION DOC]
        # Stores config and engine; no other side effects.
        self.config = config
        self.engine = engine

    @trace_perf("ppo_critic.compute_values", category="compute")
    @torch.no_grad()
    def compute_values(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Computes per-token values without gradients for a list of batch dicts (or a
        # single dict, as handled by batched_call) by applying _compute_values to each.
        # Traced with trace_perf. Returns the value tensor(s).
        # KNOWN ISSUE: annotated to accept/return lists but batched_call also handles
        # single-dict inputs, so the actual return shape follows the input form.
        return batched_call(self._compute_values, data)

    def _compute_values(self, data: dict[str, Any]) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Puts the engine in eval mode and runs engine.forward on one batch;
        # each micro-batch output's trailing size-1 dimension is squeezed and the
        # outputs are concatenated along the last dimension. Returns the value tensor.
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat([x.squeeze(-1) for x in xs], dim=-1),
        )

    @trace_perf("ppo_critic.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_critic")
    def ppo_update(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Runs _ppo_update on the given data via batched_call without unpacking (the
        # whole input is passed as one item). Wrapped in perf tracing and the
        # "ppo_critic" stats scope. Returns None.
        batched_call(self._ppo_update, data, unpack=False)

    def _ppo_update(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: One critic PPO update. Logs config scalars (mask_no_eos_with_zero,
        # eps_clip), removes the keys rewards/tot_rewards/kl_rewards/versions from data
        # in place, switches the engine to train mode (needed for gradient
        # checkpointing), splits the padded batch into ppo_n_minibatches minibatches and
        # calls engine.train_batch on each with ppo_loss_fn (eps_clip bound) and loss
        # weight = number of nonzero loss_mask tokens; each returned train stat dict is
        # logged as scalars. Performs optimizer steps on the engine.
        ########## Logging code starts ##########
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        stats_tracker.scalar(**scalars)
        ########## Logging code ends ##########

        for key in ["rewards", "tot_rewards", "kl_rewards", "versions"]:
            data.pop(key, None)

        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        for mb in mb_inputs.mbs:
            train_stat = self.engine.train_batch(
                mb,
                loss_fn=functools.partial(
                    ppo_loss_fn,
                    eps_clip=self.config.eps_clip,
                ),
                loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
            )
            stats_tracker.scalar(**train_stat)


class PPOCriticController(TrainController):
    # [CLASS DOC]
    # Controller-side proxy (TrainController) exposing the PPOCritic methods via RPC to
    # all workers.
    def compute_values(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Broadcasts a compute_values custom call to all workers via
        # _custom_function_call and returns its result.
        return self._custom_function_call(
            "compute_values", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def ppo_update(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Broadcasts a ppo_update custom call to all workers; returns None.
        self._custom_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOCriticControllerV2(GatewayTrainController):
    # [CLASS DOC]
    # Gateway-based controller (v2 training service) that forwards critic calls as HTTP
    # POSTs with serialized args/kwargs.
    def compute_values(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Serializes args (as list) and kwargs, POSTs to /ppo/critic/compute_values via
        # _gateway_post_result and returns the deserialized result.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/critic/compute_values", payload)

    def ppo_update(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Serializes args/kwargs and POSTs to /ppo/critic/update via _gateway_post;
        # returns None.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/critic/update", payload)


def ppo_loss_fn(
    value: torch.Tensor,
    input_data: dict,
    eps_clip: float,
):
    # [FUNCTION DOC]
    # Per-micro-batch clipped value loss. Inputs: value (model output, trailing dim
    # squeezed, cast to float), input_data with values (old values), returns (targets)
    # and loss_mask, and eps_clip. Delegates to ppo_critic_loss_fn. Side effects:
    # registers stats_tracker denominators (n_tokens from infer_token_denominator,
    # n_valid_tokens, clipped_tokens) and logs new/old/target values, critic_loss and
    # clip_ratio over valid tokens, plus clipped new/old values over clipped tokens.
    # Returns the scalar loss tensor.
    # KNOWN ISSUE: docstring says it returns loss and logging stats, but only the loss
    # is returned (stats go to stats_tracker).
    """Loss function for critic step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    value = value.squeeze(-1).float()
    old_value = input_data["values"].float()
    target_value = input_data["returns"].float()
    loss_mask = input_data["loss_mask"].bool()

    loss, stat = ppo_critic_loss_fn(
        value=value,
        old_value=old_value,
        target_value=target_value,
        value_eps_clip=eps_clip,
        loss_mask=loss_mask,
    )

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, value),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
    )

    stats_tracker.stat(
        new_value=value.detach(),
        old_value=old_value,
        target_value=target_value,
        critic_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        denominator="n_valid_tokens",
    )

    clip_mask = stat["clip_mask"]
    clipped_new_value = torch.where(clip_mask, value.detach(), 0.0)
    clipped_old_value = torch.where(clip_mask, old_value, 0.0)
    stats_tracker.stat(
        clipped_new_value=clipped_new_value,
        clipped_old_value=clipped_old_value,
        denominator="clipped_tokens",
    )
    return loss
