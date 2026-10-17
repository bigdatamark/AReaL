# [MODULE DOC]
# Direct Preference Optimization (DPO) training wrappers. Batches are packed sequences
# where consecutive sequence pairs are (chosen, rejected) for one example. DPOEngine
# wraps a worker-side TrainEngine (whose config supplies beta and loss_type) for
# training, evaluation and reference log-prob computation; DPOController (RPC) and
# DPOControllerV2 (HTTP gateway) forward these calls from the controller.
# compute_dpo_loss is the loss function passed to the engine. A module logger
# "DPOEngine" is created but unused.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch

from areal.api import TrainEngine
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call
from areal.utils.functional import dpo_pair_logratios, dpo_preference_loss
from areal.utils.perf_tracer import trace_perf
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

logger = logging.getLogger("DPOEngine")


def _dpo_valid_pairs(x: dict[str, Any]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: From cu_seqlens computes per-sequence lengths, reshapes into [n_pairs, 2]
    # and returns a bool [n_pairs] mask that is True where both chosen and rejected
    # sequences are non-empty (empty ones are padding). Assumes an even number of
    # sequences.
    seqlens = x["cu_seqlens"][1:] - x["cu_seqlens"][:-1]
    return seqlens.view(-1, 2).ne(0).all(dim=1)


def _dpo_loss_weight(x: dict[str, Any]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Loss weight for a micro-batch = number of valid pairs, as a float tensor.
    return _dpo_valid_pairs(x).count_nonzero().float()


def _log_empty_dpo_stats(device: torch.device) -> None:
    # [FUNCTION DOC]
    # HELPER: For a micro-batch with no valid pairs, records an n_pairs denominator of
    # one False element and zero-valued
    # loss/chosen_reward/rejected_reward/reward_accuracy/reward_margin stats so
    # stats_tracker keys stay consistent across ranks. Side effect only; returns None.
    n_pairs = torch.zeros(1, dtype=torch.bool, device=device)
    stats_tracker.denominator(n_pairs=n_pairs)
    stats_tracker.stat(
        loss=torch.zeros(1, dtype=torch.float32, device=device),
        chosen_reward=torch.zeros(1, dtype=torch.float32, device=device),
        rejected_reward=torch.zeros(1, dtype=torch.float32, device=device),
        reward_accuracy=torch.zeros(1, dtype=torch.float32, device=device),
        reward_margin=torch.zeros(1, dtype=torch.float32, device=device),
        denominator="n_pairs",
    )


class DPOEngine:
    # [CLASS DOC]
    # Worker-side DPO wrapper around a TrainEngine. Reads beta and loss_type from
    # engine.config at construction.
    def __init__(self, engine: TrainEngine):
        # [FUNCTION DOC]
        # Stores the engine and caches engine.config.beta and engine.config.loss_type
        # (raises AttributeError if the engine config lacks them).
        self.engine = engine
        self.beta = engine.config.beta
        self.loss_type = engine.config.loss_type

    @trace_perf("dpo_engine.train_dpo", category="compute")
    @stats_tracker.scope_func_wrapper("dpo")
    def train_dpo(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Runs _train_dpo once on the given data (batched_call, unpack=False) inside the
        # "dpo" stats scope with perf tracing. Returns None.
        batched_call(self._train_dpo, data, unpack=False)

    def _train_dpo(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets train mode, runs engine.train_batch with compute_dpo_loss
        # (beta/loss_type bound) and _dpo_loss_weight, then logs the returned train
        # stats as scalars. Performs an optimizer step on the engine.
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=functools.partial(
                compute_dpo_loss, beta=self.beta, loss_type=self.loss_type
            ),
            loss_weight_fn=_dpo_loss_weight,
        )
        stats_tracker.scalar(**stats)

    @trace_perf("dpo_engine.evaluate_dpo", category="compute")
    @stats_tracker.scope_func_wrapper("dpo-eval")
    def evaluate_dpo(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Runs _evaluate_dpo on the data inside the "dpo-eval" stats scope with perf
        # tracing. Returns None.
        batched_call(self._evaluate_dpo, data, unpack=False)

    def _evaluate_dpo(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets eval mode and runs engine.eval_batch with the same DPO loss and
        # weight; stats are recorded by the loss function. Result is discarded.
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=functools.partial(
                compute_dpo_loss, beta=self.beta, loss_type=self.loss_type
            ),
            loss_weight_fn=_dpo_loss_weight,
        )

    @trace_perf("dpo_engine.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        # [FUNCTION DOC]
        # Without gradients, computes per-token log-probs for the data via _compute_logp
        # (used by the reference engine to supply ref_logprobs). Returns the tensor(s)
        # per batched_call's input form; may be None on non-output ranks per the type
        # hint.
        """Compute per-token log-probabilities. Used by the ref engine to
        provide reference log-probs for DPO training."""
        return batched_call(self._compute_logp, data)

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        # [FUNCTION DOC]
        # HELPER: Eval mode, engine.forward with micro-batch outputs concatenated along
        # the last dimension. Returns the log-prob tensor (or None as the engine
        # returns).
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )


class DPOController(TrainController):
    # [CLASS DOC]
    # TrainController proxy that broadcasts DPO calls to workers with group_size=2 so
    # each chosen/rejected pair is kept together on the same data-parallel rank.
    def train_dpo(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Broadcasts a train_dpo custom call with group_size=2. Returns None.
        # dpo_collate produces 2 sequences (chosen + rejected) per example;
        # group_size=2 keeps each pair on the same DP rank.
        self._custom_function_call(
            "train_dpo", *args, group_size=2, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_dpo(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Pads the eval dispatch args (via _pad_eval_dispatch_args, group_size=2) so
        # data splits evenly across ranks, then broadcasts evaluate_dpo with
        # group_size=2. Returns None.
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=2)
        self._custom_function_call(
            "evaluate_dpo",
            *args,
            group_size=2,
            rpc_meta={"broadcast": True},
            **kwargs,
        )

    def compute_logp(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Broadcasts compute_logp with group_size=2 and returns the gathered result.
        return self._custom_function_call(
            "compute_logp", *args, group_size=2, rpc_meta={"broadcast": True}, **kwargs
        )


class DPOControllerV2(GatewayTrainController):
    # [CLASS DOC]
    # Gateway (v2 training service) controller forwarding DPO calls as HTTP POSTs with
    # serialized args/kwargs.
    def train_dpo(self, *args, **kwargs):
        # [FUNCTION DOC]
        # POSTs serialized args/kwargs to /dpo/train and discards the result.
        # KNOWN ISSUE: unlike evaluate_dpo/compute_logp (and DPOController.train_dpo) it
        # does not default group_size to 2, so pairs may be split across ranks unless
        # the gateway does so.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/dpo/train", payload)

    def evaluate_dpo(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Copies kwargs, defaults group_size to 2, POSTs to /dpo/evaluate; result
        # discarded.
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/dpo/evaluate", payload)

    def compute_logp(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Copies kwargs, defaults group_size to 2, POSTs to /dpo/compute_logp and
        # returns the deserialized result.
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/dpo/compute_logp", payload)


def compute_dpo_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor | None,
    input_: dict[str, Any],
    *,
    beta: float,
    loss_type: str = "sigmoid",
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    vocab_mean_logits: torch.Tensor | None = None,
    vocab_norm_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # DPO loss for one packed micro-batch. Inputs: logprobs (per-token policy log-
    # probs), entropy (unused), input_ with cu_seqlens, loss_mask, ref_logprobs; keyword
    # beta and loss_type (default "sigmoid"); vocab_* logit stats are accepted but
    # unused. If no valid pair exists, logs empty stats and returns a zero scalar.
    # Otherwise dpo_pair_logratios gives per-pair summed policy/ref log-probs and
    # completion lengths ([n_valid, 2], index 0 chosen, 1 rejected). Logits: for "ipo"
    # the difference of length-normalized (per-token average, length clamped >=1) chosen
    # and rejected log-ratios; otherwise (chosen - rejected policy) minus (chosen -
    # rejected ref). Per-pair loss from dpo_preference_loss(logits, beta, loss_type).
    # Under no_grad logs n_pairs denominator and loss, chosen/rejected implicit rewards
    # (beta * log-ratio, unnormalized even for IPO), reward accuracy and margin. Returns
    # the mean per-pair loss (scalar).
    device = logprobs.device
    cu_seqlens = input_["cu_seqlens"].to(device=device, dtype=torch.long)
    loss_mask = input_["loss_mask"].bool().to(device=device)
    ref_logprobs = input_["ref_logprobs"].to(device)

    valid_pairs = _dpo_valid_pairs(input_).to(device=device)
    if not valid_pairs.any():
        _log_empty_dpo_stats(device)
        return torch.zeros((), dtype=torch.float32, device=device)

    policy_logps, ref_logps, completion_lens = dpo_pair_logratios(
        logprobs, ref_logprobs, cu_seqlens, loss_mask, valid_pairs
    )

    if loss_type == "ipo":
        # IPO (Azar et al. 2023): normalize per-sequence logratios by
        # completion length (per-token average) before computing the squared
        # loss. This matches trl's author-confirmed convention so that beta
        # is comparable across variable-length sequences.
        chosen_avg = (policy_logps[:, 0] - ref_logps[:, 0]) / completion_lens[
            :, 0
        ].clamp(min=1)
        rejected_avg = (policy_logps[:, 1] - ref_logps[:, 1]) / completion_lens[
            :, 1
        ].clamp(min=1)
        logits = chosen_avg - rejected_avg
    else:
        logits = (policy_logps[:, 0] - policy_logps[:, 1]) - (
            ref_logps[:, 0] - ref_logps[:, 1]
        )

    per_pair_loss = dpo_preference_loss(logits, beta=beta, loss_type=loss_type)

    with torch.no_grad():
        chosen_rewards = beta * (policy_logps[:, 0] - ref_logps[:, 0]).float()
        rejected_rewards = beta * (policy_logps[:, 1] - ref_logps[:, 1]).float()
        stats_tracker.denominator(
            n_pairs=torch.ones(
                chosen_rewards.shape[0], dtype=torch.bool, device=device
            ),
        )
        stats_tracker.stat(
            loss=per_pair_loss.detach().float(),
            chosen_reward=chosen_rewards,
            rejected_reward=rejected_rewards,
            reward_accuracy=(chosen_rewards > rejected_rewards).float(),
            reward_margin=(chosen_rewards - rejected_rewards).float(),
            denominator="n_pairs",
        )

    return per_pair_loss.mean()
