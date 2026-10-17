# [MODULE DOC]
# Supervised fine-tuning (SFT) glue. Defines LMEngine, which wraps a TrainEngine to
# train/evaluate with a packed next-token NLL loss, the controller classes that dispatch
# those calls (RPC TrainController and HTTP-gateway V2), and compute_packed_sft_loss.
# Batches are packed sequences described by "cu_seqlens" with a per-token "loss_mask".
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.api import TrainEngine
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import stats_tracker
from areal.utils.data import batched_call
from areal.utils.perf_tracer import trace_perf
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)


class LMEngine:
    # [CLASS DOC]
    # Worker-side SFT trainer that holds a TrainEngine and exposes train_lm /
    # evaluate_lm over lists of micro-batch dicts.
    def __init__(self, engine: TrainEngine):
        # [FUNCTION DOC]
        # Stores the wrapped TrainEngine in self.engine.
        self.engine = engine

    @trace_perf("lm_engine.train_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft")
    def train_lm(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Trains on each dict in data via batched_call (unpack=False) under the "sft"
        # stats scope and perf tracing. Returns None.
        batched_call(self._train_lm, data, unpack=False)

    def _train_lm(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Trains on one batch. Sets train mode, shifts loss_mask left by one
        # (converted to bool, roll along last dim) so it aligns with next-token
        # logprobs, mutating data in place, then calls train_batch with
        # compute_packed_sft_loss and loss weight = number of masked tokens, and records
        # the returned stats as scalars.
        # KNOWN ISSUE: torch.roll wraps the first mask entry to the last position
        # instead of zero-filling, so the final token of the packed batch may inherit a
        # stale mask value.
        self.engine.train()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        stats_tracker.scalar(**stats)

    @trace_perf("lm_engine.evaluate_lm", category="compute")
    @stats_tracker.scope_func_wrapper("sft-eval")
    def evaluate_lm(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Evaluates each dict in data via batched_call under the "sft-eval" stats scope.
        # Returns None.
        batched_call(self._evaluate_lm, data, unpack=False)

    def _evaluate_lm(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Evaluates one batch: eval mode, same in-place loss_mask roll as
        # _train_lm, then eval_batch with compute_packed_sft_loss and token-count loss
        # weight. Result discarded.
        self.engine.eval()
        data["loss_mask"] = torch.roll(data["loss_mask"].bool(), shifts=-1, dims=-1)
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_packed_sft_loss,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )


class LMController(TrainController):
    # [CLASS DOC]
    # RPC controller forwarding SFT calls to workers via _custom_function_call.
    def train_lm(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Remotely invokes "train_lm" with broadcast rpc_meta. Returns None.
        self._custom_function_call(
            "train_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_lm(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Pads eval args with group_size=1 for even DP dispatch, then remotely invokes
        # "evaluate_lm" with broadcast rpc_meta. Returns None.
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=1)
        self._custom_function_call(
            "evaluate_lm", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class LMControllerV2(GatewayTrainController):
    # [CLASS DOC]
    # HTTP-gateway controller variant posting SFT calls to the training service.
    def train_lm(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Serializes args (list) and kwargs and POSTs to "/sft/train". Returns None.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/train", payload)

    def evaluate_lm(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Serializes args (list) and kwargs and POSTs to "/sft/evaluate". Returns None.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/sft/evaluate", payload)


def compute_packed_sft_loss(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_: dict[str, Any],
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
    vocab_mean_logits: torch.Tensor | None = None,
    vocab_norm_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Packed SFT loss. Inputs: per-token logprobs and entropy (1D over packed tokens),
    # input_ with "cu_seqlens" and (already shifted) "loss_mask", and optional per-token
    # vocab logit statistics. Returns the mean negative logprob over masked tokens (sum
    # divided by masked count plus 1e-5). Side effects: logs denominators n_seqs
    # (sequences with any valid token), n_tokens, n_valid_tokens, prompt_tokens, and
    # stats ppl (exp of per-sequence mean NLL), per-token loss, masked entropy, and
    # vocab min/max and mean/norm logits when both of a pair are given. The per-sequence
    # loop calls .item() (GPU sync) once per sequence; sequences with no valid tokens
    # (padding dummies) are skipped. No extra CP reduction is done because inputs are
    # already CP-replicated (see docstring).
    """Compute SFT loss from logprobs.

    CP NOTE: MegatronEngine reassembles the CP-local per-token scalars
    (logprobs / entropy / vocab_*) into full sequences via
    ``reassemble_cp_packed_logprobs`` (a 1D all-gather over the CP group) *before*
    calling this loss. The vocab reduction along the V dim happens earlier, so the
    all-gather only moves per-token scalars, not the full logits — this is the
    OOM-avoiding CP path. Consequently the tensors here are already full-sequence
    and CP-replicated; the default DP-only reduce at export time yields the correct
    global values. We must NOT additionally reduce across CP (that would multiply
    sums by cp_size). See ``reassemble_cp_packed_logprobs`` and #1242.
    """
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    loss_mask = input_["loss_mask"].bool()

    logprobs = torch.where(loss_mask, logprobs, 0)

    device = logprobs.device
    loss = -logprobs.sum() / (1e-5 + loss_mask.count_nonzero())
    with torch.no_grad():
        batch_size = cu_seqlens.shape[0] - 1
        seqlogp = torch.zeros(batch_size, dtype=torch.float64, device=device)
        n_seqs = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            m = loss_mask[cu_seqlens[i] : cu_seqlens[i + 1]]
            logp = logprobs[cu_seqlens[i] : cu_seqlens[i + 1]]
            valid_tokens = int(m.count_nonzero().item())
            if valid_tokens == 0:
                # Padded dummy sequence created in `padded_mb_input`; skip it.
                continue
            n_seqs[i] = True
            seqlogp[i] = torch.where(m, logp.detach(), 0.0).sum() / valid_tokens

    ## Logging stats
    stats_tracker.denominator(
        n_seqs=n_seqs,
        n_tokens=torch.ones(logprobs.shape[0], dtype=torch.bool, device=device),
        n_valid_tokens=loss_mask,
        prompt_tokens=loss_mask.logical_not(),
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    stats_tracker.stat(
        entropy=torch.where(loss_mask, entropy.detach().float(), 0.0),
        denominator="n_valid_tokens",
    )

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    if vocab_mean_logits is not None and vocab_norm_logits is not None:
        stats_tracker.stat(
            vocab_mean_logits=vocab_mean_logits,
            vocab_norm_logits=vocab_norm_logits,
            denominator="n_tokens",
        )

    return loss
