# [MODULE DOC]
# Reward-model (RW) training glue. Provides the worker-side RWEngine that wraps a
# generic TrainEngine to train/evaluate a pairwise (Bradley-Terry) reward model, the
# controller classes that dispatch those calls to remote workers (RPC TrainController
# and the HTTP-gateway V2 controller), and the pairwise loss compute_rw_loss. Input
# batches are packed sequences described by "cu_seqlens" where consecutive sequence
# pairs are (chosen, rejected).
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from areal.api import TrainEngine
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.utils import logging, stats_tracker
from areal.utils.data import batched_call
from areal.utils.perf_tracer import trace_perf
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

logger = logging.getLogger("RWEngine")


def _rw_valid_pairs(x: dict[str, Any]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns a bool tensor with one entry per (chosen, rejected) pair, True
    # when both sequences of the pair have nonzero length. Computes lengths from
    # x["cu_seqlens"] diffs and reshapes to [n_pairs, 2]. Assumes an even number of
    # sequences (view errors otherwise).
    seqlens = x["cu_seqlens"][1:] - x["cu_seqlens"][:-1]
    return seqlens.view(-1, 2).ne(0).all(dim=1)


def _rw_loss_weight(x: dict[str, Any]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Loss weight for a batch: number of valid pairs as a float scalar tensor.
    return _rw_valid_pairs(x).count_nonzero().float()


def _log_empty_rw_stats(device: torch.device) -> None:
    # [FUNCTION DOC]
    # HELPER: Logs zero-valued RW stats (correct_ratio, pos_score, neg_score, loss)
    # against a one-element all-False "n_pairs" denominator on the given device, so that
    # stats keys stay consistent across ranks even when a batch contains no valid pairs.
    n_pairs = torch.zeros(1, dtype=torch.bool, device=device)
    stats_tracker.denominator(n_pairs=n_pairs)
    stats_tracker.stat(
        correct_ratio=torch.zeros(1, dtype=torch.float32, device=device),
        pos_score=torch.zeros(1, dtype=torch.float32, device=device),
        neg_score=torch.zeros(1, dtype=torch.float32, device=device),
        loss=torch.zeros(1, dtype=torch.float32, device=device),
        denominator="n_pairs",
    )


class RWEngine:
    # [CLASS DOC]
    # Worker-side reward-model trainer. Holds a TrainEngine (composition) and exposes
    # train_rw / evaluate_rw that run over lists of micro-batch dicts using
    # compute_rw_loss.
    def __init__(self, engine: TrainEngine):
        # [FUNCTION DOC]
        # Stores the wrapped TrainEngine in self.engine. No other side effects.
        self.engine = engine

    @trace_perf("rw_engine.train_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw")
    def train_rw(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Trains on each dict in data via batched_call (unpack=False, so each dict is
        # passed as a single argument to _train_rw). Wrapped by perf tracing and a "rw"
        # stats_tracker scope. Returns None; side effects are model updates and stat
        # logging.
        batched_call(self._train_rw, data, unpack=False)

    def _train_rw(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Trains on one batch. If it has zero valid pairs, first logs empty RW
        # stats. Puts the engine in train mode, calls train_batch with compute_rw_loss
        # and _rw_loss_weight, then records the returned stats dict as scalars in
        # stats_tracker.
        """Train on a batch (reward model)."""
        if _rw_loss_weight(data) == 0:
            _log_empty_rw_stats(data["cu_seqlens"].device)
        self.engine.train()
        stats = self.engine.train_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=_rw_loss_weight,
        )
        stats_tracker.scalar(**stats)

    @trace_perf("rw_engine.evaluate_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw-eval")
    def evaluate_rw(self, data: list[dict[str, Any]]) -> None:
        # [FUNCTION DOC]
        # Evaluates each dict in data via batched_call under the "rw-eval" stats scope
        # and perf tracing. Returns None; stats are logged as a side effect.
        batched_call(self._evaluate_rw, data, unpack=False)

    def _evaluate_rw(self, data: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # HELPER: Evaluates one batch: logs empty stats if no valid pairs, sets eval
        # mode and calls eval_batch with compute_rw_loss and _rw_loss_weight. Return
        # value of eval_batch discarded.
        if _rw_loss_weight(data) == 0:
            _log_empty_rw_stats(data["cu_seqlens"].device)
        self.engine.eval()
        self.engine.eval_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=_rw_loss_weight,
        )


class RWController(TrainController):
    # [CLASS DOC]
    # RPC-based controller that forwards reward-model calls to TrainEngine workers
    # through TrainController._custom_function_call.
    def train_rw(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Remotely invokes "train_rw" on workers with the given args/kwargs, with
        # rpc_meta broadcast=True. Returns None (result discarded).
        self._custom_function_call(
            "train_rw", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def evaluate_rw(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Pads eval dispatch args so they split evenly with group_size=2 (keeping
        # chosen/rejected pairs together on one DP rank), then remotely invokes
        # "evaluate_rw" with group_size=2 and broadcast rpc_meta. Returns None.
        # rw_modeling_collate_fn produces 2 sequences (chosen + rejected) per
        # example; group_size=2 keeps each pair on the same DP rank.
        args, kwargs = self._pad_eval_dispatch_args(args, kwargs, group_size=2)
        self._custom_function_call(
            "evaluate_rw",
            *args,
            group_size=2,
            rpc_meta={"broadcast": True},
            **kwargs,
        )


class RWControllerV2(GatewayTrainController):
    # [CLASS DOC]
    # Gateway (HTTP) based controller variant that posts reward-model calls to a
    # training service gateway.
    def train_rw(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Serializes args (as a list) and kwargs with serialize_value and POSTs them to
        # "/rw/train" via _gateway_post_result. Returns None.
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/rw/train", payload)

    def evaluate_rw(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Copies kwargs, defaults "group_size" to 2, serializes args/kwargs and POSTs to
        # "/rw/evaluate". Returns None. Unlike RWController it does not pad dispatch
        # args.
        kwargs = dict(kwargs)
        kwargs.setdefault("group_size", 2)
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post_result("/rw/evaluate", payload)


def compute_rw_loss(scores: torch.Tensor, input_: dict[str, Any]) -> torch.Tensor:
    # [FUNCTION DOC]
    # Pairwise reward-model loss. Inputs: scores (per-token scalar scores of the packed
    # batch, first dim equal to total tokens) and input_ with "cu_seqlens". Takes the
    # score at the last token of each sequence, reshapes into [n_pairs, 2] (chosen,
    # rejected), keeps valid pairs, and returns mean of -logsigmoid(chosen - rejected)
    # as a float scalar. If no valid pair exists, logs empty stats and returns a zero
    # scalar. Side effect: logs n_pairs denominator and correct_ratio, pos_score,
    # neg_score, per-pair loss to stats_tracker. Raises AssertionError if scores length
    # does not equal total sequence length.
    device = scores.device
    cu_seqlens = input_["cu_seqlens"]
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    valid_pairs = _rw_valid_pairs(input_)
    if not valid_pairs.any():
        _log_empty_rw_stats(device)
        return torch.zeros((), dtype=torch.float32, device=device)

    valid_pair_mask = valid_pairs.to(device=device)
    terminal_indices = seqlens.cumsum(0).to(device=device) - 1

    assert scores.shape[0] == seqlens.sum(), (scores.shape, seqlens.sum())
    scores = scores[terminal_indices].view(-1, 2)[valid_pair_mask].float()

    loss = -(torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]))
    logging_loss = loss.detach()
    loss = loss.mean()

    # Logging.
    with torch.no_grad():
        stats_tracker.denominator(
            n_pairs=torch.ones(scores.shape[0], dtype=torch.bool, device=device),
        )
        stats_tracker.stat(
            correct_ratio=(scores[:, 0] > scores[:, 1]).detach().float(),
            pos_score=scores[:, 0].detach().float(),
            neg_score=scores[:, 1].detach().float(),
            loss=logging_loss.float(),
            denominator="n_pairs",
        )
    return loss
