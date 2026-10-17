# [MODULE DOC]
# CPU-only fake TrainEngine used by training-service tests in place of a real
# distributed engine. It performs no computation: methods record calls/state in private
# counters and return deterministic values derived from numeric inputs, so RPC plumbing
# can be verified end to end.
"""CPU-only fake TrainEngine for training-service integration tests."""

from __future__ import annotations

from typing import Any

from areal.api import TrainEngine


def _sum_numbers(value: Any) -> float:
    # [FUNCTION DOC]
    # HELPER: Recursively sums all numeric leaves in nested dicts, lists and tuples
    # (bools
    # counted as 0/1); any other value contributes 0.0. Returns a float.
    if isinstance(value, dict):
        return sum(_sum_numbers(v) for v in value.values())
    if isinstance(value, list):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, tuple):
        return sum(_sum_numbers(v) for v in value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


class FakeTrainEngine(TrainEngine):
    # [CLASS DOC]
    # Minimal concrete TrainEngine. Tracks initialized flag, weight version, train mode,
    # offload flag, call counters for zero_grad/optimizer step/lr step, last
    # saved/loaded
    # meta and constructor kwargs; reports them via export_stats.
    """Minimal concrete TrainEngine used by integration tests."""

    def __init__(self, *args: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Stores kwargs (e.g. world_size) and resets all tracked state. Positional args
        # are
        # ignored.
        # KNOWN ISSUE: does not call the TrainEngine base initializer.
        self._initialized = False
        self._version = 0
        self._train_mode = True
        self._offloaded = False
        self._zero_grad_calls = 0
        self._optimizer_step_calls = 0
        self._lr_step_calls = 0
        self._last_saved_meta: Any = None
        self._last_loaded_meta: Any = None
        self._init_kwargs = dict(kwargs)

    def create_process_group(self, parallel_strategy=None):
        # [FUNCTION DOC]
        # No-op; returns None (no process groups are created).
        return None

    def initialize(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Marks the engine initialized; ignores arguments; returns None.
        self._initialized = True
        return None

    @property
    def data_parallel_group(self):
        # [FUNCTION DOC]
        # Property: no data parallel group (None).
        return None

    @property
    def data_parallel_rank(self) -> int:
        # [FUNCTION DOC]
        # Property: data parallel rank is always 0.
        return 0

    @property
    def data_parallel_world_size(self) -> int:
        # [FUNCTION DOC]
        # Property: data parallel world size is always 1.
        return 1

    def current_data_parallel_head(self) -> int:
        # [FUNCTION DOC]
        # Always returns rank 0 as the DP head.
        return 0

    def is_data_parallel_head(self) -> bool:
        # [FUNCTION DOC]
        # Always True (single-process engine is the DP head).
        return True

    @property
    def context_and_model_parallel_group(self):
        # [FUNCTION DOC]
        # Property: no context/model parallel group (None).
        return None

    @property
    def cpu_group(self):
        # [FUNCTION DOC]
        # Property: no CPU process group (None).
        return None

    @property
    def initialized(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether initialize() was called.
        return self._initialized

    def train(self, mode: bool = True):
        # [FUNCTION DOC]
        # Records the train mode flag; returns None.
        self._train_mode = mode
        return None

    def update_weights(self, meta):
        # [FUNCTION DOC]
        # No-op weight update; returns None.
        return None

    def connect_engine(self, engine, meta):
        # [FUNCTION DOC]
        # No-op engine connection; returns None.
        return None

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns the input data unchanged (no rollout performed).
        return data

    def prepare_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn=None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns an empty list (no batch preparation).
        return []

    def set_version(self, version: int):
        # [FUNCTION DOC]
        # Stores the version as int.
        self._version = int(version)

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # Returns the stored version.
        return self._version

    def save(self, meta):
        # [FUNCTION DOC]
        # Records meta as the last saved meta (no I/O).
        self._last_saved_meta = meta

    def load(self, meta):
        # [FUNCTION DOC]
        # Records meta as the last loaded meta (no I/O).
        self._last_loaded_meta = meta

    def optimizer_zero_grad(self):
        # [FUNCTION DOC]
        # Increments the zero_grad call counter.
        self._zero_grad_calls += 1

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Increments the optimizer step counter and returns stats: update_successful
        # 1.0,
        # grad_norm equal to the step count, lr 1e-3.
        self._optimizer_step_calls += 1
        return {
            "update_successful": 1.0,
            "grad_norm": float(self._optimizer_step_calls),
            "lr": 1e-3,
        }

    def lr_scheduler_step(self):
        # [FUNCTION DOC]
        # Increments the LR scheduler step counter.
        self._lr_step_calls += 1

    def forward_backward_batch(
        self,
        mb_list,
        process_output_fn,
        forward_only: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # No-op; returns None.
        return None

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn=None,
        loss_weight_fn=None,
    ) -> dict[str, float]:
        # [FUNCTION DOC]
        # Returns a dict with total (sum of numeric leaves of input_), current version
        # and train
        # mode as floats; loss functions are ignored.
        return {
            "total": _sum_numbers(input_),
            "version": float(self._version),
            "train_mode": float(self._train_mode),
        }

    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn=None,
        loss_weight_fn=None,
    ) -> float:
        # [FUNCTION DOC]
        # Returns the sum of numeric leaves of input_ plus the current version.
        return _sum_numbers(input_) + self._version

    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn=None,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns total, version, train_mode and the given output_seqlens unchanged;
        # aggregate_fn
        # ignored.
        return {
            "total": _sum_numbers(input_),
            "version": self._version,
            "train_mode": self._train_mode,
            "output_seqlens": output_seqlens,
        }

    def train_lm(self, input_, **kwargs):
        # [FUNCTION DOC]
        # Ignores kwargs and delegates to train_batch.
        _ = kwargs
        return self.train_batch(input_)

    def evaluate_lm(self, input_, **kwargs):
        # [FUNCTION DOC]
        # Ignores kwargs and delegates to eval_batch.
        _ = kwargs
        return self.eval_batch(input_)

    def export_stats(self) -> dict[str, float]:
        # [FUNCTION DOC]
        # Returns float stats: version, train mode, offloaded flag, the three call
        # counters, the
        # string length of last saved/loaded meta, and world_size from constructor
        # kwargs (-1 if
        # absent).
        return {
            "version": float(self._version),
            "train_mode": float(self._train_mode),
            "offloaded": float(self._offloaded),
            "zero_grad_calls": float(self._zero_grad_calls),
            "optimizer_step_calls": float(self._optimizer_step_calls),
            "lr_step_calls": float(self._lr_step_calls),
            "saved_meta_size": float(len(str(self._last_saved_meta))),
            "loaded_meta_size": float(len(str(self._last_loaded_meta))),
            "world_size": float(self._init_kwargs.get("world_size", -1)),
        }

    def onload(self) -> None:
        # [FUNCTION DOC]
        # Clears the offloaded flag.
        self._offloaded = False

    def offload(self) -> None:
        # [FUNCTION DOC]
        # Sets the offloaded flag.
        self._offloaded = True

    def get_device_stats(self):
        # [FUNCTION DOC]
        # Returns a fixed dict reporting device cpu.
        return {"device": "cpu"}
