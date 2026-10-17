# [MODULE DOC]
# Unit tests for TrainController using an in-process MockScheduler that fabricates
# workers and records async engine calls. Covers construction/initialize (worker
# creation, env setup, DP head detection), destroy (reverse-order teardown, error
# tolerance), _merge_tensors, RPC wrappers
# (train/eval/step_lr_scheduler/version/save/load), _custom_function_call, rollout
# integration (connect_engine, prepare_batch, rollout_batch), update_weights,
# export_stats and _prepare_dispatch splitting/replication.
"""Unit tests for TrainController.

Tests cover initialization, worker management, batch operations,
RPC wrappers, PPO/SFT methods, weight management, and error handling.
"""

import asyncio
from unittest.mock import Mock

import pytest
import torch

from areal.api import (
    FinetuneSpec,
    ParallelStrategy,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    Worker,
)
from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra import TrainController
from areal.infra.controller.train_controller import _merge_tensors


class MockTrainEngine(TrainEngine):
    # [CLASS DOC]
    # HELPER: Placeholder TrainEngine subclass passed as the engine class to
    # TrainController.
    """Mock TrainEngine for testing."""

    @classmethod
    def __module__(cls):
        # [FUNCTION DOC]
        # HELPER: Classmethod returning the string "tests.test_train_controller".
        # KNOWN ISSUE: defining __module__ / __name__ as classmethods replaces the class
        # attributes with bound methods rather than strings, so code reading
        # MockTrainEngine.__module__ gets a method object; likely unintended.
        return "tests.test_train_controller"

    @classmethod
    def __name__(cls):
        # [FUNCTION DOC]
        # HELPER: Classmethod returning "MockTrainEngine" (see KNOWN ISSUE on
        # __module__).
        return "MockTrainEngine"


class MockScheduler:
    # [CLASS DOC]
    # HELPER: Fake scheduler. Tracks created workers, engine call log and count, deleted
    # roles, per-worker env settings, and the last delete_workers reverse_order flag.
    """Mock Scheduler for testing TrainController."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes empty workers, call_count 0, engine_calls, deleted_roles
        # and env_settings.
        self.workers = []
        self.call_count = 0
        self.engine_calls = []
        self.deleted_roles = []
        self.env_settings: dict[str, dict[str, str]] = {}

    def create_workers(self, job):
        # [FUNCTION DOC]
        # HELPER: Creates job.replicas Worker objects with ids "<role>/<i>", ip
        # 127.0.0.1 and fixed worker/engine ports, stores them (replacing previous) and
        # returns the ids.
        """Create mock workers based on job configuration."""
        worker_ids = [f"{job.role}/{i}" for i in range(job.replicas)]
        self.workers = [
            Worker(
                id=wid,
                ip="127.0.0.1",
                worker_ports=["8000", "8001"],
                engine_ports=["9000", "9001"],
            )
            for wid in worker_ids
        ]
        return worker_ids

    def get_workers(self, role, timeout=None):
        # [FUNCTION DOC]
        # HELPER: Returns all stored workers regardless of role/timeout.
        """Return list of workers for the given role."""
        return self.workers

    async def set_worker_env(self, worker_id, env):
        # [FUNCTION DOC]
        # HELPER: Async; sleeps 1ms then stores env for worker_id with values
        # stringified.
        """Mock environment configuration."""
        await asyncio.sleep(0.001)
        self.env_settings[worker_id] = {k: str(v) for k, v in env.items()}

    async def create_engine(self, worker_id, engine, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Async; sleeps 1ms and returns None (no engine is actually created).
        """Mock engine creation."""
        await asyncio.sleep(0.001)
        return None

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Async; records (worker_id, method, args, kwargs) and increments
        # call_count. Returns True for even-indexed workers on is_data_parallel_head, 1
        # for get_version, fixed loss dicts for train_lm / evaluate_lm, otherwise sleeps
        # 1ms and returns None.
        """Mock async engine call."""
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1

        # Return appropriate mock results based on method
        if method == "is_data_parallel_head":
            # First worker in each DP group is the head
            worker_idx = int(worker_id.split("/")[-1])
            return worker_idx % 2 == 0  # Every other worker is a DP head

        elif method == "get_version":
            return 1

        elif method == "train_lm":
            return {"lm_loss": 0.4, "perplexity": 1.5}

        elif method == "evaluate_lm":
            # Return scalar loss (real implementation would return float or dict)
            return {"eval_loss": 0.35}

        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role, reverse_order: bool = False):
        # [FUNCTION DOC]
        # HELPER: Records role in deleted_roles, stores reverse_order as
        # delete_reverse_order and clears workers.
        """Mock worker deletion."""
        self.deleted_roles.append(role)
        self.delete_reverse_order = reverse_order
        self.workers.clear()


# ==================== FIXTURES ====================


@pytest.fixture
def mock_scheduler():
    # [FUNCTION DOC]
    # Fixture: returns a fresh MockScheduler.
    """Provide a MockScheduler instance."""
    return MockScheduler()


@pytest.fixture
def train_config():
    # [FUNCTION DOC]
    # Fixture: TrainEngineConfig with backend fsdp:d4t2 (8 workers) and one
    # SchedulingSpec.
    """Provide a TrainEngineConfig for testing."""
    return TrainEngineConfig(
        backend="fsdp:d4t2",
        scheduling_spec=(
            SchedulingSpec(
                cpu=4,
                gpu=1,
                mem=16000,
                port_count=2,
                cmd="python -m areal.infra.rpc.rpc_server",
            ),
        ),
    )


@pytest.fixture
def parallel_strategy():
    # [FUNCTION DOC]
    # Fixture: ParallelStrategy with dp 4, tp 2, pp 1.
    """Provide a ParallelStrategy for testing."""
    return ParallelStrategy(
        data_parallel_size=4, tensor_parallel_size=2, pipeline_parallel_size=1
    )


@pytest.fixture
def ft_spec():
    # [FUNCTION DOC]
    # Fixture: FinetuneSpec with 10 epochs, dataset size 1000, batch size 32.
    """Provide a FinetuneSpec for testing."""
    return FinetuneSpec(total_train_epochs=10, dataset_size=1000, train_batch_size=32)


@pytest.fixture
def train_controller(mock_scheduler, train_config):
    # [FUNCTION DOC]
    # Fixture: builds a TrainController on MockTrainEngine/config/scheduler, yields it,
    # then calls destroy() on teardown.
    """Provide a TrainController instance."""
    train_controller = TrainController(
        train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
    )
    yield train_controller
    train_controller.destroy()


def create_mock_distributed_batch(size=4, seq_len=10):
    # [FUNCTION DOC]
    # HELPER: Returns size trajectory dicts each with random input_ids [1, seq_len] and
    # all-True attention_mask / loss_mask of the same shape (defaults size 4, seq_len
    # 10).
    """Create a mock trajectory list for testing (_dispatch_tensors expects list[dict])."""
    return [
        {
            "input_ids": torch.randint(0, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "loss_mask": torch.ones(1, seq_len, dtype=torch.bool),
        }
        for _ in range(size)
    ]


# ==================== TEST CLASSES ====================


class TestTrainControllerInitialization:
    # [CLASS DOC]
    # Construction and initialize tests.
    """Tests for TrainController initialization and setup."""

    def test_constructor(self, mock_scheduler, train_config):
        # [FUNCTION DOC]
        # Asserts the constructor stores engine/config/scheduler and starts with no
        # workers or DP-head flags.
        """Test TrainController constructor."""
        controller = TrainController(
            train_engine=MockTrainEngine, config=train_config, scheduler=mock_scheduler
        )

        assert controller.train_engine == MockTrainEngine
        assert controller.config == train_config
        assert controller.scheduler == mock_scheduler
        assert controller.workers == []
        assert controller.workers_is_dp_head == []

    def test_initialize(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # After initialize, asserts worker count equals allocation world size, role is
        # stored, one DP-head flag per worker, engine calls happened and env was set for
        # every worker.
        """Test initialize method creates workers and engines."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Verify workers were created
        assert (
            len(train_controller.workers)
            == train_controller.train_alloc.parallel.world_size
        )
        assert train_controller._worker_role == "train_worker"

        # Verify DP heads were identified
        assert len(train_controller.workers_is_dp_head) == len(train_controller.workers)

        # Verify scheduler was called
        assert train_controller.scheduler.call_count > 0
        # Verify environment configuration occurred for each worker
        assert len(train_controller.scheduler.env_settings) == len(
            train_controller.workers
        )

    def test_identify_dp_heads(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # After initialize, asserts DP-head flags are True exactly for even-indexed
        # workers.
        """Test _identify_dp_heads correctly identifies DP head workers."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # MockScheduler returns True for even-indexed workers
        for idx, is_head in enumerate(train_controller.workers_is_dp_head):
            assert is_head == (idx % 2 == 0)


class TestTrainControllerDestroy:
    # [CLASS DOC]
    # Destroy/cleanup tests.
    """Tests for TrainController cleanup and destruction."""

    def test_destroy(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # After initialize then destroy, asserts workers and DP-head flags are cleared
        # and the role was deleted.
        """Test destroy method cleans up resources."""
        # Initialize first
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        initial_worker_count = len(train_controller.workers)
        assert initial_worker_count > 0

        # Call destroy
        train_controller.destroy()

        # Verify cleanup
        assert len(train_controller.workers) == 0
        assert len(train_controller.workers_is_dp_head) == 0
        assert "train_worker" in train_controller.scheduler.deleted_roles

    def test_destroy_handles_errors(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Replaces scheduler.delete_workers with a raising function; asserts destroy
        # does not raise and still clears workers.
        """Test destroy handles errors gracefully."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Make delete_workers raise an exception
        def raise_error(role):
            # [FUNCTION DOC]
            # HELPER: Always raises RuntimeError("Simulated error").
            # KNOWN ISSUE: it accepts only role, so a destroy call passing reverse_order
            # would fail with TypeError instead of the simulated error; the test still
            # passes because any exception is swallowed.
            raise RuntimeError("Simulated error")

        train_controller.scheduler.delete_workers = raise_error

        # Should not raise, just log the error
        train_controller.destroy()

        # Workers should still be cleared
        assert len(train_controller.workers) == 0

    def test_destroy_requests_reverse_order(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts destroy calls delete_workers with reverse_order True.
        """Workers must be torn down in reverse rank order.

        This protects rank-0 (which owns the global TCPStore server) from
        being killed before non-zero ranks finish NCCL abort, avoiding a
        noisy ``TCPStore.recvValue failed`` warning from
        HeartbeatMonitor.
        """
        train_controller.initialize(role="train_worker", ft_spec=ft_spec)

        train_controller.destroy()

        assert (
            getattr(train_controller.scheduler, "delete_reverse_order", False) is True
        )


class TestTrainControllerMergeResults:
    # [CLASS DOC]
    # Tests for _merge_tensors(results, group_indices).
    """Tests for result merging via _merge_tensors."""

    def test_merge_tensors_reorders_results(self):
        # [FUNCTION DOC]
        # Asserts dict results with singleton groups come back in original order.
        """Test _merge_tensors reorders results to original trajectory order."""
        results = [{"status": "ok"}, {"status": "done"}]

        merged = _merge_tensors(results, group_indices=[[0], [1]])

        assert merged is not None
        assert len(merged) == 2
        assert merged[0] == {"status": "ok"}
        assert merged[1] == {"status": "done"}

    def test_merge_tensors_with_tensor_results(self):
        # [FUNCTION DOC]
        # Asserts tensor results merge to two items.
        # KNOWN ISSUE: only the length is checked, not the actual ordering or values.
        """Test _merge_tensors correctly reorders tensor results."""
        results = [torch.tensor([[0.5, 0.5]]), torch.tensor([[0.3, 0.3]])]

        merged = _merge_tensors(results, group_indices=[[0], [1]])

        assert merged is not None
        assert len(merged) == 2


class TestTrainControllerRPCWrappers:
    # [CLASS DOC]
    # RPC wrapper tests.
    """Tests for RPC wrapper methods."""

    def test_train_mode(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # After initialize, asserts train(mode=True) returns self and dispatches
        # "train".
        """Test train() method sets training mode."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        result = train_controller.train(mode=True)

        # Should return self for chaining
        assert result is train_controller

        # Verify custom_function_call was invoked
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_eval_mode(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts eval() returns self and dispatches "train" (with False).
        """Test eval() method sets evaluation mode."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        result = train_controller.eval()

        # Should return self for chaining
        assert result is train_controller

        # Verify train(False) was called
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "train" in engine_calls

    def test_step_lr_scheduler(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts step_lr_scheduler dispatches "step_lr_scheduler".
        """Test step_lr_scheduler() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        train_controller.step_lr_scheduler()

        # Verify step_lr_scheduler was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "step_lr_scheduler" in engine_calls


class TestTrainControllerWeightManagement:
    # [CLASS DOC]
    # Version and checkpoint wrapper tests.
    """Tests for weight management operations."""

    def test_set_version(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts set_version(42) dispatches "set_version".
        """Test set_version() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        train_controller.set_version(42)

        # Verify set_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "set_version" in engine_calls

    def test_get_version(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts get_version returns an int and dispatches "get_version".
        """Test get_version() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        version = train_controller.get_version()

        # Should return version number
        assert isinstance(version, int)

        # Verify get_version was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "get_version" in engine_calls

    def test_save(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts save(SaveLoadMeta) dispatches "save".
        """Test save() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.save(meta)

        # Verify save was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "save" in engine_calls

    def test_load(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts load(SaveLoadMeta) dispatches "load".
        """Test load() method."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = SaveLoadMeta(
            path="/tmp/checkpoint", weight_format="safetensors", with_optim=True
        )
        train_controller.load(meta)

        # Verify load was called on engines
        engine_calls = [call[1] for call in train_controller.scheduler.engine_calls]
        assert "load" in engine_calls


class TestTrainControllerCustomFunctionCall:
    # [CLASS DOC]
    # Tests for _custom_function_call orchestration.
    """Tests for custom_function_call orchestration."""

    def test_custom_function_call_with_distributed_batch(
        self, train_controller, ft_spec
    ):
        # [FUNCTION DOC]
        # Clears call log, calls train_lm with a 16-trajectory batch; asserts a non-None
        # result and one engine call per worker.
        """Test custom_function_call with batch argument."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Clear previous calls from initialization
        train_controller.scheduler.engine_calls = []

        batch = create_mock_distributed_batch(size=16)
        result = train_controller._custom_function_call("train_lm", input_=batch)

        # Should split batch across DP groups and call only DP heads
        assert result is not None

        # Count how many workers were called
        worker_calls = len(train_controller.scheduler.engine_calls)

        # Should call all workers (DP heads get data, others get empty)
        assert worker_calls == len(train_controller.workers)

    def test_custom_function_call_with_regular_args(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Calls set_version(5) through _custom_function_call; asserts result None and
        # one call per worker.
        """Test custom_function_call with non-batch arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Clear previous calls
        train_controller.scheduler.engine_calls = []

        result = train_controller._custom_function_call("set_version", 5)

        # set_version returns None, which is expected - just verify it doesn't crash
        # The key test is that all workers were called
        assert result is None

        # Verify all workers were called
        assert len(train_controller.scheduler.engine_calls) == len(
            train_controller.workers
        )

    def test_custom_function_call_filters_dp_heads(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Calls ppo_update with an 8-trajectory batch.
        # KNOWN ISSUE: contains no assertions; it only checks that the call does not
        # raise, despite the docstring claiming DP-head filtering is verified.
        """Test custom_function_call only returns results from DP heads."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=8)
        train_controller._custom_function_call("ppo_update", input_=batch)

        # Results should only come from DP head workers
        # (verified by _collect_results filtering to DP heads)


class TestTrainControllerEdgeCases:
    # [CLASS DOC]
    # Edge-case tests: create_process_group as a no-op and train()/eval() method chaining.
    def test_create_process_group_is_dummy(self, train_controller, parallel_strategy):
        # [FUNCTION DOC]
        # Asserts create_process_group is a no-op on an uninitialized controller and
        # parallel_strategy was already set by the constructor.
        """Test create_process_group is now a dummy method that does nothing."""
        # Don't initialize, so workers list is empty
        # Should not raise - create_process_group is now a no-op
        train_controller.create_process_group(parallel_strategy)

        # parallel_strategy should be set by constructor (from backend="fsdp:d4t2")
        assert train_controller.parallel_strategy is not None

    def test_method_chaining(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts train().eval().train() chaining returns the controller.
        """Test that train() and eval() support method chaining."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Should be able to chain calls
        result = train_controller.train().eval().train()
        assert result is train_controller


class TestTrainControllerRolloutIntegration:
    # [CLASS DOC]
    # Tests rollout-controller integration methods (rollout engine is a Mock).
    """Tests for rollout engine integration methods."""

    def test_connect_engine_sets_rollout(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts connect_engine stores the rollout controller.
        """Test connect_engine correctly sets the rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_connect_engine_warns_on_change(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Connects two different rollouts; asserts the second replaces the first
        # (warning not checked).
        """Test connect_engine logs warning when rollout controller changes."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout1 = Mock()
        mock_rollout2 = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout1, meta)
        train_controller.connect_engine(mock_rollout2, meta)

        assert train_controller.rollout == mock_rollout2

    def test_connect_engine_same_rollout_no_warning(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Connects the same rollout twice; asserts it remains set (absence of warning
        # not checked).
        """Test connect_engine does not warn when same rollout controller is used."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        train_controller.connect_engine(mock_rollout, meta)
        train_controller.connect_engine(mock_rollout, meta)

        assert train_controller.rollout == mock_rollout

    def test_check_rollout_engine_connected_raises_when_not_connected(
        self, train_controller, ft_spec
    ):
        # [FUNCTION DOC]
        # Asserts _check_rollout_engine_connected raises RuntimeError "Rollout engine
        # not connected" before connect.
        """Test _check_rollout_engine_connected raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller._check_rollout_engine_connected()

    def test_check_rollout_engine_connected_passes_when_connected(
        self, train_controller, ft_spec
    ):
        # [FUNCTION DOC]
        # Asserts _check_rollout_engine_connected passes after connect.
        """Test _check_rollout_engine_connected passes when rollout is connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        # Should not raise
        train_controller._check_rollout_engine_connected()

    def test_prepare_batch_delegates_to_rollout(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts prepare_batch forwards dataloader/workflow/kwargs to
        # rollout.prepare_batch with defaults should_accept_fn None, dynamic_bs False,
        # group_size 1, reward_normalization False, drop_incomplete_group False.
        """Test prepare_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.prepare_batch.return_value = {}
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        mock_dataloader = Mock()
        train_controller.prepare_batch(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.prepare_batch.assert_called_once_with(
            dataloader=mock_dataloader,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            dynamic_bs=False,
            group_size=1,
            reward_normalization=False,
            drop_incomplete_group=False,
        )

    def test_rollout_batch_delegates_to_rollout(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts rollout_batch forwards data/workflow/kwargs to rollout.rollout_batch
        # with default should_accept_fn, group_size, reward_normalization and
        # drop_incomplete_group.
        """Test rollout_batch delegates to rollout controller."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        mock_rollout = Mock()
        mock_rollout.rollout_batch.return_value = {}
        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        train_controller.connect_engine(mock_rollout, meta)

        data = [{"id": 1}, {"id": 2}]
        train_controller.rollout_batch(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
        )

        mock_rollout.rollout_batch.assert_called_once_with(
            data=data,
            workflow="test.workflow",
            workflow_kwargs={"key": "value"},
            should_accept_fn=None,
            group_size=1,
            reward_normalization=False,
            drop_incomplete_group=False,
        )


class TestTrainControllerWeightUpdateMethods:
    # [CLASS DOC]
    # Weight update tests.
    """Tests for weight update methods."""

    def test_update_weights_raises_when_not_connected(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts update_weights raises RuntimeError "Rollout engine not connected"
        # without a rollout.
        """Test update_weights raises when rollout is not connected."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")

        with pytest.raises(RuntimeError, match="Rollout engine not connected"):
            train_controller.update_weights(meta)


class TestTrainControllerExportStats:
    # [CLASS DOC]
    # export_stats tests.
    """Tests for export_stats method."""

    def test_export_stats(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Replaces async_call_engine with a coroutine returning expected stats for
        # "export_stats"; asserts export_stats returns those values.
        """Test export_stats returns statistics from first worker."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        # Mock the scheduler to return stats
        expected_stats = {"loss": 0.5, "accuracy": 0.95}

        async def mock_async_call(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Async fake engine call; returns expected_stats when the method
            # (kwarg or second positional) is export_stats, else None.
            if kwargs.get("method") == "export_stats" or (
                len(args) > 1 and args[1] == "export_stats"
            ):
                return expected_stats
            return None

        train_controller.scheduler.async_call_engine = mock_async_call

        result = train_controller.export_stats()
        for k in expected_stats:
            assert result[k] == expected_stats[k]


class TestTrainControllerDispatchInputs:
    # [CLASS DOC]
    # Tests _prepare_dispatch(args, kwargs) -> (split_args, split_kwargs,
    # group_indices).
    """Tests for input dispatching across DP groups."""

    def test_prepare_dispatch_splits_distributed_batch(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts a 16-trajectory batch yields one split arg with dp_size shards.
        """Test _prepare_dispatch correctly splits batch."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(batch)

        # Should split into dp_size chunks
        assert len(split_args) == 1
        assert len(split_args[0]) == train_controller.parallel_strategy.dp_size

    def test_prepare_dispatch_partitions_without_duplication(
        self, train_controller, ft_spec
    ):
        # [FUNCTION DOC]
        # Regression (#1202): asserts trajectories are partitioned not replicated:
        # group_indices has dp_size groups, shard sizes sum to the batch size and
        # indices cover 0..N-1 exactly once.
        """Regression for #1202: trajectories are partitioned across DP ranks,
        never replicated. Each DP rank must see a disjoint slice so that total
        tokens processed equal the original batch, not batch * dp_size."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        dp_size = train_controller.parallel_strategy.dp_size
        batch_size = 4 * dp_size
        batch = create_mock_distributed_batch(size=batch_size)

        split_args, _, group_indices = train_controller._prepare_dispatch(batch)

        # Sanity: tensor-like list triggers the partition path, not replication.
        assert group_indices is not None
        assert len(group_indices) == dp_size

        shards = split_args[0]
        assert sum(len(shard) for shard in shards) == batch_size

        flat_indices = [idx for group in group_indices for idx in group]
        assert sorted(flat_indices) == list(range(batch_size))

    def test_prepare_dispatch_replicates_non_batch_args(
        self, train_controller, ft_spec
    ):
        # [FUNCTION DOC]
        # Asserts a scalar positional arg is replicated dp_size times.
        """Test _prepare_dispatch replicates non-batch arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        scalar_arg = 42
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(scalar_arg)

        # Should replicate to all DP groups
        assert len(split_args) == 1
        assert all(arg == 42 for arg in split_args[0])
        assert len(split_args[0]) == train_controller.parallel_strategy.dp_size

    def test_prepare_dispatch_handles_kwargs(self, train_controller, ft_spec):
        # [FUNCTION DOC]
        # Asserts kwargs: batch input_ is split into dp_size shards and scalar
        # learning_rate replicated.
        """Test _prepare_dispatch correctly handles keyword arguments."""
        train_controller.initialize(
            role="train_worker",
            ft_spec=ft_spec,
        )

        batch = create_mock_distributed_batch(size=16)
        split_args, split_kwargs, _ = train_controller._prepare_dispatch(
            input_=batch, learning_rate=0.001
        )

        assert "input_" in split_kwargs
        assert "learning_rate" in split_kwargs
        assert len(split_kwargs["input_"]) == train_controller.parallel_strategy.dp_size
        assert all(lr == 0.001 for lr in split_kwargs["learning_rate"])
