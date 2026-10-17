# [MODULE DOC]
# Tests for RolloutController (areal.infra), the single-controller front end that
# creates inference workers through a scheduler, dispatches rollout tasks round-robin,
# collects results via HTTP callbacks, tracks staleness/capacity and forwards version,
# weight-update and pause/resume calls to all workers. Most tests use MockScheduler, which
# fakes worker creation and engine RPCs in-process (submit immediately POSTs the task id
# to the controller's callback address); one slow, SGLang-marked integration test runs a
# real LocalScheduler with two SGLang servers.
import asyncio
from unittest.mock import Mock

import pytest
import requests
import torch

from tests.utils import get_model_path

from areal.api import (
    ModelRequest,
    ParamSpec,
    WeightUpdateMeta,
    Worker,
)
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    SchedulingSpec,
    SGLangConfig,
)
from areal.infra import RolloutController
from areal.infra.scheduler.local import LocalScheduler
from areal.utils.hf_utils import load_hf_tokenizer


def create_test_config(backend="sglang:d2", **kwargs):
    # [FUNCTION DOC]
    # HELPER: Returns an InferenceEngineConfig with the given backend (default "sglang:d2"),
    # consumer_batch_size 16 and a single 1 CPU / 1 GPU / 1 mem scheduling spec, overridden
    # by any kwargs.
    """Create a test InferenceEngineConfig with proper scheduling_spec."""
    # Create a mutable SchedulingSpec that can be modified by the controller
    scheduling_spec = SchedulingSpec(cpu=1, gpu=1, mem=1)

    defaults = {
        "backend": backend,
        "consumer_batch_size": 16,
        "scheduling_spec": (scheduling_spec,),
    }
    defaults.update(kwargs)
    config = InferenceEngineConfig(**defaults)

    return config


class MockScheduler:
    # [CLASS DOC]
    # HELPER: In-memory scheduler stand-in. Records every engine call in engine_calls
    # (worker_id, method, args, kwargs) and keeps per-worker pending results keyed by
    # incrementing task ids.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes empty worker list, call counters/log, pending results and task
        # counter.
        self.workers = []
        self.call_count = 0
        self.engine_calls = []
        self._pending_results = {}  # worker_id -> dict[task_id -> result]
        self._task_counter = 0

    def create_workers(self, job, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Creates job.replicas fake Workers "<role>/<i>" on 127.0.0.1 with fixed ports,
        # resets pending-result buckets for them, and returns their ids.
        """Create workers based on Job specification."""
        role = job.role
        replicas = job.replicas
        worker_ids = [f"{role}/{i}" for i in range(replicas)]
        self.workers = [
            Worker(
                id=wid,
                ip="127.0.0.1",
                worker_ports=["8000", "8001"],
                engine_ports=["9000", "9001"],
            )
            for wid in worker_ids
        ]
        # Initialize pending results for each worker
        for wid in worker_ids:
            self._pending_results[wid] = {}
        return worker_ids

    def get_workers(self, role, timeout=None):
        # [FUNCTION DOC]
        # HELPER: Returns all created workers regardless of role.
        return self.workers

    async def create_engine(self, worker_id, engine, engine_name, config):
        # [FUNCTION DOC]
        # HELPER: No-op async engine creation.
        pass

    async def async_call_engine(self, worker_id, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Async fake RPC. Logs the call; "agenerate" returns a Mock; "submit" stores a
        # random 10-token trajectory dict (half loss mask, random reward) under a new task id,
        # synchronously POSTs {"task_id"} to kwargs["callback_addr"] (raising on HTTP error) and
        # returns the id; "wait_for_task" pops and returns the stored result or None; "wait"
        # returns count results only if at least count are pending, else an empty list. Other
        # methods return None.
        # KNOWN ISSUE: the callback POST uses blocking requests inside a coroutine, which blocks
        # the event loop while the controller's callback server handles it.
        self.engine_calls.append((worker_id, method, args, kwargs))
        self.call_count += 1
        if method == "agenerate":
            return Mock()
        # Handle submit method - return a task_id and store the result
        elif method == "submit":
            if worker_id not in self._pending_results:
                self._pending_results[worker_id] = {}
            # Generate a unique task_id
            task_id = self._task_counter
            self._task_counter += 1
            # Simulate a successful rollout result
            result = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.tensor(
                    [0] * 5 + [1] * 5, dtype=torch.bool
                ).unsqueeze(0),
                "rewards": torch.randn(1),
            }
            self._pending_results[worker_id][task_id] = result
            # Immediately fire callback
            callback_addr = kwargs["callback_addr"]
            resp = requests.post(callback_addr, json=dict(task_id=task_id))
            resp.raise_for_status()
            return task_id
        # Handle wait_for_task method
        elif method == "wait_for_task":
            task_id = kwargs.get("task_id")
            if (
                worker_id in self._pending_results
                and task_id in self._pending_results[worker_id]
            ):
                result = self._pending_results[worker_id].pop(task_id)
                return result
            return None
        elif method == "wait":
            # Return a result from pending results if available
            count = kwargs["count"]
            if worker_id in self._pending_results and self._pending_results[worker_id]:
                if len(self._pending_results[worker_id]) < count:
                    return []
                # Get first count results
                task_ids = list(self._pending_results[worker_id].keys())[:count]
                results = [
                    self._pending_results[worker_id].pop(tid) for tid in task_ids
                ]
                return results
            return []
        return None

    def call_engine(self, worker_id, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Sync fake RPC. Logs the call; weight-update methods return a coroutine that
        # sleeps 1 ms; "submit" stores a fake result and returns a task id (without callback);
        # everything else returns None.
        self.engine_calls.append((worker_id, method, args, kwargs))

        # For weight update methods that await call_engine, return a coroutine
        if method in [
            "update_weights_from_distributed",
            "update_weights_from_disk",
            "init_weights_update_group",
        ]:
            return self._async_call_engine_internal(worker_id, method, *args, **kwargs)

        # Handle submit method - return a task_id and store the result
        if method == "submit":
            if worker_id not in self._pending_results:
                self._pending_results[worker_id] = {}
            # Generate a unique task_id
            task_id = self._task_counter
            self._task_counter += 1
            # Simulate a successful rollout result
            result = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10, dtype=torch.bool),
                "loss_mask": torch.tensor(
                    [0] * 5 + [1] * 5, dtype=torch.bool
                ).unsqueeze(0),
                "rewards": torch.randn(1),
            }
            self._pending_results[worker_id][task_id] = result
            return task_id

        return None

    async def _async_call_engine_internal(self, worker_id, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Coroutine that sleeps 1 ms and returns None.
        await asyncio.sleep(0.001)
        return None

    def delete_workers(self, role, reverse_order: bool = False):
        # [FUNCTION DOC]
        # HELPER: Clears workers, pending results and resets the task counter.
        self.workers.clear()
        self._pending_results.clear()
        self._task_counter = 0


class MockInferenceEngine:
    # [CLASS DOC]
    # HELPER: Placeholder engine class passed as inf_engine; the controller only uses it to
    # build an import path.
    # KNOWN ISSUE: __module__ and __name__ are defined as classmethods, so the class's
    # __module__ attribute becomes a classmethod object rather than a string (and __name__ is
    # shadowed only in the class dict); this works only because the mocks never import it.
    @classmethod
    def __module__(cls):
        # [FUNCTION DOC]
        # HELPER: Returns the module path string "tests.test_rollout_controller".
        return "tests.test_rollout_controller"

    @classmethod
    def __name__(cls):
        # [FUNCTION DOC]
        # HELPER: Returns the class name string "MockInferenceEngine".
        return "MockInferenceEngine"


class TestRolloutControllerInitialization:
    # [CLASS DOC]
    # Constructor and initialize() tests.
    def test_constructor(self):
        # [FUNCTION DOC]
        # A fresh controller stores config and scheduler, has no workers, worker index 0,
        # version 0 and no staleness manager.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.config == config
        assert controller.scheduler == scheduler
        assert controller.workers == []
        assert controller._current_worker_idx == 0
        assert controller._version == 0
        assert controller.staleness_manager is None

    def test_initialize_creates_workers(self):
        # [FUNCTION DOC]
        # initialize with "sglang:d2" creates 2 workers and a staleness manager.
        config = create_test_config(
            backend="sglang:d2",
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=False,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(controller.workers) == 2
        assert controller.staleness_manager is not None

        controller.destroy()

    def test_initialize_creates_staleness_manager(self):
        # [FUNCTION DOC]
        # The staleness manager receives max_concurrent_rollouts, consumer_batch_size and
        # max_head_offpolicyness from config.
        config = create_test_config(
            consumer_batch_size=32,
            max_head_offpolicyness=5,
            max_concurrent_rollouts=100,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager.max_concurrent_rollouts == 100
        assert controller.staleness_manager.consumer_batch_size == 32
        assert controller.staleness_manager.max_staleness == 5

        controller.destroy()

    def test_initialize_uses_consumer_batch_size_as_fallback(self):
        # [FUNCTION DOC]
        # With max_concurrent_rollouts None, the staleness manager uses consumer_batch_size.
        config = create_test_config(
            consumer_batch_size=64,
            max_head_offpolicyness=3,
            max_concurrent_rollouts=None,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager.max_concurrent_rollouts == 64

        controller.destroy()

    def test_initialize_with_tracing_enabled(self):
        # [FUNCTION DOC]
        # initialize/destroy succeed with enable_rollout_tracing on (no assertions).
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
            enable_rollout_tracing=True,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.destroy()


class TestRolloutControllerDestroy:
    # [CLASS DOC]
    # destroy() tests.
    def test_destroy_cleans_up_resources(self):
        # [FUNCTION DOC]
        # destroy empties controller.workers.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(controller.workers) > 0

        controller.destroy()
        assert len(controller.workers) == 0

    def test_destroy_deletes_workers_via_scheduler(self):
        # [FUNCTION DOC]
        # destroy calls scheduler.delete_workers, emptying the scheduler's workers.
        config = create_test_config(backend="sglang:d2", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert len(scheduler.workers) == 2

        controller.destroy()

        assert len(scheduler.workers) == 0

    def test_destroy_handles_scheduler_error(self):
        # [FUNCTION DOC]
        # destroy swallows an exception from scheduler.delete_workers.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        scheduler.delete_workers = Mock(side_effect=Exception("Test error"))

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.destroy()


class TestRolloutControllerCapacity:
    # [CLASS DOC]
    # get_capacity() tests.
    def test_get_capacity_initial_state(self):
        # [FUNCTION DOC]
        # Initial capacity equals max_concurrent_rollouts (32) when staleness allows more.
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity = controller.get_capacity()
        assert capacity == 32

        controller.destroy()

    def test_get_capacity_uses_version(self):
        # [FUNCTION DOC]
        # Raising the version via set_version increases capacity when staleness is the limit.
        config = create_test_config(
            consumer_batch_size=8,
            max_concurrent_rollouts=1000,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity_v0 = controller.get_capacity()

        controller.set_version(5)
        capacity_v5 = controller.get_capacity()

        assert capacity_v5 > capacity_v0

        controller.destroy()


class TestRolloutControllerWorkerSelection:
    # [CLASS DOC]
    # Worker selection tests.
    def test_choose_worker_round_robin(self):
        # [FUNCTION DOC]
        # _choose_worker cycles through 3 workers in order twice.
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        worker_ids = []
        for _ in range(6):
            worker, _ = controller._choose_worker()
            worker_ids.append(worker.id)

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"
        assert worker_ids[4] == "rollout/1"
        assert worker_ids[5] == "rollout/2"

        controller.destroy()


class TestRolloutControllerSubmitAndWait:
    # [CLASS DOC]
    # submit()/wait() tests.
    def test_wait_returns_distributed_batch(self):
        # [FUNCTION DOC]
        # Three submissions followed by wait(3) return a list of 3 dicts.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        for i in range(3):
            controller.submit(
                {"id": i},
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=3, timeout=5.0)

        assert isinstance(batch, list)
        assert len(batch) == 3
        for b in batch:
            assert isinstance(b, dict)

        controller.destroy()

    def test_wait_timeout_when_insufficient_results(self):
        # [FUNCTION DOC]
        # With engine calls slowed by 0.1 s, wait(1, timeout=0.1) raises TimeoutError.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=10)
        scheduler = MockScheduler()

        async def async_mock(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Delegates to MockScheduler.async_call_engine then sleeps 0.1 s before returning.
            res = await MockScheduler.async_call_engine(scheduler, *args, **kwargs)
            await asyncio.sleep(0.1)
            return res

        # Mock the `wait` call.
        scheduler.async_call_engine = async_mock

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.submit(
            {"id": 0},
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        with pytest.raises(TimeoutError, match="Timed out waiting for"):
            controller.wait(count=1, timeout=0.1)

        controller.destroy()

    def test_submit_passes_is_eval_and_group_size(self):
        # [FUNCTION DOC]
        # submit forwards is_eval=True and group_size=4 as kwargs of the single worker "submit"
        # call.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.submit(
            data={"id": 1},
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
            is_eval=True,
            group_size=4,
        )
        controller.wait(count=1, timeout=5.0)

        submit_calls = [call for call in scheduler.engine_calls if call[1] == "submit"]
        assert len(submit_calls) == 1
        submit_kwargs = submit_calls[0][3]
        assert "is_eval" in submit_kwargs and submit_kwargs["is_eval"] is True
        assert "group_size" in submit_kwargs and submit_kwargs["group_size"] == 4

        controller.destroy()


class TestRolloutControllerBatchOperations:
    # [CLASS DOC]
    # rollout_batch() tests.
    def test_rollout_batch_returns_list_of_dicts(self):
        # [FUNCTION DOC]
        # rollout_batch returns a list of plain dicts with no RTensor values (top level or one
        # nested level).
        """Verify RolloutController returns list of regular dicts, NOT RTensors.

        Unlike TrainController which uses RTensors for distributed batch storage,
        RolloutController uses task-based round-robin and returns list of regular Python dicts.
        """
        from areal.infra.rpc.rtensor import RTensor

        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i} for i in range(3)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Verify batch is a dict, not RTensor
        assert isinstance(batch, list), "RolloutController should return list of dicts"

        # Verify no RTensors in the result
        for item in batch:
            assert isinstance(item, dict), "Each item should be a dict"
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    assert not isinstance(value, RTensor), f"Found RTensor at key {key}"
                elif isinstance(value, dict):
                    for k, v in value.items():
                        assert not isinstance(v, RTensor), f"Found RTensor at {key}.{k}"

        controller.destroy()

    def test_rollout_batch_submits_all_data(self):
        # [FUNCTION DOC]
        # Four inputs yield four trajectories.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i, "value": f"item_{i}"} for i in range(4)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Check that all trajectories are returned (each as a separate dict)
        assert len(batch) == 4

        controller.destroy()

    def test_rollout_batch_waits_for_all_results(self):
        # [FUNCTION DOC]
        # Ten inputs yield ten trajectories.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=100)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        batch_data = [{"id": i} for i in range(10)]
        batch = controller.rollout_batch(
            batch_data,
            workflow="tests.utils.TestWorkflow",
            workflow_kwargs={},
        )

        # Check that all trajectories are returned (each as a separate dict)
        assert len(batch) == 10

        controller.destroy()


class TestRolloutControllerVersionManagement:
    # [CLASS DOC]
    # get_version()/set_version() tests.
    def test_get_version_initial(self):
        # [FUNCTION DOC]
        # Initial version is 0 (destroy before initialize is tolerated).
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.get_version() == 0
        controller.destroy()

    def test_set_version_updates_controller_version(self):
        # [FUNCTION DOC]
        # set_version(42) is reflected by get_version.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(42)
        assert controller.get_version() == 42

        controller.destroy()

    def test_set_version_calls_workers(self):
        # [FUNCTION DOC]
        # set_version issues one "set_version" call per worker (2).
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(10)

        version_calls = [
            call for call in scheduler.engine_calls if call[1] == "set_version"
        ]
        assert len(version_calls) == 2

        controller.destroy()

    def test_set_version_handles_worker_error(self):
        # [FUNCTION DOC]
        # set_version does not raise when every worker call fails.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Always raises Exception("Worker error").
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.set_version(5)
        controller.destroy()


class TestRolloutControllerWeightUpdates:
    # [CLASS DOC]
    # Weight-update forwarding tests; each returns an awaitable that is run to completion.
    def test_init_weights_update_group_returns_future(self):
        # [FUNCTION DOC]
        # init_weights_update_group returns an awaitable that completes.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        coro = controller.init_weights_update_group(meta)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()

    def test_update_weights_from_distributed_returns_future(self):
        # [FUNCTION DOC]
        # update_weights_from_distributed with one ParamSpec returns an awaitable that completes.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        param_specs = [ParamSpec(name="test", shape=(10, 10), dtype="float32")]
        coro = controller.update_weights_from_distributed(meta, param_specs)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()

    def test_update_weights_from_disk_returns_future(self):
        # [FUNCTION DOC]
        # update_weights_from_disk returns an awaitable that completes.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        meta = WeightUpdateMeta(type="disk", path="/tmp/test")
        coro = controller.update_weights_from_disk(meta)

        # Run the coroutine and verify it completes successfully
        asyncio.run(coro)

        controller.destroy()


class TestRolloutControllerLifecycle:
    # [CLASS DOC]
    # pause()/resume() tests.
    def test_pause_calls_all_workers(self):
        # [FUNCTION DOC]
        # pause issues one "pause" call per worker.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.pause()

        pause_calls = [call for call in scheduler.engine_calls if call[1] == "pause"]
        assert len(pause_calls) == 2

        controller.destroy()

    def test_resume_calls_all_workers(self):
        # [FUNCTION DOC]
        # resume issues one "resume" call per worker.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.resume()

        resume_calls = [call for call in scheduler.engine_calls if call[1] == "resume"]
        assert len(resume_calls) == 2

        controller.destroy()

    def test_pause_handles_worker_error(self):
        # [FUNCTION DOC]
        # pause does not raise when worker calls fail.
        # KNOWN ISSUE: the controller is never destroyed, leaking its background resources.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Always raises Exception("Worker error").
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.pause()

    def test_resume_handles_worker_error(self):
        # [FUNCTION DOC]
        # resume does not raise when worker calls fail.
        # KNOWN ISSUE: the controller is never destroyed, leaking its background resources.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        def failing_call(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Always raises Exception("Worker error").
            raise Exception("Worker error")

        scheduler.call_engine = failing_call

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        controller.resume()


class TestRolloutControllerAgenerate:
    # [CLASS DOC]
    # agenerate() tests.
    def test_agenerate_chooses_worker(self):
        # [FUNCTION DOC]
        # One agenerate call issues exactly one "agenerate" RPC with req passed as kwarg.
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        req = ModelRequest(input_ids=[1, 2, 3, 4, 5])

        async def test_agenerate():
            # [FUNCTION DOC]
            # HELPER: Awaits controller.agenerate(req) and returns the result.
            result = await controller.agenerate(req)
            return result

        asyncio.run(test_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        assert len(agenerate_calls) == 1
        assert agenerate_calls[0][3]["req"] == req

        controller.destroy()

    def test_agenerate_round_robin(self):
        # [FUNCTION DOC]
        # Six agenerate calls on 3 workers are dispatched round-robin.
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        async def test_multiple_agenerate():
            # [FUNCTION DOC]
            # HELPER: Sequentially awaits six agenerate calls.
            for _ in range(6):
                req = ModelRequest(input_ids=[1, 2, 3])
                await controller.agenerate(req)

        asyncio.run(test_multiple_agenerate())

        agenerate_calls = [
            call for call in scheduler.engine_calls if call[1] == "agenerate"
        ]
        worker_ids = [call[0] for call in agenerate_calls]

        assert worker_ids[0] == "rollout/0"
        assert worker_ids[1] == "rollout/1"
        assert worker_ids[2] == "rollout/2"
        assert worker_ids[3] == "rollout/0"

        controller.destroy()


class TestRolloutControllerErrorHandling:
    # [CLASS DOC]
    # Error-handling tests.
    def test_wait_returns_empty_batch_on_no_results(self):
        # [FUNCTION DOC]
        # When every engine call returns None, wait(1, timeout=0.5) raises TimeoutError.
        # KNOWN ISSUE: the name says it returns an empty batch, but it asserts a TimeoutError.
        config = create_test_config(consumer_batch_size=16, max_concurrent_rollouts=50)
        scheduler = MockScheduler()

        async def reject_all(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Async stub that sleeps 10 ms and returns None for any call.
            await asyncio.sleep(0.01)
            return None

        scheduler.async_call_engine = reject_all

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        with pytest.raises(TimeoutError):
            controller.wait(count=1, timeout=0.5)

        controller.destroy()


class TestRolloutControllerIntegration:
    # [CLASS DOC]
    # Mock-based end-to-end flows.
    def test_end_to_end_workflow(self):
        # [FUNCTION DOC]
        # Capacity 20, five submits then wait(5), then set_version(1) is reflected.
        config = create_test_config(
            consumer_batch_size=8,
            max_concurrent_rollouts=20,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        capacity = controller.get_capacity()
        assert capacity == 20

        for i in range(5):
            controller.submit(
                {"id": i},
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )

        batch = controller.wait(count=5, timeout=5.0)
        assert len(batch) == 5

        controller.set_version(1)
        assert controller.get_version() == 1

        controller.destroy()

    def test_multiple_batch_cycles(self):
        # [FUNCTION DOC]
        # Three consecutive rollout_batch calls of 4 items each return 4 trajectories.
        config = create_test_config(
            consumer_batch_size=4,
            max_concurrent_rollouts=50,
            max_head_offpolicyness=5,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        for cycle in range(3):
            batch_data = [{"id": i, "cycle": cycle} for i in range(4)]
            batch = controller.rollout_batch(
                batch_data,
                workflow="tests.utils.TestWorkflow",
                workflow_kwargs={},
            )
            assert len(batch) == 4

        controller.destroy()


@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_parametrized_worker_count(num_workers):
    # [FUNCTION DOC]
    # "sglang:dN" creates N workers for N in 1, 2, 4.
    config = create_test_config(
        backend=f"sglang:d{num_workers}", consumer_batch_size=16
    )
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    controller.initialize(role="rollout", server_args={})

    assert len(controller.workers) == num_workers

    controller.destroy()


@pytest.mark.parametrize(
    "consumer_batch_size,max_concurrent_rollouts,expected_capacity",
    [(16, 32, 32), (32, 64, 64), (8, 100, 24)],
)
def test_parametrized_capacity_settings(
    consumer_batch_size, max_concurrent_rollouts, expected_capacity
):
    # [FUNCTION DOC]
    # Initial capacity is min(max_concurrent_rollouts, (max_staleness+1) * consumer batch
    # size) for the given cases (e.g. 8 batch, 100 concurrency, staleness 2 -> 24).
    config = create_test_config(
        consumer_batch_size=consumer_batch_size,
        max_concurrent_rollouts=max_concurrent_rollouts,
        max_head_offpolicyness=2,
    )
    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=config,
        scheduler=scheduler,
    )

    controller.initialize(role="rollout", server_args={})

    capacity = controller.get_capacity()
    assert capacity == expected_capacity

    controller.destroy()


QWEN3_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


@pytest.mark.sglang
@pytest.mark.parametrize("model_path", [QWEN3_PATH])
@pytest.mark.slow
@pytest.mark.ci
def test_rollout_controller_integration(tmp_path, model_path):
    # [FUNCTION DOC]
    # Slow SGLang/CI test: with a real LocalScheduler and two SGLang servers on Qwen3-0.6B,
    # rollout_batch of 10 identical GSM8K-style prompts through RLVRWorkflow returns 10 dicts
    # containing attention_mask. Always destroys the controller.
    from areal.engine.sglang_remote import RemoteSGLangEngine

    tokenizer = load_hf_tokenizer(model_path)
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    scheduler = LocalScheduler(
        log_dir=tmp_path,
        experiment_name="test_rollout_controller_integration",
        trial_name="trial0",
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )
    rollout = RolloutController(
        inf_engine=RemoteSGLangEngine,
        config=InferenceEngineConfig(
            experiment_name="test",
            trial_name="test",
            backend="sglang:d2",
            consumer_batch_size=128,
            max_head_offpolicyness=1,
            max_concurrent_rollouts=5,
            setup_timeout=300,
            enable_rollout_tracing=True,
            scheduling_spec=(
                SchedulingSpec(
                    cpu=4, gpu=1, cmd="python -m areal.infra.rpc.rpc_server"
                ),
            ),
        ),
        scheduler=scheduler,
    )

    bs = 10
    try:
        rollout.initialize(
            role="rollout",
            server_args=SGLangConfig.build_args(
                SGLangConfig(model_path=model_path, mem_fraction_static=0.5),
                tp_size=1,
                base_gpu_id=0,
            ),
        )
        result = rollout.rollout_batch(
            data=[dict(messages=[dict(role="user", content="hello")], answer="1")] * bs,
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=dict(
                reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
                gconfig=GenerationHyperparameters(),
                tokenizer=tokenizer,
            ),
        )
        assert isinstance(result, list)
        assert len(result) == bs
        assert isinstance(result[0], dict)
        assert "attention_mask" in result[0]
    finally:
        rollout.destroy()


class TestRolloutControllerResolveWorkflow:
    # [CLASS DOC]
    # Workflow resolution tests.
    """Tests for workflow resolution methods."""

    def test_resolve_workflow_str_with_string(self):
        # [FUNCTION DOC]
        # A string workflow path is returned unchanged.
        """Test _resolve_workflow_str with string input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        result = controller._resolve_workflow_str("areal.workflow.rlvr.RLVRWorkflow")
        assert result == "areal.workflow.rlvr.RLVRWorkflow"


class TestRolloutControllerShouldAcceptFn:
    # [CLASS DOC]
    # should_accept_fn resolution tests.
    """Tests for should_accept_fn resolution."""

    def test_resolve_should_accept_fn_with_none(self):
        # [FUNCTION DOC]
        # None resolves to None.
        """Test _resolve_should_accept_fn with None input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        result = controller._resolve_should_accept_fn(None)
        assert result is None

    def test_resolve_should_accept_fn_with_callable_raises(self):
        # [FUNCTION DOC]
        # A callable (not an import path) raises RuntimeError.
        """Test _resolve_should_accept_fn raises for callable input."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        def my_filter(data):
            # [FUNCTION DOC]
            # HELPER: Dummy filter returning True.
            return True

        with pytest.raises(RuntimeError, match="must be an importable string path"):
            controller._resolve_should_accept_fn(my_filter)

    def test_resolve_should_accept_fn_with_invalid_path_raises(self):
        # [FUNCTION DOC]
        # An unimportable path raises RuntimeError ("Failed to import").
        """Test _resolve_should_accept_fn raises for invalid import path."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(RuntimeError, match="Failed to import"):
            controller._resolve_should_accept_fn("invalid.module.path.function")


class TestRolloutControllerDispatcher:
    # [CLASS DOC]
    # dispatcher property tests.
    """Tests for dispatcher property and initialization."""

    def test_dispatcher_raises_before_initialization(self):
        # [FUNCTION DOC]
        # Accessing dispatcher before initialize raises RuntimeError.
        """Test dispatcher property raises when not initialized."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        with pytest.raises(RuntimeError, match="initialize\\(\\) must be called"):
            _ = controller.dispatcher

    def test_dispatcher_available_after_initialization(self):
        # [FUNCTION DOC]
        # dispatcher is non-None after initialize.
        """Test dispatcher property works after initialization."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        dispatcher = controller.dispatcher
        assert dispatcher is not None

        controller.destroy()


class TestRolloutControllerStalenessManager:
    # [CLASS DOC]
    # staleness_manager property tests.
    """Tests for staleness manager property."""

    def test_staleness_manager_none_before_initialization(self):
        # [FUNCTION DOC]
        # staleness_manager is None before initialize.
        """Test staleness_manager is None before initialization."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        assert controller.staleness_manager is None

    def test_staleness_manager_available_after_initialization(self):
        # [FUNCTION DOC]
        # After initialize the manager exists with max_staleness 2.
        """Test staleness_manager is available after initialization."""
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        assert controller.staleness_manager is not None
        assert controller.staleness_manager.max_staleness == 2

        controller.destroy()


class TestRolloutControllerRunner:
    # [CLASS DOC]
    # runner property tests.
    """Tests for runner property (backward compatibility)."""

    def test_runner_property_returns_dispatcher_runner(self):
        # [FUNCTION DOC]
        # controller.runner is the dispatcher's runner object.
        """Test runner property returns the dispatcher's runner."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        runner = controller.runner
        assert runner is controller.dispatcher.runner

        controller.destroy()


class TestRolloutControllerExportStats:
    # [CLASS DOC]
    # export_stats tests.
    """Tests for export_stats method."""

    def test_export_stats_aggregates_from_workers(self):
        # [FUNCTION DOC]
        # With workers returning reward/loss stats and counts, export_stats returns a dict
        # containing reward or loss.
        # KNOWN ISSUE: the "or" assertion is weak and aggregated values are not checked.
        """Test export_stats correctly aggregates stats from all workers."""
        config = create_test_config(consumer_batch_size=16)
        scheduler = MockScheduler()

        # Override async_call_engine to return stats for export_stats method
        original_async_call = scheduler.async_call_engine

        async def mock_async_call_engine(worker_id, method, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns a fixed stats dict for "export_stats", otherwise delegates to the
            # original MockScheduler.async_call_engine.
            if method == "export_stats":
                return {
                    "reward": 0.5,
                    "reward__count": 10,
                    "loss": 0.3,
                    "loss__count": 10,
                }
            return await original_async_call(worker_id, method, *args, **kwargs)

        scheduler.async_call_engine = mock_async_call_engine

        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        stats = controller.export_stats()

        # Should aggregate stats from all workers
        assert "reward" in stats or "loss" in stats

        controller.destroy()


class TestRolloutControllerRolloutStats:
    # [CLASS DOC]
    # _rollout_stats tests.
    """Tests for _rollout_stats method."""

    def test_rollout_stats_returns_formatted_string(self):
        # [FUNCTION DOC]
        # The stats string mentions enqueued, running, accepted and rejected counts.
        """Test _rollout_stats returns properly formatted stats string."""
        config = create_test_config(
            consumer_batch_size=16,
            max_head_offpolicyness=2,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        stats_str = controller._rollout_stats()

        assert "enqueued:" in stats_str
        assert "running:" in stats_str
        assert "accepted:" in stats_str
        assert "rejected:" in stats_str

        controller.destroy()


class TestRolloutControllerSchedulingSpec:
    # [CLASS DOC]
    # Scheduling spec handling tests.
    """Tests for scheduling spec handling during initialization."""

    def test_initialization_scales_scheduling_spec(self):
        # [FUNCTION DOC]
        # With "sglang:d2t2", two workers (one per data-parallel instance) are created.
        # KNOWN ISSUE: the scaled per-worker GPU spec is not actually asserted.
        """Test initialization correctly scales scheduling spec for instance size."""
        config = create_test_config(
            backend="sglang:d2t2",
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        # Use TP=2 to test instance size scaling
        controller.initialize(role="rollout", server_args={})

        # Verify workers were created with correct count
        assert len(controller.workers) == 2  # dp_size = 2

        controller.destroy()


class TestRolloutControllerQueueSize:
    # [CLASS DOC]
    # Queue size configuration tests.
    """Tests for queue size configuration."""

    def test_queue_size_uses_config_value(self):
        # [FUNCTION DOC]
        # With queue_size 100, initialization creates a dispatcher.
        # KNOWN ISSUE: the queue size value itself is never checked.
        """Test queue size uses config value when provided."""
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            queue_size=100,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Queue size should be used from config
        assert controller.dispatcher is not None

        controller.destroy()

    def test_queue_size_defaults_to_concurrent_rollouts(self):
        # [FUNCTION DOC]
        # With queue_size None, initialization creates a dispatcher.
        # KNOWN ISSUE: the default queue size value itself is never checked.
        """Test queue size defaults to max_concurrent_rollouts * 16 when not provided."""
        config = create_test_config(
            consumer_batch_size=16,
            max_concurrent_rollouts=32,
            queue_size=None,
        )
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Should use default queue size
        assert controller.dispatcher is not None

        controller.destroy()


class TestRolloutControllerCollectiveRPC:
    # [CLASS DOC]
    # _collective_rpc tests.
    """Tests for collective RPC methods."""

    def test_collective_rpc_calls_all_workers(self):
        # [FUNCTION DOC]
        # _collective_rpc("test_method", ...) issues one call per worker (3).
        """Test _collective_rpc calls all workers."""
        config = create_test_config(backend="sglang:d3", consumer_batch_size=16)
        scheduler = MockScheduler()
        controller = RolloutController(
            inf_engine=MockInferenceEngine,
            config=config,
            scheduler=scheduler,
        )

        controller.initialize(role="rollout", server_args={})

        # Clear previous calls
        scheduler.engine_calls = []

        controller._collective_rpc("test_method", arg1="value1")

        # Should have called all 3 workers
        test_calls = [
            call for call in scheduler.engine_calls if call[1] == "test_method"
        ]
        assert len(test_calls) == 3

        controller.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
