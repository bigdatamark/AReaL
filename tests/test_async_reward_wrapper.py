# [MODULE DOC]
# Unit tests for AsyncRewardWrapper (areal.api.reward_api), which runs reward functions
# in shared ProcessPoolExecutors keyed by max_workers with timeout, retry and broken-
# pool recreation. Reward functions are module-level so they are picklable. An autouse
# fixture resets the shared executors around every test. Spawns real worker processes.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest

from areal.api.reward_api import AsyncRewardWrapper


# Module-level: ProcessPoolExecutor requires picklable callables
def _add_reward(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # HELPER: Picklable reward fn returning a + b.
    return a + b


def _slow_reward(seconds: float) -> float:
    # [FUNCTION DOC]
    # HELPER: Sleeps for the given seconds and returns 1.0; used to trigger timeouts.
    time.sleep(seconds)
    return 1.0


def _crash_worker() -> float:
    # [FUNCTION DOC]
    # HELPER: Hard-exits the worker process with status 1 to break the process pool.
    os._exit(1)


def _raise_value_error() -> float:
    # [FUNCTION DOC]
    # HELPER: Always raises ValueError('intentional error from reward fn').
    raise ValueError("intentional error from reward fn")


@pytest.fixture(autouse=True)
def _isolate_executor_state():
    # [FUNCTION DOC]
    # Autouse fixture: calls AsyncRewardWrapper._atexit_shutdown_all before and after
    # each test so the class-level executor map starts and ends empty.
    """Shut down and clear all shared executors before and after each test."""
    AsyncRewardWrapper._atexit_shutdown_all()
    yield
    AsyncRewardWrapper._atexit_shutdown_all()


class TestAsyncRewardWrapperBasic:
    # [CLASS DOC]
    # Basic result and executor-sharing behavior.
    @pytest.mark.asyncio
    async def test_returns_correct_result(self):
        # [FUNCTION DOC]
        # Awaiting the wrapper of _add_reward(3, 4) returns 7.0.
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        result = await wrapper(3.0, 4.0)
        assert result == 7.0

    @pytest.mark.asyncio
    async def test_multiple_calls_return_correct_results(self):
        # [FUNCTION DOC]
        # Five sequential calls return the expected sums.
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=2, max_retries=0)
        results = [await wrapper(float(i), 1.0) for i in range(5)]
        assert results == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_shared_executor_for_same_max_workers(self):
        # [FUNCTION DOC]
        # Two wrappers with the same max_workers share one executor key and only one
        # executor exists in the class map.
        w1 = AsyncRewardWrapper(_add_reward, max_workers=2)
        w2 = AsyncRewardWrapper(_add_reward, max_workers=2)
        assert w1._executor_key == w2._executor_key
        assert AsyncRewardWrapper._executors.get(2) is not None
        assert len(AsyncRewardWrapper._executors) == 1

    def test_different_executor_for_different_max_workers(self):
        # [FUNCTION DOC]
        # Wrappers with max_workers 1 and 2 create two executors.
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2


class TestAsyncRewardWrapperTimeout:
    # [CLASS DOC]
    # Timeout handling.
    @pytest.mark.asyncio
    async def test_timeout_returns_zero_after_retries(self):
        # [FUNCTION DOC]
        # With timeout 0.1s and one retry, a 10s reward returns float 0.0 instead of
        # raising.
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=1
        )
        result = await wrapper(10.0)
        assert isinstance(result, float)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_timeout_no_retries_returns_zero(self):
        # [FUNCTION DOC]
        # With timeout 0.1s and no retries, a 10s reward returns float 0.0.
        wrapper = AsyncRewardWrapper(
            _slow_reward, timeout_seconds=0.1, max_workers=1, max_retries=0
        )
        result = await wrapper(10.0)
        assert isinstance(result, float)
        assert result == 0.0


class TestAsyncRewardWrapperBrokenPool:
    # [CLASS DOC]
    # Behavior when a worker process dies.
    @pytest.mark.asyncio
    async def test_crash_raises_broken_process_pool(self):
        # [FUNCTION DOC]
        # A crashing worker with no retries surfaces BrokenProcessPool to the caller.
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=0, timeout_seconds=5
        )
        with pytest.raises(BrokenProcessPool):
            await wrapper()

    @pytest.mark.asyncio
    async def test_recreation_replaces_executor(self):
        # [FUNCTION DOC]
        # With one retry, a crashing worker still raises BrokenProcessPool, and the
        # class-level executor for max_workers=1 is replaced by a new instance.
        wrapper = AsyncRewardWrapper(
            _crash_worker, max_workers=1, max_retries=1, timeout_seconds=5
        )
        executor_before = AsyncRewardWrapper._executors.get(1)
        assert executor_before is not None

        with pytest.raises(BrokenProcessPool):
            await wrapper()

        executor_after = AsyncRewardWrapper._executors.get(1)
        assert executor_after is not executor_before


class TestAsyncRewardWrapperExceptionHandling:
    # [CLASS DOC]
    # Exception propagation and post-shutdown behavior.
    @pytest.mark.asyncio
    async def test_reward_fn_exception_propagates(self):
        # [FUNCTION DOC]
        # A ValueError from the reward fn propagates without retries.
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=0)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_reward_fn_exception_retries_then_raises(self):
        # [FUNCTION DOC]
        # With max_retries=2 the ValueError still propagates (no swallowing).
        wrapper = AsyncRewardWrapper(_raise_value_error, max_workers=1, max_retries=2)
        with pytest.raises(ValueError, match="intentional error"):
            await wrapper()

    @pytest.mark.asyncio
    async def test_shutdown_then_call_raises_runtime_error(self):
        # [FUNCTION DOC]
        # After _atexit_shutdown_all, calling an existing wrapper raises RuntimeError
        # containing 'has been shut down'.
        wrapper = AsyncRewardWrapper(_add_reward, max_workers=1, max_retries=0)
        AsyncRewardWrapper._atexit_shutdown_all()
        with pytest.raises(RuntimeError, match="has been shut down"):
            await wrapper(1.0, 2.0)


class TestAsyncRewardWrapperAtexitCleanup:
    # [CLASS DOC]
    # Executor cleanup at exit.
    def test_atexit_clears_all_executors(self):
        # [FUNCTION DOC]
        # Two executors exist after creating wrappers; _atexit_shutdown_all empties the
        # map.
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper(_add_reward, max_workers=2)
        assert len(AsyncRewardWrapper._executors) == 2

        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0

    def test_atexit_is_idempotent(self):
        # [FUNCTION DOC]
        # Calling _atexit_shutdown_all twice is safe and leaves no executors.
        AsyncRewardWrapper(_add_reward, max_workers=1)
        AsyncRewardWrapper._atexit_shutdown_all()
        AsyncRewardWrapper._atexit_shutdown_all()
        assert len(AsyncRewardWrapper._executors) == 0


class TestRecreateExecutorRaceSafety:
    # [CLASS DOC]
    # Compare-and-swap semantics of _recreate_executor.
    def test_recreate_skips_when_already_replaced(self):
        # [FUNCTION DOC]
        # If the map already holds a different executor than the one passed as broken,
        # _recreate_executor returns the current replacement without creating another.
        # Shuts the replacement down afterwards.
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        # Simulate a concurrent thread having already replaced the executor
        replacement = ProcessPoolExecutor(max_workers=1)
        AsyncRewardWrapper._executors[1] = replacement

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is replacement

        replacement.shutdown(wait=False)

    def test_recreate_replaces_when_identity_matches(self):
        # [FUNCTION DOC]
        # If the map still holds the passed executor, _recreate_executor creates and
        # stores a new one and returns it.
        AsyncRewardWrapper(_add_reward, max_workers=1)
        original = AsyncRewardWrapper._executors[1]

        result = AsyncRewardWrapper._recreate_executor(1, 1, original)
        assert result is not None
        assert result is not original
        assert AsyncRewardWrapper._executors[1] is result
