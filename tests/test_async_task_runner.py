# [MODULE DOC]
# Unit tests for the generic AsyncTaskRunner (areal.infra.async_task_runner), which runs
# submitted coroutine functions on a background event-loop thread and returns their
# results through wait(count, timeout) in shuffled order. Covers submission,
# pause/resume, timeouts, concurrency, error handling, queue sizes, result
# caching/ordering and generic result types. Each test creates, initializes and destroys
# its own runner; timing-based assertions make some tests sensitive to slow machines.
# KNOWN ISSUE: runner.destroy() is not in a finally block, so a failing assertion leaks
# the background thread for the rest of the session.
"""Unit tests for the generic AsyncTaskRunner.

This test suite verifies the functionality of the AsyncTaskRunner without
any AReaL-specific dependencies, ensuring it can be used as a generic
async task executor.
"""

import asyncio
import time

import pytest

from areal.infra.async_task_runner import AsyncTaskRunner


class TestAsyncTaskRunnerBasic:
    # [CLASS DOC]
    # Basic submit/wait behavior.
    """Test basic functionality of AsyncTaskRunner."""

    def test_initialization_and_shutdown(self):
        # [FUNCTION DOC]
        # Initializes a runner, sleeps 0.1s to let the thread start and destroys it
        # without errors.
        """Test that runner can be initialized and shut down cleanly."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()
        time.sleep(0.1)  # Allow thread to start
        runner.destroy()

    def test_simple_task_submission(self):
        # [FUNCTION DOC]
        # Submits one async add(5, 10) and asserts wait(count=1) returns [15].
        """Test submitting and waiting for a single task."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def add(a: int, b: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns a + b.
            await asyncio.sleep(0.01)
            return a + b

        runner.submit(add, 5, 10, task_id=1)
        results = runner.wait(count=1, timeout=2.0)

        assert len(results) == 1
        assert results[0] == 15
        runner.destroy()

    def test_multiple_task_submission(self):
        # [FUNCTION DOC]
        # Submits compute(i) for i in 0..9 and asserts the result set equals the doubled
        # values.
        """Test submitting and waiting for multiple tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def compute(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x * 2.
            await asyncio.sleep(0.01)
            return x * 2

        # Submit 10 tasks
        for i in range(10):
            runner.submit(compute, i, task_id=i)

        results = runner.wait(count=10, timeout=5.0)

        assert len(results) == 10
        # Results are shuffled, so check that all expected values are present
        assert set(results) == {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
        runner.destroy()

    def test_with_kwargs(self):
        # [FUNCTION DOC]
        # Submits tasks with keyword-only and mixed positional/keyword args and asserts
        # results {15, 28}.
        """Test task submission with keyword arguments."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def multiply(a: int, b: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns a * b.
            await asyncio.sleep(0.01)
            return a * b

        runner.submit(multiply, a=5, b=3, task_id=1)
        runner.submit(multiply, 4, b=7, task_id=2)

        results = runner.wait(count=2, timeout=2.0)

        assert len(results) == 2
        assert set(results) == {15, 28}
        runner.destroy()


class TestAsyncTaskRunnerPauseResume:
    # [CLASS DOC]
    # Pause/resume semantics.
    """Test pause/resume functionality."""

    def test_pause_prevents_new_tasks(self):
        # [FUNCTION DOC]
        # Submits five 0.2s tasks, pauses, submits five more, and asserts wait(count=10,
        # timeout=0.5) raises TimeoutError after at least 0.5s; fails if it returns.
        """Test that pause prevents new tasks from starting."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def slow_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 0.2s and returns x.
            await asyncio.sleep(0.2)
            return x

        # Submit some tasks
        for i in range(5):
            runner.submit(slow_task, i, task_id=i)

        # Pause immediately
        runner.pause()
        time.sleep(0.1)

        # Submit more tasks while paused
        for i in range(5, 10):
            runner.submit(slow_task, i, task_id=i)

        # Should not be able to get all results quickly because paused tasks don't start
        start = time.time()
        try:
            runner.wait(count=10, timeout=0.5)
            pytest.fail("Should have timed out")
        except TimeoutError:
            elapsed = time.time() - start
            assert elapsed >= 0.5

        runner.destroy()

    def test_resume_allows_tasks(self):
        # [FUNCTION DOC]
        # Pauses before submitting five tasks, resumes, and asserts all five doubled
        # results arrive.
        """Test that resume allows paused tasks to start."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def fast_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x * 2.
            await asyncio.sleep(0.01)
            return x * 2

        # Pause before submitting
        runner.pause()

        # Submit tasks while paused
        for i in range(5):
            runner.submit(fast_task, i, task_id=i)

        # Resume
        runner.resume()

        # Now should be able to get results
        results = runner.wait(count=5, timeout=2.0)
        assert len(results) == 5
        assert set(results) == {0, 2, 4, 6, 8}

        runner.destroy()


class TestAsyncTaskRunnerTimeout:
    # [CLASS DOC]
    # wait() timeout behavior.
    """Test timeout behavior."""

    def test_timeout_on_insufficient_results(self):
        # [FUNCTION DOC]
        # A single 5s task with wait timeout 0.5s raises TimeoutError matching 'Timed
        # out waiting for 1 results'.
        """Test that wait() times out if not enough results are ready."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def slow_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 5s and returns x.
            await asyncio.sleep(5.0)  # Very slow task
            return x

        runner.submit(slow_task, 1, task_id=1)

        with pytest.raises(TimeoutError, match="Timed out waiting for 1 results"):
            runner.wait(count=1, timeout=0.5)

        runner.destroy()

    def test_partial_results_timeout(self):
        # [FUNCTION DOC]
        # Submits three fast and two 10s tasks; wait(count=5, timeout=1) raises
        # TimeoutError mentioning 'only received 3'.
        """Test timeout when only partial results are available."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()
        time.sleep(0.1)  # Give thread time to start

        async def fast_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x.
            await asyncio.sleep(0.01)
            return x

        async def slow_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10s and returns x.
            await asyncio.sleep(10.0)
            return x

        # Submit 3 fast and 2 slow
        for i in range(3):
            runner.submit(fast_task, i, task_id=i)
        for i in range(2):
            runner.submit(slow_task, i + 100, task_id=i + 100)

        # Should timeout waiting for all 5
        with pytest.raises(TimeoutError, match="only received 3"):
            runner.wait(count=5, timeout=1.0)

        runner.destroy()


class TestAsyncTaskRunnerConcurrency:
    # [CLASS DOC]
    # Concurrency of execution.
    """Test concurrent task execution."""

    def test_concurrent_execution(self):
        # [FUNCTION DOC]
        # With fast polling (poll_wait_time 0.05, poll_sleep_time 0.01) five 0.2s tasks
        # must complete in under 0.8s total, proving concurrent execution.
        """Test that tasks execute concurrently, not sequentially."""
        # Use faster polling for this test
        runner = AsyncTaskRunner[float](
            max_queue_size=50,
            poll_wait_time=0.05,
            poll_sleep_time=0.01,  # Very fast polling for this test
        )
        runner.initialize()
        time.sleep(0.1)  # Give thread time to start

        async def timed_task(duration: float) -> float:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps duration seconds and returns the measured elapsed
            # time.
            start = time.time()
            await asyncio.sleep(duration)
            return time.time() - start

        # Submit 5 tasks that each take 0.2 seconds
        for i in range(5):
            runner.submit(timed_task, 0.2, task_id=i)

        start = time.time()
        results = runner.wait(count=5, timeout=2.0)
        total_time = time.time() - start

        # If concurrent, should take ~0.2s. If sequential, would take ~1.0s
        assert total_time < 0.8, f"Tasks run sequentially ({total_time}s)"
        assert len(results) == 5

        runner.destroy()

    def test_many_concurrent_tasks(self):
        # [FUNCTION DOC]
        # Submits 100 squaring tasks and asserts all 100 squares are returned within
        # 10s.
        """Test handling many concurrent tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=200)
        runner.initialize()

        async def compute(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x squared.
            await asyncio.sleep(0.01)
            return x * x

        # Submit 100 tasks
        for i in range(100):
            runner.submit(compute, i, task_id=i)

        results = runner.wait(count=100, timeout=10.0)

        assert len(results) == 100
        assert set(results) == {i * i for i in range(100)}

        runner.destroy()


class TestAsyncTaskRunnerErrorHandling:
    # [CLASS DOC]
    # Errors and shutdown.
    """Test error handling and edge cases."""

    def test_queue_full_error(self):
        # [FUNCTION DOC]
        # With max_queue_size=2 and two long tasks submitted, a third submit raises
        # RuntimeError 'Input queue full'.
        # KNOWN ISSUE: Relies on the background thread not having drained the input
        # queue yet; if tasks are picked up immediately the third submit may succeed,
        # making the test racy.
        """Test that submitting to a full queue raises an error."""
        runner = AsyncTaskRunner[int](max_queue_size=2)
        runner.initialize()

        async def task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10s and returns x.
            await asyncio.sleep(10.0)  # Long delay to keep tasks pending
            return x

        # Fill the queue
        runner.submit(task, 1, task_id=1)
        runner.submit(task, 2, task_id=2)

        # Next submission should fail
        with pytest.raises(RuntimeError, match="Input queue full"):
            runner.submit(task, 3, task_id=3)

        runner.destroy()

    def test_task_exception_handling(self):
        # [FUNCTION DOC]
        # A task raising ValueError yields a None result (first wait) while a later
        # normal task still returns 42 (second wait); the runner thread remains healthy
        # per _check_thread_health.
        # KNOWN ISSUE: Assumes the failing task (10ms) finishes before the working one
        # (20ms); scheduling jitter could reorder results.
        """Test that task exceptions are properly handled and don't stop the runner."""
        runner = AsyncTaskRunner[int](max_queue_size=10)
        runner.initialize()

        async def failing_task() -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms then raises ValueError('Task failed!').
            await asyncio.sleep(0.01)
            raise ValueError("Task failed!")

        async def working_task() -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 20ms and returns 42.
            await asyncio.sleep(0.02)
            return 42

        # Submit a failing task and a working task
        runner.submit(failing_task, task_id=1)
        runner.submit(working_task, task_id=2)

        # The working task should still complete and its result should be retrievable.
        # The failing task will return None but won't crash the runner.
        results = runner.wait(count=1, timeout=2.0)
        assert results == [None]
        results = runner.wait(count=1, timeout=2.0)
        assert results == [42]

        # Check that the runner thread is still alive
        runner._check_thread_health()

        runner.destroy()

    def test_shutdown_with_pending_tasks(self):
        # [FUNCTION DOC]
        # Submits ten 10s tasks and asserts destroy() returns cleanly (cancelling
        # pending work).
        """Test clean shutdown with pending tasks."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def long_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10s and returns x.
            await asyncio.sleep(10.0)
            return x

        # Submit tasks that won't complete
        for i in range(10):
            runner.submit(long_task, i, task_id=i)

        # Should shut down cleanly, cancelling pending tasks
        runner.destroy()


class TestAsyncTaskRunnerQueueSizes:
    # [CLASS DOC]
    # Queue size reporting.
    """Test queue size tracking."""

    def test_get_queue_sizes(self):
        # [FUNCTION DOC]
        # get_queue_sizes is (0, 0) on a fresh runner; after submitting five tasks the
        # input queue size is at most 5.
        """Test that queue sizes can be retrieved."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 0.1s and returns x.
            await asyncio.sleep(0.1)
            return x

        # Initially, queues should be empty
        input_size, output_size = runner.get_queue_sizes()
        assert input_size == 0
        assert output_size == 0

        # Submit some tasks
        for i in range(5):
            runner.submit(task, i, task_id=i)

        time.sleep(0.05)
        input_size, output_size = runner.get_queue_sizes()
        # Some tasks should have been picked up from input queue
        assert input_size <= 5

        runner.destroy()


class TestAsyncTaskRunnerResultOrdering:
    # [CLASS DOC]
    # Result caching across wait() calls and ordering.
    """Test result ordering and caching."""

    def test_result_cache_accumulation(self):
        # [FUNCTION DOC]
        # Ten tasks are consumed through wait calls of 3, 4 and 3; together they cover
        # 0..9.
        """Test that results accumulate in cache across multiple wait() calls."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def fast_task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x.
            await asyncio.sleep(0.01)
            return x

        # Submit 10 tasks
        for i in range(10):
            runner.submit(fast_task, i, task_id=i)

        # Wait for first 3
        results1 = runner.wait(count=3, timeout=2.0)
        assert len(results1) == 3

        # Wait for next 4
        results2 = runner.wait(count=4, timeout=2.0)
        assert len(results2) == 4

        # Wait for remaining 3
        results3 = runner.wait(count=3, timeout=2.0)
        assert len(results3) == 3

        # All results combined should be the full set
        all_results = results1 + results2 + results3
        assert set(all_results) == set(range(10))

        runner.destroy()

    def test_results_are_shuffled(self):
        # [FUNCTION DOC]
        # Twenty tasks' results are asserted not to come back in submission order.
        # KNOWN ISSUE: Probabilistic; a random shuffle can reproduce the identity order
        # (very unlikely), making the test theoretically flaky.
        """Test that results are returned in random order."""
        runner = AsyncTaskRunner[int](max_queue_size=50)
        runner.initialize()

        async def task(x: int) -> int:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x.
            await asyncio.sleep(0.01)
            return x

        # Submit tasks in order
        for i in range(20):
            runner.submit(task, i, task_id=i)

        results = runner.wait(count=20, timeout=5.0)

        # Results should not be in strict sequential order (extremely unlikely
        # with shuffle)
        # Check that at least some results are out of order
        assert results != list(range(20)), "Results should be shuffled"

        runner.destroy()


class TestAsyncTaskRunnerWithDifferentTypes:
    # [CLASS DOC]
    # Generic result types.
    """Test AsyncTaskRunner with various result types."""

    def test_with_string_results(self):
        # [FUNCTION DOC]
        # String results: concat('hello', ' world') returns 'hello world'.
        """Test with string return types."""
        runner = AsyncTaskRunner[str](max_queue_size=10)
        runner.initialize()

        async def concat(a: str, b: str) -> str:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns a + b.
            await asyncio.sleep(0.01)
            return a + b

        runner.submit(concat, "hello", " world", task_id=1)
        results = runner.wait(count=1, timeout=2.0)

        assert results[0] == "hello world"
        runner.destroy()

    def test_with_dict_results(self):
        # [FUNCTION DOC]
        # Dict results: both {'a': 1} and {'b': 2} are returned.
        """Test with dictionary return types."""
        runner = AsyncTaskRunner[dict](max_queue_size=10)
        runner.initialize()

        async def create_dict(key: str, value: int) -> dict:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns {key: value}.
            await asyncio.sleep(0.01)
            return {key: value}

        runner.submit(create_dict, "a", 1, task_id=1)
        runner.submit(create_dict, "b", 2, task_id=2)

        results = runner.wait(count=2, timeout=2.0)

        assert len(results) == 2
        assert {"a": 1} in results
        assert {"b": 2} in results

        runner.destroy()

    def test_with_none_results(self):
        # [FUNCTION DOC]
        # Optional results: None values are returned as results (not dropped); the non-
        # None values are the even numbers 0..8.
        """Test with None return type (e.g., for filtering)."""
        runner = AsyncTaskRunner[int | None](max_queue_size=20)
        runner.initialize()

        async def filter_even(x: int) -> int | None:
            # [FUNCTION DOC]
            # HELPER: Async; sleeps 10ms and returns x if even, else None.
            await asyncio.sleep(0.01)
            return x if x % 2 == 0 else None

        # Submit 10 numbers
        for i in range(10):
            runner.submit(filter_even, i, task_id=i)

        # Get all results (including None values)
        results = runner.wait(count=10, timeout=2.0)

        assert len(results) == 10
        # Filter None values
        even_numbers = [x for x in results if x is not None]
        assert set(even_numbers) == {0, 2, 4, 6, 8}

        runner.destroy()
