# [MODULE DOC]
# Vendored scaffolding scheduler. ScaffoldingLlm owns a private asyncio event loop
# running in a daemon thread; callers submit prompts (thread-safe), each request clones
# the prototype controller and runs its generator, dispatching yielded task lists to
# workers by worker_tag and running ParallelProcess branches concurrently, with a cap on
# concurrently running requests. Results are delivered through ScaffoldingResult.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.scaffolding_llm

import asyncio
import threading
import traceback
from collections import deque
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Any

from .controller import Controller, ParallelProcess
from .result import ScaffoldingResult
from .task import Task
from .worker import Worker


@dataclass(frozen=True)
class ScaffoldingRequest:
    # [CLASS DOC]
    # Frozen dataclass describing one queued request: prompt, kwargs for
    # controller.generate, the (cloned) controller instance, and the ScaffoldingResult
    # to fill.
    prompt: str
    kwargs: Mapping[str, Any]
    controller: Controller
    result: "ScaffoldingResult"


class ScaffoldingLlm:
    # [CLASS DOC]
    # Scheduler/front-end tying controllers to workers. Key state: workers (worker_tag
    # -> Worker), loop and loop thread, task_queue (asyncio.Queue of requests, None =
    # stop), running_req_count / max_parallel_requests / pending_queue for admission
    # control, output_task_collection flag.
    def __init__(
        self,
        prototype_controller: Controller,
        workers: Mapping[str, Worker],  # map of role to worker instance,
        max_parallel_requests: int = 64,
    ):
        # [FUNCTION DOC]
        # Stores the prototype controller and the worker mapping (keys must equal the
        # worker_tag values controllers assign), creates a new event loop, starts the
        # loop thread and blocks until it has created its asyncio primitives;
        # initializes scheduling counters (max_parallel_requests default 64) and
        # disables task-collection output.
        self.prototype_controller = prototype_controller
        self.workers = workers

        # Always create a dedicated event loop in a separate thread.
        # This avoids deadlocks when ScaffoldingLlm is used inside another
        # event loop (e.g. AsyncTaskRunner's uvloop), where fire-and-forget
        # tasks created via create_task() would never get executed.
        #
        # asyncio primitives (Queue, Event) are created inside the loop
        # thread to ensure they are bound to the correct event loop.
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._run_main_loop_thread()
        self._ready.wait()

        # For top scheduler
        self.running_req_count = 0
        self.max_parallel_requests = max_parallel_requests
        self.pending_queue = deque()

        self.output_task_collection = False

    def __enter__(self):
        # [FUNCTION DOC]
        # Context-manager entry; returns self.
        return self

    def __exit__(self):
        # [FUNCTION DOC]
        # Context-manager exit; calls shutdown().
        # KNOWN ISSUE: Signature takes no exception arguments, so using the object in a
        # with-statement raises TypeError on exit.
        self.shutdown()

    async def _handle_controller_generator(
        self, gen: Generator, request: ScaffoldingRequest = None
    ):
        # [FUNCTION DOC]
        # HELPER: Iterates a controller generator synchronously; for each yielded item
        # awaits either _handle_parallel_process (ParallelProcess) or _handle_task_list
        # (anything else, assumed list of Tasks). The generator's return value is
        # discarded here.
        """Handle a controller generator, processing tasks and parallel processes."""
        for obj in gen:
            if isinstance(obj, ParallelProcess):
                await self._handle_parallel_process(obj, request)
            else:
                await self._handle_task_list(obj, request)

    async def _handle_task_list(
        self, tasks: list[Task], request: ScaffoldingRequest = None
    ):
        # [FUNCTION DOC]
        # HELPER: Runs worker.run_task for every task concurrently (worker chosen by
        # task.worker_tag; KeyError if missing) and waits for all. Afterwards, for tasks
        # with streaming_output_flag, forwards each buffered streaming output to the
        # request's result and clears the buffer.
        """Execute a list of tasks concurrently."""
        async_tasks = [
            asyncio.create_task(self.workers[task.worker_tag].run_task(task))
            for task in tasks
        ]
        await asyncio.gather(*async_tasks)
        for task in tasks:
            if task.streaming_output_flag:
                for output in task.streaming_output_list:
                    request.result.set_output_streaming(output)
                task.streaming_output_list = []

    async def _handle_parallel_process(
        self, tasks: ParallelProcess, request: ScaffoldingRequest = None
    ):
        # [FUNCTION DOC]
        # HELPER: Runs every sub-generator of the ParallelProcess concurrently via
        # _handle_controller_generator and waits for all.
        """Handle parallel execution of multiple generators."""
        async_tasks = [
            asyncio.create_task(self._handle_controller_generator(sub_gen, request))
            for sub_gen in tasks.sub_gens
        ]
        await asyncio.gather(*async_tasks)

    async def _handle_single_request(self, request: ScaffoldingRequest):
        # [FUNCTION DOC]
        # HELPER: Runs one request to completion. On exception prints the error and
        # traceback to stdout, terminates the result with no output, and re-raises
        # (inside a fire-and-forget task). Always decrements running_req_count and tries
        # to schedule pending requests.
        """Process a single scaffolding request."""
        try:
            gen = self._create_controller_generator(request)
            await self._handle_controller_generator(gen, request)
        except Exception as e:
            print(f"ScaffoldingLLM request exception: {e}")
            traceback.print_exc()
            request.result.set_output(None)
            raise
        finally:
            self.running_req_count -= 1
            self._maybe_schedule()

    def _create_controller_generator(self, request: ScaffoldingRequest):
        # [FUNCTION DOC]
        # HELPER: Generator wrapper: delegates to controller.generate(prompt, **kwargs),
        # capturing its returned ScaffoldingOutput; then, if output_task_collection is
        # enabled, attaches controller.task_collections to the result, and finally sets
        # the output (which also terminates the result stream).
        """Create a generator wrapper for the controller."""
        scaffolding_output = yield from request.controller.generate(
            request.prompt, **request.kwargs
        )

        if self.output_task_collection:
            request.result.set_task_collections(request.controller.task_collections)
        request.result.set_output(scaffolding_output)

    def _schedule_request(self, request: ScaffoldingRequest):
        # [FUNCTION DOC]
        # HELPER: Starts _handle_single_request as a fire-and-forget asyncio task on the
        # current loop and increments running_req_count.
        """Schedule a single request for execution."""
        asyncio.create_task(self._handle_single_request(request))
        self.running_req_count += 1

    def _maybe_schedule(self, request: ScaffoldingRequest = None):
        # [FUNCTION DOC]
        # HELPER: Unless shutdown_event is set, appends request (if given) to
        # pending_queue and starts pending requests FIFO while running_req_count <
        # max_parallel_requests.
        """Schedule pending requests if capacity allows."""
        if self.shutdown_event.is_set():
            return

        if request is not None:
            self.pending_queue.append(request)

        while (
            self.running_req_count < self.max_parallel_requests and self.pending_queue
        ):
            next_request = self.pending_queue.popleft()
            self._schedule_request(next_request)

    async def _handle_event_loop(self):
        # [FUNCTION DOC]
        # HELPER: Consumes task_queue forever: None ends the loop, ScaffoldingRequest is
        # admitted via _maybe_schedule, anything else raises ValueError.
        """Main event handling loop."""
        while True:
            item = await self.task_queue.get()

            if item is None:
                return
            elif isinstance(item, ScaffoldingRequest):
                self._maybe_schedule(item)
            else:
                raise ValueError(f"Unsupported task_queue item type: {type(item)}")

    async def _main_loop_async_func(self):
        # [FUNCTION DOC]
        # HELPER: Runs the event-handling loop to completion, then sets
        # main_loop_stop_event.
        """Main async loop function."""
        handle_event_task = asyncio.create_task(self._handle_event_loop())
        await handle_event_task
        self.main_loop_stop_event.set()

    def _run_main_loop_thread(self):
        # [FUNCTION DOC]
        # HELPER: Starts a daemon thread that sets self.loop as its event loop, creates
        # task_queue, main_loop_stop_event and shutdown_event, signals readiness via the
        # threading Event, and runs the main async function until it completes.
        def main_loop_thread():
            # [FUNCTION DOC]
            # HELPER: Thread target described above.
            asyncio.set_event_loop(self.loop)
            # Create asyncio primitives inside the loop thread.
            self.task_queue = asyncio.Queue()
            self.main_loop_stop_event = asyncio.Event()
            self.shutdown_event = asyncio.Event()
            self._ready.set()
            self.loop.run_until_complete(self._main_loop_async_func())

        self.main_loop_thread = threading.Thread(target=main_loop_thread, daemon=True)
        self.main_loop_thread.start()

    def generate_async(self, prompt: str) -> ScaffoldingResult:
        # [FUNCTION DOC]
        # Submits one prompt without waiting: creates a ScaffoldingResult, clones the
        # prototype controller synchronously, and schedules (thread-safe) a coroutine on
        # the loop that enqueues a ScaffoldingRequest with empty kwargs. Returns the
        # result handle immediately.
        result = ScaffoldingResult()
        # Clone synchronously here (before any async handoff) to avoid race
        # conditions where concurrent callers mutate prototype_controller state
        # between this call and when put_request actually runs on self.loop.
        cloned_controller = self.prototype_controller.clone()

        async def put_request():
            # [FUNCTION DOC]
            # HELPER: Builds the request and puts it on task_queue; if construction
            # fails, puts None (which stops the whole main loop) and prints the error.
            try:
                request = ScaffoldingRequest(
                    prompt=prompt,
                    kwargs={},
                    result=result,
                    controller=cloned_controller,
                )
            except Exception as e:
                await self.task_queue.put(None)
                print(
                    f"Error: build ScaffoldingRequest failed: {e} \n {traceback.format_exc()}"
                )
            else:
                await self.task_queue.put(request)

        asyncio.run_coroutine_threadsafe(put_request(), self.loop)

        return result

    def generate(
        self, prompts: str | list[str]
    ) -> ScaffoldingResult | list[ScaffoldingResult]:
        # [FUNCTION DOC]
        # Synchronous batch API: submits each prompt via generate_async, then blocks on
        # each result(). Returns a single ScaffoldingResult for a str input or a list
        # for a list input.
        unbatched = not isinstance(prompts, list)
        batched_prompts = [prompts] if unbatched else prompts

        scaffolding_results = []
        for prompt in batched_prompts:
            scaffolding_results.append(self.generate_async(prompt))

        for scaffolding_result in scaffolding_results:
            scaffolding_result.result()

        return scaffolding_results[0] if unbatched else scaffolding_results

    def enable_output_task_collection(self):
        # [FUNCTION DOC]
        # Enables attaching each controller's task_collections to its ScaffoldingResult.
        self.output_task_collection = True

    def shutdown(self, shutdown_workers=False):
        # [FUNCTION DOC]
        # Stops the scheduler: schedules a coroutine on the loop that enqueues the stop
        # sentinel, waits for the main loop to stop and then async-shuts-down each
        # worker; joins the loop thread. If shutdown_workers is true, also calls each
        # worker's synchronous shutdown() from the caller thread.
        # KNOWN ISSUE: The loop's run_until_complete returns as soon as the main loop
        # finishes, so the scheduled coroutine's worker.async_shutdown calls may never
        # run. shutdown_event is never set anywhere.
        def shutdown_workers_func():
            # [FUNCTION DOC]
            # HELPER: Calls shutdown() on every worker.
            for worker in self.workers.values():
                worker.shutdown()

        async def stop_task_on_loop():
            # [FUNCTION DOC]
            # HELPER: Coroutine run on the scheduler loop that performs the stop
            # sequence described above.
            await self.task_queue.put(None)
            await self.main_loop_stop_event.wait()
            for worker in self.workers.values():
                await worker.async_shutdown()

        asyncio.run_coroutine_threadsafe(stop_task_on_loop(), self.loop)
        self.main_loop_thread.join()

        if shutdown_workers:
            shutdown_workers_func()
