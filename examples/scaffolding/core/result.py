# [MODULE DOC]
# Vendored scaffolding result types: ScaffoldingOutput (final text/tokens/data) and
# ScaffoldingResult, a thread-safe future/stream handle that the ScaffoldingLlm loop
# thread fills and callers consume synchronously, by awaiting, or by async iteration.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.result

import asyncio
import queue
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class ScaffoldingOutput:
    # [CLASS DOC]
    # Dataclass for one output: text, token_ids (list of int) and optional arbitrary
    # data payload.
    text: str
    token_ids: list[int]
    data: Any = None


class ScaffoldingResult:
    # [CLASS DOC]
    # Handle for a request's result. Producer puts ScaffoldingOutput items (streaming
    # updates) then None (terminator) on an internal queue.Queue; consumers drain it,
    # keeping only the latest output in outputs[0] and setting a done flag on None.
    """Result object for scaffolding requests.

    Uses a thread-safe ``queue.Queue`` for cross-thread communication so
    that producers (ScaffoldingLlm loop thread) and consumers
    (caller's event loop) can safely exchange data.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Creates the queue, outputs initialized to one empty ScaffoldingOutput, done
        # False, task_collections None.
        super().__init__()
        self._queue: queue.Queue = queue.Queue()
        self.outputs = []
        # only support one output for now, so use an empty obj to init
        self.outputs.append(ScaffoldingOutput("", []))
        self._done = False
        self.task_collections = None

    def set_output(self, output: ScaffoldingOutput | Any):
        # [FUNCTION DOC]
        # Producer API: enqueues output if it is a ScaffoldingOutput (non-outputs are
        # dropped), then always enqueues the None terminator.
        if isinstance(output, ScaffoldingOutput):
            self.set_output_streaming(output)
        # terminate
        self.set_output_streaming(None)

    def set_output_streaming(self, output: ScaffoldingOutput | Any):
        # [FUNCTION DOC]
        # Producer API: enqueues output (or None as terminator) without blocking.
        self._queue.put_nowait(output)

    def set_task_collections(self, task_collections: Mapping[str, Any]):
        # [FUNCTION DOC]
        # Stores the controller's task collections mapping (name -> TaskCollection) for
        # the consumer to inspect after completion.
        self.task_collections = task_collections

    async def _aresult_step(self):
        # [FUNCTION DOC]
        # HELPER: Waits for one queue item in the default thread-pool executor
        # (non-blocking for the event loop); None marks done, otherwise replaces
        # outputs[0].
        """Asynchronously wait for the next item from the thread-safe queue."""
        loop = asyncio.get_running_loop()
        obj = await loop.run_in_executor(None, self._queue.get)
        if obj is None:
            self._done = True
        else:  # obj is ScaffoldingOutput
            self.outputs[0] = obj

    def result(self, timeout: float | None = None) -> "ScaffoldingResult":
        # [FUNCTION DOC]
        # Synchronous consumer: blocks draining the queue until the terminator; timeout
        # applies per item and on expiry returns early (possibly not done). Returns
        # self.
        while not self._done:
            try:
                obj = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if obj is None:
                self._done = True
            else:
                self.outputs[0] = obj
        return self

    async def aresult(self) -> "ScaffoldingResult":
        # [FUNCTION DOC]
        # Async consumer: awaits items until done; returns self.
        while not self._done:
            await self._aresult_step()
        return self

    def __await__(self):
        # [FUNCTION DOC]
        # Makes the object awaitable (delegates to aresult).
        return self.aresult().__await__()

    def __aiter__(self):
        # [FUNCTION DOC]
        # Async iterator protocol; returns self.
        return self

    async def __anext__(self):
        # [FUNCTION DOC]
        # Raises StopAsyncIteration once done; otherwise awaits one queue item and
        # returns self (so each iteration yields the latest streaming state).
        # KNOWN ISSUE: The iteration that consumes the terminator still returns self
        # once more after done is set.
        if self._done:
            raise StopAsyncIteration

        await self._aresult_step()
        return self
