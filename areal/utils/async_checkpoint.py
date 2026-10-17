# [MODULE DOC]
# Async distributed-checkpoint (torch DCP) save management. Defines AsyncMode and
# AsyncCheckpointManager, which performs synchronous dcp.save or process-based
# dcp.async_save with pinned/shared-memory staging, and runs optional post-save work
# (e.g. consolidation) on a single background thread.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.checkpoint.state_dict_saver import (
    AsyncCheckpointerType,
    AsyncSaveResponse,
)
from torch.distributed.checkpoint.storage import StorageWriter

from areal.utils.logging import getLogger

logger = getLogger("AsyncCheckpoint")


class AsyncMode(str, enum.Enum):
    # [CLASS DOC]
    # String enum of checkpoint modes: AUTO, SYNC, ASYNC. AUTO is not handled by the
    # manager itself (callers are expected to resolve it); with AUTO save() does nothing
    # but post_fn.
    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


class AsyncCheckpointManager:
    # [CLASS DOC]
    # Per-engine async checkpoint manager. State: mode, futures for upload, staging and
    # background post-work, a lazily created DefaultStager, a 1-worker
    # ThreadPoolExecutor (ASYNC only), and two dedicated gloo process groups (one for
    # async_save on the main thread, one for consolidation barriers on the background
    # thread).
    """Manage async DCP checkpoint saves for a single engine.

    Supports two modes:
    - SYNC: synchronous dcp.save()
    - ASYNC: process-based async with pinned memory staging,
      extra CPU pinned memory proportional to per-rank model shard size
    """

    def __init__(self, async_mode: AsyncMode):
        # [FUNCTION DOC]
        # Stores the mode, initializes futures to None, and in ASYNC mode creates the
        # executor and, if torch.distributed is initialized, two new gloo groups
        # (collective call; all ranks must construct the manager). Side effects: thread
        # pool and process group creation.
        self.async_mode = async_mode
        self._save_future: Future | None = None
        self._staging_future: Future | None = None
        self._stager: DefaultStager | None = None
        self._bg_future: Future | None = None

        # Sequential executor for post-upload work (consolidation).
        self._executor: ThreadPoolExecutor | None = None
        if async_mode == AsyncMode.ASYNC:
            self._executor = ThreadPoolExecutor(max_workers=1)

        # Two separate gloo PGs: eliminates need for gating.
        # _pg: used by dcp.async_save on the main thread.
        # _consolidation_pg: used by consolidation barriers on the bg thread.
        self._pg: dist.ProcessGroup | None = None
        self._consolidation_pg: dist.ProcessGroup | None = None
        if async_mode == AsyncMode.ASYNC and dist.is_initialized():
            self._pg = dist.new_group(backend="gloo")
            self._consolidation_pg = dist.new_group(backend="gloo")

    @property
    def is_async(self) -> bool:
        # [FUNCTION DOC]
        # True iff mode is ASYNC.
        return self.async_mode == AsyncMode.ASYNC

    @property
    def process_group(self) -> dist.ProcessGroup | None:
        # [FUNCTION DOC]
        # Returns the gloo group used for async saves (None unless ASYNC with dist
        # initialized).
        return self._pg

    @property
    def consolidation_process_group(self) -> dist.ProcessGroup | None:
        # [FUNCTION DOC]
        # Returns the gloo group reserved for background consolidation barriers (or
        # None).
        return self._consolidation_pg

    def save(
        self,
        state_dict: dict,
        *,
        storage_writer: StorageWriter | None = None,
        checkpoint_id: str | None = None,
        post_fn: Callable | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Starts a checkpoint save of state_dict with optional storage_writer,
        # checkpoint_id and post_fn. Raises RuntimeError in ASYNC mode without a process
        # group. First re-raises any finished background error, blocks (with a warning)
        # on still-running background work, then waits for previous staging and upload.
        # SYNC: blocking dcp.save. ASYNC: creates the stager on first use (pinned +
        # shared memory, async staging, non-blocking copy), calls dcp.async_save with
        # the process-type checkpointer and stores upload/staging futures. Finally
        # submits post_fn (if any) to run after the upload. Returns None.
        """Initiate a save. Sync mode blocks; async mode returns immediately."""
        if self.async_mode == AsyncMode.ASYNC and self._pg is None:
            raise RuntimeError(
                "Async checkpoint requires a process group, but dist was not "
                "initialized when AsyncCheckpointManager was created"
            )

        self._check_bg_error()

        if self._bg_future is not None:
            logger.warning("Background consolidation still running. Blocking.")
            try:
                self._bg_future.result()
            finally:
                self._bg_future = None

        self._wait_for_staging()
        self._wait_for_upload()

        if self.async_mode == AsyncMode.SYNC:
            dcp.save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
            )
        elif self.async_mode == AsyncMode.ASYNC:
            if self._stager is None:
                self._stager = DefaultStager(
                    StagingOptions(
                        use_pinned_memory=True,
                        use_shared_memory=True,
                        use_async_staging=True,
                        use_non_blocking_copy=True,
                    )
                )
            result = dcp.async_save(
                state_dict,
                storage_writer=storage_writer,
                checkpoint_id=checkpoint_id,
                process_group=self._pg,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                async_stager=self._stager,
            )
            assert isinstance(result, AsyncSaveResponse)
            self._save_future = result.upload_completion
            self._staging_future = result.staging_completion

        if post_fn is not None:
            self._submit_post_fn(post_fn)

    def _submit_post_fn(self, fn: Callable) -> None:
        # [FUNCTION DOC]
        # HELPER: Schedules fn to run after the current upload future completes (upload
        # errors are logged, fn still runs). Uses the background executor if present,
        # otherwise runs synchronously in the caller thread. Stores the background
        # future.
        """Submit a bg job: wait for upload -> run *fn*."""
        upload_future = self._save_future

        def _upload_then_consolidate():
            # [FUNCTION DOC]
            # HELPER: Nested job body: waits for the captured upload future (logging any
            # failure), then calls fn; exceptions from fn propagate into the background
            # future.
            if upload_future is not None:
                try:
                    upload_future.result()
                except Exception:
                    logger.error("Async checkpoint upload failed", exc_info=True)
            fn()

        if self._executor is not None:
            self._bg_future = self._executor.submit(_upload_then_consolidate)
        else:
            _upload_then_consolidate()

    def maybe_wait_for_staging(self) -> None:
        # [FUNCTION DOC]
        # Called by the training loop before parameters change: surfaces finished
        # background errors and blocks until GPU-to-CPU staging is done.
        """Wait for staging (GPU->CPU) + check for bg errors.

        Called by the training loop before optimizer.step().
        """
        self._check_bg_error()
        self._wait_for_staging()

    def _check_bg_error(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If the background future is done, retrieves its result (re-raising its
        # exception) and clears it. Non-blocking otherwise.
        """Non-blocking: only raises if bg job is done and failed."""
        if self._bg_future is not None and self._bg_future.done():
            try:
                self._bg_future.result()
            finally:
                self._bg_future = None

    def _wait_for_staging(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Blocks on the staging future if any, logging (not raising) failures,
        # then clears it.
        """Wait for staging (GPU->CPU) to complete."""
        if self._staging_future is not None:
            try:
                self._staging_future.result()
            except Exception:
                logger.error("Async checkpoint staging failed", exc_info=True)
            finally:
                self._staging_future = None

    def _wait_for_upload(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Blocks on the previous upload future if any, swallowing errors (logged
        # by the background job), then clears it, so shared staging memory can be
        # reused.
        """Wait for the previous upload to complete.

        Needed before new staging to ensure shared memory is free.
        """
        if self._save_future is not None:
            try:
                self._save_future.result()
            except Exception:
                pass  # error is logged by the bg thread
            finally:
                self._save_future = None

    def finalize(self) -> None:
        # [FUNCTION DOC]
        # End-of-training cleanup: waits for staging, waits for background work (logs
        # and re-raises its error), waits for upload; always shuts down the executor,
        # closes the stager and destroys both gloo process groups.
        """Call at training end: wait for pending operations and release resources."""
        try:
            self._wait_for_staging()
            if self._bg_future is not None:
                try:
                    self._bg_future.result()
                except Exception:
                    logger.error("Final bg work failed", exc_info=True)
                    raise
                finally:
                    self._bg_future = None
            self._wait_for_upload()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            if self._stager is not None:
                self._stager.close()
                self._stager = None
            if self._pg is not None:
                dist.destroy_process_group(self._pg)
                self._pg = None
            if self._consolidation_pg is not None:
                dist.destroy_process_group(self._consolidation_pg)
                self._consolidation_pg = None
