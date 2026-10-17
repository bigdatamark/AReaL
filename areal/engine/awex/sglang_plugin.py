# [MODULE DOC]
# SGLang scheduler plugin enabling AWEX colocated weight updates. On import it rejects
# PYTORCH_CUDA_ALLOC_CONF with expandable_segments (RuntimeError) since that disables
# SGLang's memory saver. register_awex_plugin monkey-patches Scheduler.__init__ (inside
# the scheduler child process) to bind an AwexSchedulerPlugin, which adds awex_* RPC
# methods, starts a background thread that gates per-version updates via the MetaServer
# (env AWEX_META_SERVER_ADDR), and replaces the scheduler event loops so queued updates
# are applied on the main thread while the engine is paused. Can also be run as a module
# to launch an SGLang server with the patched scheduler entry point.
# KNOWN ISSUE: the module docstring says the background thread fetches and queues IPC
# data, but it now only queues version markers; the awex reader collects IPC handles
# itself.
# SPDX-License-Identifier: Apache-2.0

"""AWEX SGLang scheduler plugin for colocated weight transfer.

Patches SGLang's scheduler to inject CUDA IPC weight receiving capabilities.
When AWEX_META_SERVER_ADDR env var is set, starts a background thread that
fetches IPC handles from MetaServer (CPU I/O) and queues them for the
scheduler's main loop to process (CUDA copy on main thread).

Weight transfer flow (mirrors the AWEX reference colocate mode):
  1. Training side: convert params → cuda_ipc_serialize → MetaServer put
  2. Background thread: MetaServer get → queue IPC data (CPU only)
  3. Scheduler main loop: release_memory → deserialize + copy → resume_memory
  4. Main loop: signal done → train side releases shared tensors

Usage:
    # Option 1: Register plugin then launch SGLang
    from areal.engine.awex.sglang_plugin import register_awex_plugin
    register_awex_plugin()

    # Option 2: Run as entry module (replaces sglang.launch_server)
    # python3 -m areal.engine.awex.sglang_plugin --model-path ...
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def assert_alloc_conf_supports_memory_saver(conf: str) -> None:
    # [FUNCTION DOC]
    # Raises RuntimeError if the allocator config string contains
    # expandable_segments:true (case/space-insensitive); otherwise returns None.
    """Reject allocator configs that silently disable SGLang's memory saver.

    torch_memory_saver disables itself when it sees expandable_segments, so
    release/resume becomes a no-op: the rollout never hands its GPU back and
    weight pages stay mapped, which surfaces much later as a colocate OOM or an
    invalid CUDA IPC target. Colocated roles must therefore carry their own
    scheduling_spec env_vars rather than share the actor's.
    """
    if "expandable_segments:true" in conf.lower().replace(" ", ""):
        raise RuntimeError(
            "SGLang's memory saver cannot unmap/remap expandable segments, so "
            f"it would disable itself (PYTORCH_CUDA_ALLOC_CONF={conf!r}). Give "
            "the rollout role its own scheduling_spec env_vars without "
            "expandable_segments instead of sharing the actor's."
        )


assert_alloc_conf_supports_memory_saver(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))


from areal.utils import pkg_version  # noqa: E402
from areal.utils.logging import getLogger  # noqa: E402

logger = getLogger("AwexSGLangPlugin")


SUPPORTED_SGLANG_VERSIONS = ("0.5.9", "0.5.10.post1")


def assert_supported_sglang_version() -> None:
    # [FUNCTION DOC]
    # Raises RuntimeError unless the installed sglang version is exactly one of
    # SUPPORTED_SGLANG_VERSIONS.
    """Refuse to patch a SGLang build whose internals were not verified.

    This plugin reaches into scheduler internals: it wraps Scheduler.__init__,
    replaces both event loops, and backports a model-worker task dispatcher.
    Those touch points move between releases, so a silent mismatch surfaces as
    a hang or a corrupt transfer rather than an import error. Fail loudly
    instead, and extend the tuple only after re-checking the patched surfaces.
    """
    installed = pkg_version.get_version("sglang")
    if installed not in SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            f"AWEX colocate patches SGLang internals and was verified against "
            f"{', '.join(SUPPORTED_SGLANG_VERSIONS)}, but found {installed}. "
            f"Re-check Scheduler.__init__, the event loops, and "
            f"execute_task_in_model_worker before allowing this version."
        )


def _float_env(name: str, default: float) -> float:
    # [FUNCTION DOC]
    # HELPER: Returns float(env[name]) or default when unset/empty; logs a warning and
    # returns default on parse failure.
    value = os.environ.get(name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, value, default)
        return default


class AwexSchedulerPlugin:
    # [CLASS DOC]
    # Per-scheduler plugin object. Holds the scheduler, a lazily created
    # AwexColocateReader, a background thread, a thread-safe queue of version markers,
    # the last applied version, and the paused-loop poll interval (env
    # AWEX_PAUSED_POLL_INTERVAL_S, default 0.01 s, clamped >= 0).
    """Binds awex weight-receive to a SGLang Scheduler instance.

    Architecture: background thread handles MetaServer I/O (CPU only),
    scheduler main loop handles CUDA weight copy (via process_awex_queue).
    """

    def __init__(self, scheduler: Any) -> None:
        # [FUNCTION DOC]
        # Initialises state as described for the class. No I/O beyond reading the env
        # var.
        self._scheduler = scheduler
        self._receiver = None
        self._bg_thread: threading.Thread | None = None
        self._weight_queue: queue.Queue = queue.Queue()
        self._version = 0
        self._paused_poll_interval_s = max(
            0.0, _float_env("AWEX_PAUSED_POLL_INTERVAL_S", 0.01)
        )

    def bind(self) -> None:
        # [FUNCTION DOC]
        # Attaches the awex_* methods and process_awex_queue onto the scheduler instance
        # as attributes, logs, and if AWEX_META_SERVER_ADDR is set, starts the
        # background worker and patches the event loops.
        methods = [
            "awex_init_receiver",
            "awex_receive_weights",
            "awex_release_memory",
            "awex_resume_memory",
            "awex_get_weight_metadata",
            "awex_get_parallelism",
            "process_awex_queue",
        ]
        for name in methods:
            setattr(self._scheduler, name, getattr(self, name))
        logger.info(
            f"[AWEX] AwexSchedulerPlugin bound {len(methods)} methods to scheduler",
        )

        meta_server_addr = os.environ.get("AWEX_META_SERVER_ADDR")
        if meta_server_addr:
            self._start_background_worker(meta_server_addr)
            self._patch_event_loop()

    def _require_receiver(self):
        # [FUNCTION DOC]
        # HELPER: Lazily constructs and returns the AwexColocateReader for the
        # scheduler.
        if self._receiver is None:
            from areal.engine.awex.colocate_reader import AwexColocateReader

            self._receiver = AwexColocateReader(self._scheduler)
        return self._receiver

    def awex_init_receiver(self, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # Forwards keyword arguments to the reader's initialize.
        self._require_receiver().initialize(**kwargs)

    def awex_receive_weights(self, version: int = 0) -> None:
        # [FUNCTION DOC]
        # Runs the reader's update_weights for the given version.
        self._require_receiver().update_weights(version)

    def awex_release_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Forwards to the reader's release_memory with tags.
        self._require_receiver().release_memory(tags)

    def awex_resume_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Forwards to the reader's resume_memory with tags.
        self._require_receiver().resume_memory(tags)

    def awex_get_weight_metadata(self) -> list:
        # [FUNCTION DOC]
        # Returns the reader's inference parameters metadata.
        return self._require_receiver().get_weight_metadata()

    def awex_get_parallelism(self) -> dict:
        # [FUNCTION DOC]
        # Returns the reader's parallelism dict.
        return self._require_receiver().get_parallelism()

    # ── Main loop hook: process queued weight updates ─────────────────

    def process_awex_queue(self) -> None:
        # [FUNCTION DOC]
        # Main-loop hook; must be called by all TP ranks together. Checks whether the
        # local queue is non-empty and, with TP > 1, all-reduces MIN over the TP CPU
        # group so it proceeds only when every rank has an item. Then pops the version
        # marker, waits for the writer's offload signal, resumes SGLang 'weights'
        # memory, runs the reader's update_weights (logging and re-raising exceptions),
        # signals finished, and records the version as applied. Side effects:
        # collectives, GPU memory changes, MetaServer I/O, logging.
        """Called from scheduler main loop. Processes pending weight updates.

        This is a TP-collective operation: ALL TP ranks must call it together
        (since it's called between recv_requests() calls which use broadcast_pyobj).

        Uses all_reduce(MIN) to check if all TP ranks have a pending update.
        Only proceeds when ALL ranks have queued an update, preventing the deadlock
        where one rank blocks in CUDA ops while others wait in broadcast_pyobj.

        We act as the awex *driver* layer (the community SGLang scheduler has no
        ``execute_task_in_model_worker`` driver). The collect-IPC + StreamBatch
        transport + writer handshake is delegated to the awex-native worker reader
        (``AwexColocateReader.update_weights`` -> ``NCCLWorkerWeightsReader``). We
        only own the driver-equivalent steps around it:
          1. Wait for all_training_offloaded_weights (= driver _pre_update_weights)
          2. resume_memory_occupation(weights) — re-allocate infer weight buffers
          3. reader.update_weights(version) — awex worker reader does the rest:
             collect IPC + StreamBatch transport + put weights_update_finished
             + barrier + get_then_delete write_finished + flush_cache
          4. signal_finished_weights_update (= driver _resume_kvcache)
        """
        import torch
        import torch.distributed

        tp_cpu_group = self._scheduler.tp_cpu_group
        tp_size = self._scheduler.tp_size

        has_item = 1 if not self._weight_queue.empty() else 0

        if tp_size > 1:
            has_item_tensor = torch.tensor([has_item], dtype=torch.int32)
            torch.distributed.all_reduce(
                has_item_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=tp_cpu_group,
            )
            all_ready = has_item_tensor.item() == 1
        else:
            all_ready = has_item == 1

        if not all_ready:
            return

        item = self._weight_queue.get_nowait()
        version = item["version"]
        gpu_id = getattr(self._scheduler, "gpu_id", "?")
        logger.info(
            f"[AWEX] main loop: processing weight update v{version} (gpu_id={gpu_id})",
        )

        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        receiver = self._require_receiver()

        # Step 1: Wait for writer to offload its model weights first (= awex driver
        # _pre_update_weights). Ensures no 2x model weights on GPU simultaneously.
        # The background thread already gated on this, so this returns immediately;
        # kept for driver-equivalent clarity.
        logger.info(
            f"[AWEX] main loop: waiting for all_training_offloaded_weights (gpu_id={gpu_id})",
        )
        receiver.wait_for_training_offloaded(version)
        logger.info(
            f"[AWEX] main loop: writer offloaded weights confirmed (gpu_id={gpu_id})",
        )

        # Step 2: Resume weight memory (memory_saver re-allocates buffers).
        resume_req = ResumeMemoryOccupationReqInput(tags=["weights"])
        self._scheduler.resume_memory_occupation(resume_req)
        logger.info(
            f"[AWEX] main loop: resumed weight memory for v{version} (gpu_id={gpu_id})",
        )

        # Step 3: Delegate the whole collect-IPC + StreamBatch transport + writer
        # handshake (put weights_update_finished + barrier + get_then_delete
        # write_finished + flush_cache) to the awex-native worker reader.
        try:
            receiver.update_weights(version)
            logger.info(
                f"[AWEX] main loop: weight update done for v{version} (gpu_id={gpu_id})",
            )
        except Exception:
            logger.exception(
                "AWEX main loop failed to update weights v%s on gpu_id=%s",
                version,
                gpu_id,
            )
            raise

        # Step 4: Signal that this infer engine finished weight update, so the
        # writer can resume kv_cache (= awex driver _resume_kvcache).
        receiver.signal_finished_weights_update()
        self._version = version

    # ── Patch scheduler event loop to call process_awex_queue ─────────

    def _patch_event_loop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces scheduler.event_loop_overlap and event_loop_normal with
        # versions that, whenever the engine is paused, call process_awex_queue and
        # sleep the poll interval instead of running batches; otherwise they replicate
        # SGLang's normal/overlap scheduling logic. If the scheduler has
        # log_decode_stats hooks, wraps them to record the decode counter at their last
        # call so decode metrics skipped by the patched loop can be re-emitted.
        # KNOWN ISSUE: the replicated loops copy SGLang internals and must be kept in
        # sync with the supported versions; the saved original loop references are
        # unused.
        """Inject process_awex_queue into scheduler's event loops.

        SGLang uses event_loop_overlap by default. Patch both for safety.
        Weight updates process when engine is paused (no ongoing inference).
        """
        scheduler = self._scheduler
        plugin = self

        _decode_hooks_available = hasattr(scheduler, "log_decode_stats") and hasattr(
            scheduler, "log_decode_stats_every_iteration"
        )
        if _decode_hooks_available:
            _orig_log_decode_stats = scheduler.log_decode_stats
            _orig_log_decode_stats_every_iteration = (
                scheduler.log_decode_stats_every_iteration
            )

            def _tracked_log_decode_stats(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Records forward_ct_decode on the scheduler, then calls the
                # original log_decode_stats.
                scheduler._areal_awex_last_decode_stats_ct = getattr(
                    scheduler, "forward_ct_decode", None
                )
                return _orig_log_decode_stats(*args, **kwargs)

            def _tracked_log_decode_stats_every_iteration(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Records forward_ct_decode, then calls the original per-
                # iteration decode stats logger.
                scheduler._areal_awex_last_decode_stats_every_iter_ct = getattr(
                    scheduler, "forward_ct_decode", None
                )
                return _orig_log_decode_stats_every_iteration(*args, **kwargs)

            scheduler.log_decode_stats = _tracked_log_decode_stats
            scheduler.log_decode_stats_every_iteration = (
                _tracked_log_decode_stats_every_iteration
            )
        else:
            logger.warning(
                "[AWEX] sglang scheduler has no log_decode_stats hooks "
                "(removed in sglang>=0.5.10); skipping decode-stats tracking"
            )

        def _maybe_restore_decode_metrics(stage, batch, result):
            # [FUNCTION DOC]
            # HELPER: After processing a DECODE batch result, when hooks exist,
            # AREAL_AWEX_FORCE_SGLANG_METRICS is '1' (default) and scheduler metrics are
            # enabled: calls log_decode_stats if the decode counter hits
            # decode_log_interval and it was not already logged for this counter, and
            # calls the per-iteration logger if not already called for this counter.
            if not _decode_hooks_available:
                return
            if os.environ.get("AREAL_AWEX_FORCE_SGLANG_METRICS", "1") != "1":
                return
            if stage != "after_process_batch_result" or batch is None:
                return
            mode = getattr(getattr(batch, "forward_mode", None), "name", None)
            if mode != "DECODE":
                return
            if not getattr(scheduler, "current_scheduler_metrics_enabled", False):
                return

            current_ct = getattr(scheduler, "forward_ct_decode", None)
            interval = (
                getattr(
                    getattr(scheduler, "server_args", None), "decode_log_interval", 1
                )
                or 1
            )
            should_log_decode = current_ct is not None and current_ct % interval == 0

            if (
                should_log_decode
                and getattr(scheduler, "_areal_awex_last_decode_stats_ct", None)
                != current_ct
            ):
                can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
                logger.debug(
                    f"[AWEX-METRICS] restoring native log_decode_stats "
                    f"gpu_id={getattr(scheduler, 'gpu_id', '?')} "
                    f"forward_ct_decode={current_ct}",
                )
                scheduler.log_decode_stats(can_run_cuda_graph, running_batch=batch)

            if (
                getattr(scheduler, "_areal_awex_last_decode_stats_every_iter_ct", None)
                != current_ct
            ):
                scheduler.log_decode_stats_every_iteration(
                    batch,
                    num_accepted_tokens=getattr(result, "num_accepted_tokens", 0),
                )

        # Patch event_loop_overlap (the one actually used by SGLang)
        _orig_overlap = scheduler.event_loop_overlap

        def _patched_overlap():
            # [FUNCTION DOC]
            # HELPER: Overlap event loop (never returns): receive and process requests
            # (logging control requests and every 500 loops), handle pause by processing
            # the awex queue, else get the next batch, run it, queue its result and
            # process the previous batch's result (overlapped unless disabled for this
            # batch), self-check when idle, launch sampling for generation models, and
            # track last_batch.
            """Patched overlap loop that checks awex queue when paused."""
            from collections import deque

            scheduler.result_queue = deque()
            _loop_count = 0
            _paused_reported = False

            def pop_and_process():
                # [FUNCTION DOC]
                # HELPER: Pops the oldest (batch, result) from the scheduler's result
                # queue, processes it and restores decode metrics.
                tmp_batch, tmp_result = scheduler.result_queue.popleft()
                scheduler.process_batch_result(tmp_batch, tmp_result)
                _maybe_restore_decode_metrics(
                    "after_process_batch_result", tmp_batch, tmp_result
                )

            logger.info(
                f"[AWEX] _patched_overlap STARTING (gpu_id={getattr(scheduler, 'gpu_id', '?')})",
            )

            while True:
                recv_reqs = scheduler.recv_requests()
                if recv_reqs:
                    req_types = [type(r).__name__ for r in recv_reqs]
                    has_control = any(
                        t
                        not in (
                            "TokenizedGenerateReqInput",
                            "TokenizedEmbeddingReqInput",
                        )
                        for t in req_types
                    )
                    if has_control or _loop_count % 500 == 0:
                        logger.info(
                            f"[AWEX] loop gpu_id={getattr(scheduler, 'gpu_id', '?')}: "
                            f"recv {len(recv_reqs)} reqs, types={req_types[:5]}, "
                            f"_engine_paused={scheduler._engine_paused}, loop={_loop_count}",
                        )

                was_paused = scheduler._engine_paused
                scheduler.process_input_requests(recv_reqs)
                if scheduler._engine_paused != was_paused:
                    logger.info(
                        f"[AWEX] _engine_paused CHANGED: {was_paused} → {scheduler._engine_paused} "
                        f"(gpu_id={getattr(scheduler, 'gpu_id', '?')}, loop={_loop_count})",
                    )

                if scheduler._engine_paused:
                    if not _paused_reported:
                        logger.info(
                            f"[AWEX] _patched_overlap: _engine_paused=True detected! "
                            f"(gpu_id={getattr(scheduler, 'gpu_id', '?')}, loop_count={_loop_count})",
                        )
                        _paused_reported = True
                    plugin.process_awex_queue()
                    time.sleep(plugin._paused_poll_interval_s)
                    continue

                _loop_count += 1
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                disable_overlap_for_batch = scheduler.is_disable_overlap_for_batch(
                    batch
                )

                if disable_overlap_for_batch:
                    pop_and_process()

                if batch:
                    batch_result = scheduler.run_batch(batch)
                    scheduler.result_queue.append((batch.copy(), batch_result))
                else:
                    batch_result = None

                if scheduler.last_batch:
                    if not disable_overlap_for_batch:
                        pop_and_process()
                elif batch is None:
                    scheduler.self_check_during_idle()

                if scheduler.is_generation:
                    scheduler.launch_batch_sample_if_needed(batch_result)

                scheduler.last_batch = batch

        scheduler.event_loop_overlap = _patched_overlap

        # Also patch event_loop_normal as fallback
        _orig_normal = scheduler.event_loop_normal

        def _patched_normal():
            # [FUNCTION DOC]
            # HELPER: Non-overlap event loop (never returns): receive/process requests,
            # process the awex queue while paused, otherwise run the next batch
            # synchronously and process its result, or self-check when idle.
            logger.info(
                f"[AWEX] _patched_normal STARTING (gpu_id={getattr(scheduler, 'gpu_id', '?')})",
            )
            while True:
                recv_reqs = scheduler.recv_requests()
                scheduler.process_input_requests(recv_reqs)
                if scheduler._engine_paused:
                    plugin.process_awex_queue()
                    time.sleep(plugin._paused_poll_interval_s)
                    continue
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                if batch:
                    result = scheduler.run_batch(batch)
                    scheduler.process_batch_result(batch, result)
                    _maybe_restore_decode_metrics(
                        "after_process_batch_result", batch, result
                    )
                else:
                    scheduler.self_check_during_idle()
                scheduler.last_batch = batch

        scheduler.event_loop_normal = _patched_normal
        logger.info(
            "[AWEX] Patched event_loop_overlap + event_loop_normal with awex queue",
        )

    # ── Background thread: MetaServer I/O only (no CUDA ops) ─────────

    def _start_background_worker(self, meta_server_addr: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Starts a daemon thread running _background_worker with the MetaServer
        # address and logs it.
        self._bg_thread = threading.Thread(
            target=self._background_worker,
            args=(meta_server_addr,),
            daemon=True,
        )
        self._bg_thread.start()
        gpu_id = int(getattr(self._scheduler, "gpu_id", -1))
        logger.info(
            f"[AWEX] Started background worker thread "
            f"(gpu_id={gpu_id}, meta_server={meta_server_addr})",
        )

    def _background_worker(self, meta_server_addr: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Thread body. Sets the CUDA device to the scheduler's gpu_id,
        # initialises the receiver (logs and exits the thread on failure), reads the
        # writer's starting version from MetaServer key
        # awex_writer_version_<ip>_<physical gpu>. Then loops: wait until that version's
        # IPC weights key exists, enqueue a version marker, busy-wait (0.1 s) until the
        # main loop applied it, increment the version. On exceptions logs, retries with
        # exponential backoff (capped at 30 s) up to AWEX_READER_MAX_RETRIES (default
        # 1000) consecutive failures, then gives up.
        # KNOWN ISSUE: if process_awex_queue raises on the main thread, _version never
        # advances and this thread spins forever in the wait loop.
        """Initialize the reader, then gate weight-update triggers to the main loop.

        This thread does NOT perform any CUDA memory writes. It only:
        1. Connects to MetaServer and initializes the awex worker reader
        2. Blocks on the per-version writer-offload signal (a set-size wait)
        3. Enqueues a version marker so the TP-collective main-loop gate fires
           (the awex worker reader collects the IPC handles itself inside
           update_weights, so no large payload is prefetched here)
        """
        import torch

        gpu_id = int(getattr(self._scheduler, "gpu_id", 0))
        torch.cuda.set_device(gpu_id)
        logger.info(f"[AWEX] background worker: set CUDA device to {gpu_id}")

        try:
            self._init_receiver_from_meta_server(meta_server_addr)
        except Exception:
            logger.exception("AWEX background worker initialization failed")
            return

        logger.info(
            "[AWEX] background worker: initialization complete, entering fetch loop",
        )
        # Recover can resume weight-transfer versions from the checkpoint step
        # instead of v1, so sync the first version from the writer.
        from awex.meta.meta_server import MetaServerClient as _MSC
        from awex.util.common import get_ip_address as _get_ip

        _host, _port = meta_server_addr.rsplit(":", 1)
        _ver_client = _MSC(_host, int(_port))
        from areal.engine.awex.colocate_writer import (
            awex_colocate_timeout_s,
            resolve_physical_gpu_id,
        )

        # Shares a key namespace with the training writer, so it needs the
        # physical GPU id rather than the mask-relative index.
        _ver_key = f"awex_writer_version_{_get_ip()}_{resolve_physical_gpu_id(gpu_id)}"

        version = int(
            _ver_client.get_object(
                _ver_key,
                timeout=awex_colocate_timeout_s(),
            )
        )
        logger.info(
            f"[AWEX] background worker: writer stream starts at v{version}",
        )
        retries = 0
        # Slow online environments can spend tens of minutes between updates.
        # Keep the reader alive across transient wait timeouts.
        max_retries = int(os.environ.get("AWEX_READER_MAX_RETRIES", "1000"))

        while True:
            try:
                # Block on THIS version's writer-published IPC handles
                # (existence-only probe, no deserialization). This is the
                # per-version trigger: the writer only publishes v+1's key in the
                # next training cycle, so the background thread cannot fire early
                # off a stale unversioned set and dead-lock the main loop. See
                # AwexColocateReader.wait_for_weights_ready for the full rationale.
                logger.info(
                    f"[AWEX] background worker: waiting for writer weights v{version}",
                )
                receiver = self._require_receiver()
                receiver.wait_for_weights_ready(version)
                logger.info(
                    f"[AWEX] background worker: writer published v{version}, "
                    f"queuing for main loop",
                )

                # Queue a version marker for the main loop (no CUDA ops here).
                self._weight_queue.put({"version": version})

                # Wait for main loop to finish processing before gating the next.
                while self._version < version:
                    time.sleep(0.1)

                version += 1
                retries = 0
            except Exception as e:
                retries += 1
                logger.exception(
                    "AWEX background worker failed while waiting for writer "
                    "weights v%s (attempt %s/%s): %s",
                    version,
                    retries,
                    max_retries,
                    e,
                )
                if retries >= max_retries:
                    logger.info(
                        f"[AWEX] background worker: giving up after {max_retries} failures",
                    )
                    break
                time.sleep(min(2**retries, 30))

    def _init_receiver_from_meta_server(self, meta_server_addr: str):
        # [FUNCTION DOC]
        # HELPER: Connects a MetaServerClient (up to 60 attempts, 5 s apart;
        # RuntimeError if all fail), waits for awex_train_info, sets infer world size
        # equal to train world size, computes transfer_rank = SLURM_NODEID * (world /
        # SLURM_NNODES) + gpu_id, and initialises the receiver with that rank and the
        # physical GPU id.
        # KNOWN ISSUE: the transfer rank formula assumes equal GPUs per node and SLURM
        # env vars; outside SLURM multi-node runs all nodes would compute colliding
        # ranks.
        """Connect to MetaServer, get train info, initialize colocate receiver."""
        from awex.meta.meta_server import MetaServerClient

        host, port = meta_server_addr.rsplit(":", 1)

        client = None
        for attempt in range(60):
            try:
                client = MetaServerClient(host, int(port))
                break
            except Exception:
                if attempt % 10 == 0:
                    logger.info(
                        f"[AWEX] background worker: MetaServer not ready, retrying... "
                        f"(attempt {attempt + 1}, addr={meta_server_addr})",
                    )
                time.sleep(5)
        if client is None:
            raise RuntimeError(
                f"Failed to connect to MetaServer at {meta_server_addr} after 60 attempts"
            )

        logger.info(
            f"[AWEX] background worker: connected to MetaServer at {meta_server_addr}",
        )

        receiver = self._require_receiver()

        # `gpu_id` is node-local. Multi-node colocate needs a globally unique
        # transfer rank that stays physically paired with the training process.
        gpu_id = int(getattr(self._scheduler, "gpu_id", 0))
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))

        logger.info(
            f"[AWEX] background worker: waiting for awex_train_info "
            f"(gpu_id={gpu_id}, node_id={node_id}, nnodes={nnodes})",
        )
        # The driver publishes awex_train_info only after rollout init finishes,
        # so large models need the same timeout budget as the weight path.
        from areal.engine.awex.colocate_writer import (
            awex_colocate_timeout_s,
            resolve_physical_gpu_id,
        )

        train_info = client.get_object(
            "awex_train_info",
            timeout=awex_colocate_timeout_s(),
        )
        train_world_size = train_info["train_world_size"]
        # In colocate mode train and infer share the same N physical GPUs, so the
        # global infer NCCL world spans the same N ranks (numerically == train
        # world). This is a *physical* coincidence (same GPUs), NOT a requirement
        # that train/infer parallel topologies match: the infer side decomposes
        # into num_infer_engines DP replicas inside receiver.initialize().
        infer_world_size = train_world_size

        n_gpus_per_node = max(1, infer_world_size // nnodes)
        transfer_rank = node_id * n_gpus_per_node + gpu_id

        logger.info(
            f"[AWEX] background worker: got train_world_size={train_world_size}, "
            f"infer_world_size={infer_world_size}, n_gpus_per_node={n_gpus_per_node}, "
            f"transfer_rank={transfer_rank}",
        )

        # transfer_rank stays a relative index so it lines up with the infer
        # NCCL world, but the CUDA IPC keys are shared with the training writer
        # and must therefore use physical GPU ids.
        receiver.initialize(
            meta_server_addr=meta_server_addr,
            transfer_rank=transfer_rank,
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            local_gpu_id=resolve_physical_gpu_id(gpu_id),
        )
        logger.info(
            f"[AWEX] background worker: receiver initialized "
            f"(transfer_rank={transfer_rank}, infer_world_size={infer_world_size})",
        )


@dataclass
class ModelWorkerTask:
    # [CLASS DOC]
    # Dataclass describing a callable plus kwargs to run inside the model worker
    # (backport of SGLang PR #13595).
    """Task for execute_task_in_model_worker (PR #13595 backport for SGLang 0.5.9)."""

    task_func: Callable
    kwargs: dict = field(default_factory=dict)


def register_awex_plugin() -> None:
    # [FUNCTION DOC]
    # Checks the SGLang version, then wraps Scheduler.__init__ so every new scheduler
    # binds an AwexSchedulerPlugin and gets execute_task_in_model_worker. Must run in
    # the scheduler child process. Side effect: global monkey-patch, logging.
    """Patch Scheduler.__init__ to inject awex plugin after construction.

    Must be called INSIDE the scheduler child process (not the parent),
    because SGLang spawns scheduler processes via mp.Process with "spawn"
    start method, which doesn't inherit parent-process monkey-patches.
    """
    assert_supported_sglang_version()

    from sglang.srt.managers.scheduler import Scheduler

    _orig_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls the original __init__, binds the plugin and patches the task
        # executor.
        _orig_init(self, *args, **kwargs)
        AwexSchedulerPlugin(self).bind()
        _patch_execute_task_in_model_worker(self)

    Scheduler.__init__ = _patched_init
    logger.info("[AWEX] Patched Scheduler.__init__ with awex plugin")


def _patch_execute_task_in_model_worker(scheduler) -> None:
    # [FUNCTION DOC]
    # HELPER: Attaches execute_task_in_model_worker to the scheduler and, if it has a
    # request dispatcher, registers it as the handler for ModelWorkerTask.
    """Add execute_task_in_model_worker to Scheduler (backport from PR #13595)."""

    def execute_task_in_model_worker(task_spec: ModelWorkerTask):
        # [FUNCTION DOC]
        # HELPER: Calls task_spec.task_func with its kwargs plus model_context (tp
        # rank/size, server args, scheduler), the model and the model runner; returns
        # its result.
        model_context = dict(
            tp_rank=scheduler.tp_rank,
            tp_size=scheduler.tp_size,
            server_args=scheduler.server_args,
            scheduler=scheduler,
        )
        kwargs = dict(task_spec.kwargs)
        kwargs["model_context"] = model_context
        kwargs["model"] = scheduler.tp_worker.model_runner.model
        kwargs["model_runner"] = scheduler.tp_worker.model_runner
        return task_spec.task_func(**kwargs)

    scheduler.execute_task_in_model_worker = execute_task_in_model_worker

    if hasattr(scheduler, "_request_dispatcher"):
        scheduler._request_dispatcher._mapping[ModelWorkerTask] = (
            execute_task_in_model_worker
        )
        logger.info("[AWEX] Registered execute_task_in_model_worker in dispatcher")


def awex_run_scheduler_process(*args, **kwargs):
    # [FUNCTION DOC]
    # Scheduler process entry point: registers the plugin if AWEX_META_SERVER_ADDR is
    # set (else logs and skips), then delegates to SGLang's run_scheduler_process with
    # all arguments and returns its result.
    """Scheduler process entry point that registers awex plugin.

    Memory management (pause/resume weights, KV cache, CUDA graphs) is handled
    at runtime by AWEX's release_memory/resume_memory, matching the AWEX
    reference integration.
    No init-time memory patching needed.
    """
    import os

    meta_addr = os.environ.get("AWEX_META_SERVER_ADDR")
    if meta_addr:
        register_awex_plugin()
    else:
        logger.info(
            "[AWEX] No AWEX_META_SERVER_ADDR, skipping plugin registration",
        )
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


if __name__ == "__main__":
    import os
    import sys

    logger.info("[AWEX] awex_sglang_plugin __main__ starting")

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(
            server_args,
            run_scheduler_process_func=awex_run_scheduler_process,
        )
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
