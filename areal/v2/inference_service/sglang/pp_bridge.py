# [MODULE DOC]
# PPSchedulerBridge: adds per-pipeline-stage routing of NCCL weight-update groups to an
# SGLang scheduler when pp_size > 1, by replacing methods on the scheduler's TpWorker
# and ModelRunner instances (no module-level monkey-patching). The target pp rank is the
# numeric suffix of the group name; workers at other pp ranks record a None sentinel and
# skip update/destroy calls for that group.
# KNOWN ISSUE: uses stdlib logging with __name__ instead of areal's getLogger, contrary
# to project conventions.
# SPDX-License-Identifier: Apache-2.0

"""PPSchedulerBridge: compose per-PP-rank weight-update routing onto SGLang Scheduler.

When sglang runs with pipeline parallelism (pp_size > 1), weight updates from
the training side come per-PP-stage.  Each training PP rank creates a separate
NCCL group that only includes the sglang workers at the matching PP rank.

This bridge intercepts the existing weight-update methods on the TpWorker and
ModelRunner instances to add per-PP-rank routing, without any monkey-patching
of sglang source modules.  The pp_rank is derived from the group name suffix
(all engines use ``update_weight_group_{pp_rank}`` when PP>1).

The approach is identical to :class:`AwexSchedulerBridge`:
  1. Created after ``Scheduler.__init__()`` in :func:`areal_run_scheduler_process`
  2. :meth:`bind` attaches wrapped methods via ``setattr`` on TpWorker/ModelRunner
  3. No inheritance, no monkey-patch on module-level classes

Backward compatibility
----------------------
When ``pp_size == 1``, the bridge is a complete no-op.  When ``pp_size > 1``
but the group name does not end with a digit suffix, the bridge falls through
to the original method -- identical to original sglang behaviour.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from typing import Any

logger = logging.getLogger(__name__)


def _extract_pp_rank_from_group_name(group_name: str) -> int | None:
    # [FUNCTION DOC]
    # HELPER: Returns the integer after the last underscore of group_name, or None if it
    # is not an integer.
    """Extract pp_rank from group name suffix (e.g. ``update_weight_group_0`` -> 0).

    Returns None if the group name does not end with ``_{digit}``.
    """
    try:
        suffix = group_name.rsplit("_", 1)[-1]
        return int(suffix)
    except (ValueError, IndexError):
        return None


class PPSchedulerBridge:
    # [CLASS DOC]
    # Holds the scheduler and pp_size; bind installs the wrappers.
    """Compose per-PP-rank weight-update capabilities onto a Scheduler instance.

    This bridge wraps the TpWorker's ``init_weights_update_group`` method and
    the ModelRunner's ``update_weights_from_distributed`` and
    ``destroy_weights_update_group`` methods to add pp_rank-aware routing.

    No inheritance.  No monkey-patch on module-level classes.  The scheduler,
    tp_worker, and model_runner instances remain plain sglang objects.
    """

    def __init__(self, scheduler: Any, server_args: Any) -> None:
        # [FUNCTION DOC]
        # Stores scheduler and server_args.pp_size.
        self._scheduler = scheduler
        self._pp_size = server_args.pp_size

    def bind(self) -> None:
        # [FUNCTION DOC]
        # No-op when pp_size <= 1. Otherwise looks up scheduler.tp_worker and its
        # model_runner (warns and returns if missing), installs the tp_worker and
        # model_runner wrappers, and logs pp/tp ranks.
        """Attach per-PP-rank wrapped methods to the scheduler's tp_worker and model_runner.

        When pp_size == 1, this method is a no-op.
        """
        if self._pp_size <= 1:
            logger.debug("PPSchedulerBridge: pp_size=%d, skipping bind.", self._pp_size)
            return

        scheduler = self._scheduler

        tp_worker = getattr(scheduler, "tp_worker", None)
        if tp_worker is None:
            logger.warning(
                "PPSchedulerBridge: scheduler has no tp_worker; skipping bind."
            )
            return

        model_runner = getattr(tp_worker, "model_runner", None)
        if model_runner is None:
            logger.warning(
                "PPSchedulerBridge: tp_worker has no model_runner; skipping bind."
            )
            return

        self._bind_tp_worker(tp_worker, model_runner)
        self._bind_model_runner(model_runner)

        logger.info(
            "PPSchedulerBridge bound (pp_size=%d, pp_rank=%d, tp_rank=%d).",
            self._pp_size,
            getattr(model_runner, "pp_rank", -1),
            getattr(model_runner, "tp_rank", -1),
        )

    def _bind_tp_worker(self, tp_worker: Any, model_runner: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces tp_worker.init_weights_update_group with a wrapper (see
        # nested).
        """Wrap ``tp_worker.init_weights_update_group`` for per-PP-rank routing."""
        _orig_tp_init = tp_worker.init_weights_update_group

        def _pp_init_weights_update_group(recv_req):
            # [FUNCTION DOC]
            # HELPER: If the group name's pp suffix differs from this worker's pp_rank,
            # stores a None sentinel in model_runner's update-group map and returns
            # (True, skip message) without joining. Otherwise starts a daemon watchdog
            # thread that warns after 30s, 90s and 210s cumulative if still blocked,
            # calls the original init, stops the watchdog, logs duration (or logs and
            # re-raises on exception) and returns the original result.
            group_name = recv_req.group_name
            pp_rank_from_name = _extract_pp_rank_from_group_name(group_name)

            if (
                pp_rank_from_name is not None
                and model_runner.pp_rank != pp_rank_from_name
            ):
                # This worker is at a different PP rank -- skip group creation
                # and store a None sentinel so update/destroy can short-circuit.
                model_runner._model_update_group[group_name] = None
                logger.info(
                    "Skipping group '%s': target pp_rank=%d but worker is pp_rank=%d.",
                    group_name,
                    pp_rank_from_name,
                    model_runner.pp_rank,
                )
                return True, (
                    f"Skipped group creation (pp_rank mismatch: "
                    f"target={pp_rank_from_name}, local={model_runner.pp_rank})."
                )

            if pp_rank_from_name is not None:
                logger.info(
                    "Worker pp_rank=%d tp_rank=%d joining per-PP-rank group '%s'.",
                    model_runner.pp_rank,
                    model_runner.tp_rank,
                    group_name,
                )

            # ---- BEGIN AREAL: watchdog for NCCL init hang detection ----
            _t0 = _time.monotonic()
            _stop = threading.Event()

            def _wd():
                # [FUNCTION DOC]
                # HELPER: Watchdog loop waiting 30, 60, then 120 seconds on the stop
                # event, logging a "BLOCKED" warning with elapsed time after each
                # unsignaled wait.
                for s in [30, 60, 120]:
                    if _stop.wait(s):
                        return
                    elapsed = _time.monotonic() - _t0
                    logger.warning(
                        "init_weights_update_group BLOCKED %.0fs: pp=%s tp=%s group=%s",
                        elapsed,
                        getattr(model_runner, "pp_rank", "?"),
                        getattr(model_runner, "tp_rank", "?"),
                        group_name,
                    )

            _wd_t = threading.Thread(target=_wd, daemon=True)
            _wd_t.start()
            # ---- END AREAL ----

            try:
                result = _orig_tp_init(recv_req)
            except Exception as e:
                _stop.set()
                logger.error(
                    "init_weights_update_group EXCEPTION after %.2fs: "
                    "pp=%s tp=%s group=%s: %s",
                    _time.monotonic() - _t0,
                    model_runner.pp_rank,
                    model_runner.tp_rank,
                    group_name,
                    e,
                    exc_info=True,
                )
                raise

            _stop.set()
            _elapsed = _time.monotonic() - _t0
            logger.info(
                "init_weights_update_group completed in %.2fs: pp=%s tp=%s group=%s",
                _elapsed,
                model_runner.pp_rank,
                model_runner.tp_rank,
                group_name,
            )
            return result

        tp_worker.init_weights_update_group = _pp_init_weights_update_group

    def _bind_model_runner(self, model_runner: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces model_runner.update_weights_from_distributed and
        # destroy_weights_update_group with sentinel-aware wrappers.
        """Wrap ModelRunner's update and destroy methods for per-PP-rank routing."""

        # ---- update_weights_from_distributed ----
        _orig_update = model_runner.update_weights_from_distributed

        def _pp_update_weights_from_distributed(
            names,
            dtypes,
            shapes,
            group_name,
            load_format=None,
        ):
            # [FUNCTION DOC]
            # HELPER: Returns (True, skip message) if this worker holds a None sentinel
            # for group_name; otherwise calls the original with the same args.
            """Skip broadcast receive if this worker did not join *group_name*."""
            pg = model_runner._model_update_group.get(group_name)
            if pg is None and group_name in model_runner._model_update_group:
                # Sentinel: this worker was deliberately excluded from the group.
                logger.debug(
                    "Skipping update_weights_from_distributed for group '%s': "
                    "worker pp_rank=%d did not join.",
                    group_name,
                    model_runner.pp_rank,
                )
                return True, (
                    f"Skipped weight update (worker pp_rank={model_runner.pp_rank} "
                    f"did not join group '{group_name}')."
                )
            return _orig_update(names, dtypes, shapes, group_name, load_format)

        model_runner.update_weights_from_distributed = (
            _pp_update_weights_from_distributed
        )

        # ---- destroy_weights_update_group ----
        _orig_destroy = model_runner.destroy_weights_update_group

        def _pp_destroy_weights_update_group(group_name):
            # [FUNCTION DOC]
            # HELPER: If a None sentinel exists for group_name, removes it and returns
            # (True, skip message); otherwise calls the original destroy.
            """Skip destruction if this worker holds a None sentinel."""
            pg = model_runner._model_update_group.get(group_name)
            if pg is None and group_name in model_runner._model_update_group:
                model_runner._model_update_group.pop(group_name, None)
                logger.debug(
                    "Skipping destroy for group '%s': worker pp_rank=%d did not join.",
                    group_name,
                    model_runner.pp_rank,
                )
                return True, (
                    f"Skipped group destruction (worker pp_rank={model_runner.pp_rank} "
                    f"did not join group '{group_name}')."
                )
            return _orig_destroy(group_name)

        model_runner.destroy_weights_update_group = _pp_destroy_weights_update_group
