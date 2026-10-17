# [MODULE DOC]
# CPU-only unit tests for MegatronCheckpointManager's async-save logic in
# areal/engine/megatron_utils/checkpointer.py. The checkpointer module is loaded from
# disk under stubbed megatron/areal parent modules; AsyncCallsQueue,
# save/load_dist_checkpointing, torch.distributed rank/barrier and CUDA cache calls are
# patched. Also tests the module helpers
# _inspect_retained_payload/_release_retained_payload (freeing tensor payloads retained
# by an async request while keeping finalize closures working) and the
# _UnsupportedMCoreAsyncLayout guard.
"""Unit tests for the Megatron checkpoint manager's async-save state machine.

These tests do NOT exercise real Megatron dist_checkpointing or distributed
process groups. They patch the queue and `save_dist_checkpointing` to verify
that the manager correctly:

- skips queue creation when async_save is False
- routes the AsyncRequest to AsyncCallsQueue.schedule_async_request when True
- finalizes completed saves non-blockingly on each new save call
- blocks on load_checkpoint / close
- treats close as idempotent
- emits the async-only metric (queue_depth on schedule)
- emits no async metrics on the sync path
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
import types
import weakref
from unittest.mock import MagicMock, patch

import pytest


def _import_checkpointer():
    # [FUNCTION DOC]
    # HELPER: Returns the cached checkpointer module from sys.modules if present.
    # Otherwise installs empty stub modules (setdefault) for
    # megatron.core.dist_checkpointing.* and areal parent packages, fills required
    # attributes with MagicMocks (mpu, tensor_parallel, ShardedObject, default load/save
    # strategies, AsyncCallsQueue/AsyncRequest, fully-parallel wrappers,
    # current_platform with no device, getLogger, a stub stats_tracker with a MagicMock
    # scalar), then loads the real checkpointer.py via importlib spec under its dotted
    # name, registers it in sys.modules and returns it.
    # KNOWN ISSUE: Permanently inserts stub "areal", "areal.utils" etc. modules into
    # sys.modules if they were not yet imported, which can break later tests in the same
    # process; and if the module is already cached the stats_tracker mock is not
    # installed (hence the skipped metric tests).
    """Import the checkpointer module without triggering the full areal package."""
    if "areal.engine.megatron_utils.checkpointer" in sys.modules:
        return sys.modules["areal.engine.megatron_utils.checkpointer"]

    # Stub out heavy/optional dependencies so the module loads on a CPU box
    # without Megatron or a Stager-capable torch build.
    for path in (
        "megatron",
        "megatron.core",
        "megatron.core.dist_checkpointing",
        "megatron.core.dist_checkpointing.mapping",
        "megatron.core.dist_checkpointing.serialization",
        "megatron.core.dist_checkpointing.strategies",
        "megatron.core.dist_checkpointing.strategies.async_utils",
        "megatron.core.dist_checkpointing.strategies.fully_parallel",
        "areal",
        "areal.engine",
        "areal.engine.megatron_utils",
        "areal.infra",
        "areal.infra.platforms",
        "areal.utils",
        "areal.utils.logging",
    ):
        sys.modules.setdefault(path, types.ModuleType(path))

    sys.modules["megatron.core"].dist_checkpointing = sys.modules[
        "megatron.core.dist_checkpointing"
    ]
    sys.modules["megatron.core"].mpu = MagicMock()
    sys.modules["megatron.core"].tensor_parallel = MagicMock()
    sys.modules["megatron.core.dist_checkpointing.mapping"].ShardedObject = MagicMock()
    sys.modules[
        "megatron.core.dist_checkpointing.serialization"
    ].get_default_load_sharded_strategy = MagicMock()
    sys.modules[
        "megatron.core.dist_checkpointing.serialization"
    ].get_default_save_sharded_strategy = MagicMock()
    async_utils = sys.modules["megatron.core.dist_checkpointing.strategies.async_utils"]
    async_utils.AsyncCallsQueue = MagicMock
    async_utils.AsyncRequest = MagicMock
    fp_mod = sys.modules["megatron.core.dist_checkpointing.strategies.fully_parallel"]
    fp_mod.FullyParallelLoadStrategyWrapper = MagicMock
    fp_mod.FullyParallelSaveStrategyWrapper = MagicMock
    sys.modules["areal.infra.platforms"].current_platform = MagicMock(
        device_type="cuda", is_available=lambda: False
    )
    sys.modules["areal.utils.logging"].getLogger = lambda *_a, **_k: MagicMock()

    # stats_tracker.scalar is called by the manager to report latency.
    stats_mod = types.ModuleType("areal.utils.stats_tracker")
    stats_mod.scalar = MagicMock()
    sys.modules["areal.utils.stats_tracker"] = stats_mod

    # The checkpointer imports `from areal.utils import logging, stats_tracker`.
    # Make sure the parent `areal.utils` package exposes both as attributes.
    sys.modules["areal.utils"].logging = sys.modules["areal.utils.logging"]
    sys.modules["areal.utils"].stats_tracker = stats_mod

    # Load the real checkpointer module from disk under the stubbed parents.
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "areal.engine.megatron_utils.checkpointer",
        repo_root / "areal" / "engine" / "megatron_utils" / "checkpointer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["areal.engine.megatron_utils.checkpointer"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def patched_checkpointer():
    # [FUNCTION DOC]
    # Fixture: imports the module, builds a MagicMock queue (0 unfinalized calls,
    # finalize returns [], schedule returns 0), patches get_rank to 0 and
    # AsyncCallsQueue to return that queue, constructs a manager with async_save=True
    # and MagicMock model/optimizer, and yields (module, manager, queue) with patches
    # active.
    mod = _import_checkpointer()

    queue = MagicMock()
    queue.get_num_unfinalized_calls.return_value = 0
    queue.maybe_finalize_async_calls.return_value = []
    queue.schedule_async_request.return_value = 0

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch.object(mod, "AsyncCallsQueue", return_value=queue),
    ):
        manager = mod.MegatronCheckpointManager(
            model=MagicMock(),
            optimizer=MagicMock(),
            lr_scheduler=None,
            async_save=True,
        )
        yield mod, manager, queue


def test_async_disabled_creates_no_queue():
    # [FUNCTION DOC]
    # With async_save=False the manager creates no queue (_async_queue None), and
    # _reap_finished_async_saves, wait_async_saves and close are safe no-ops.
    mod = _import_checkpointer()
    with patch("torch.distributed.get_rank", return_value=0):
        m = mod.MegatronCheckpointManager(
            model=MagicMock(),
            optimizer=MagicMock(),
            lr_scheduler=None,
            async_save=False,
        )
    assert m._async_queue is None
    m._reap_finished_async_saves()
    m.wait_async_saves()
    m.close()


def test_save_schedules_async_request(patched_checkpointer, tmp_path):
    # [FUNCTION DOC]
    # save_checkpoint with async enabled calls save_dist_checkpointing once with
    # async_save=True, inspects the returned request's retained payload, schedules the
    # request on the queue, and releases the inspected tensor lists.
    mod, manager, queue = patched_checkpointer
    fake_request = object()
    tensor_lists = [[]]

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(
            mod, "save_dist_checkpointing", return_value=fake_request
        ) as save_fn,
        patch.object(
            mod, "_inspect_retained_payload", return_value=tensor_lists
        ) as inspect_fn,
        patch.object(mod, "_release_retained_payload") as release_fn,
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    save_fn.assert_called_once()
    assert save_fn.call_args.kwargs["async_save"] is True
    inspect_fn.assert_called_once_with(fake_request)
    queue.schedule_async_request.assert_called_once_with(fake_request)
    release_fn.assert_called_once_with(tensor_lists)


def test_save_reaps_before_scheduling_next(patched_checkpointer, tmp_path):
    # [FUNCTION DOC]
    # Two consecutive saves each call maybe_finalize_async_calls non-blocking (2 calls,
    # blocking False) and schedule 2 requests.
    mod, manager, queue = patched_checkpointer

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", side_effect=["r1", "r2"]),
        patch.object(mod, "_inspect_retained_payload", return_value=[]),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))
        manager.save_checkpoint(str(tmp_path / "step1"))

    calls = queue.maybe_finalize_async_calls.call_args_list
    assert len(calls) == 2
    assert all(call.kwargs.get("blocking", False) is False for call in calls)
    assert queue.schedule_async_request.call_count == 2


def test_load_blocks_on_pending_saves(patched_checkpointer, tmp_path):
    # [FUNCTION DOC]
    # With one pending save, load_checkpoint first calls
    # maybe_finalize_async_calls(blocking=True). The load itself is expected to fail
    # (AssertionError or KeyError) because mocked load returns an empty dict.
    mod, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch("os.path.exists", return_value=True),
        patch.object(manager, "generate_state_dict", return_value={}),
        patch.object(mod, "load_dist_checkpointing", return_value={}),
    ):
        with pytest.raises((AssertionError, KeyError)):
            manager.load_checkpoint(str(tmp_path / "step0"))

    queue.maybe_finalize_async_calls.assert_called_with(blocking=True)


def test_close_is_idempotent(patched_checkpointer):
    # [FUNCTION DOC]
    # Calling close twice does not raise and leaves _async_queue None.
    _, manager, queue = patched_checkpointer
    queue.get_num_unfinalized_calls.return_value = 0

    manager.close()
    manager.close()

    assert manager._async_queue is None


@pytest.mark.skip(
    reason="Fixture is not isolated across test files: if test_megatron_engine "
    "(or any test that imports MegatronEngine) runs first, "
    "areal.engine.megatron_utils.checkpointer is already cached in sys.modules, "
    "so _import_checkpointer's stub-installation branch (which mocks "
    "areal.utils.stats_tracker.scalar) is skipped. Tracked in a follow-up issue."
)
def test_async_save_reports_queue_depth_only(patched_checkpointer, tmp_path):
    # [FUNCTION DOC]
    # (Skipped: module caching breaks stats mock isolation.) Two async saves report only
    # the "ckpt/async_save_queue_depth" metric key via stats_tracker.scalar.
    """async_save emits ckpt/async_save_queue_depth on schedule and no other metric.

    Successful finalize is observable as queue_depth returning to 0; a failing
    background save raises from wait_async_saves, so an explicit count metric
    would be redundant.
    """
    mod, manager, queue = patched_checkpointer
    stats_scalar = sys.modules["areal.utils.stats_tracker"].scalar
    stats_scalar.reset_mock()

    queue.schedule_async_request.return_value = 42
    queue.get_num_unfinalized_calls.return_value = 1

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", side_effect=["r1", "r2"]),
        patch.object(mod, "_inspect_retained_payload", return_value=[]),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

        # On the next save, reap returns [42] -> still no extra metrics.
        queue.maybe_finalize_async_calls.return_value = [42]
        queue.schedule_async_request.return_value = 43
        manager.save_checkpoint(str(tmp_path / "step1"))

    all_keys = set()
    for c in stats_scalar.call_args_list:
        all_keys.update(c.kwargs.keys())
    assert all_keys == {"ckpt/async_save_queue_depth"}


@pytest.mark.skip(
    reason="Fixture is not isolated across test files: if test_megatron_engine "
    "(or any test that imports MegatronEngine) runs first, "
    "areal.engine.megatron_utils.checkpointer is already cached in sys.modules, "
    "so _import_checkpointer's stub-installation branch (which mocks "
    "areal.utils.stats_tracker.scalar) is skipped. Tracked in a follow-up issue."
)
def test_sync_save_emits_no_async_metrics(patched_checkpointer, tmp_path):
    # [FUNCTION DOC]
    # (Skipped for the same reason.) A synchronous save (async disabled) emits no
    # stats_tracker.scalar calls.
    """Sync save path stays metric-free; trainer-side `timeperf/save` is sufficient."""
    mod, manager, _ = patched_checkpointer
    manager.async_save = False
    manager._async_queue = None
    stats_scalar = sys.modules["areal.utils.stats_tracker"].scalar
    stats_scalar.reset_mock()

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=None),
        patch("torch.cuda.empty_cache"),
        patch("torch.distributed.barrier"),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    stats_scalar.assert_not_called()


def test_generate_state_dict_requests_dp_reshardable_sharding(patched_checkpointer):
    # [FUNCTION DOC]
    # generate_state_dict with only the optimizer requests the optimizer's
    # sharded_state_dict with metadata distrib_optim_sharding_type "dp_reshardable" and
    # is_loading False, and the result has an "optimizer" key.
    _, manager, _ = patched_checkpointer

    with patch("torch.distributed.barrier"):
        state_dict = manager.generate_state_dict(
            with_model=False, with_optimizer=True, with_rng=False
        )

    kwargs = manager.optimizer.sharded_state_dict.call_args.kwargs
    assert kwargs["metadata"] == {"distrib_optim_sharding_type": "dp_reshardable"}
    assert kwargs["is_loading"] is False
    assert "optimizer" in state_dict


def test_load_checkpoint_builds_optimizer_template_with_is_loading(
    patched_checkpointer, tmp_path
):
    # [FUNCTION DOC]
    # load_checkpoint (optimizer only) builds the optimizer template with is_loading
    # True and dp_reshardable metadata, then calls optimizer.load_state_dict with the
    # loaded {"step": 1}.
    mod, manager, _ = patched_checkpointer

    with (
        patch("os.path.exists", return_value=True),
        patch("torch.distributed.barrier"),
        patch.object(
            mod, "load_dist_checkpointing", return_value={"optimizer": {"step": 1}}
        ),
    ):
        manager.load_checkpoint(
            str(tmp_path / "step0"),
            with_model=False,
            with_optimizer=True,
            with_rng=False,
        )

    kwargs = manager.optimizer.sharded_state_dict.call_args.kwargs
    assert kwargs["is_loading"] is True
    assert kwargs["metadata"] == {"distrib_optim_sharding_type": "dp_reshardable"}
    manager.optimizer.load_state_dict.assert_called_once_with({"step": 1})


class _FakeRequest:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for a megatron AsyncRequest with async_fn_args,
    # preload_fn and finalize_fns.
    def __init__(self, args, preload_fn, finalize_fns):
        # [FUNCTION DOC]
        # HELPER: Stores the three attributes.
        self.async_fn_args = args
        self.preload_fn = preload_fn
        self.finalize_fns = finalize_fns


def _request_with_pending_payload(buckets, finalize_fns=None):
    # [FUNCTION DOC]
    # HELPER: Builds a _FakeRequest whose args are (0, buckets, "results_queue"), whose
    # preload_fn returns buckets, and whose finalize_fns default to a single no-op.
    return _FakeRequest(
        args=(0, buckets, "results_queue"),
        preload_fn=lambda: buckets,
        finalize_fns=finalize_fns or [lambda: None],
    )


def test_release_retained_payload_clears_tensor_data_and_preserves_finalize():
    # [FUNCTION DOC]
    # Builds a write bucket holding a bytes item list and a tensor item list (with a
    # weakly referenced payload object), inspects the request, deletes the strong ref
    # and releases: asserts the tensor is garbage-collected, the bucket list object is
    # unchanged but its tensor list is emptied, request args/preload/finalize are
    # intact, and the finalize closure still sees the bytes items.
    mod = _import_checkpointer()

    class Payload:
        # [CLASS DOC]
        # HELPER: Empty class used as a weak-referenceable fake tensor.
        pass

    tensor = Payload()
    tensor_ref = weakref.ref(tensor)
    writer = types.SimpleNamespace(
        write_buckets=[
            (
                "file",
                "key",
                ([("byte-item", b"payload")], [("tensor-item", tensor)]),
            )
        ]
    )
    buckets = writer.write_buckets

    def finalize_fn():
        # [FUNCTION DOC]
        # HELPER: Finalize closure returning the bucket count and the first bucket's
        # bytes item list.
        return len(writer.write_buckets), writer.write_buckets[0][2][0]

    request = _request_with_pending_payload(buckets, [finalize_fn])
    tensor_lists = mod._inspect_retained_payload(request)
    del tensor

    mod._release_retained_payload(tensor_lists)
    gc.collect()

    assert tensor_ref() is None
    assert writer.write_buckets is buckets
    assert writer.write_buckets == [("file", "key", ([("byte-item", b"payload")], []))]
    assert request.async_fn_args == (0, buckets, "results_queue")
    assert request.preload_fn() is buckets
    assert request.finalize_fns == [finalize_fn]
    assert finalize_fn() == (1, [("byte-item", b"payload")])


def test_save_unknown_payload_layout_fails_before_scheduling(
    patched_checkpointer, tmp_path
):
    # [FUNCTION DOC]
    # With an unrecognized bucket layout and megatron-core version patched to 0.18.0,
    # save_checkpoint raises _UnsupportedMCoreAsyncLayout mentioning the version and
    # write_buckets[0], never schedules the request, and leaves buckets untouched.
    mod, manager, queue = patched_checkpointer
    buckets = ["opaque-bucket"]
    request = _request_with_pending_payload(buckets)

    with (
        patch.object(manager, "generate_state_dict", return_value={"model": {}}),
        patch.object(mod, "save_dist_checkpointing", return_value=request),
        patch.object(mod, "_mcore_version", return_value="0.18.0"),
        pytest.raises(
            mod._UnsupportedMCoreAsyncLayout,
            match=r"megatron-core=0\.18\.0.*write_buckets\[0\]",
        ),
    ):
        manager.save_checkpoint(str(tmp_path / "step0"))

    queue.schedule_async_request.assert_not_called()
    assert buckets == ["opaque-bucket"]
