# [MODULE DOC]
# Shared tests and file-loaders for AWEX DTE (delta weight transfer). Target modules
# under areal/v2/weight_update/awex (delta_config, delta_detect, sglang_adapter,
# megatron_adapter) are executed from their file paths with stub modules in sys.modules
# so no full AReaL runtime import is needed. Covers env-driven DTERuntimeConfig, wire-
# dtype synchronization, DTE world-size validation, AdamW inversion, and memory
# telemetry. Loader helpers are reused by other test modules (e.g.
# test_awex_separation_delta). Requires torch.
# SPDX-License-Identifier: Apache-2.0
"""Shared Stage-3 tests and loaders for AWEX DTE delta transfer.

The runtime adapters import optional packages such as AWEX, DTE, and httpx.
These tests file-load the target modules with narrow stubs so separation logic
can be unit-tested without importing the full AReaL runtime.
"""

from __future__ import annotations

import importlib.util
import logging as stdlib_logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parent.parent
_DC_PATH = _ROOT / "areal/v2/weight_update/awex/delta_config.py"


def _make_environ_stub():
    # [FUNCTION DOC]
    # HELPER: Builds a stub "areal.utils.environ" module exposing get_env_var and
    # get_bool_env_var with the real helpers' signatures; returns the module.
    environ_mod = types.ModuleType("areal.utils.environ")

    def get_env_var(name, default=None, *, fallback_names=(), allow_empty=False):
        # [FUNCTION DOC]
        # HELPER: Returns the first of name and fallback_names whose os.environ value is
        # set and (unless allow_empty) not blank; otherwise default.
        for candidate in (name, *fallback_names):
            value = os.environ.get(candidate)
            if value is not None and (allow_empty or value.strip() != ""):
                return value
        return default

    def get_bool_env_var(
        name,
        default="false",
        *,
        fallback_names=(),
        truthy_values=("true", "1"),
        falsy_values=("false", "0"),
        strip_value=False,
    ):
        # [FUNCTION DOC]
        # HELPER: Reads the variable via get_env_var (default "false"), optionally
        # strips whitespace, and returns whether its lowercased value is in
        # truthy_values. falsy_values is accepted but ignored.
        # KNOWN ISSUE: a None default would crash on strip/lower; stub only, so values
        # outside truthy_values are simply False.
        del falsy_values
        value = get_env_var(name, default, fallback_names=fallback_names)
        value = value.strip() if strip_value else value
        return value.lower() in truthy_values

    environ_mod.get_env_var = get_env_var
    environ_mod.get_bool_env_var = get_bool_env_var
    return environ_mod


def _load_delta_config():
    # [FUNCTION DOC]
    # HELPER: Executes areal/v2/weight_update/awex/delta_config.py as module
    # "awex_delta_config", temporarily installing stub areal, areal.utils and environ
    # modules, then restoring (or removing) the previous sys.modules entries. The loaded
    # module stays registered in sys.modules under its spec name. Returns the module.
    spec = importlib.util.spec_from_file_location("awex_delta_config", _DC_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    module_names = ("areal", "areal.utils", "areal.utils.environ")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    sys.modules["areal"] = types.ModuleType("areal")
    sys.modules["areal.utils"] = types.ModuleType("areal.utils")
    sys.modules["areal.utils.environ"] = _make_environ_stub()
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
    return mod


@pytest.fixture
def dc():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a freshly loaded delta_config module.
    return _load_delta_config()


def _encode_like_adapter(tracker, params, version):
    # [FUNCTION DOC]
    # HELPER: Mirrors adapter sender logic: if tracker.full_sync_reason(version) is not
    # None, seeds the tracker with all params and returns all names/tensors; otherwise
    # returns names/tensors from tracker.encode. Input params is an ordered dict
    # name->tensor.
    """Mirror the sender-side DeltaTracker contract used by adapters."""
    params_list = list(params.items())
    reason = tracker.full_sync_reason(version)
    if reason is not None:
        tracker.seed(params_list, version)
        return list(params.keys()), list(params.values())
    encoded = tracker.encode(params_list, version)
    return encoded.names, encoded.tensors


def _over_the_wire(names, tensors):
    # [FUNCTION DOC]
    # HELPER: Simulates IPC transfer by returning a dict mapping names to cloned
    # tensors.
    """Stand in for cuda_ipc serialize/deserialize: clone to detach storage."""
    return dict(zip(names, [t.clone() for t in tensors]))


def _weights(seed: int):
    # [FUNCTION DOC]
    # HELPER: Returns deterministic fake weights (embed 32x16, layer 16x16, bias 16)
    # from a seeded CPU generator.
    generator = torch.Generator().manual_seed(seed)
    return {
        "embed.weight": torch.randn(32, 16, generator=generator),
        "layer.weight": torch.randn(16, 16, generator=generator),
        "layer.bias": torch.randn(16, generator=generator),
    }


def test_delta_config_env_gates(dc, monkeypatch):
    # [FUNCTION DOC]
    # Asserts DTERuntimeConfig.from_env delta_transfer defaults False, follows legacy
    # AWEX_DELTA_TRANSFER, and is overridden by DTE_DELTA_TRANSFER; likewise
    # anchor_interval follows AWEX_ then DTE_ anchor env vars.
    """DTE env vars must honor DTE_* overrides over legacy AWEX_* names."""
    monkeypatch.delenv("DTE_DELTA_TRANSFER", raising=False)
    monkeypatch.delenv("AWEX_DELTA_TRANSFER", raising=False)
    assert dc.DTERuntimeConfig.from_env().delta_transfer is False
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", "1")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is True
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "0")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is False
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "1")
    assert dc.DTERuntimeConfig.from_env().delta_transfer is True

    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "5")
    assert dc.DTERuntimeConfig.from_env().anchor_interval == 5
    monkeypatch.setenv("DTE_DELTA_ANCHOR_INTERVAL", "7")
    assert dc.DTERuntimeConfig.from_env().anchor_interval == 7


def test_delta_runtime_config_preserves_legacy_values_and_snapshots(dc, monkeypatch):
    # [FUNCTION DOC]
    # With blank DTE_DELTA_TRANSFER, legacy " yes " and DTE_SEPARATION_WEIGHT_UPDATE
    # "on": asserts delta transfer, separation and enabled are True. Then asserts a
    # later env change does not mutate the existing config (snapshot) but a fresh
    # from_env sees enabled False.
    monkeypatch.setenv("DTE_DELTA_TRANSFER", "")
    monkeypatch.setenv("AWEX_DELTA_TRANSFER", " yes ")
    monkeypatch.setenv("DTE_SEPARATION_WEIGHT_UPDATE", "on")

    config = dc.DTERuntimeConfig.from_env()

    assert config.delta_transfer is True
    assert config.separation_weight_update is True
    assert config.enabled is True

    monkeypatch.setenv("DTE_DELTA_TRANSFER", "0")
    assert config.enabled is True
    assert dc.DTERuntimeConfig.from_env().enabled is False


def test_wire_dtype_union_is_global_sorted_and_includes_empty_local_rank(
    dc, monkeypatch
):
    # [FUNCTION DOC]
    # Empty local plan on one rank: monkeypatches dist.all_gather_object/get_world_size
    # and asserts synchronize_wire_dtypes returns the global union sorted canonically
    # (bfloat16, float32).
    """Every rank derives the same rounds even when its local plan is empty."""
    plan = SimpleNamespace(operations={})
    group = object()

    def _all_gather_object(output, local_names, group=None):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather_object asserting an empty local list and non-None
        # group, filling outputs with one empty and one two-dtype list.
        assert local_names == []
        assert group is not None
        output[:] = [[], ["torch.float32", "torch.bfloat16"]]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    dtypes = dc.synchronize_wire_dtypes(plan, group)

    assert dtypes == (torch.bfloat16, torch.float32)


def test_wire_dtype_union_ignores_rank_local_discovery_order(dc, monkeypatch):
    # [FUNCTION DOC]
    # Asserts the dtype union order does not depend on per-rank insertion order.
    """Canonical ordering is independent of each rank's metadata insertion order."""
    op = SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=torch.float32))
    plan = SimpleNamespace(operations={1: [op]})

    def _all_gather_object(output, local_names, group=None):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather_object asserting the local list is ["torch.float32"]
        # and returning two rank lists in different orders.
        assert local_names == ["torch.float32"]
        output[:] = [
            ["torch.float32", "torch.bfloat16"],
            ["torch.bfloat16", "torch.float32"],
        ]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    dtypes = dc.synchronize_wire_dtypes(plan, object())

    assert dtypes == (torch.bfloat16, torch.float32)


def test_wire_dtype_union_preserves_single_bfloat16_round(dc, monkeypatch):
    # [FUNCTION DOC]
    # Asserts a BF16-only plan on all ranks yields exactly (bfloat16,).
    """The established BF16-only path retains exactly one protocol round."""
    op = SimpleNamespace(recv_shard_meta=SimpleNamespace(dtype=torch.bfloat16))
    plan = SimpleNamespace(operations={1: [op]})

    def _all_gather_object(output, local_names, group=None):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather_object asserting local ["torch.bfloat16"] and
        # returning BF16 for both ranks.
        assert local_names == ["torch.bfloat16"]
        output[:] = [["torch.bfloat16"], ["torch.bfloat16"]]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    assert dc.synchronize_wire_dtypes(plan, object()) == (torch.bfloat16,)


def test_wire_dtype_union_rejects_globally_empty_plan(dc, monkeypatch):
    # [FUNCTION DOC]
    # Asserts ValueError "no wire dtypes" when all ranks report empty plans.
    monkeypatch.setattr(dc.dist, "get_world_size", lambda group: 2)

    def _all_gather_object(output, local_names, group=None):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather_object filling outputs with two empty lists.
        output[:] = [[], []]

    monkeypatch.setattr(dc.dist, "all_gather_object", _all_gather_object)

    with pytest.raises(ValueError, match="no wire dtypes"):
        dc.synchronize_wire_dtypes(SimpleNamespace(operations={}), object())


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
def test_dte_world_size_accepts_recursive_scheduler_sizes(dc, world_size):
    # [FUNCTION DOC]
    # Parametrized power-of-two world sizes split evenly; asserts
    # validate_dte_world_size does not raise.
    dc.validate_dte_world_size(world_size, world_size // 2, world_size // 2)


@pytest.mark.parametrize("world_size", [3, 5, 6])
def test_dte_world_size_rejects_incomplete_recursive_schedules(dc, world_size):
    # [FUNCTION DOC]
    # Parametrized non-power-of-two sizes 3, 5, 6; asserts ValueError whose message
    # lists inference, training and combined sizes.
    infer_world_size = world_size // 2
    train_world_size = world_size - infer_world_size

    with pytest.raises(
        ValueError,
        match=(
            f"inference={infer_world_size}, training={train_world_size}, "
            f"combined={world_size}"
        ),
    ):
        dc.validate_dte_world_size(
            world_size,
            infer_world_size,
            train_world_size,
        )


@pytest.mark.parametrize(
    ("combined_world_size", "infer_world_size", "train_world_size"),
    [(1, 1, 0), (4, 1, 2)],
)
def test_dte_world_size_rejects_missing_side_or_inconsistent_total(
    dc, combined_world_size, infer_world_size, train_world_size
):
    # [FUNCTION DOC]
    # Asserts ValueError for a missing training side (1,1,0) and an inconsistent total
    # (4,1,2).
    with pytest.raises(ValueError, match=f"combined={combined_world_size}"):
        dc.validate_dte_world_size(
            combined_world_size,
            infer_world_size,
            train_world_size,
        )


def test_factory_builds_dte_tracker(dc, monkeypatch):
    # [FUNCTION DOC]
    # Requires dte. With anchor interval 0, asserts create_delta_tracker returns an
    # object with encode and seed methods.
    """The lazy factory should build a DTE tracker when DTE is available."""
    pytest.importorskip("dte")
    monkeypatch.setenv("AWEX_DELTA_ANCHOR_INTERVAL", "0")
    tracker = dc.DTERuntimeConfig.from_env().create_delta_tracker()
    assert hasattr(tracker, "encode") and hasattr(tracker, "seed")


def test_invert_adamw_roundtrip():
    # [FUNCTION DOC]
    # Requires dte. Runs one AdamW step on a random 512-vector and asserts
    # dte.core.invert_adamw recovers the pre-step weights within tolerance.
    """DTE AdamW inversion recovers pre-step weights on CPU."""
    pytest.importorskip("dte")
    from dte.core import invert_adamw

    torch.manual_seed(0)
    theta_prev = torch.randn(512, dtype=torch.float32)
    param = theta_prev.clone().requires_grad_(True)
    lr, wd, b1, b2, eps = 1e-3, 0.01, 0.9, 0.999, 1e-8
    opt = torch.optim.AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    param.grad = torch.randn_like(param)
    opt.step()
    state = opt.state[param]
    recovered = invert_adamw(
        param.detach().clone(),
        state["exp_avg"],
        state["exp_avg_sq"],
        float(state["step"]),
        lr,
        wd,
        b1,
        b2,
        eps,
    )
    torch.testing.assert_close(recovered, theta_prev, rtol=1e-3, atol=1e-4)


def _stub_colocate_device(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Installs a stub colocate_device module (device_mapping_key, fixed local
    # IP, physical device id from local index) via monkeypatch.
    colocate_device_mod = types.ModuleType(
        "areal.v2.weight_update.awex.colocate_device"
    )
    colocate_device_mod.device_mapping_key = lambda ip, device: f"{ip}_{device}"
    colocate_device_mod.get_colocate_ip_address = lambda: "127.0.0.1"
    colocate_device_mod.get_physical_cuda_device_id = lambda local_index=None: str(
        local_index or 0
    )
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.colocate_device",
        colocate_device_mod,
    )


def _stub_areal_packages(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Installs stub modules for httpx, areal package hierarchy, awex package
    # helpers, colocate_device, areal.utils.logging (stdlib getLogger),
    # areal.infra.platforms (no-op synchronize) and weight_digest (no-op
    # log_tensor_digest).
    monkeypatch.setitem(sys.modules, "httpx", types.ModuleType("httpx"))
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.v2", types.ModuleType("areal.v2"))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update",
        types.ModuleType("areal.v2.weight_update"),
    )

    awex_mod = types.ModuleType("areal.v2.weight_update.awex")
    awex_mod.awex_wu_use_group = lambda: False
    awex_mod.fetch_kv_metadata = lambda *args, **kwargs: ([], [])
    awex_mod.load_kv_metadata_file = lambda *args, **kwargs: None
    awex_mod.resolve_physical_gpu_id = lambda *args, **kwargs: 0
    awex_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "areal.v2.weight_update.awex", awex_mod)
    _stub_colocate_device(monkeypatch)

    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = stdlib_logging.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    infra_mod = types.ModuleType("areal.infra")
    platforms_mod = types.ModuleType("areal.infra.platforms")
    platforms_mod.current_platform = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "areal.infra", infra_mod)
    monkeypatch.setitem(sys.modules, "areal.infra.platforms", platforms_mod)

    weight_digest_mod = types.ModuleType("areal.v2.weight_update.awex.weight_digest")
    weight_digest_mod.log_tensor_digest = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.weight_digest",
        weight_digest_mod,
    )


def _load_delta_detect(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Installs stub areal/logging/environ/v2 packages, a freshly loaded
    # delta_config module, then executes delta_detect.py as "awex_delta_detect" and
    # returns it.
    """Load delta_detect.py without the full AReaL runtime."""
    fake = types.ModuleType("areal.utils.logging")
    fake.getLogger = stdlib_logging.getLogger
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", types.ModuleType("areal.utils"))
    monkeypatch.setitem(sys.modules, "areal.utils.logging", fake)
    monkeypatch.setitem(sys.modules, "areal.utils.environ", _make_environ_stub())
    for package in (
        "areal.v2",
        "areal.v2.weight_update",
        "areal.v2.weight_update.awex",
    ):
        monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        _load_delta_config(),
    )
    path = _ROOT / "areal/v2/weight_update/awex/delta_detect.py"
    spec = importlib.util.spec_from_file_location("awex_delta_detect", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cuda_mem_stats_mb_without_cuda_returns_sentinel(monkeypatch):
    # [FUNCTION DOC]
    # Asserts _cuda_mem_stats_mb returns non-negative MB values with CUDA, else (-1.0,
    # -1.0) sentinels (also with reset_peak=False).
    """CPU-only hosts must not fail while reporting DTE memory telemetry."""
    mod = _load_delta_detect(monkeypatch)
    alloc_mb, peak_mb = mod._cuda_mem_stats_mb()
    if torch.cuda.is_available():
        assert alloc_mb >= 0.0
        assert peak_mb >= 0.0
    else:
        assert (alloc_mb, peak_mb) == (-1.0, -1.0)
        assert mod._cuda_mem_stats_mb(reset_peak=False) == (-1.0, -1.0)


def test_adamw_hparams_require_and_prefer_recorded_step_lr(monkeypatch):
    # [FUNCTION DOC]
    # Asserts _adamw_hparams uses the recorded "_areal_last_step_lr" over the group's
    # lr, and returns None once that key is removed.
    """A scheduler's next-step LR cannot silently drive AdamW inversion."""
    mod = _load_delta_detect(monkeypatch)
    param_group = {
        "lr": 2e-6,
        "_areal_last_step_lr": 3e-6,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    hparams = mod._adamw_hparams(param_group)

    assert hparams is not None
    assert hparams[0] == 3e-6
    del param_group["_areal_last_step_lr"]
    assert mod._adamw_hparams(param_group) is None


def test_missing_recorded_step_lr_forces_dense_reconstruction(monkeypatch):
    # [FUNCTION DOC]
    # Installs a fake dte.core whose invert_adamw fails the test if called. Sets up an
    # AdamWInversionDetector with bound fake param names and a stored fingerprint, steps
    # a plain AdamW (which does not record the step LR), and asserts
    # _reconstruct_pre_step_mcore with a fake distributed optimizer returns None (dense
    # fallback) without attempting inversion.
    """Missing LR metadata must not fall back to the scheduler's current LR."""

    def _unexpected_invert_adamw(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fails the test if invoked.
        del args, kwargs
        pytest.fail("AdamW inversion must not run without a recorded step LR")

    dte_mod = types.ModuleType("dte")
    dte_mod.__path__ = []
    dte_core_mod = types.ModuleType("dte.core")
    dte_core_mod.invert_adamw = _unexpected_invert_adamw
    dte_mod.core = dte_core_mod
    monkeypatch.setitem(sys.modules, "dte", dte_mod)
    monkeypatch.setitem(sys.modules, "dte.core", dte_core_mod)

    mod = _load_delta_detect(monkeypatch)
    param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    before = param.detach().clone()

    inversion = mod.AdamWInversionDetector(
        SimpleNamespace(_offloaded_optimizer_states={})
    )
    _bind_inversion_param_names(inversion, ("w", param))
    inversion._last_synced_steps = {"w": None}
    inversion._last_synced_fingerprints = {"w": mod._tensor_fingerprint(before)}

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert "_areal_last_step_lr" not in optimizer.param_groups[0]

    class _FakeDistOpt:
        # [CLASS DOC]
        # HELPER: Minimal Megatron distributed-optimizer stand-in holding the param in
        # fp32/fp16 groups and the real optimizer.
        shard_fp32_from_float16_groups = [[param]]
        model_float16_groups = [[param]]
        model_param_group_index_map = None
        data_parallel_group = None

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Stores the real optimizer.
            self.optimizer = optimizer

        def _get_model_param_range_map(self, model_param):
            # [FUNCTION DOC]
            # HELPER: Asserts the expected param and returns a range map covering all
            # its elements.
            assert model_param is param
            return {"param": SimpleNamespace(start=0, end=param.numel())}

    assert inversion._reconstruct_pre_step_mcore([_FakeDistOpt()]) is None


def _bind_inversion_param_names(inv, *named_params):
    # [FUNCTION DOC]
    # HELPER: Overrides inv._module_param_key_maps to return a mapping from param id to
    # the given names and an empty second map.
    """Bind fake mcore parameter names for CPU-only inversion tests."""
    id2key = {id(param): name for name, param in named_params}
    inv._module_param_key_maps = lambda: (id2key, {})


def _load_sglang_adapter(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Requires awex weight_meta and sglang_sharding. Stubs AReaL packages, a
    # delta_config with DTE disabled, AwexInferenceAdapter and nccl_group, then executes
    # sglang_adapter.py and returns the module.
    """Load sglang_adapter.py with only its AReaL imports stubbed."""
    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.sglang_sharding")

    _stub_areal_packages(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")

    class _DTERuntimeConfig:
        # [CLASS DOC]
        # HELPER: Stub runtime config class.
        @classmethod
        def from_env(cls):
            # [FUNCTION DOC]
            # HELPER: Returns a namespace with enabled=False.
            return SimpleNamespace(enabled=False)

    delta_config_mod.DTERuntimeConfig = _DTERuntimeConfig
    delta_config_mod.synchronize_wire_dtypes = lambda *args, **kwargs: ()
    delta_config_mod.validate_dte_world_size = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    inference_adapter_mod = types.ModuleType("areal.v2.weight_update.inference_adapter")

    class _AwexInferenceAdapter:
        # [CLASS DOC]
        # HELPER: Empty stub base class.
        pass

    inference_adapter_mod.AwexInferenceAdapter = _AwexInferenceAdapter
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.inference_adapter",
        inference_adapter_mod,
    )

    nccl_group_mod = types.ModuleType("areal.v2.weight_update.nccl_group")
    nccl_group_mod.init_weights_update_group = lambda *args, **kwargs: None
    nccl_group_mod.setup_batch_isend_irecv = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.nccl_group",
        nccl_group_mod,
    )

    path = _ROOT / "areal/v2/weight_update/awex/sglang_adapter.py"
    spec = importlib.util.spec_from_file_location("awex_sglang_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_megatron_adapter(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Requires several awex submodules. Stubs AReaL packages, a delta_config
    # with DTE enabled (tracker factory returning None), delta_detect, nccl_group and
    # AwexTrainingAdapter, then executes megatron_adapter.py and returns the module.
    """Load megatron_adapter.py with runtime-heavy AReaL imports stubbed."""
    pytest.importorskip("awex.meta.weight_meta")
    pytest.importorskip("awex.sharding.param_sharding")
    pytest.importorskip("awex.transfer.transfer_plan")
    pytest.importorskip("awex.util.tensor_util")

    _stub_areal_packages(monkeypatch)

    delta_config_mod = types.ModuleType("areal.v2.weight_update.awex.delta_config")

    class _DTERuntimeConfig:
        # [CLASS DOC]
        # HELPER: Stub runtime config class.
        @classmethod
        def from_env(cls):
            # [FUNCTION DOC]
            # HELPER: Returns a namespace with enabled=True and a create_delta_tracker
            # returning None.
            return SimpleNamespace(
                enabled=True,
                create_delta_tracker=lambda: None,
            )

    delta_config_mod.DTERuntimeConfig = _DTERuntimeConfig
    delta_config_mod.synchronize_wire_dtypes = lambda *args, **kwargs: ()
    delta_config_mod.validate_dte_world_size = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_config",
        delta_config_mod,
    )

    delta_detect_mod = types.ModuleType("areal.v2.weight_update.awex.delta_detect")
    delta_detect_mod.AdamWInversionDetector = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.awex.delta_detect",
        delta_detect_mod,
    )

    nccl_group_mod = types.ModuleType("areal.v2.weight_update.nccl_group")
    nccl_group_mod.init_weights_update_group = lambda *args, **kwargs: None
    nccl_group_mod.setup_batch_isend_irecv = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.nccl_group",
        nccl_group_mod,
    )

    training_adapter_mod = types.ModuleType("areal.v2.weight_update.training_adapter")

    class _AwexTrainingAdapter:
        # [CLASS DOC]
        # HELPER: Empty stub base class.
        pass

    training_adapter_mod.AwexTrainingAdapter = _AwexTrainingAdapter
    monkeypatch.setitem(
        sys.modules,
        "areal.v2.weight_update.training_adapter",
        training_adapter_mod,
    )

    path = _ROOT / "areal/v2/weight_update/awex/megatron_adapter.py"
    spec = importlib.util.spec_from_file_location("awex_megatron_adapter_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
