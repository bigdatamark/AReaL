# [MODULE DOC]
# Tests resolve_broadcast_target (areal.infra.rpc.guard.engine_blueprint): returns
# (group, device) for broadcasting; offloaded engines that expose
# cpu_model_parallel_group use it with device "cpu", otherwise
# context_and_model_parallel_group with the given device.
"""Offloaded engines broadcast over CPU only when they expose a CPU mirror group."""

from types import SimpleNamespace

from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target


def _engine(is_offload: bool, with_cpu_group: bool):
    # [FUNCTION DOC]
    # HELPER: Builds a fake engine (SimpleNamespace) with is_offload,
    # context_and_model_parallel_group="device-group" and, if with_cpu_group,
    # cpu_model_parallel_group="cpu-group".
    fields = {
        "is_offload": is_offload,
        "context_and_model_parallel_group": "device-group",
    }
    if with_cpu_group:
        fields["cpu_model_parallel_group"] = "cpu-group"
    return SimpleNamespace(**fields)


def test_offloaded_engine_uses_cpu_mirror_group():
    # [FUNCTION DOC]
    # Offloaded engine with CPU group: asserts ("cpu-group", "cpu").
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=True), device="cuda:0"
    )
    assert group == "cpu-group"
    assert device == "cpu"


def test_engine_without_cpu_group_keeps_device_broadcast():
    # [FUNCTION DOC]
    # Offloaded engine without CPU group (FSDP case): asserts ("device-group",
    # "cuda:0").
    """FSDP tracks is_offload but has no CPU mirror group."""
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=False), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"


def test_resident_engine_keeps_device_broadcast():
    # [FUNCTION DOC]
    # Non-offloaded engine with CPU group: asserts device broadcast ("device-group",
    # "cuda:0").
    group, device = resolve_broadcast_target(
        _engine(is_offload=False, with_cpu_group=True), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"
