# [MODULE DOC]
# Unit tests for resolve_physical_gpu_id (areal.engine.awex.colocate_writer), which maps
# a logical CUDA device index to a physical GPU id via CUDA_VISIBLE_DEVICES. Tests
# mutate the env var through monkeypatch only.
from areal.engine.awex.colocate_writer import resolve_physical_gpu_id


def test_physical_gpu_id_maps_through_visible_devices(monkeypatch):
    # [FUNCTION DOC]
    # With CUDA_VISIBLE_DEVICES="4,5,6,7", asserts logical 0 -> 4 and 3 -> 7.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert resolve_physical_gpu_id(0) == 4
    assert resolve_physical_gpu_id(3) == 7


def test_physical_gpu_id_is_identity_without_visible_devices(monkeypatch):
    # [FUNCTION DOC]
    # With CUDA_VISIBLE_DEVICES unset, asserts the mapping is identity (2 -> 2).
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert resolve_physical_gpu_id(2) == 2


def test_physical_gpu_id_falls_back_on_non_integer_entries(monkeypatch):
    # [FUNCTION DOC]
    # With non-integer entries (GPU UUIDs), asserts fallback to the logical index.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,GPU-def")

    assert resolve_physical_gpu_id(1) == 1


def test_physical_gpu_id_falls_back_when_index_out_of_range(monkeypatch):
    # [FUNCTION DOC]
    # With an index beyond the visible list length, asserts fallback to the logical
    # index.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    assert resolve_physical_gpu_id(2) == 2
