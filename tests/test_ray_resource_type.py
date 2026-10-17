# [MODULE DOC]
# Unit tests for ray_resource_type in areal.infra.utils.ray, which picks the Ray resource
# name ("GPU", "NPU" or "CPU") for scheduling: local accelerators win; a CPU-only driver
# connected to Ray consults cluster_resources; a disconnected CPU driver returns "CPU".
# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.infra.utils import ray as ray_utils


@pytest.fixture
def cpu_only_driver(monkeypatch):
    # [FUNCTION DOC]
    # Fixture: makes the driver look CPU-only (no NPU, no CUDA) and Ray initialized.
    import torch

    monkeypatch.setattr(ray_utils.ray, "is_initialized", lambda: True)
    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_local_gpu_driver_reports_gpu(monkeypatch):
    # [FUNCTION DOC]
    # With local CUDA available, the type is "GPU".
    import torch

    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert ray_utils.ray_resource_type() == "GPU"


def test_cpu_only_driver_reports_cluster_gpu(cpu_only_driver, monkeypatch):
    # [FUNCTION DOC]
    # A CPU-only connected driver reports "GPU" when the cluster has GPUs.
    monkeypatch.setattr(
        ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0, "GPU": 64.0}
    )

    assert ray_utils.ray_resource_type() == "GPU"


def test_cpu_only_driver_reports_cluster_npu(cpu_only_driver, monkeypatch):
    # [FUNCTION DOC]
    # A CPU-only connected driver reports "NPU" when the cluster has NPUs.
    monkeypatch.setattr(
        ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0, "NPU": 16.0}
    )

    assert ray_utils.ray_resource_type() == "NPU"


def test_cpu_only_driver_without_cluster_accelerators_reports_cpu(
    cpu_only_driver, monkeypatch
):
    # [FUNCTION DOC]
    # Without accelerators in the cluster the type is "CPU".
    monkeypatch.setattr(ray_utils.ray, "cluster_resources", lambda: {"CPU": 32.0})

    assert ray_utils.ray_resource_type() == "CPU"


def test_disconnected_cpu_driver_does_not_query_cluster(monkeypatch):
    # [FUNCTION DOC]
    # When Ray is not initialized, cluster_resources must not be called and "CPU" is returned.
    import torch

    monkeypatch.setattr("areal.infra.platforms.is_npu_available", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ray_utils.ray, "is_initialized", lambda: False)

    def fail():
        # [FUNCTION DOC]
        # HELPER: Raises AssertionError if the cluster is queried.
        raise AssertionError("cluster_resources must not be queried when disconnected")

    monkeypatch.setattr(ray_utils.ray, "cluster_resources", fail)

    assert ray_utils.ray_resource_type() == "CPU"
