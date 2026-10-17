# [MODULE DOC]
# Ray helpers: cluster init, discovering a master address inside a placement group,
# building Ray resource specs, and detecting the accelerator resource type.
# SPDX-License-Identifier: Apache-2.0

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from areal.utils.network import find_free_ports, gethostip


def initialize_ray() -> None:
    # [FUNCTION DOC]
    # No-op if Ray is initialized; otherwise tries ray.init(address="auto") to join an
    # existing cluster and on ConnectionError starts a local Ray instance.
    """Connect to an existing Ray cluster, or start a local one."""
    if ray.is_initialized():
        return

    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)


def get_placement_group_master_ip_and_port(
    placement_group: PlacementGroup, placement_group_bundle_index: int = 0
):
    # [FUNCTION DOC]
    # Schedules a tiny Ray task (1 CPU, 0 GPU, 10MB memory) on the given bundle index of
    # placement_group that returns (host IP, a free port in 10000-32767); blocks on
    # ray.get and returns that tuple. Used to pick a rendezvous address on the node.
    # KNOWN ISSUE: the inline comment says "Convert MB to bytes" but the value is a
    # fixed 10MB literal; the port is only free at probe time (race).
    def _master_ip_and_port():
        # [FUNCTION DOC]
        # HELPER: remote body; returns this node's IP and one free port.
        host_ip = gethostip()
        port = find_free_ports(1, (10000, 32767))[0]
        return host_ip, port

    future = ray.remote(
        num_cpus=1,
        num_gpus=0,
        memory=10 * 1024 * 1024,  # Convert MB to bytes
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            placement_group_bundle_index=placement_group_bundle_index,
        ),
    )(_master_ip_and_port).remote()
    return ray.get(future)


def create_resource_spec(device, cpu: int, gpu: int, mem_in_bytes: int):
    # [FUNCTION DOC]
    # Returns Ray resource kwargs: num_cpus=cpu, memory=mem_in_bytes, plus num_gpus=gpu
    # when device is "GPU", or a custom resources dict {device: gpu} for other non-CPU
    # devices (e.g. "NPU").
    res = {"num_cpus": cpu, "memory": mem_in_bytes}
    if device == "GPU":
        res["num_gpus"] = float(gpu)
    elif device != "CPU":
        res["resources"] = {device: float(gpu)}
    return res


def ray_resource_type():
    # [FUNCTION DOC]
    # Returns "NPU" if an NPU is available locally (checked first since MindSpeed
    # patches CUDA availability), else "GPU" if CUDA is available, else the cluster-wide
    # accelerator type or "CPU". Lazily imports torch and platform detection.
    # npu before cuda because mindspeed patches cuda.is_available
    import torch

    from areal.infra.platforms import is_npu_available

    if is_npu_available:
        return "NPU"

    if torch.cuda.is_available():
        return "GPU"

    return _cluster_accelerator_type() or "CPU"


def _cluster_accelerator_type() -> str | None:
    # [FUNCTION DOC]
    # HELPER: if Ray is initialized, returns "NPU" or "GPU" when the cluster resources
    # report a nonzero amount (NPU checked first), else None.
    # The driver may sit on a device-free node (e.g. a CPU-only head pod) while
    # accelerators live on remote workers, so local probing is not conclusive.
    if not ray.is_initialized():
        return None

    resources = ray.cluster_resources()
    for device in ("NPU", "GPU"):
        if resources.get(device, 0):
            return device
    return None
