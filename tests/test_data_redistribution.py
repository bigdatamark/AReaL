# [MODULE DOC]
# Multi-GPU test for data redistribution across ranks. Launches
# tests/torchrun/redistribute.py via torchrun, which pickles per-rank results (objects
# with all_data, group_indices, data attributes) to redistributed{rank}.pkl, then checks
# consistency on the host.
import pickle
import subprocess
import sys

import pytest
import torch
from torch.testing import assert_close

from areal.infra.platforms import current_platform
from areal.utils.environ import is_in_ci
from areal.utils.network import find_free_ports


def assert_tensor_container_close(x1, x2):
    # [FUNCTION DOC]
    # HELPER: Recursively asserts two containers are equal: same type, tensors compared
    # with torch.testing.assert_close, lists compared element-wise (same length), dicts
    # by identical keys then per-value, anything else by ==. Raises AssertionError on
    # mismatch; returns None.
    assert type(x1) is type(x2), (type(x1), type(x2))
    if torch.is_tensor(x1):
        assert_close(x1, x2)
        return
    if isinstance(x1, list):
        assert len(x1) == len(x2), (len(x1), len(x2))
        [assert_tensor_container_close(xx1, xx2) for xx1, xx2 in zip(x1, x2)]
        return
    if isinstance(x1, dict):
        assert x1.keys() == x2.keys(), (x1.keys(), x2.keys())
        for k in x1.keys():
            assert_tensor_container_close(x1[k], x2[k])
        return
    assert x1 == x2


@pytest.mark.skipif(is_in_ci(), reason="CI machine will crash with all_gather_object")
@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_redistribute(world_size, tmp_path):
    # [FUNCTION DOC]
    # Parametrized world_size in {2,4,8}; skipped in CI and when fewer GPUs are
    # available. Picks a free port, runs torchrun with --dump-path=tmp_path (pytest.fail
    # on non-zero exit), loads every rank's pickle, asserts all ranks agree on all_data
    # and group_indices, and that each rank's data equals the all_data items selected by
    # its group_indices entry.
    # KNOWN ISSUE: stderr is redirected to sys.stdout so e.stderr in the failure message
    # is always None.
    if current_platform.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/redistribute.py",
                f"--dump-path={str(tmp_path)}",
            ],
            check=True,
            text=True,
            stderr=sys.stdout,
            stdout=sys.stdout,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")

    redistributed_data = []
    for i in range(world_size):
        with open(tmp_path / f"redistributed{i}.pkl", "rb") as f:
            redistributed_data.append(pickle.load(f))

    for x in redistributed_data[1:]:
        assert_tensor_container_close(x.all_data, redistributed_data[0].all_data)
    for x in redistributed_data[1:]:
        assert_tensor_container_close(
            x.group_indices, redistributed_data[0].group_indices
        )

    all_data = redistributed_data[0].all_data
    group_indices = redistributed_data[0].group_indices
    for x, indices in zip(redistributed_data, group_indices):
        expected_data = [all_data[i] for i in indices]
        assert_tensor_container_close(x.data, expected_data)
