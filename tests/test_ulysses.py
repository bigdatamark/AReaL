# [MODULE DOC]
# Multi-GPU wrapper test for Ulysses sequence parallelism: launches
# tests/torchrun/run_ulysses.py under torchrun and fails on a nonzero exit.
import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(world_size):
    # [FUNCTION DOC]
    # HELPER: Runs torchrun with world_size processes on a free localhost port executing
    # the Ulysses worker script, capturing output; on CalledProcessError calls
    # pytest.fail with stderr.
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_ulysses.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size 2; skips if fewer devices, otherwise runs the torchrun
    # worker.
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} gpus")
    _run_test_with_torchrun(world_size=world_size)
