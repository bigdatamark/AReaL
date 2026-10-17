# [MODULE DOC]
# Tests FSDP Ulysses sequence-parallel train_batch, delegated to
# tests/torchrun/run_fsdp_ulysses_train_batch.py under torchrun.
import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(n_gpus: int):
    # [FUNCTION DOC]
    # HELPER: Runs the Ulysses train_batch torchrun script with nproc_per_node=n_gpus on
    # a free port; pytest.fail with stderr on non-zero exit.
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_fsdp_ulysses_train_batch.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("world_size", [2])
def test_fsdp_ulysses_train_batch_2gpu(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size=2; skips if fewer GPUs, otherwise runs the script.
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} gpus")
    _run_test_with_torchrun(world_size)


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [1])
def test_fsdp_ulysses_train_batch_1gpu(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size=1; runs the script on a single process without a GPU-count
    # check.
    _run_test_with_torchrun(world_size)
