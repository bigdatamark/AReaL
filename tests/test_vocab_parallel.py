# [MODULE DOC]
# Multi-GPU pytest wrapper that launches tests/torchrun/run_vocab_parallel.py under
# torchrun to verify vocab-parallel logprob/entropy computations with tensor parallel.
import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(world_size: int):
    # [FUNCTION DOC]
    # HELPER: Picks a free port and runs torchrun (single node, world_size procs,
    # master localhost) on the vocab-parallel script, capturing output. Spawns
    # subprocesses; on non-zero exit calls pytest.fail with captured stdout/stderr.
    """Run tensor parallel tests with torchrun."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_vocab_parallel.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error:\nstdout: {e.stdout}\nstderr: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_vocab_parallel(world_size: int):
    # [FUNCTION DOC]
    # Parametrized over world_size 2 and 4 (marked multi_gpu). Skips if the platform has
    # fewer devices than world_size, otherwise delegates to the torchrun helper.
    """Test vocab parallel logprobs and entropy with TP."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size=world_size)
