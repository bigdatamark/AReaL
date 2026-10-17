# [MODULE DOC]
# Multi-GPU test for Archon context parallelism (Ulysses): launches the torchrun worker
# run_cp_forward.py. Module skipped without CUDA.
"""Context Parallelism (CP/Ulysses) tests for Archon Engine.

Run tests:
    pytest tests/experimental/archon/test_distributed_cp.py -v -m multi_gpu
"""

import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_cp_test_with_torchrun(n_gpus: int, cp_size: int):
    # [FUNCTION DOC]
    # HELPER: Runs torchrun on one node with n_gpus processes, a free master port, and
    # the run_cp_forward.py script with --cp_size. check=True; a non-zero exit fails the
    # test showing stderr. No output file is checked; success is the exit code.
    """Run FSDP+CP (Ulysses) Archon forward test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/experimental/archon/torchrun/run_cp_forward.py",
                f"--cp_size={cp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_cp_forward_2gpu():
    # [FUNCTION DOC]
    # Multi-GPU, slow. Skips with fewer than 2 devices; runs the CP forward worker with
    # 2 processes and cp_size 2 (dp 1).
    """Test Archon Engine forward with CP (Ulysses) on 2 GPUs (dp=1, cp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_cp_test_with_torchrun(2, 2)
