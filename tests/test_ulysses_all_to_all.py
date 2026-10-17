# [MODULE DOC]
# Multi-GPU wrapper tests for the Ulysses all_to_all_tensor primitive: each test
# launches tests/torchrun/run_ulysses_all_to_all.py under torchrun with a test name
# (correctness, backward, compile) and fails if the worker exits nonzero.
"""Pytest wrapper for ulysses all_to_all_tensor tests.

Tests:
- test_ulysses_all_to_all_correctness: Verify output matches reference implementation
- test_ulysses_all_to_all_backward: Verify autograd backward pass works
- test_ulysses_all_to_all_compile: Verify torch.compile compatibility
"""

import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(world_size: int, test_name: str):
    # [FUNCTION DOC]
    # HELPER: Runs torchrun with world_size processes on a free port executing the all-
    # to-all worker with --test_name, prints captured stdout on success, and on
    # CalledProcessError calls pytest.fail with stderr and stdout.
    port = find_free_ports(1)[0]
    try:
        result = subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_ulysses_all_to_all.py",
                f"--test_name={test_name}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}\n{e.stdout}")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_correctness(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size 2; skips if fewer devices; runs the "correctness" worker
    # test.
    """Test that all_to_all_tensor produces correct results."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "correctness")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_backward(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size 2; skips if fewer devices; runs the "backward" worker
    # test.
    """Test that autograd backward pass works correctly."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "backward")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_compile(world_size):
    # [FUNCTION DOC]
    # Parametrized world_size 2; skips if fewer devices; runs the "compile" worker test.
    """Test torch.compile compatibility (no graph breaks)."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "compile")
