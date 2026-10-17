# [MODULE DOC]
# Tests the distributed lock utility by launching tests/torchrun/run_lock.py under
# torchrun with a given world size and backend; the worker performs the actual lock
# acquisition/contention checks and these tests only verify its exit code.
import subprocess

import pytest

from areal.utils.network import find_free_ports


def _run_lock_test(world_size: int, backend: str = "nccl", iters: int = 10):
    # [FUNCTION DOC]
    # HELPER: Picks a free port via find_free_ports, then runs torchrun (single node,
    # localhost) on tests/torchrun/run_lock.py with --backend (default "nccl"), --iters
    # (default 10), timeout 5.0 s, hold-time 0.001 s and backoff 0.005 s. Blocks until
    # completion (no timeout) and calls pytest.fail with command, return code, stdout
    # and stderr if the process exits non-zero. Returns None. Relative script path
    # assumes the repo root as cwd.
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_lock.py",
        "--backend",
        backend,
        "--iters",
        str(iters),
        "--timeout",
        "5.0",
        "--hold-time",
        "0.001",
        "--backoff",
        "0.005",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "\n".join(
                [
                    "Distributed lock test failed",
                    f"Command: {' '.join(cmd)}",
                    f"Return code: {result.returncode}",
                    f"STDOUT:\n{result.stdout}",
                    f"STDERR:\n{result.stderr}",
                ]
            )
        )


@pytest.mark.parametrize("world_size", [1])
def test_distributed_lock_single_rank(world_size):
    # [FUNCTION DOC]
    # Runs the lock worker with a single rank (NCCL backend, so it needs a GPU).
    _run_lock_test(world_size)


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_distributed_lock_multi_rank(world_size):
    # [FUNCTION DOC]
    # Runs the lock worker with 2 ranks (multi_gpu marker) to exercise contention.
    _run_lock_test(world_size)
