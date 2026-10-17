# [MODULE DOC]
# Regression tests (#1292) for FSDP optimizer_dtype handling; the actual checks run
# inside tests/torchrun/run_fsdp_optimizer_dtype.py under torchrun.
"""FSDP optimizer dtype invariants test.

Verifies that:
- optimizer_dtype controls underlying param storage
- AdamW optimizer states inherit optimizer_dtype
- FSDP2 MixedPrecisionPolicy still casts forward to compute dtype
- HF export weights are cast back to compute dtype
- `_cast_to_compute_dtype` maps fp32 storage to compute dtype (xccl path)

Regression test for #1292.
"""

import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_with_torchrun(n_gpus: int, dtype: str, optimizer_dtype: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Runs the dtype torchrun script on a free port with nproc_per_node=n_gpus,
    # --dtype, --optimizer_dtype and --world_size. Calls pytest.fail with stdout/stderr
    # if the return code is non-zero. Returns None.
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_fsdp_optimizer_dtype.py",
        "--dtype",
        dtype,
        "--optimizer_dtype",
        optimizer_dtype,
        "--world_size",
        str(n_gpus),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"torchrun failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_fsdp_fp32_master_weights_2gpu():
    # [FUNCTION DOC]
    # Skips with <2 GPUs; runs bf16 compute with fp32 optimizer/master weights on 2
    # GPUs.
    """Default config: bf16 compute + fp32 master weights (the fix)."""
    if current_platform.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    _run_with_torchrun(2, "bfloat16", "float32")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_fsdp_legacy_bf16_storage_2gpu():
    # [FUNCTION DOC]
    # Skips with <2 GPUs; runs legacy bf16 storage (optimizer_dtype bf16) on 2 GPUs.
    """Legacy config: bf16 storage. Confirms backward-compatible path."""
    if current_platform.device_count() < 2:
        pytest.skip("requires 2 GPUs")
    _run_with_torchrun(2, "bfloat16", "bfloat16")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_fsdp_fp32_master_weights_1gpu():
    # [FUNCTION DOC]
    # Skips without a GPU; runs bf16 compute with fp32 master weights on 1 GPU.
    # KNOWN ISSUE: marked multi_gpu although it only needs one GPU.
    """Single-GPU sanity check (no sharding)."""
    if current_platform.device_count() < 1:
        pytest.skip("requires 1 GPU")
    _run_with_torchrun(1, "bfloat16", "float32")
