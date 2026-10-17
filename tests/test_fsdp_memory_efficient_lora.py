# [MODULE DOC]
# GPU test for memory-efficient LoRA in the FSDP engine, delegated to
# tests/torchrun/run_fsdp_memory_efficient_lora.py which writes "Passed" to an output
# file.
import subprocess

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(alloc_mode: str, output: str):
    # [FUNCTION DOC]
    # HELPER: Runs the LoRA torchrun script with nproc_per_node from alloc_mode's world
    # size on a free port, passing backend and output path. pytest.fail with captured
    # output on failure; asserts the output file reads "Passed".
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_fsdp_memory_efficient_lora.py",
                f"--backend={alloc_mode}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}, {e.stdout}")
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.slow
def test_fsdp_memory_efficient_lora(tmp_path_factory):
    # [FUNCTION DOC]
    # Skips without a GPU; runs the script with fsdp:d1t1c1.
    if current_platform.device_count() < 1:
        pytest.skip("Test requires at least 1 GPU")
    output = tmp_path_factory.mktemp("test_output") / "fsdp_memory_efficient_lora.out"
    _run_test_with_torchrun("fsdp:d1t1c1", str(output))
