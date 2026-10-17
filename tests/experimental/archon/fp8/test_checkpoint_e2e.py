# [MODULE DOC]
# End-to-end GPU test that loads a real Qwen3-1.7B FP8 checkpoint through Archon by
# launching the torchrun worker script run_checkpoint.py. Model paths come from env vars
# AREAL_FP8_MODEL_PATH / AREAL_BF16_MODEL_PATH (with cluster defaults). Whole module is
# skipped when CUDA is unavailable.
import os
import subprocess

import pytest
import torch

from areal.utils.network import find_free_ports

FP8_MODEL_PATH = os.environ.get(
    "AREAL_FP8_MODEL_PATH", "/storage/openpsi/models/Qwen__Qwen3-1.7B-FP8"
)
BF16_MODEL_PATH = os.environ.get(
    "AREAL_BF16_MODEL_PATH", "/storage/openpsi/models/Qwen__Qwen3-1.7B"
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_torchrun_test(
    script: str,
    n_gpus: int,
    output_file: str,
    extra_args: list[str] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: Launches a torchrun job on a single node. Inputs: script path, n_gpus
    # (nproc_per_node), output_file (passed as --output), optional extra_args appended
    # to the command. Picks a free master port, runs the subprocess synchronously
    # capturing output, prints stdout/stderr, and fails the pytest test if the exit code
    # is non-zero or if the output file content (stripped) is not exactly "Passed". Side
    # effects: spawns processes, uses GPUs, reads the output file.
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script,
        f"--output={output_file}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(f"--- STDOUT ---\n{result.stdout}")
    if result.stderr:
        print(f"--- STDERR ---\n{result.stderr}")

    if result.returncode != 0:
        pytest.fail(
            f"torchrun exited with code {result.returncode}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    with open(output_file) as f:
        content = f.read().strip()
    if content != "Passed":
        pytest.fail(
            f"Test wrote '{content}' to output (expected 'Passed').\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


class TestFP8CheckpointE2E:
    # [CLASS DOC]
    # End-to-end FP8 checkpoint loading test class; every test auto-skips when the FP8
    # model directory is absent.
    @pytest.fixture(autouse=True)
    def _check_fp8_model(self):
        # [FUNCTION DOC]
        # HELPER: Autouse fixture that skips the test if FP8_MODEL_PATH does not exist
        # on disk.
        if not os.path.exists(FP8_MODEL_PATH):
            pytest.skip(f"FP8 model not found at {FP8_MODEL_PATH}")

    @pytest.mark.ci
    def test_fp8_load_and_forward_dp8(self, tmp_path):
        # [FUNCTION DOC]
        # Marked ci. Requires at least 2 GPUs (skips otherwise). Runs run_checkpoint.py
        # with all visible GPUs, passing --fp8_model_path and, if present,
        # --bf16_model_path for a logits comparison. Passes when the worker writes
        # "Passed".
        # KNOWN ISSUE: The name says dp8 but it uses however many GPUs are visible
        # (minimum 2), not necessarily 8.
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need at least 2 GPUs")
        extra = [f"--fp8_model_path={FP8_MODEL_PATH}"]
        if os.path.exists(BF16_MODEL_PATH):
            extra.append(f"--bf16_model_path={BF16_MODEL_PATH}")
        _run_torchrun_test(
            "tests/experimental/archon/fp8/torchrun/run_checkpoint.py",
            n_gpus,
            str(tmp_path / "result.out"),
            extra_args=extra,
        )
