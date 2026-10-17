# [MODULE DOC]
# Multi-GPU end-to-end test that MegatronEngine's global grad norm is invariant to
# tensor-parallel degree. Spawns the worker script
# tests/megatron/torchrun/run_grad_norm_tp.py via torchrun at TP=1 and TP=2, reads each
# run's JSON result, and compares grad norms; also checks that duplicated params were
# demoted from tensor_model_parallel at TP=2. Skipped without CUDA or with fewer than 2
# devices.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that duplicated params do not inflate the TP grad norm.

The global grad norm is a property of the full model + data and must be
invariant to the tensor-parallel degree. If replicated params are left marked
``tensor_model_parallel=True`` (the TE default that
``MegatronEngine._mark_duplicated_params`` fixes), they get counted on every TP
rank and SUM-reduced, so the reported grad norm grows with TP.

This launches the real MegatronEngine via torchrun at TP=1 and TP=2 on the same
deterministic input and asserts the two grad norms agree. It also asserts that
the fix actually demoted at least one real duplicated param at TP=2.

Requires >= 2 GPUs; skipped otherwise.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_WORKER = "tests/megatron/torchrun/run_grad_norm_tp.py"


def _run_tp(tp: int, output: str) -> dict:
    # [FUNCTION DOC]
    # HELPER: Launches torchrun on a free port with nproc_per_node=tp running the worker
    # script with --tp and --output, PYTHONPATH prefixed with the project root,
    # stdout/stderr forwarded to the current stdout. Waits for it; on any exception
    # (including KeyboardInterrupt) kills the process tree and re-raises. Calls
    # pytest.fail on nonzero exit. Returns the parsed JSON dict written to output (keys
    # grad_norm, num_duplicated, structural_ok).
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [
            "torchrun",
            f"--nproc_per_node={tp}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            _WORKER,
            f"--tp={tp}",
            f"--output={output}",
        ],
        text=True,
        stderr=sys.stdout,
        stdout=sys.stdout,
        env=env,
    )
    try:
        proc.wait()
    except BaseException:
        kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        pytest.fail(f"torchrun (tp={tp}) exited with code {proc.returncode}")

    with open(output) as f:
        return json.load(f)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_grad_norm_is_tp_invariant(tmp_path_factory):
    # [FUNCTION DOC]
    # Skips if fewer than 2 devices. Runs TP=1 and TP=2 into a temp dir, asserts TP=2
    # result is structural_ok with num_duplicated > 0, and that the two grad norms match
    # within 2% relative tolerance.
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    out_dir = tmp_path_factory.mktemp("grad_norm_tp")
    res1 = _run_tp(1, str(out_dir / "tp1.json"))
    res2 = _run_tp(2, str(out_dir / "tp2.json"))

    # The fix must have demoted at least one real duplicated param at TP=2.
    assert res2["structural_ok"], "a duplicated param kept tensor_model_parallel=True"
    assert res2["num_duplicated"] > 0, "no duplicated params detected in the model"

    gn1, gn2 = res1["grad_norm"], res2["grad_norm"]
    assert gn1 == pytest.approx(gn2, rel=0.02), (
        f"grad norm not TP-invariant: tp1={gn1}, tp2={gn2} "
        "(duplicated params likely double-counted)"
    )
