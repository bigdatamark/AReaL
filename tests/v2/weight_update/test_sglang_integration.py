# [MODULE DOC]
# GPU integration tests for AReaL's custom SGLang server
# (areal.v2.inference_service.sglang.launch_server) that exposes /awex/* weight-update
# endpoints. A module-scoped fixture spawns the server subprocess on a free port with a
# small Qwen3-0.6B model; tests hit /health, /awex/report_parallelism and
# /awex/report_weight_meta via httpx. Marked slow and sglang and skipped without CUDA.
# KNOWN ISSUE: the module docstring's run command references
# test_sglang_server_integration.py, but this file is test_sglang_integration.py.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for AReaL's custom SGLang server with /awex/* endpoints.

Requires GPU. Marked @pytest.mark.slow and @pytest.mark.sglang to exclude
from default CI. Run manually:
    uv run pytest tests/v2/weight_update/test_sglang_server_integration.py -v -s
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx
import pytest
import torch

from areal.utils.network import find_free_ports

pytestmark = [
    pytest.mark.slow,
    pytest.mark.sglang,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

SERVER_STARTUP_TIMEOUT = 180


def _get_test_model_path() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the local Qwen3-0.6B directory under /storage/openpsi/models if it
    # exists, otherwise the HF hub id 'Qwen/Qwen3-0.6B'. Reads the filesystem only.
    """Get a small model path for testing — prefer local, fall back to HF."""
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def sglang_server():
    # [FUNCTION DOC]
    # Module-scoped fixture. Picks a free port, launches the custom SGLang server via
    # 'python -m' (tp-size 1, mem-fraction 0.7, host 127.0.0.1, log-level warning) with
    # output forwarded to the test's stdout, then polls GET /health every 2s for up to
    # SERVER_STARTUP_TIMEOUT (180s), ignoring connect/read timeouts. Fails the test if
    # the process exits early or never becomes healthy (killing it in the latter case).
    # Yields (base_url, process); on teardown sends SIGTERM, waits 30s, then kills.
    # KNOWN ISSUE: stdout/stderr are redirected to sys.stdout rather than PIPE, so
    # process.stdout/stderr are None and the early-exit failure message always shows
    # empty output. Also launches 'python' rather than sys.executable.
    """Launch AReaL's custom SGLang server and yield (base_url, process).

    The server is launched as a subprocess using the custom entry point
    that registers /awex/* weight update endpoints.
    """
    port = find_free_ports(1)[0]
    model_path = _get_test_model_path()

    process = subprocess.Popen(
        [
            "python",
            "-m",
            "areal.v2.inference_service.sglang.launch_server",
            "--model-path",
            model_path,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.7",
            "--log-level",
            "warning",
        ],
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    base_url = f"http://127.0.0.1:{port}"

    deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT
    healthy = False
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass

        if process.poll() is not None:
            stdout = process.stdout.read().decode() if process.stdout else ""
            stderr = process.stderr.read().decode() if process.stderr else ""
            pytest.fail(
                f"Server process exited prematurely (code {process.returncode}).\n"
                f"stdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}"
            )
        time.sleep(2.0)

    if not healthy:
        process.kill()
        process.wait(timeout=10)
        pytest.fail(f"Server failed to become healthy within {SERVER_STARTUP_TIMEOUT}s")

    yield base_url, process

    os.kill(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


class TestSGLangServerHealth:
    # [CLASS DOC]
    # Server health check tests.
    def test_server_healthy(self, sglang_server):
        # [FUNCTION DOC]
        # Asserts GET /health returns 200 after startup.
        """Verify /health returns 200 after startup."""
        base_url, _ = sglang_server
        resp = httpx.get(f"{base_url}/health", timeout=10.0)
        assert resp.status_code == 200


class TestAwexEndpointsRegistered:
    # [CLASS DOC]
    # Verifies the custom /awex/* endpoints are registered and respond sensibly.
    def test_report_parallelism_endpoint_exists(self, sglang_server):
        # [FUNCTION DOC]
        # Asserts GET /awex/report_parallelism returns 200 with a 'world_size' key.
        """GET /awex/report_parallelism should return parallelism info."""
        base_url, _ = sglang_server
        resp = httpx.get(f"{base_url}/awex/report_parallelism", timeout=30.0)
        assert resp.status_code == 200
        data = resp.json()
        assert "world_size" in data

    def test_report_weight_meta_endpoint_exists(self, sglang_server):
        # [FUNCTION DOC]
        # Asserts POST /awex/report_weight_meta returns 200 with status 'ok' and a
        # 'meta' key (60s timeout).
        """POST /awex/report_weight_meta should return weight metadata."""
        base_url, _ = sglang_server
        resp = httpx.post(f"{base_url}/awex/report_weight_meta", timeout=60.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"
        assert "meta" in data

    def test_report_parallelism_returns_valid_parallelism(self, sglang_server):
        # [FUNCTION DOC]
        # Asserts /awex/report_parallelism returns an integer world_size >= 1.
        """world_size must be a positive integer (>= 1)."""
        base_url, _ = sglang_server
        resp = httpx.get(f"{base_url}/awex/report_parallelism", timeout=30.0)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["world_size"], int)
        assert data["world_size"] >= 1
