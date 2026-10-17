# [MODULE DOC]
# Shared helpers and constants for inference-service integration tests: model paths for
# Qwen3-0.6B and Qwen3-VL-2B (local cluster storage preferred, HuggingFace fallback),
# GPU detection, server health checks and launching a standalone vLLM server subprocess.
# areal.* imports are deferred into function bodies for Python 3.10 compatibility.
"""Integration test utilities for the experimental gateway.

This module provides helper functions and constants for integration testing
of the gateway module. All imports from areal.* are lazy (inside functions)
for Python 3.10 compatibility.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from typing import Any

import requests
import torch
from huggingface_hub import snapshot_download

# =============================================================================
# Constants
# =============================================================================

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
SERVER_STARTUP_TIMEOUT = 180  # seconds
EXPR_NAME = "test_gateway_controller_integration"
TRIAL_NAME = "trial_0"

VLM_LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen3-VL-2B-Instruct"
VLM_HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


# =============================================================================
# Helper Functions
# =============================================================================


def _get_model_path(local_path: str, hf_id: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns local_path if it exists on disk; otherwise downloads hf_id with
    # huggingface_hub.snapshot_download (skipping *.gguf, *.ggml, consolidated* files)
    # and returns the download directory. Network/disk side effects. Raises RuntimeError
    # (chained) if the download fails.
    """Get model path, preferring local storage over HuggingFace Hub.

    If local_path exists, returns it directly. Otherwise downloads
    the model from HuggingFace Hub using snapshot_download.

    Args:
        local_path: Local path to check first.
        hf_id: HuggingFace model ID to download if local not found.

    Returns:
        Path to the model (either local or downloaded).
    """
    if os.path.exists(local_path):
        return local_path

    try:
        downloaded_path = snapshot_download(
            repo_id=hf_id,
            ignore_patterns=["*.gguf", "*.ggml", "consolidated*"],
        )
        return downloaded_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {hf_id}: {e}") from e


def has_gpu() -> bool:
    # [FUNCTION DOC]
    # Returns True iff torch reports CUDA available and at least one device.
    """Check if GPU is available.

    Returns:
        bool: True if CUDA is available and at least one GPU device is present.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def get_test_model_path() -> str:
    # [FUNCTION DOC]
    # Returns the Qwen3-0.6B model path via _get_model_path (may download).
    """Get the model path for tests (lazy evaluation).

    Returns:
        str: Path to the test model, falling back to HuggingFace if not available locally.
    """
    return _get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def get_vlm_test_model_path() -> str:
    # [FUNCTION DOC]
    # Returns the Qwen3-VL-2B-Instruct model path via _get_model_path (may download).
    """Get the VLM model path for tests (Qwen3-VL-2B-Instruct).

    Returns:
        str: Path to the VLM test model, falling back to HuggingFace if not available locally.
    """
    return _get_model_path(VLM_LOCAL_MODEL_PATH, VLM_HF_MODEL_ID)


def check_server_health(base_url: str) -> bool:
    # [FUNCTION DOC]
    # GETs {base_url}/health with a 30s timeout; returns True on HTTP 200, False on
    # non-200 or any requests exception.
    """Check if the inference server is healthy.

    Args:
        base_url: The base URL of the server (e.g., "http://localhost:8000").

    Returns:
        bool: True if the server responds with a 200 status to the /health endpoint.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def launch_vllm_server(
    model_path: str,
    startup_timeout: int = SERVER_STARTUP_TIMEOUT,
) -> tuple[subprocess.Popen, dict[str, Any]]:
    # [FUNCTION DOC]
    # Launches a vLLM OpenAI server subprocess for model_path on a free port of the
    # local host IP, using vLLMConfig.build_cmd with gpu_memory_utilization 0.3 and
    # tp/pp size 1. Environment: copies os.environ, gives TRITON_CACHE_PATH and
    # VLLM_CACHE_ROOT unique uuid subdirectories, and enables
    # VLLM_ALLOW_RUNTIME_LORA_UPDATING. Output goes to stdout. Polls check_server_health
    # once per second up to startup_timeout seconds; on timeout kills the process tree
    # gracefully and raises RuntimeError. Returns (Popen, info) where info has host,
    # port and base_url; the caller must kill the process tree on cleanup.
    """Launch a vLLM server subprocess and wait for it to become healthy.

    Returns:
        Tuple of (process, info_dict) where info_dict has keys
        ``host``, ``port``, ``base_url``.  Caller is responsible for
        calling ``kill_process_tree(process.pid)`` on cleanup.
    """
    from areal.api.cli_args import vLLMConfig
    from areal.infra.utils.launcher import TRITON_CACHE_PATH, VLLM_CACHE_ROOT
    from areal.utils import network

    host = network.gethostip()
    (port,) = network.find_free_ports(1)

    cmd = vLLMConfig.build_cmd(
        vllm_config=vLLMConfig(
            model=model_path,
            gpu_memory_utilization=0.3,
        ),
        tp_size=1,
        pp_size=1,
        host=host,
        port=port,
    )

    env = os.environ.copy()
    env["TRITON_CACHE_PATH"] = os.path.join(
        env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH), str(uuid.uuid4())
    )
    env["VLLM_CACHE_ROOT"] = os.path.join(
        env.get("VLLM_CACHE_ROOT", VLLM_CACHE_ROOT), str(uuid.uuid4())
    )
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    process = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stdout)
    base_url = f"http://{host}:{port}"

    t0 = time.time()
    while time.time() - t0 < startup_timeout:
        if check_server_health(base_url):
            break
        time.sleep(1)
    else:
        from areal.infra.utils.proc import kill_process_tree

        kill_process_tree(process.pid, graceful=True)
        raise RuntimeError(
            f"vLLM server did not become healthy within {startup_timeout}s"
        )

    return process, {"host": host, "port": port, "base_url": base_url}
