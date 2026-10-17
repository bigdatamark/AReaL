# [MODULE DOC]
# Slow GPU integration tests for the OpenAI proxy rollout architecture: launches a real
# SGLang server (Qwen3-0.6B), builds a RolloutController with RemoteSGLangEngine on a
# LocalScheduler, starts proxy workers and runs the SimpleAgent workflow end-to-end via
# rollout_batch. Also includes in-process checks of proxy_rollout_server.SessionData
# staleness and lifecycle. Marked sglang and slow; skipped without a GPU.
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
import requests
import torch

from tests.utils import get_model_path

from areal.api import LocalInfServerInfo
from areal.api.cli_args import (
    InferenceEngineConfig,
    SGLangConfig,
)
from areal.engine import RemoteSGLangEngine
from areal.infra import RolloutController
from areal.infra.rpc.rtensor import RTensor
from areal.infra.scheduler.local import LocalScheduler
from areal.infra.utils.proc import kill_process_tree
from areal.utils import network, seeding

pytestmark = pytest.mark.sglang
# =============================================================================
# Test Configuration
# =============================================================================

EXPR_NAME = "test_proxy_integration"
TRIAL_NAME = "trial_0"
LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
RUN_SERVER_TIMEOUT = 180


def get_test_model_path() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the local Qwen3-0.6B path if present, else the HF hub id (resolved
    # lazily via get_model_path).
    """Get the model path for tests (lazy evaluation)."""
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def has_local_model() -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the local model directory exists. Unused within this
    # module.
    """Check if the model is available locally (without network)."""

    return os.path.exists(LOCAL_MODEL_PATH)


def check_server_health(base_url: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: GETs <base_url>/health with a 30s timeout; True iff status 200, False on
    # any requests exception.
    """Check if the inference server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def has_gpu() -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if CUDA is available with at least one device.
    """Check if GPU is available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sglang_server():
    # [FUNCTION DOC]
    # Module-scoped fixture: skips without GPU, seeds RNGs, picks host IP and two free
    # ports, launches SGLang (skip tokenizer init, mem fraction 0.3, tp 1, GPU 0) as a
    # subprocess, polls health each second for up to 180s (killing the process and
    # failing the test on timeout), yields a dict with host, port, base_url and process,
    # and finally kills the process tree.
    """Launch an SGLang server for testing."""
    if not has_gpu():
        pytest.skip("GPU required for SGLang server")

    seeding.set_random_seed(1, EXPR_NAME)

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=get_test_model_path(),
            mem_fraction_static=0.3,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    base_url = f"http://{host}:{port}"

    # Wait for server to be ready
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)

    if time.time() - tik >= RUN_SERVER_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server failed to start within timeout")

    yield {"host": host, "port": port, "base_url": base_url, "process": process}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture
def local_scheduler(tmp_path):
    # [FUNCTION DOC]
    # Fixture creating a LocalScheduler on GPU 0 with log dir, fileroot and name-resolve
    # directories under tmp_path; skips without GPU. Yields it and deletes all its
    # workers on teardown. Side effects: creates directories, may spawn worker
    # processes.
    """Create a LocalScheduler for testing."""
    if not has_gpu():
        pytest.skip("GPU required for LocalScheduler")

    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()
    scheduler = LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )
    yield scheduler
    scheduler.delete_workers(None)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required for integration tests")
class TestProxyIntegration:
    # [CLASS DOC]
    # End-to-end proxy workflow test class (slow, GPU only).
    """Integration tests for the proxy architecture."""

    def test_proxy_workflow_roundtrip(self, sglang_server, local_scheduler, tmp_path):
        # [FUNCTION DOC]
        # Builds an InferenceEngineConfig (sglang:d1, 2 concurrent rollouts, consumer
        # batch 1, tokenizer path set) and a RolloutController over RemoteSGLangEngine
        # and the scheduler. Initializes it against the already running server
        # (LocalInfServerInfo without process), starts proxy workers, asserts proxy 0's
        # address is http, runs rollout_batch on one message sample with workflow
        # tests.experimental.openai.utils.SimpleAgent, and asserts the result is a
        # non-empty list of dicts whose input_ids is a 2-D RTensor. Always destroys the
        # controller.
        """Test a complete roundtrip through the proxy architecture.

        This test:
        1. Creates a RolloutController with RemoteSGLangEngine
        2. Initializes rollout workers connected to the SGLang server
        3. Starts proxy workers
        4. Runs a simple agent workflow through OpenAIProxyWorkflow
        5. Verifies the result contains expected tensor fields
        """

        # Create RolloutController
        config = InferenceEngineConfig(
            experiment_name=EXPR_NAME,
            trial_name=TRIAL_NAME,
            backend="sglang:d1",
            max_concurrent_rollouts=2,
            consumer_batch_size=1,
            tokenizer_path=get_test_model_path(),
        )

        rollout = RolloutController(
            inf_engine=RemoteSGLangEngine,
            config=config,
            scheduler=local_scheduler,
        )

        try:
            # Initialize rollout workers connected to existing SGLang server
            server_info = LocalInfServerInfo(
                process=None,
                host=sglang_server["host"],
                port=sglang_server["port"],
            )

            rollout.initialize(
                role="rollout",
                server_infos=[server_info],
            )

            # Start proxy workers
            rollout.start_proxy()

            # Get proxy address
            proxy_addr = rollout.get_proxy_addr(0)
            assert proxy_addr.startswith("http://")

            # Create test data
            test_data = [
                {
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "answer": "4",
                }
            ]

            # Run workflow
            result = rollout.rollout_batch(
                data=test_data,
                workflow="tests.experimental.openai.utils.SimpleAgent",
            )

            # Verify result structure - rollout_batch now returns list[dict]
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], dict)
            assert "input_ids" in result[0]
            assert isinstance(result[0]["input_ids"], RTensor)
            assert result[0]["input_ids"].ndim == 2  # [batch, seq_len]

        finally:
            rollout.destroy()


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required for integration tests")
class TestProxyServerEndpoints:
    # [CLASS DOC]
    # Direct tests of proxy server session data structures (slow, GPU-gated though they
    # need no GPU).
    """Test proxy server HTTP endpoints directly."""

    def test_session_timeout_cleanup(self):
        # [FUNCTION DOC]
        # Inserts a SessionData into the module-global _session_cache, asserts it is not
        # stale, backdates its private _last_access_time past SESSION_TIMEOUT_SECONDS
        # and asserts it is stale, then removes it from the cache.
        # KNOWN ISSUE: named "cleanup" but never exercises any cleanup routine; it only
        # checks is_stale. If an assertion fails the cache entry is not removed.
        """Test that stale sessions are cleaned up."""
        from areal.experimental.openai.proxy.proxy_rollout_server import (
            SessionData,
            _session_cache,
        )
        from areal.experimental.openai.proxy.server import SESSION_TIMEOUT_SECONDS

        # Create a session
        session = SessionData(session_id="test-session-1")
        _session_cache["test-session-1"] = session

        # Session should not be stale immediately
        assert not session.is_stale()

        # Manually set last access time to past
        import time

        session._last_access_time = time.time() - SESSION_TIMEOUT_SECONDS - 1

        # Now it should be stale
        assert session.is_stale()

        # Clean up the test session
        _session_cache.pop("test-session-1", None)

    @pytest.mark.asyncio
    async def test_session_data_lifecycle(self):
        # [FUNCTION DOC]
        # Asserts a new SessionData has its id and is not completed, update_last_access
        # advances _last_access_time, finish sets _completed and _end_time, and
        # wait_for_finish(timeout 0.1) returns True immediately once finished.
        """Test SessionData class methods."""
        from areal.experimental.openai.proxy.proxy_rollout_server import SessionData

        session = SessionData(session_id="test-lifecycle")

        # Test initial state
        assert session.session_id == "test-lifecycle"
        assert not session._completed

        # Test update_last_access
        import time

        old_time = session._last_access_time
        time.sleep(0.01)
        session.update_last_access()
        assert session._last_access_time > old_time

        # Test finish
        session.finish()
        assert session._completed
        assert session._end_time is not None

        # Test wait_for_finish (should return immediately since already finished)
        result = await session.wait_for_finish(timeout=0.1)
        assert result is True
