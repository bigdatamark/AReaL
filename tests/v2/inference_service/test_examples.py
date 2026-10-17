# [MODULE DOC]
# Slow end-to-end example test for the experimental inference service: runs the tau2
# rollout example (examples/experimental/inference_service/tau2_rollout.py) against a
# separately launched SGLang "user LLM" server. Module-level pytestmark marks all tests
# slow; logs via the "InferenceServiceExamples" logger.
import os
import re
import subprocess
import time

import pytest

from tests.test_examples import run_example
from tests.utils import get_model_path

from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging

logger = logging.getLogger("InferenceServiceExamples")

pytestmark = pytest.mark.slow


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_tau2_rollout(tmp_path_factory):
    # [FUNCTION DOC]
    # SGLang multi-GPU test. Skips if the tau2 package is missing, TAU2_DATA_DIR is
    # unset or nonexistent, the Qwen3 non-thinking chat template file is missing, or
    # fewer than 3 devices exist. Resolves Qwen3-0.6B, launches an sglang.launch_server
    # subprocess on the last visible device (port 30081, qwen25 tool-call parser, mem
    # fraction 0.8, output discarded), sleeps a fixed 60s for startup, then runs the
    # example via run_example with backend sglang:d2 on 2 GPUs, temp
    # experiment/name_resolve dirs, batch size 2, the user LLM base URL and wandb
    # disabled, waiting up to 600s for the "Rollout complete" log pattern. Asserts
    # success; always force-kills the user LLM process tree afterwards.
    # KNOWN ISSUE: startup is a fixed 60s sleep with no health check, so a slow server
    # load can cause flaky failures.
    tau2 = pytest.importorskip("tau2")
    del tau2

    tau2_data_dir = os.environ.get("TAU2_DATA_DIR")
    if not tau2_data_dir:
        pytest.skip("TAU2_DATA_DIR environment variable not set. Skipping tau2 test.")
    if not os.path.exists(tau2_data_dir):
        pytest.skip(
            f"TAU2_DATA_DIR ({tau2_data_dir}) does not exist. Skipping tau2 test."
        )

    chat_template_path = "/storage/openpsi/data/qwen3_nonthinking.jinja"
    if not os.path.exists(chat_template_path):
        pytest.skip(f"Chat template not found at {chat_template_path}")

    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for user LLM, 2 for rollout) to run."
        )

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    user_llm_gpu = visible_devices[-1]
    user_llm_port = 30081

    _env = os.environ.copy()
    _env[current_platform.device_control_env_var] = user_llm_gpu

    logger.info(
        f"Launching user LLM server on GPU {user_llm_gpu}, port {user_llm_port}"
    )
    user_llm_proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(user_llm_port),
            "--tool-call-parser",
            "qwen25",
            "--chat-template",
            chat_template_path,
            "--dp-size",
            "1",
            "--mem-fraction-static",
            "0.8",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_env,
    )

    try:
        logger.info("Waiting for user LLM server to start...")
        time.sleep(60)

        user_llm_base_url = f"http://localhost:{user_llm_port}/v1/"
        success = run_async_task(
            run_example,
            "examples/experimental/inference_service/tau2_rollout.py",
            "examples/experimental/inference_service/tau2_rollout.yaml",
            "rollout.backend=sglang:d2",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            "rollout.admin_api_key=test-admin-key",
            f"model_path={model_path}",
            "train_dataset.batch_size=2",
            "train_dataset.path=tau2/train",
            f"econfig.user_llm_base_url={user_llm_base_url}",
            "econfig.user_llm=openai/self-hosted-qwen3",
            "stats_logger.wandb.mode=disabled",
            timeout=600,
            success_pattern=re.compile(r"Rollout complete"),
        )
        assert success, "Tau2 rollout example failed"
    finally:
        logger.info("Shutting down user LLM server...")
        kill_process_tree(user_llm_proc.pid, graceful=False)
