# [MODULE DOC]
# Unit tests for LocalLauncher.submit command construction (areal.infra.launcher.local),
# focusing on env var quoting and whether stdbuf line buffering is prepended when
# LD_PRELOAD is present. subprocess.Popen is patched.
from __future__ import annotations

from collections import defaultdict
from unittest.mock import Mock, patch

import pytest

from areal.infra.launcher.local import LocalLauncher


@pytest.fixture
def launcher(tmp_path):
    # [FUNCTION DOC]
    # Pytest fixture: builds a LocalLauncher via object.__new__ (skipping __init__ and
    # GPU probing), sets experiment/trial names, fileroot=tmp_path, empty job tables, a
    # GPU counter of 0, no GPU devices and a Mock wait(). Yields it and clears
    # jobs/counters afterwards.
    """Create a LocalLauncher without probing host GPU devices."""
    local_launcher = object.__new__(LocalLauncher)
    local_launcher.experiment_name = "experiment"
    local_launcher.trial_name = "trial"
    local_launcher.fileroot = str(tmp_path)
    local_launcher._jobs = {}
    local_launcher._job_counter = defaultdict(int)
    local_launcher._job_states = {}
    local_launcher._gpu_counter = 0
    local_launcher._gpu_devices = []
    local_launcher.wait = Mock()
    yield local_launcher
    local_launcher._jobs.clear()
    local_launcher._job_counter.clear()


def _submitted_command(launcher: LocalLauncher, env_vars: dict[str, str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Calls launcher.submit for job 'trainer' running 'python train.py' with the
    # given env_vars while Popen is patched to return a fake process (pid 1234); returns
    # the shell command string passed to Popen.
    process = Mock(pid=1234)
    with patch(
        "areal.infra.launcher.local.subprocess.Popen", return_value=process
    ) as mock_popen:
        launcher.submit(
            job_name="trainer",
            cmd="python train.py",
            env_vars=env_vars,
        )

    return mock_popen.call_args.args[0]


def test_submit_regular_env_uses_target_stdbuf_and_quotes_value(launcher, monkeypatch):
    # [FUNCTION DOC]
    # With no LD_PRELOAD in the environment, asserts the command starts with the quoted
    # env assignment followed by 'stdbuf -oL python train.py 2>&1 | tee -a '.
    """Regular local jobs retain target buffering and shell-safe values."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    command = _submitted_command(launcher, {"WORKER_LABEL": "actor one"})

    assert command.startswith(
        "WORKER_LABEL='actor one' stdbuf -oL python train.py 2>&1 | tee -a "
    )


def test_submit_explicit_preload_skips_target_stdbuf(launcher, monkeypatch):
    # [FUNCTION DOC]
    # With LD_PRELOAD given in env_vars, asserts env assignments are shell-quoted, the
    # command runs without stdbuf before the pipe.
    """An explicit TMS preload is not extended with libstdbuf."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    command = _submitted_command(
        launcher,
        {
            "LD_PRELOAD": "/opt/tms hooks/tms preload.so",
            "WORKER_LABEL": "actor one",
        },
    )

    assert command.startswith(
        "LD_PRELOAD='/opt/tms hooks/tms preload.so' "
        "WORKER_LABEL='actor one' python train.py 2>&1 | tee -a "
    )
    assert "stdbuf" not in command.split(" 2>&1", maxsplit=1)[0]


def test_submit_inherited_preload_skips_target_stdbuf(launcher, monkeypatch):
    # [FUNCTION DOC]
    # With LD_PRELOAD set in the launcher's own environment, asserts no stdbuf is
    # prepended and the plain env assignment is used.
    """A preload inherited by LocalLauncher is not extended with libstdbuf."""
    monkeypatch.setenv("LD_PRELOAD", "/opt/tms/torch_memory_saver.so")

    command = _submitted_command(launcher, {"WORKER_LABEL": "actor"})

    assert command.startswith("WORKER_LABEL=actor python train.py 2>&1 | tee -a ")
    assert "stdbuf" not in command.split(" 2>&1", maxsplit=1)[0]
