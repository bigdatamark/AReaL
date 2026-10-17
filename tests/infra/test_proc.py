# [MODULE DOC]
# Unit tests for build_streaming_log_cmd and run_with_streaming_logs in
# areal.infra.utils.proc. They verify the shell pipeline built to tee worker output into
# per-worker and merged logs, in particular that the target command gets 'stdbuf -oL'
# only when no LD_PRELOAD is in play (explicit, command-level, Popen env or inherited),
# while the sed stage keeps stdbuf. shutil.which and subprocess.Popen are patched.
from __future__ import annotations

from unittest.mock import Mock, patch

from areal.infra.utils.proc import build_streaming_log_cmd, run_with_streaming_logs

MODULE = "areal.infra.utils.proc"


def _target_command(shell_command: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the part of a shell command before ' 2>&1', i.e. the target
    # command with its env prefix.
    return shell_command.split(" 2>&1", maxsplit=1)[0]


def test_build_streaming_log_cmd_regular_env_uses_target_stdbuf(monkeypatch):
    # [FUNCTION DOC]
    # No LD_PRELOAD: target is prefixed with the quoted env assignment and 'stdbuf -oL',
    # and the sed stage uses stdbuf.
    """Regular targets retain line buffering and quote environment values."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    with patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"):
        command = build_streaming_log_cmd(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env_vars={"WORKER_LABEL": "actor one"},
        )

    assert (
        _target_command(command)
        == "WORKER_LABEL='actor one' stdbuf -oL python worker.py"
    )
    assert "stdbuf -oL sed" in command


def test_build_streaming_log_cmd_command_preload_skips_target_stdbuf(monkeypatch):
    # [FUNCTION DOC]
    # LD_PRELOAD given in env_vars: env assignments are quoted and the target runs
    # without stdbuf.
    """A command-prefixed preload reaches the target without libstdbuf."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    with patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"):
        command = build_streaming_log_cmd(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env_vars={
                "LD_PRELOAD": "/opt/tms hooks/tms preload.so",
                "WORKER_LABEL": "actor one",
            },
        )

    assert _target_command(command) == (
        "LD_PRELOAD='/opt/tms hooks/tms preload.so' "
        "WORKER_LABEL='actor one' python worker.py"
    )
    assert "stdbuf -oL sed" in command


def test_build_streaming_log_cmd_empty_preload_still_skips_target_stdbuf(
    monkeypatch,
):
    # [FUNCTION DOC]
    # An empty LD_PRELOAD in env_vars still suppresses target stdbuf and is emitted as
    # LD_PRELOAD=''.
    """An explicitly empty preload is still treated as caller-owned state."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    with patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"):
        command = build_streaming_log_cmd(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env_vars={"LD_PRELOAD": ""},
        )

    assert _target_command(command) == "LD_PRELOAD='' python worker.py"
    assert "stdbuf -oL sed" in command


def test_run_with_streaming_logs_popen_preload_skips_target_stdbuf(monkeypatch):
    # [FUNCTION DOC]
    # run_with_streaming_logs with env containing LD_PRELOAD: target has no stdbuf and
    # Popen receives the same env object.
    """A preload supplied through Popen's environment is preserved unchanged."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    child_env = {
        "LD_PRELOAD": "/opt/tms/torch_memory_saver.so",
        "WORKER_LABEL": "actor",
    }

    with (
        patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"),
        patch(f"{MODULE}.subprocess.Popen", return_value=Mock()) as mock_popen,
    ):
        run_with_streaming_logs(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env=child_env,
        )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "python worker.py"
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is child_env


def test_run_with_streaming_logs_command_preload_skips_target_stdbuf(monkeypatch):
    # [FUNCTION DOC]
    # run_with_streaming_logs with env_vars_in_cmd containing LD_PRELOAD: the assignment
    # is prefixed to the target without stdbuf and Popen env is None.
    """A LocalScheduler-style command preload reaches the target unchanged."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    command_env = {"LD_PRELOAD": "/opt/tms/torch_memory_saver.so"}

    with (
        patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"),
        patch(f"{MODULE}.subprocess.Popen", return_value=Mock()) as mock_popen,
    ):
        run_with_streaming_logs(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env_vars_in_cmd=command_env,
        )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == (
        "LD_PRELOAD=/opt/tms/torch_memory_saver.so python worker.py"
    )
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is None


def test_run_with_streaming_logs_inherited_preload_skips_target_stdbuf(
    monkeypatch,
):
    # [FUNCTION DOC]
    # Parent process has LD_PRELOAD and no env is passed: target runs without stdbuf and
    # Popen env is None (inherits).
    """A default Popen environment accounts for the parent's preload."""
    monkeypatch.setenv("LD_PRELOAD", "/opt/tms/torch_memory_saver.so")

    with (
        patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"),
        patch(f"{MODULE}.subprocess.Popen", return_value=Mock()) as mock_popen,
    ):
        run_with_streaming_logs(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
        )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "python worker.py"
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is None


def test_run_with_streaming_logs_empty_env_does_not_inherit_parent_preload(
    monkeypatch,
):
    # [FUNCTION DOC]
    # Parent has LD_PRELOAD but an explicit empty env is passed: the child does not
    # inherit it, so target gets stdbuf; Popen env is the given dict.
    """An explicit empty Popen environment remains isolated from its parent."""
    monkeypatch.setenv("LD_PRELOAD", "/opt/tms/torch_memory_saver.so")
    child_env: dict[str, str] = {}

    with (
        patch(f"{MODULE}.shutil.which", return_value="/usr/bin/stdbuf"),
        patch(f"{MODULE}.subprocess.Popen", return_value=Mock()) as mock_popen,
    ):
        run_with_streaming_logs(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
            env=child_env,
        )

    command = mock_popen.call_args.args[0]
    assert _target_command(command) == "stdbuf -oL python worker.py"
    assert "stdbuf -oL sed" in command
    assert mock_popen.call_args.kwargs["env"] is child_env


def test_build_streaming_log_cmd_without_stdbuf_preserves_fallback(monkeypatch):
    # [FUNCTION DOC]
    # When stdbuf is not found on PATH, the command contains no stdbuf at all.
    """Systems without stdbuf retain the existing unbuffered shell pipeline."""
    monkeypatch.delenv("LD_PRELOAD", raising=False)

    with patch(f"{MODULE}.shutil.which", return_value=None):
        command = build_streaming_log_cmd(
            ["python", "worker.py"],
            "/tmp/worker.log",
            "/tmp/merged.log",
            "actor",
        )

    assert _target_command(command) == "python worker.py"
    assert "stdbuf" not in command
