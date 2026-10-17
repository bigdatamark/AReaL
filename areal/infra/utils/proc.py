# [MODULE DOC]
# Process utilities: building shell commands with env prefixes and line buffering,
# streaming child output to the terminal plus role-specific and merged (role-prefixed)
# log files, and killing process trees with psutil. Logger "ProcUtils".
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from areal.utils import logging
from areal.utils.logging import LOG_PREFIX_WIDTH

logger = logging.getLogger("ProcUtils")

if TYPE_CHECKING:
    from typing import IO


def build_target_cmd(
    cmd: str | list[str],
    *,
    env_vars: dict[str, str] | None = None,
    use_stdbuf: bool = False,
    target_env: dict[str, str] | None = None,
) -> str:
    # [FUNCTION DOC]
    # Returns a single shell string: optional "K=V ..." prefix from env_vars (values
    # shell-quoted), then "stdbuf -oL" when use_stdbuf is set and LD_PRELOAD is absent
    # from both env_vars and the effective target environment (target_env, or os.environ
    # when None), then the command (a list is shlex-quoted and joined).
    """Build a shell command with optional environment and line buffering.

    GNU ``stdbuf`` injects ``libstdbuf`` through ``LD_PRELOAD``. Avoid wrapping a
    target whose effective environment already contains ``LD_PRELOAD`` so the
    caller-provided preload value reaches the target unchanged. A ``None`` target
    environment follows :class:`subprocess.Popen` semantics and inherits
    :data:`os.environ`; an explicit empty mapping does not.
    """
    if isinstance(cmd, list):
        cmd_str = " ".join(shlex.quote(str(part)) for part in cmd)
    else:
        cmd_str = cmd

    command_parts: list[str] = []
    if env_vars:
        command_parts.append(
            " ".join(
                f"{key}={shlex.quote(str(value))}" for key, value in env_vars.items()
            )
        )

    inherited_env = os.environ if target_env is None else target_env
    target_has_preload = (
        "LD_PRELOAD" in (env_vars or {}) or "LD_PRELOAD" in inherited_env
    )
    if use_stdbuf and not target_has_preload:
        command_parts.append("stdbuf -oL")
    command_parts.append(cmd_str)
    return " ".join(command_parts)


def build_streaming_log_cmd(
    cmd: str | list[str],
    log_file: str | Path,
    merged_log: str | Path,
    role: str,
    *,
    env_vars: dict[str, str] | None = None,
    target_env: dict[str, str] | None = None,
) -> str:
    # [FUNCTION DOC]
    # Builds a bash command running the target (via build_target_cmd, using stdbuf if
    # available on PATH) with stderr merged into stdout, piped to tee which appends to
    # log_file and, via process substitution, to a sed that prefixes each line with
    # "[role]" padded to LOG_PREFIX_WIDTH and appends to merged_log; tee also echoes to
    # stdout. Returns the string.
    # KNOWN ISSUE: log_file/merged_log paths and role are not shell-quoted; spaces or
    # special characters (e.g. "/" in role for sed) break the command.
    """Build a shell command that streams output to stdout and log files.

    The command uses tee/sed pattern:
    - stdout streams to terminal in real-time
    - Output appends to role-specific log file (no prefix)
    - Output appends to merged log with [role] prefix

    Parameters
    ----------
    cmd : str | list[str]
        Command to execute. If list, will be shell-escaped and joined.
    log_file : str | Path
        Path to role-specific log file (appended with -a)
    merged_log : str | Path
        Path to merged log file (appended with prefix)
    role : str
        Role name for log prefix (e.g., "actor", "master")
    env_vars : dict[str, str] | None
        Optional environment variables to prefix the command with KEY=VALUE
    target_env : dict[str, str] | None
        Optional environment passed directly to the target process. ``None``
        means that the target inherits :data:`os.environ`.

    Returns
    -------
    str
        Shell command string ready for execution with bash
    """
    # Check if stdbuf is available (not present on macOS by default)
    _has_stdbuf = shutil.which("stdbuf") is not None

    full_cmd = build_target_cmd(
        cmd,
        env_vars=env_vars,
        use_stdbuf=_has_stdbuf,
        target_env=target_env,
    )

    # Build log prefix for merged log
    log_prefix = f"[{role}]".ljust(LOG_PREFIX_WIDTH)

    # Construct tee/sed pipeline
    if _has_stdbuf:
        sed_prefix = f"stdbuf -oL sed 's/^/{log_prefix}/'"
    else:
        sed_prefix = f"sed 's/^/{log_prefix}/'"
    shell_cmd = f"{full_cmd} 2>&1 | tee -a {log_file} >({sed_prefix} >> {merged_log})"
    return shell_cmd


def run_with_streaming_logs(
    cmd: str | list[str],
    log_file: str | Path,
    merged_log: str | Path,
    role: str,
    *,
    env: dict[str, str] | None = None,
    env_vars_in_cmd: dict[str, str] | None = None,
    stdout: IO | None = None,
) -> subprocess.Popen:
    # [FUNCTION DOC]
    # Builds the streaming command and launches it with Popen(shell=True, /bin/bash)
    # using env, sending stdout and stderr to the given stream (default sys.stdout).
    # Returns the Popen handle; does not wait.
    """Run a command with streaming output to stdout and log files.

    Parameters
    ----------
    cmd : str | list[str]
        Command to execute
    log_file : str | Path
        Path to role-specific log file
    merged_log : str | Path
        Path to merged log file
    role : str
        Role name for log prefix
    env : dict[str, str] | None
        Environment dict for Popen (passed to subprocess)
    env_vars_in_cmd : dict[str, str] | None
        Environment variables to prefix in the shell command (KEY=VALUE format)
    stdout : IO | None
        File object for stdout. Defaults to sys.stdout

    Returns
    -------
    subprocess.Popen
        The spawned process
    """
    shell_cmd = build_streaming_log_cmd(
        cmd,
        str(log_file),
        str(merged_log),
        role,
        env_vars=env_vars_in_cmd,
        target_env=env,
    )

    return subprocess.Popen(
        shell_cmd,
        shell=True,
        executable="/bin/bash",
        env=env,
        stdout=stdout or sys.stdout,
        stderr=stdout or sys.stdout,
    )


def kill_process_tree(
    parent_pid: int | None = None,
    timeout: int = 5,
    include_parent: bool = True,
    skip_pid: int | None = None,
    graceful: bool = True,
) -> None:
    # [FUNCTION DOC]
    # Terminates parent_pid (default: current process, in which case include_parent is
    # forced False) and all recursive children, excluding skip_pid. In the main thread
    # first resets SIGCHLD to default. Returns early (INFO log) if the parent does not
    # exist. Graceful mode: SIGTERM to children and parent, wait up to timeout seconds,
    # SIGKILL survivors and wait 1s more, logging progress. Non-graceful: SIGKILL
    # children; if include_parent, SIGKILL the parent (calling sys.exit(0) if it is the
    # current process) and additionally send SIGQUIT for processes immune to SIGKILL
    # (e.g. PID 1 in containers). NoSuchProcess errors are ignored.
    # KNOWN ISSUE: killing the current process with SIGKILL means sys.exit(0) is
    # effectively unreachable.
    # Remove sigchld handler to avoid spammy logs (only in main thread)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    # Handle None parent_pid - defaults to current process
    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    # Get process tree
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        logger.info(f"Process {parent_pid} already terminated")
        return

    # Filter skip_pid from children
    if skip_pid is not None:
        children = [c for c in children if c.pid != skip_pid]

    # Terminate based on mode
    if graceful:
        # Graceful mode: SIGTERM → wait → SIGKILL
        logger.info(
            f"Sending SIGTERM to process {parent_pid} and {len(children)} children"
        )

        # Send SIGTERM to all children
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        # Send SIGTERM to parent if requested
        if include_parent:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                # Parent is already gone, but we still need to wait for children.
                pass

        # Wait for graceful shutdown
        procs_to_wait = children + ([parent] if include_parent else [])
        gone, alive = psutil.wait_procs(procs_to_wait, timeout=timeout)

        # Force kill any remaining processes
        if alive:
            logger.warning(
                f"Force killing {len(alive)} processes that didn't terminate gracefully"
            )
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass

            # Final wait to ensure they're gone
            psutil.wait_procs(alive, timeout=1)

        logger.info(f"Successfully cleaned up process tree for PID {parent_pid}")
    else:
        # Aggressive mode: immediate SIGKILL
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

        if include_parent:
            try:
                if parent_pid == os.getpid():
                    parent.kill()
                    sys.exit(0)

                parent.kill()

                # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
                # so we send an additional signal to kill them.
                parent.send_signal(signal.SIGQUIT)
            except psutil.NoSuchProcess:
                pass
