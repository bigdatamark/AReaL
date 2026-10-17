# [MODULE DOC]
# Click command "areal agent stop": terminates all processes of an agent service and
# removes its persisted state.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.agent.state import ServiceState
from areal.v2.cli.process import kill_pids


@click.command(name="stop", help="Stop an agent service.")
@click.option("--service", default=None)
@click.option("--grace-period", type=float, default=10.0, show_default=True)
@click.option("--keep-state", is_flag=True)
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
def stop_cmd(
    service: str | None, grace_period: float, keep_state: bool, force: bool
) -> None:
    # [FUNCTION DOC]
    # Click entry point. Options: --service (default current), --grace-period seconds
    # before SIGKILL (10.0), --keep-state (do not delete the state file), --force
    # (SIGKILL immediately). Exits via SystemExit with do_stop's return code.
    raise SystemExit(
        do_stop(
            service=service,
            grace_period=grace_period,
            keep_state=keep_state,
            force=force,
        )
        or 0
    )


def do_stop(
    *,
    service: str | None,
    grace_period: float,
    keep_state: bool,
    force: bool,
) -> int:
    # [FUNCTION DOC]
    # Resolves the service name; if no state file exists prints "not running" and
    # returns 0. If the state cannot be loaded, deletes it (unless keep_state) and
    # reports stale state removed. Otherwise collects positive pids from all components
    # and kills them via kill_pids with a grace of 0 when force else grace_period,
    # deletes the state unless keep_state, prints "stopped" and returns 0.
    # KNOWN ISSUE: the "removed stale state" message is printed even when keep_state
    # prevented removal.
    name = agent_lifecycle.resolve_service_name(service)
    path = agent_lifecycle.state_path(name)
    if not path.exists():
        click.echo(f"service {name!r} is not running")
        return 0

    try:
        state = agent_lifecycle.load_state(name)
    except Exception:
        if not keep_state:
            ServiceState.remove(name)
        click.echo(f"removed stale state for {name!r}")
        return 0

    pids = [pid for _, h in state.components() if (pid := h.pid) > 0]
    kill_pids(pids, grace_s=0.0 if force else grace_period)
    if not keep_state:
        ServiceState.remove(name)
    click.echo(f"service {name!r} stopped")
    return 0
