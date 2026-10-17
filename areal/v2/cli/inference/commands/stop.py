# [MODULE DOC]
# Click command "areal inf stop": terminates every process of an inference service and
# removes its state files.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.inference.common import logger, terminate_runtime_state
from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.inference.state import ModelState, ServiceState


@click.command(name="stop", help="Stop an inference service.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--grace", type=float, default=10.0, show_default=True)
@click.option("--force", is_flag=True, help="SIGKILL immediately.")
@click.option("--keep-state", is_flag=True, help="Keep service state after stopping.")
def stop_cmd(service: str | None, grace: float, force: bool, keep_state: bool) -> None:
    # [FUNCTION DOC]
    # Click entry point. Options: --service, --grace (10s), --force (SIGKILL now),
    # --keep-state. Exits via SystemExit with do_stop's return code.
    raise SystemExit(do_stop(grace, force, service=service, keep_state=keep_state) or 0)


def do_stop(
    grace: float,
    force: bool,
    *,
    service: str | None = None,
    keep_state: bool = False,
) -> int:
    # [FUNCTION DOC]
    # Resolves the service; prints "not running" and returns 0 if no state file. If
    # state fails to load, logs a warning and removes service and model state unless
    # keep_state. Otherwise calls terminate_runtime_state (grace, force), removes both
    # state files unless keep_state, prints "stopped" and returns 0.
    # KNOWN ISSUE: the stale-state warning says "removing" even when keep_state is set.
    service_name = inf_lifecycle.resolve_service_name(service)
    if not inf_lifecycle.state_path(service_name).exists():
        click.echo(f"service {service_name!r} not running")
        return 0

    try:
        state = inf_lifecycle.load_state(service_name)
    except Exception:
        logger.warning("stale state; removing")
        if not keep_state:
            ServiceState.remove(service_name)
            ModelState.remove(service_name)
        return 0

    terminate_runtime_state(state, grace_s=grace, force=force)

    if not keep_state:
        ServiceState.remove(service_name)
        ModelState.remove(service_name)
    click.echo(f"service {service_name!r} stopped")
    return 0
