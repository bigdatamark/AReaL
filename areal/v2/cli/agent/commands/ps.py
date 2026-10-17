# [MODULE DOC]
# Implements `areal agent ps`: lists locally known agent services from the lifecycle
# state directory, as JSON or a text table.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.agent.lifecycle import agent_lifecycle
from areal.v2.cli.process import pid_alive
from areal.v2.cli.utils import json_or_table


@click.command(name="ps", help="List locally known agent services.")
@click.option("--json", "as_json", is_flag=True)
@click.option("--all", "include_all", is_flag=True, help="Include stale services.")
def ps_cmd(as_json: bool, include_all: bool) -> None:
    # [FUNCTION DOC]
    # Click command wrapper: calls do_ps with the --json and --all flags and exits the
    # process with its return code (SystemExit).
    raise SystemExit(do_ps(as_json=as_json, include_all=include_all) or 0)


def do_ps(*, as_json: bool, include_all: bool) -> int:
    # [FUNCTION DOC]
    # For each service known to the agent lifecycle, loads its state; unreadable state
    # counts as "stale" (listed only with include_all). A service is "running" if its
    # gateway PID is alive; stale ones are listed only with include_all. Rows hold
    # service, status, gateway_url and agent. Prints via JSON or _print_table. Returns 0.
    rows: list[dict] = []
    for service in agent_lifecycle.list_services():
        try:
            state = agent_lifecycle.load_state(service)
        except Exception:
            if include_all:
                rows.append({"service": service, "status": "stale"})
            continue
        running = pid_alive(state.gateway.pid)
        if running or include_all:
            rows.append(
                {
                    "service": service,
                    "status": "running" if running else "stale",
                    "gateway_url": state.gateway.url,
                    "agent": state.agent,
                }
            )

    json_or_table(rows, as_json=as_json, table_renderer=_print_table)
    return 0


def _print_table(rows: list[dict]) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints rows as a left-aligned, two-space-separated table with columns
    # SERVICE, STATUS, GATEWAY, AGENT (widths fit the longest value), or
    # "no agent services" if empty.
    if not rows:
        click.echo("no agent services")
        return
    cols = ("SERVICE", "STATUS", "GATEWAY", "AGENT")
    table = [
        (
            row["service"],
            row["status"],
            row.get("gateway_url", ""),
            row.get("agent", ""),
        )
        for row in rows
    ]
    widths = [max(len(str(r[i])) for r in (cols, *table)) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
