# [MODULE DOC]
# Click command "areal inf models": lists the models registered in a running inference
# service as JSON or a table.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict

import click

from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.utils import json_or_table


@click.command(name="models", help="List registered models.")
@click.option("--service", default=None, help="Target service instance.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def models_cmd(service: str | None, as_json: bool) -> None:
    # [FUNCTION DOC]
    # Click entry point. Options: --service (default current), --json. Exits via
    # SystemExit with do_models's return code.
    raise SystemExit(do_models(as_json, service=service) or 0)


def do_models(as_json: bool, *, service: str | None = None) -> int:
    # [FUNCTION DOC]
    # Gets the running state (None if not running, in which case prints "[]" for JSON or
    # "service not running"). Otherwise builds one dict per model (name plus the
    # dataclass fields of its entry) and prints it as JSON or via _print_models_table.
    # Returns 0.
    state = inf_lifecycle.running_state(service)
    if state is None:
        click.echo("[]" if as_json else "service not running")
        return 0

    payload = [{"name": name, **asdict(entry)} for name, entry in state.models.items()]
    json_or_table(payload, as_json=as_json, table_renderer=_print_models_table)
    return 0


def _print_models_table(rows: list[dict]) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an aligned NAME / BACKEND / WORKERS table (workers = replica
    # count), or "no models registered" when rows is empty. Column widths fit the
    # longest cell.
    if not rows:
        click.echo("no models registered")
        return
    cols = ("NAME", "BACKEND", "WORKERS")
    table = [(row["name"], row["backend"], str(len(row["replicas"]))) for row in rows]
    widths = [max(len(r[i]) for r in (cols, *table)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*cols))
    for row in table:
        click.echo(fmt.format(*row))
