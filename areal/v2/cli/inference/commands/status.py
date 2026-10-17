# [MODULE DOC]
# Click command "areal inf status": per-component health drill-down for one inference
# service, as a table or JSON.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict

import click

from areal.v2.cli.inference.common import (
    format_gpu_count,
    format_placement,
    format_ref,
)
from areal.v2.cli.inference.lifecycle import inf_lifecycle
from areal.v2.cli.inference.scheduler import TaskHandle
from areal.v2.cli.inference.state import RuntimeState
from areal.v2.cli.state import SupportsComponentProbe
from areal.v2.cli.status import ColumnSpec, StatusReporter


@click.command(
    name="status",
    help="Show inference service status — per-component drill-down for one service.",
)
@click.option("--service", default=None, help="Target service instance.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def status_cmd(service: str | None, as_json: bool) -> None:
    # [FUNCTION DOC]
    # Click entry point. Options: --service, --json. Exits via SystemExit with
    # do_status's code.
    raise SystemExit(do_status(as_json, service=service) or 0)


def do_status(as_json: bool, *, service: str | None = None) -> int:
    # [FUNCTION DOC]
    # Resolves the service; if no state file, prints a not-running message (or JSON with
    # running False) and returns 0. Loading failures raise ClickException. Otherwise
    # probes all components via StatusReporter and prints either the JSON snapshot or a
    # table with a header (service, backend, gateway addr, model count, running yes/no).
    # Returns 0.
    service_name = inf_lifecycle.resolve_service_name(service)
    if not inf_lifecycle.state_path(service_name).exists():
        if as_json:
            click.echo(
                json.dumps({"service": service_name, "running": False}, indent=2)
            )
        else:
            click.echo(f"service {service_name!r} not running")
        return 0

    try:
        state = inf_lifecycle.load_state(service_name)
    except Exception as exc:
        raise click.ClickException(f"failed to load state: {exc}") from exc

    components = list(state.components())
    reporter = StatusReporter(components, _columns(state.backend))
    alive = reporter.probe_all()

    if as_json:
        click.echo(json.dumps(_json_snapshot(state, reporter, alive), indent=2))
        return 0

    gateway_addr = state.gateway_handle.addr or "-"
    reporter.print_table(
        reporter.render_rows(alive),
        header_line=(
            f"service: {state.service}   "
            f"backend: {state.backend}   "
            f"gateway: {gateway_addr}   "
            f"models: {len(state.models)}   "
            f"running: {'yes' if state.gateway_alive() else 'no'}"
        ),
    )
    return 0


def _columns(backend: str) -> list[ColumnSpec]:
    # [FUNCTION DOC]
    # HELPER: Builds column specs COMPONENT, PLACEMENT, GPUS, ADDR, REF, ALIVE;
    # placement and ref text depend on the scheduler backend. Returns list of
    # ColumnSpec.
    def _component(label: str, _: SupportsComponentProbe, __: bool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the component label.
        return label

    def _placement(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        # SupportsComponentProbe doesn't promise the TaskHandle surface,
        # but in practice every inf component handle IS a TaskHandle.
        # [FUNCTION DOC]
        # HELPER: Returns format_placement(backend, handle) or "-" if empty.
        return format_placement(backend, handle) or "-"  # type: ignore[arg-type]

    def _gpus(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the GPU count text for the handle.
        return format_gpu_count(handle)  # type: ignore[arg-type]

    def _addr(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the handle address or "-".
        return handle.addr or "-"

    def _ref(_: str, handle: SupportsComponentProbe, __: bool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the backend-specific reference text (pid/pod/job).
        return format_ref(backend, handle)  # type: ignore[arg-type]

    def _alive(_: str, __: SupportsComponentProbe, alive: bool) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns "yes" if alive else "no".
        return "yes" if alive else "no"

    return [
        ColumnSpec("COMPONENT", _component),
        ColumnSpec("PLACEMENT", _placement),
        ColumnSpec("GPUS", _gpus),
        ColumnSpec("ADDR", _addr),
        ColumnSpec("REF", _ref),
        ColumnSpec("ALIVE", _alive),
    ]


def _json_snapshot(
    state: RuntimeState, reporter: StatusReporter, alive: list[bool]
) -> dict:
    # [FUNCTION DOC]
    # HELPER: Builds the JSON status dict: service, backend, running, gateway/router
    # handles as dicts, started_at, models (as dicts), and per-component rows from the
    # reporter snapshot each augmented with its label. Raises ValueError if lengths of
    # the snapshot and component list differ (strict zip).
    handle_pairs: list[tuple[str, TaskHandle]] = list(state.components())  # type: ignore[arg-type]
    rows = [
        {**snap, "label": label}
        for snap, (label, _) in zip(
            reporter.json_snapshot(alive), handle_pairs, strict=True
        )
    ]
    return {
        "service": state.service,
        "backend": state.backend,
        "running": state.gateway_alive(),
        "gateway_handle": asdict(state.gateway_handle),
        "router_handle": asdict(state.router_handle),
        "started_at": state.started_at,
        "models": {name: asdict(entry) for name, entry in state.models.items()},
        "components": rows,
    }
