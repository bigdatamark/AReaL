# [MODULE DOC]
# Status table helper: concurrently probes components' /health endpoints and renders
# caller-defined columns as an aligned text table or JSON rows.
# SPDX-License-Identifier: Apache-2.0

"""Status drill-down table with parallel /health probing.

A ``StatusReporter`` is constructed with the components to inspect plus
a list of ``ColumnSpec`` describing the columns to render. It probes
every component's ``/health`` endpoint concurrently and produces either
a text table or a JSON payload.

Both the column set and the per-column value extractors are caller-
supplied — subcommand CLIs decide what to show. Scaffold provides only
the orchestration (parallel probe + table rendering).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import click

from areal.v2.cli.state import SupportsComponentProbe


@dataclass
class ColumnSpec:
    # [CLASS DOC]
    # One table column: header text and a value function mapping (label, handle, alive)
    # to the cell string ("-" conventionally means not applicable).
    """One column in the status table.

    ``value`` receives the row's ``(label, handle, alive)`` tuple and
    returns the rendered string. Returning ``"-"`` for "not applicable"
    is the conventional placeholder.
    """

    header: str
    value: Callable[[str, SupportsComponentProbe, bool], str]


class StatusReporter:
    # [CLASS DOC]
    # Holds the component list, column specs, per-probe timeout (3s) and max probe
    # threads (8).
    def __init__(
        self,
        components: list[tuple[str, SupportsComponentProbe]],
        columns: list[ColumnSpec],
        *,
        probe_timeout: float = 3.0,
        max_workers: int = 8,
    ) -> None:
        # [FUNCTION DOC]
        # Copies components and columns into lists and stores probe settings.
        self.components = list(components)
        self.columns = list(columns)
        self.probe_timeout = probe_timeout
        self.max_workers = max_workers

    def probe_all(self) -> list[bool]:
        # [FUNCTION DOC]
        # Probes every component's <addr>/health in a thread pool (size min(max_workers,
        # count), at least 1). A component is alive if the response status is below 500;
        # empty addr or network/timeout errors mean not alive. Returns bools aligned
        # with components ([] if none).
        # KNOWN ISSUE: urlopen raises HTTPError for 4xx too, so 4xx responses count as
        # not alive, contrary to the "non-5xx counts as alive" docstring.
        """Probe every component's ``/health`` endpoint concurrently.

        Returns a list of bools aligned with ``self.components``. A
        non-5xx response within ``probe_timeout`` counts as alive; any
        network error or 5xx counts as not alive.
        """

        if not self.components:
            return []
        addrs = [handle.addr or "" for _, handle in self.components]

        def probe(addr: str) -> bool:
            # [FUNCTION DOC]
            # HELPER: Returns the health result for one address (see above).
            if not addr:
                return False
            try:
                with urllib.request.urlopen(
                    f"{addr.rstrip('/')}/health", timeout=self.probe_timeout
                ) as resp:
                    return resp.status < 500
            except (
                urllib.error.URLError,
                ConnectionError,
                TimeoutError,
                OSError,
            ):
                return False

        with ThreadPoolExecutor(
            max_workers=max(1, min(self.max_workers, len(addrs)))
        ) as pool:
            return list(pool.map(probe, addrs))

    def render_rows(self, alive_flags: list[bool] | None = None) -> list[list[str]]:
        # [FUNCTION DOC]
        # Returns one list of cell strings per component by applying each column's value
        # function; probes first if alive_flags is None. Raises ValueError on length
        # mismatch.
        """Build per-row string lists using each column's value extractor."""

        if alive_flags is None:
            alive_flags = self.probe_all()
        rows: list[list[str]] = []
        for (label, handle), alive in zip(self.components, alive_flags, strict=True):
            rows.append([col.value(label, handle, alive) for col in self.columns])
        return rows

    def print_table(
        self,
        rows: list[list[str]] | None = None,
        *,
        header_line: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Echoes optional header_line plus a blank line, then a left-aligned table with
        # two-space gaps and widths from the longest cell per column. Renders rows if
        # not given.
        # KNOWN ISSUE: with an empty rows list the width computation calls max on a
        # single int and raises TypeError.
        """Print a left-aligned table with column widths derived from
        the longest cell in each column. Optional ``header_line`` is
        echoed before the table (e.g. ``"service: foo  backend: local"``).
        """

        if rows is None:
            rows = self.render_rows()
        if header_line:
            click.echo(header_line)
            click.echo()
        headers = [c.header for c in self.columns]
        widths = [
            max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        click.echo(fmt.format(*headers))
        for row in rows:
            click.echo(fmt.format(*row))

    def json_snapshot(self, alive_flags: list[bool] | None = None) -> list[dict]:
        # [FUNCTION DOC]
        # Returns a list of dicts (header -> cell value) per component; probes if
        # alive_flags is None.
        """Return the table content as a list of dicts (one per row)
        keyed by column header — convenient for ``--json`` output."""

        if alive_flags is None:
            alive_flags = self.probe_all()
        snapshot = []
        for (label, handle), alive in zip(self.components, alive_flags, strict=True):
            snapshot.append(
                {col.header: col.value(label, handle, alive) for col in self.columns}
            )
        return snapshot
