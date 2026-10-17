# [MODULE DOC]
# Factory for the generic "<group> logs" click command: resolves a service, locates a
# component's log file in the service's log directory and tails it.
# SPDX-License-Identifier: Apache-2.0

"""``<group> logs`` verb factory.

Every service-style CLI has a ``logs`` command that does the same
thing: resolve the service, find the log directory, tail one component
file. This class packages that pattern. Subcommand CLIs do::

    from areal.v2.cli.commands.logs import LogsCommand
    from areal.v2.cli.inference.lifecycle import inf_lifecycle

    logs_cmd = LogsCommand(lifecycle=inf_lifecycle).build()
    inf.add_command(logs_cmd)
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from areal.v2.cli.lifecycle import ServiceLifecycle


class LogsCommand:
    # [CLASS DOC]
    # Builds and executes a "logs" command bound to a ServiceLifecycle. Class
    # attributes: default_component ("gateway") and default_lines (200) used as option
    # defaults; subclasses may override them and log_target.
    default_component: str = "gateway"
    default_lines: int = 200

    def __init__(self, *, lifecycle: ServiceLifecycle) -> None:
        # [FUNCTION DOC]
        # Stores the lifecycle (provides service-name resolution and the state store).
        self.lifecycle = lifecycle

    def build(self) -> click.Command:
        @click.command(name="logs", help="Tail a service log file.")
        @click.option("--service", default=None)
        @click.option(
            "--component",
            default=self.default_component,
            show_default=True,
            help="Component log to tail (gateway / router / worker-N / etc.).",
        )
        @click.option("-f", "--follow", is_flag=True, help="Stream appended lines.")
        @click.option(
            "-n",
            "--lines",
            type=int,
            default=self.default_lines,
            show_default=True,
            help="Number of trailing lines to show before optional follow.",
        )
        # [FUNCTION DOC]
        # Creates and returns a click command named "logs" with options --service,
        # --component (default default_component), -f/--follow and -n/--lines (default
        # default_lines); its callback delegates to execute.
        def logs_cmd(
            service: str | None, component: str, follow: bool, lines: int
        ) -> None:
            # [FUNCTION DOC]
            # Click callback forwarding service, component, follow and lines to
            # self.execute.
            self.execute(service, component, follow, lines)

        return logs_cmd

    def execute(
        self,
        service: str | None,
        component: str,
        follow: bool,
        lines: int,
    ) -> None:
        # [FUNCTION DOC]
        # Resolves the service name, computes the log path via log_target and, if it
        # exists, replaces the current process with tail. If missing, raises
        # click.ClickException listing the available component names (stems of *.log
        # files) or saying no logs exist.
        name = self.lifecycle.resolve_service_name(service)
        target = self.log_target(name, component)
        if not target.exists():
            available = sorted(p.stem for p in target.parent.glob("*.log"))
            if not available:
                raise click.ClickException(f"no logs found under {target.parent}")
            raise click.ClickException(
                f"no log named {component!r}; available: {', '.join(available)}"
            )
        self._exec_tail(target, lines=lines, follow=follow)

    def log_target(self, service: str, component: str) -> Path:
        # [FUNCTION DOC]
        # Returns <logs_dir(service)>/<component>.log from the lifecycle's store.
        # Overridable hook for component aliases.
        """Resolve the log file path. Subclasses may override to add a
        component name alias / translation layer (e.g. accepting
        ``qwen/0/worker`` as shorthand for ``qwen-worker-0``)."""

        return self.lifecycle.store.logs_dir(service) / f"{component}.log"

    def _exec_tail(self, target: Path, *, lines: int, follow: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces the current process (os.execvp, never returns) with "tail
        # -n<lines>", adding -F when follow is set, on target.
        cmd = ["tail", f"-n{lines}"]
        if follow:
            cmd.append("-F")
        cmd.append(str(target))
        os.execvp(cmd[0], cmd)
