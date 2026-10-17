# [MODULE DOC]
# Generic service lifecycle helper shared by CLI groups: resolves service names, loads
# state through a state class, checks whether a service is running, and force-replaces
# an existing service slot.
# SPDX-License-Identifier: Apache-2.0

"""Service lifecycle helpers shared across subcommand CLIs.

A ``ServiceLifecycle`` bundles a CLI's namespace + state class + stop
command name, then exposes the three "is this service running"
predicates (``running_state`` / ``load_running_state`` /
``refuse_if_running``) plus ``force_replace_slot`` for ``run --force``.

Subcommand CLIs subclass to inject their state class and may override
``gateway_alive`` if they don't want the default
``state.gateway_alive()`` delegation.
"""

from __future__ import annotations

from pathlib import Path

import click

from areal.v2.cli.process import kill_pids
from areal.v2.cli.state import DEFAULT_SERVICE, NamespacedStateStore


class ServiceLifecycle:
    # [CLASS DOC]
    # Configured by namespace, state_class (must provide classmethod load(service) and
    # instance gateway_alive/components), stop_command (shown in error messages) and
    # default_service; owns a NamespacedStateStore.
    namespace: str = ""
    state_class: type = type(None)
    stop_command: str = ""
    default_service: str = DEFAULT_SERVICE

    def __init__(
        self,
        *,
        namespace: str | None = None,
        state_class: type | None = None,
        stop_command: str | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Applies constructor overrides of namespace/state_class/stop_command and raises
        # ValueError if namespace or stop_command is empty. Creates the store.
        if namespace is not None:
            self.namespace = namespace
        if state_class is not None:
            self.state_class = state_class
        if stop_command is not None:
            self.stop_command = stop_command
        if not self.namespace:
            raise ValueError("ServiceLifecycle requires a namespace")
        if not self.stop_command:
            raise ValueError("ServiceLifecycle requires a stop_command")
        self.store = NamespacedStateStore(self.namespace)

    # ------------------------------------------------------------------
    # Hooks subclasses may override

    def gateway_alive(self, state) -> bool:
        # [FUNCTION DOC]
        # Hook: returns state.gateway_alive().
        """Default delegates to ``state.gateway_alive()`` (the
        ServiceStateBase contract). Subclasses may override if their
        state object can't or shouldn't implement that method itself."""

        return state.gateway_alive()

    def state_path(self, service: str) -> Path:
        # [FUNCTION DOC]
        # Returns the service-state file path for service.
        return self.store.service_state_path(service)

    def resolve_service_name(self, explicit: str | None) -> str:
        # [FUNCTION DOC]
        # Returns the explicit name, else the store's current service, else
        # default_service (as implemented by the store).
        return self.store.resolve_service_name(explicit, fallback=self.default_service)

    def list_services(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns the names of services with state files in the namespace.
        return self.store.list_service_names()

    def load_state(self, service: str):
        # [FUNCTION DOC]
        # Loads state via state_class.load(service); errors propagate.
        return self.state_class.load(service)

    # ------------------------------------------------------------------
    # Public API

    def running_state(self, service: str | None = None):
        # [FUNCTION DOC]
        # Returns the loaded state if its file exists, loads without error and the
        # gateway is alive; otherwise None. Never raises for those cases.
        """Return the loaded state iff it exists, parses, and reports
        the gateway alive. Returns None otherwise — never raises."""

        name = self.resolve_service_name(service)
        if not self.state_path(name).exists():
            return None
        try:
            state = self.load_state(name)
        except Exception:
            return None
        if not self.gateway_alive(state):
            return None
        return state

    def load_running_state(self, service: str | None = None):
        # [FUNCTION DOC]
        # Same checks as running_state but raises click.ClickException describing which
        # check failed (not running, failed to load, gateway not alive). Returns the
        # state.
        """Like ``running_state`` but raises ``ClickException`` on
        every failure — for command bodies that need a state to proceed."""

        name = self.resolve_service_name(service)
        if not self.state_path(name).exists():
            raise click.ClickException(f"service {name!r} is not running")
        try:
            state = self.load_state(name)
        except Exception as exc:
            raise click.ClickException(f"failed to load state: {exc}") from exc
        if not self.gateway_alive(state):
            raise click.ClickException(f"service {name!r} gateway is not alive")
        return state

    def refuse_if_running(self, service: str | None = None) -> None:
        # [FUNCTION DOC]
        # Raises ClickException suggesting the stop command if the service is running;
        # else no-op.
        """Used by ``run`` to refuse double-start. No-op if no running
        service exists; raises ``ClickException`` if one does."""

        state = self.running_state(service)
        if state is None:
            return
        raise click.ClickException(
            f"service {state.service!r} already running. "
            f"Run `{self.stop_command} --service {state.service}` first."
        )

    def force_replace_slot(self, service: str, *, grace_s: float = 5.0) -> None:
        # [FUNCTION DOC]
        # If a state file exists: collects component pids from the loaded state, or
        # falls back to the store's raw-JSON pid recovery (empty on failure), kills them
        # with grace_s, deletes the state file and clears the current-service pointer.
        # No-op if no state file.
        """``run --force`` path: tear down any existing children for
        ``service`` (loading the state for an orderly kill, falling back
        to the raw-JSON PID walk if the state file is corrupted), then
        unlink the state file so the next spawn starts clean.

        Caller is still responsible for the subsequent fresh spawn.
        """

        path = self.state_path(service)
        if not path.exists():
            return
        pids: list[int] = []
        try:
            state = self.load_state(service)
            pids = self._collect_pids(state)
        except Exception:
            try:
                pids = self.store.recover_pids_from_raw_state(service)
            except Exception:
                pids = []
        if pids:
            kill_pids(pids, grace_s=grace_s)
        path.unlink(missing_ok=True)
        self.store.clear_current_service(service)

    def _collect_pids(self, state) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Returns positive pids of every component yielded by
        # state.components().
        """Default implementation pulls ``.pid`` off every component
        ``state.components()`` yields. Subclasses with extra state files
        (e.g. inf's ``models/<svc>.json``) may override to extend the
        list."""

        return [pid for _, h in state.components() if (pid := h.pid) > 0]
