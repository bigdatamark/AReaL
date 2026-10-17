# [MODULE DOC]
# Persisted on-disk state of an agent service (gateway, router and worker/proxy pairs),
# stored as JSON in the "agent" namespace of the CLI state store.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from areal.v2.cli.process import pid_alive
from areal.v2.cli.state import (
    NamespacedStateStore,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
)

AGENT_NAMESPACE = "agent"

# Module-level store — every save/load/remove routes paths through it,
# tests with ``AREAL_HOME`` work because the store re-resolves on each
# call.
store = NamespacedStateStore(AGENT_NAMESPACE)


@dataclass
class ProcessState:
    # [CLASS DOC]
    # Record of one spawned component: component name, pid, base url and log_file path.
    component: str
    pid: int
    url: str
    log_file: str

    @property
    def addr(self) -> str:
        # [FUNCTION DOC]
        # Property returning the URL as the component's address (for status probing).
        return self.url


@dataclass
class PairState:
    # [CLASS DOC]
    # Record of one worker/data-proxy pair: its index and the two ProcessState entries.
    index: int
    worker: ProcessState
    data_proxy: ProcessState


@dataclass
class ServiceState(ServiceStateBase):
    # [CLASS DOC]
    # Full agent service state: service name, launch_mode, agent import path, admin API
    # key, gateway and router processes, list of pairs, session/health/drain timing
    # settings, and started_at timestamp (defaults to now). Implements the
    # ServiceStateBase contract.
    service: str
    launch_mode: str
    agent: str
    admin_api_key: str
    gateway: ProcessState
    router: ProcessState
    pairs: list[PairState]
    session_timeout: float = 1800.0
    health_poll_interval: float = 5.0
    drain_timeout: float = 30.0
    started_at: float = field(default_factory=time.time)

    def save(self) -> None:
        # [FUNCTION DOC]
        # Atomically writes the dataclass as JSON to the service's state path and marks
        # this service as the current one in the store. Side effect: filesystem writes.
        atomic_write_json(store.service_state_path(self.service), asdict(self))
        store.set_current_service(self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        # [FUNCTION DOC]
        # Reads and parses the JSON state file for service, rebuilding nested
        # ProcessState and PairState objects. Renames a legacy created_at key to
        # started_at, drops leftover created_at and legacy
        # inf_addr/inf_api_key/inf_model keys. Returns a ServiceState. Raises
        # OSError/JSON/TypeError errors on missing or malformed files.
        # KNOWN ISSUE: the inline comment says "forward compatibility" but handling old
        # files is backward compatibility.
        with open(store.service_state_path(service)) as f:
            raw = json.load(f)
        raw["gateway"] = ProcessState(**raw["gateway"])
        raw["router"] = ProcessState(**raw["router"])
        raw["pairs"] = [
            PairState(
                index=pair["index"],
                worker=ProcessState(**pair["worker"]),
                data_proxy=ProcessState(**pair["data_proxy"]),
            )
            for pair in raw.get("pairs", [])
        ]
        # ``created_at`` is the legacy field name; accept it for forward
        # compatibility with any state files written before the rename.
        if "created_at" in raw and "started_at" not in raw:
            raw["started_at"] = raw.pop("created_at")
        raw.pop("created_at", None)
        for legacy in ("inf_addr", "inf_api_key", "inf_model"):
            raw.pop(legacy, None)
        return cls(**raw)

    @classmethod
    def remove(cls, service: str) -> None:
        # [FUNCTION DOC]
        # Deletes the service's state file if present and clears the current-service
        # pointer if it names this service.
        path = store.service_state_path(service)
        if path.exists():
            path.unlink()
        store.clear_current_service(service)

    def gateway_alive(self) -> bool:
        # The CLI treats "running" as "gateway process still alive". The
        # gateway is the entry point — without it, the rest of the stack
        # is unreachable even if some children survived.
        # [FUNCTION DOC]
        # Returns True if the gateway pid is alive; the service counts as running only
        # then.
        return pid_alive(self.gateway.pid)

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        # [FUNCTION DOC]
        # Yields (label, handle) pairs: gateway, router, then for each pair the worker
        # and proxy using their component names as labels.
        yield "gateway", self.gateway
        yield "router", self.router
        for pair in self.pairs:
            yield pair.worker.component, pair.worker
            yield pair.data_proxy.component, pair.data_proxy
