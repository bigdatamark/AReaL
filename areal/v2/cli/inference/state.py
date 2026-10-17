# [MODULE DOC]
# Persisted state of inference services. Each service has a service-state JSON (gateway
# and router handles) and a model-state JSON (model registry with per-replica handles),
# combined into RuntimeState for the CLI verbs. Also defines the inf-specific state
# store.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.v2.cli.inference.scheduler import TaskHandle
from areal.v2.cli.process import pid_alive
from areal.v2.cli.state import (
    DEFAULT_SERVICE,
    NamespacedStateStore,
    ServiceStateBase,
    SupportsComponentProbe,
    atomic_write_json,
)
from areal.v2.cli.utils import file_lock

INF_NAMESPACE = "inf"


class InferenceStateStore(NamespacedStateStore):
    # [CLASS DOC]
    # NamespacedStateStore for the "inf" namespace adding the models/ directory, per-
    # service model-state file and lock file, and a PID recovery that walks both files.
    """Inference-specific extension of the scaffold ``NamespacedStateStore``.

    The inference service splits its state across two files per service —
    the service-state (gateway / router handles, inherited from
    scaffold) and the model-state (model registry, locked for
    register / deregister). This class adds the model-state file paths
    + lock + overrides ``recover_pids_from_raw_state`` to walk both
    files when ``run --force`` falls back to raw-JSON PID recovery.
    """

    def __init__(self, namespace: str = INF_NAMESPACE) -> None:
        # [FUNCTION DOC]
        # Initializes the base store with the namespace (default "inf").
        super().__init__(namespace)

    # --- model-state file layout ----------------------------------------

    def models_dir(self) -> Path:
        # [FUNCTION DOC]
        # Returns <root>/models, creating it if needed (filesystem side effect).
        """``$AREAL_HOME/<namespace>/models`` — created lazily."""

        d = self.root() / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def models_state_path(self, service: str) -> Path:
        # [FUNCTION DOC]
        # Returns models/<service>.json.
        return self.models_dir() / f"{service}.json"

    def models_lock_path(self, service: str) -> Path:
        # [FUNCTION DOC]
        # Returns models/<service>.lock.
        return self.models_dir() / f"{service}.lock"

    # --- locking --------------------------------------------------------

    @contextmanager
    def lock_model_state(self, service: str) -> Iterator[None]:
        # [FUNCTION DOC]
        # Context manager holding an exclusive file lock on the service's model lock
        # file for the duration of the block, serializing registry read-modify-write
        # across processes.
        """Per-service flock on the model-registry file. Held across
        register / deregister so concurrent CLI calls cannot race the
        JSON read-modify-write."""

        with file_lock(self.models_lock_path(service)):
            yield

    # --- best-effort recovery (override) --------------------------------

    def recover_pids_from_raw_state(self, service: str) -> list[int]:
        # [FUNCTION DOC]
        # Best-effort: parses the service-state and model-state JSON files (skipping
        # missing ones) and recursively collects positive int values (or lists of them)
        # under "pid"/"pids" keys anywhere in the structure. Returns de-duplicated pids
        # in first-seen order. JSON/IO errors propagate.
        """Walk BOTH service-state and model-state JSON for ``pid`` /
        ``pids`` keys. Inference overrides scaffold's single-file walker
        because the second (model-state) file is invisible to it."""

        pids: list[int] = []
        pid_keys = {"pid", "pids"}

        def add(value) -> None:
            # [FUNCTION DOC]
            # HELPER: Appends value if a positive int, or recurses into a list.
            if isinstance(value, int) and value > 0:
                pids.append(value)
            elif isinstance(value, list):
                for item in value:
                    add(item)

        def walk(value) -> None:
            # [FUNCTION DOC]
            # HELPER: Recursively visits dicts/lists, sending values under pid keys to
            # add.
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in pid_keys:
                        add(item)
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for path in (self.service_state_path(service), self.models_state_path(service)):
            if not path.exists():
                continue
            with open(path) as f:
                walk(json.load(f))

        seen: set[int] = set()
        unique: list[int] = []
        for pid in pids:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique


# Module-level singleton — every dataclass / verb routes path resolution
# through this. Tests use ``AREAL_HOME`` to isolate; the store re-resolves
# on every call so the env override takes effect without rebuilding.
store = InferenceStateStore()


# ``DEFAULT_SERVICE`` is re-exported above so existing subcommand imports
# (`from ...inference.state import DEFAULT_SERVICE`) keep working.
__all__ = [
    "DEFAULT_SERVICE",
    "INF_NAMESPACE",
    "InferenceStateStore",
    "ModelEntry",
    "ModelReplica",
    "ModelState",
    "RuntimeState",
    "ServiceState",
    "store",
]


def _handle_from_dict(raw: dict) -> TaskHandle:
    # [FUNCTION DOC]
    # HELPER: Builds a TaskHandle from a JSON dict (host required; ports, gpu_devices,
    # ref optional, copied).
    return TaskHandle(
        host=raw["host"],
        ports=list(raw.get("ports", [])),
        gpu_devices=list(raw.get("gpu_devices", [])),
        ref=dict(raw.get("ref", {})),
    )


@dataclass
class ModelReplica:
    # [CLASS DOC]
    # One data-parallel replica: data_proxy handle (front) and worker handle (behind).
    """One DP replica: data-proxy in front, worker behind. Field order
    matches the request-flow direction used by stop / status output."""

    data_proxy: TaskHandle
    worker: TaskHandle


@dataclass
class ModelEntry:
    # [CLASS DOC]
    # A registered model: backend spec string and list of replicas.
    backend: str = ""
    replicas: list[ModelReplica] = field(default_factory=list)

    def all_workers(self) -> list[TaskHandle]:
        # [FUNCTION DOC]
        # Returns the worker handles of all replicas.
        return [r.worker for r in self.replicas]

    def all_data_proxies(self) -> list[TaskHandle]:
        # [FUNCTION DOC]
        # Returns the data-proxy handles of all replicas.
        return [r.data_proxy for r in self.replicas]

    def gpu_devices(self) -> list[int]:
        # [FUNCTION DOC]
        # Returns all GPU ids used by this model's workers (flattened list).
        return [g for w in self.all_workers() for g in w.gpu_devices]

    @classmethod
    def from_dict(cls, raw: dict) -> ModelEntry:
        # [FUNCTION DOC]
        # Builds a ModelEntry from its JSON dict form (backend defaults to "", replicas
        # to []).
        replicas = [
            ModelReplica(
                data_proxy=_handle_from_dict(r["data_proxy"]),
                worker=_handle_from_dict(r["worker"]),
            )
            for r in raw.get("replicas", [])
        ]
        return cls(backend=raw.get("backend", ""), replicas=replicas)


@dataclass
class ServiceState:
    # [CLASS DOC]
    # Service-level state: service, scheduler backend, gateway and router handles, admin
    # key, started_at and launch_mode (default "detached").
    service: str
    backend: str
    gateway_handle: TaskHandle
    router_handle: TaskHandle
    admin_api_key: str
    started_at: float
    launch_mode: str = "detached"

    @property
    def gateway_pid(self) -> int:
        # [FUNCTION DOC]
        # Property: gateway handle pid.
        return self.gateway_handle.pid

    @property
    def gateway_url(self) -> str:
        # [FUNCTION DOC]
        # Property: gateway handle address.
        return self.gateway_handle.addr

    @property
    def router_pid(self) -> int:
        # [FUNCTION DOC]
        # Property: router handle pid.
        return self.router_handle.pid

    @property
    def router_url(self) -> str:
        # [FUNCTION DOC]
        # Property: router handle address.
        return self.router_handle.addr

    def save(self) -> None:
        # [FUNCTION DOC]
        # Atomically writes the state as JSON and marks the service as current.
        atomic_write_json(store.service_state_path(self.service), asdict(self))
        store.set_current_service(self.service)

    @classmethod
    def load(cls, service: str) -> ServiceState:
        # [FUNCTION DOC]
        # Reads the service-state JSON for service; raises FileNotFoundError if absent,
        # KeyError for missing required fields. Returns a ServiceState with handles
        # rebuilt.
        p = store.service_state_path(service)
        if not p.exists():
            raise FileNotFoundError(f"No service state at {p}")
        with open(p) as f:
            raw = json.load(f)
        return cls(
            service=raw.get("service", service),
            backend=raw["backend"],
            gateway_handle=_handle_from_dict(raw["gateway_handle"]),
            router_handle=_handle_from_dict(raw["router_handle"]),
            admin_api_key=raw["admin_api_key"],
            started_at=float(raw["started_at"]),
            launch_mode=raw.get("launch_mode", "detached"),
        )

    @classmethod
    def remove(cls, service: str) -> None:
        # [FUNCTION DOC]
        # Deletes the service-state file if present and clears the current-service
        # pointer if it refers to this service.
        p = store.service_state_path(service)
        if p.exists():
            p.unlink()
        store.clear_current_service(service)


@dataclass
class ModelState:
    # [CLASS DOC]
    # Model registry for one service: mapping model name -> ModelEntry.
    service: str
    models: dict[str, ModelEntry] = field(default_factory=dict)

    def save(self) -> None:
        # [FUNCTION DOC]
        # Atomically writes the registry JSON to models/<service>.json.
        atomic_write_json(store.models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str) -> ModelState:
        # [FUNCTION DOC]
        # Loads the registry JSON; returns an empty ModelState if the file is missing.
        # Rebuilds ModelEntry objects.
        p = store.models_state_path(service)
        if not p.exists():
            return cls(service=service)
        with open(p) as f:
            raw = json.load(f)
        models = {
            name: ModelEntry.from_dict(entry)
            for name, entry in (raw.pop("models", None) or {}).items()
        }
        return cls(service=raw.get("service", service), models=models)

    @classmethod
    def remove(cls, service: str) -> None:
        # [FUNCTION DOC]
        # Deletes the registry file if present.
        p = store.models_state_path(service)
        if p.exists():
            p.unlink()

    def all_workers(self) -> list[TaskHandle]:
        # [FUNCTION DOC]
        # Returns worker handles across all models.
        return [h for entry in self.models.values() for h in entry.all_workers()]

    def all_data_proxies(self) -> list[TaskHandle]:
        # [FUNCTION DOC]
        # Returns data-proxy handles across all models.
        return [h for entry in self.models.values() for h in entry.all_data_proxies()]

    def occupied_gpus(self) -> set[int]:
        # [FUNCTION DOC]
        # Returns the set of GPU ids used by any registered model.
        return {g for entry in self.models.values() for g in entry.gpu_devices()}


@dataclass
class RuntimeState(ServiceStateBase):
    # [CLASS DOC]
    # Composite of ServiceState and ModelState implementing ServiceStateBase; read-only
    # properties delegate to the service state, models to the model state.
    """Composite of ServiceState (gateway/router) and ModelState (model
    registry). The single object the CLI verbs operate on; the lifecycle
    contract is implemented against ``RuntimeState`` so scaffold helpers
    enumerate every component in one pass."""

    service_state: ServiceState
    model_state: ModelState

    @property
    def service(self) -> str:
        # [FUNCTION DOC]
        # Property: service name.
        return self.service_state.service

    @property
    def backend(self) -> str:
        # [FUNCTION DOC]
        # Property: scheduler backend.
        return self.service_state.backend

    @property
    def gateway_handle(self) -> TaskHandle:
        # [FUNCTION DOC]
        # Property: gateway TaskHandle.
        return self.service_state.gateway_handle

    @property
    def router_handle(self) -> TaskHandle:
        # [FUNCTION DOC]
        # Property: router TaskHandle.
        return self.service_state.router_handle

    @property
    def gateway_pid(self) -> int:
        # [FUNCTION DOC]
        # Property: gateway pid.
        return self.service_state.gateway_pid

    @property
    def gateway_url(self) -> str:
        # [FUNCTION DOC]
        # Property: gateway URL.
        return self.service_state.gateway_url

    @property
    def router_pid(self) -> int:
        # [FUNCTION DOC]
        # Property: router pid.
        return self.service_state.router_pid

    @property
    def router_url(self) -> str:
        # [FUNCTION DOC]
        # Property: router URL.
        return self.service_state.router_url

    @property
    def admin_api_key(self) -> str:
        # [FUNCTION DOC]
        # Property: admin API key.
        return self.service_state.admin_api_key

    @property
    def started_at(self) -> float:
        # [FUNCTION DOC]
        # Property: service start timestamp.
        return self.service_state.started_at

    @property
    def launch_mode(self) -> str:
        # [FUNCTION DOC]
        # Property: launch mode.
        return self.service_state.launch_mode

    @property
    def models(self) -> dict[str, ModelEntry]:
        # [FUNCTION DOC]
        # Property: the model registry dict (mutable, shared with model_state).
        return self.model_state.models

    def save(self) -> None:
        # [FUNCTION DOC]
        # Saves both the service state and the model state.
        self.service_state.save()
        self.model_state.save()

    @classmethod
    def load(cls, service: str) -> RuntimeState:
        # [FUNCTION DOC]
        # Loads the service state (raises if missing) and then the model state for its
        # service name.
        service_state = ServiceState.load(service)
        return cls(
            service_state=service_state,
            model_state=ModelState.load(service_state.service),
        )

    # --- ServiceStateBase contract ----------------------------------------

    def gateway_alive(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if the gateway pid is alive (always False for non-local handles
        # without pid).
        # Non-local backends will need their own liveness probe — PID-based
        # check returns False for them (no pid in handle.ref).
        return pid_alive(self.gateway_pid)

    def components(self) -> Iterable[tuple[str, SupportsComponentProbe]]:
        # [FUNCTION DOC]
        # Yields (label, handle): gateway, router, then for each model and replica index
        # i the "data_proxy[<model>/<i>]" and "worker[<model>/<i>]" handles.
        yield "gateway", self.gateway_handle
        yield "router", self.router_handle
        for name, entry in self.model_state.models.items():
            for i, replica in enumerate(entry.replicas):
                yield f"data_proxy[{name}/{i}]", replica.data_proxy
                yield f"worker[{name}/{i}]", replica.worker
