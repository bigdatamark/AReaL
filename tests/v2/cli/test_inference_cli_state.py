# [MODULE DOC]
# Tests for the v2 inference CLI state store (areal.v2.cli.inference.state): per-service
# ServiceState/ModelState JSON files under AREAL_HOME, current-service resolution, the
# `inf models` / `inf ps` click commands, and raw-state PID recovery. Every test points
# AREAL_HOME at pytest's tmp_path via monkeypatch so the filesystem state is isolated.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

from click.testing import CliRunner

from areal.v2.cli.inference.scheduler import TaskHandle
from areal.v2.cli.inference.state import (
    ModelEntry,
    ModelState,
    ServiceState,
    store,
)
from areal.v2.cli.main import cli


def _service_path(service: str):
    # [FUNCTION DOC]
    # HELPER: Returns store.service_state_path(service), the JSON path for a service's
    # state.
    return store.service_state_path(service)


def _current_path():
    # [FUNCTION DOC]
    # HELPER: Returns store.current_service_path(), the file holding the current service
    # name.
    return store.current_service_path()


def _resolve(name: str | None) -> str:
    # [FUNCTION DOC]
    # HELPER: Delegates to store.resolve_service_name(name); None means auto-resolve.
    return store.resolve_service_name(name)


def _save_service(service: str, *, gateway_pid: int | None = None) -> None:
    # [FUNCTION DOC]
    # HELPER: Builds and saves (writes to disk) a ServiceState for `service` with
    # backend "local", gateway handle on 127.0.0.1:8080 and router handle on
    # 127.0.0.1:9000, both with ref {"pid": pid}, admin key "admin", started_at 1.0. pid
    # defaults to the current process pid (so the service looks alive); pass gateway_pid
    # (e.g. -1) to simulate a dead process. Saving also updates the current-service
    # pointer as a side effect of ServiceState.save.
    pid = os.getpid() if gateway_pid is None else gateway_pid
    ServiceState(
        service=service,
        backend="local",
        gateway_handle=TaskHandle(
            host="127.0.0.1", ports=[8080], gpu_devices=[], ref={"pid": pid}
        ),
        router_handle=TaskHandle(
            host="127.0.0.1", ports=[9000], gpu_devices=[], ref={"pid": pid}
        ),
        admin_api_key="admin",
        started_at=1.0,
    ).save()


def _placeholder_model() -> ModelEntry:
    # [FUNCTION DOC]
    # HELPER: Returns a ModelEntry with backend "sglang:d1" and no replicas.
    return ModelEntry(backend="sglang:d1", replicas=[])


def test_service_and_model_state_are_per_service(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Saves service "svc-a" plus a ModelState with model "m-a"; asserts the service and
    # models state files exist, the current-service file contains "svc-a", and loading
    # both round-trips service name, backend "local" and the model list.
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    model_state = ModelState(service="svc-a")
    model_state.models["m-a"] = _placeholder_model()
    model_state.save()

    assert _service_path("svc-a").exists()
    assert store.models_state_path("svc-a").exists()
    assert _current_path().read_text().strip() == "svc-a"

    loaded_service = ServiceState.load("svc-a")
    loaded_models = ModelState.load("svc-a")
    assert loaded_service.service == "svc-a"
    assert loaded_service.backend == "local"
    assert list(loaded_models.models) == ["m-a"]


def test_resolve_service_uses_current_then_single_running(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Saves service "current"; asserts resolve(None) returns it, then deletes the
    # current-service file and asserts resolution falls back to the single running
    # service.
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("current")
    assert _resolve(None) == "current"

    _current_path().unlink()
    assert _resolve(None) == "current"


def test_models_command_is_service_scoped(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Saves two services each with one model, invokes `inf models --service svc-a
    # --json` via CliRunner, and asserts exit code 0 and that only "m-a" is listed.
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    _save_service("svc-b")
    for service, model in (("svc-a", "m-a"), ("svc-b", "m-b")):
        model_state = ModelState(service=service)
        model_state.models[model] = _placeholder_model()
        model_state.save()

    result = CliRunner().invoke(cli, ["inf", "models", "--service", "svc-a", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [entry["name"] for entry in payload] == ["m-a"]


def test_ps_lists_services_and_stale_state(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Saves a live service (current pid) and a stale one (pid -1); invokes `inf ps --all
    # --json` and asserts the status map is live->running, stale->stale.
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("live")
    _save_service("stale", gateway_pid=-1)

    result = CliRunner().invoke(cli, ["inf", "ps", "--all", "--json"])

    assert result.exit_code == 0
    payload = {entry["service"]: entry["status"] for entry in json.loads(result.output)}
    assert payload == {"live": "running", "stale": "stale"}


def test_recover_pids_from_raw_state_walks_handles(tmp_path, monkeypatch):
    # [FUNCTION DOC]
    # Hand-writes v1-schema service and models JSON files with nested ref.pid values and
    # asserts store.recover_pids_from_raw_state("svc") returns [100, 101, 300, 200],
    # i.e. the JSON walk order (gateway, router, then replica data_proxy before worker).
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    # Hand-write the v1 schema so we exercise recover_pids_from_raw_state
    # against TaskHandle.ref.pid nesting. Order of returned pids follows the
    # JSON walk order: service file first, then models file; within each
    # ModelReplica data_proxy is declared before worker.
    _service_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "backend": "local",
                "gateway_handle": {
                    "host": "127.0.0.1",
                    "ports": [8080],
                    "gpu_devices": [],
                    "ref": {"pid": 100},
                },
                "router_handle": {
                    "host": "127.0.0.1",
                    "ports": [9000],
                    "gpu_devices": [],
                    "ref": {"pid": 101},
                },
                "admin_api_key": "admin",
                "started_at": 1.0,
            }
        )
    )
    store.models_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "models": {
                    "m": {
                        "backend": "sglang:d1",
                        "replicas": [
                            {
                                "data_proxy": {
                                    "host": "127.0.0.1",
                                    "ports": [5001],
                                    "gpu_devices": [],
                                    "ref": {"pid": 300},
                                },
                                "worker": {
                                    "host": "127.0.0.1",
                                    "ports": [5000],
                                    "gpu_devices": [0],
                                    "ref": {"pid": 200},
                                },
                            }
                        ],
                    }
                },
            }
        )
    )

    assert store.recover_pids_from_raw_state("svc") == [100, 101, 300, 200]
