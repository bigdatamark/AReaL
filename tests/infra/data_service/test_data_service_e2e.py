# [MODULE DOC]
# Slow end-to-end tests of the full data service stack: a worker, a router and a gateway
# each run as real uvicorn servers on free localhost ports in daemon threads, the worker
# is registered with the router, and tests drive the gateway over HTTP with admin and
# per-dataset bearer keys. Uses real datasets/models (GSM8K, Geometry3K, Qwen3-0.6B,
# Qwen2.5-VL-3B) from /storage/openpsi or the HuggingFace Hub, skipping when
# unavailable. Constants: ADMIN_KEY (shared admin token), BATCH_SIZE.
# KNOWN ISSUE: BATCH_SIZE is defined but never used.
from __future__ import annotations

# pyright: reportMissingImports=false
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from huggingface_hub import snapshot_download

from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig
from areal.infra.data_service.router.app import create_router_app
from areal.infra.data_service.router.config import RouterConfig
from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

pytestmark = pytest.mark.slow

ADMIN_KEY = "areal-data-admin"
BATCH_SIZE = 256


def _resolve_path(local: str, hf_id: str, repo_type: str = "dataset") -> str:
    # [FUNCTION DOC]
    # HELPER: Returns local if that path exists; otherwise downloads hf_id via
    # snapshot_download with the given repo_type (default 'dataset') and returns the
    # snapshot path. On any download error calls pytest.skip (the trailing
    # AssertionError is unreachable).
    if os.path.exists(local):
        return local
    try:
        return snapshot_download(repo_id=hf_id, repo_type=repo_type)
    except Exception as exc:
        pytest.skip(f"Required test artifact unavailable: {hf_id} ({exc})")
        raise AssertionError("unreachable")


def _free_port() -> int:
    # [FUNCTION DOC]
    # HELPER: Binds a TCP socket to 127.0.0.1 port 0 and returns the OS-assigned port
    # number, closing the socket.
    # KNOWN ISSUE: The port is released before the server binds it, so another process
    # could grab it (benign race typical of tests).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_uvicorn(
    app: object, host: str, port: int
) -> tuple[uvicorn.Server, threading.Thread]:
    # [FUNCTION DOC]
    # HELPER: Creates a uvicorn Server for app on host:port with warning-level logging
    # and starts server.run in a daemon thread. Returns (server, thread).
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread


def _wait_healthy(base_url: str, timeout: float = 10.0) -> None:
    # [FUNCTION DOC]
    # HELPER: Polls GET <base_url>/health every 50 ms until it returns 200 or timeout
    # (default 10 s) elapses, ignoring httpx errors. Raises TimeoutError if never
    # healthy.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise TimeoutError(f"Service did not become healthy: {base_url}")


def _admin_headers() -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Returns the Authorization header with the admin bearer token.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def _dataset_headers(api_key: str) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Returns the Authorization header with the given dataset api key.
    return {"Authorization": f"Bearer {api_key}"}


def _register_dataset(
    client: httpx.Client,
    *,
    dataset_id: str,
    dataset_path: str,
    dataset_type: str = "rl",
    tokenizer_or_processor_path: str = "",
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: POSTs /v1/datasets/register as admin with dataset_id, dataset_path,
    # dataset_type (default 'rl'), split 'test', seed 42, shuffle False and
    # tokenizer_or_processor_path (default empty), timeout 120 s. Asserts 200 and
    # matching dataset_id; returns the JSON payload (api_key, dataset_size, ...).
    resp = client.post(
        "/v1/datasets/register",
        headers=_admin_headers(),
        json={
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "dataset_type": dataset_type,
            "split": "test",
            "seed": 42,
            "shuffle": False,
            "tokenizer_or_processor_path": tokenizer_or_processor_path,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, f"register failed: {resp.text}"
    payload = resp.json()
    assert payload["dataset_id"] == dataset_id
    return payload


def _unique_dataset_id(prefix: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns '<prefix>-<8 hex chars of a uuid4>' so each test registers a
    # distinct dataset id.
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def data_service_stack(tmp_path_factory: pytest.TempPathFactory):
    # [FUNCTION DOC]
    # Module-scoped fixture: resolves dataset/model paths, picks free ports, starts the
    # worker, waits healthy, starts the router (poll interval 0.2 s), registers the
    # worker with it, starts the gateway, then yields a dict of addresses, a temp
    # state_dir and the resolved paths. On teardown sends an admin /v1/shutdown to the
    # gateway (ignoring HTTP errors) and stops servers in reverse order by setting
    # should_exit and joining threads (5 s each).
    dataset_path = _resolve_path(
        "/storage/openpsi/data/gsm8k", "openai/gsm8k", repo_type="dataset"
    )
    geometry3k_path = _resolve_path(
        "/storage/openpsi/data/hiyouga__geometry3k/",
        "hiyouga/geometry3k",
        repo_type="dataset",
    )
    model_path = _resolve_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        repo_type="model",
    )
    vlm_model_path = _resolve_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        repo_type="model",
    )

    worker_port = _free_port()
    router_port = _free_port()
    gateway_port = _free_port()

    worker_addr = f"http://127.0.0.1:{worker_port}"
    router_addr = f"http://127.0.0.1:{router_port}"
    gateway_addr = f"http://127.0.0.1:{gateway_port}"
    state_dir = tmp_path_factory.mktemp("data-service-state")

    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    worker_app = create_worker_app(
        DataWorkerConfig(
            host="127.0.0.1",
            port=worker_port,
            rank=0,
            world_size=1,
            dataloader_num_workers=0,
        )
    )
    servers.append(_start_uvicorn(worker_app, "127.0.0.1", worker_port))
    _wait_healthy(worker_addr)

    router_app = create_router_app(
        RouterConfig(
            host="127.0.0.1",
            port=router_port,
            admin_api_key=ADMIN_KEY,
            poll_interval=0.2,
        )
    )
    servers.append(_start_uvicorn(router_app, "127.0.0.1", router_port))
    _wait_healthy(router_addr)

    register_resp = httpx.post(
        f"{router_addr}/register",
        headers=_admin_headers(),
        json={"worker_addr": worker_addr},
        timeout=3.0,
    )
    assert register_resp.status_code == 200

    gateway_app = create_gateway_app(
        GatewayConfig(
            host="127.0.0.1",
            port=gateway_port,
            admin_api_key=ADMIN_KEY,
            router_addr=router_addr,
        )
    )
    servers.append(_start_uvicorn(gateway_app, "127.0.0.1", gateway_port))
    _wait_healthy(gateway_addr)

    yield {
        "worker_addr": worker_addr,
        "router_addr": router_addr,
        "gateway_addr": gateway_addr,
        "state_dir": state_dir,
        "dataset_path": dataset_path,
        "geometry3k_path": geometry3k_path,
        "model_path": model_path,
        "vlm_model_path": vlm_model_path,
    }

    try:
        httpx.post(
            f"{gateway_addr}/v1/shutdown",
            headers=_admin_headers(),
            timeout=3.0,
        )
    except httpx.HTTPError:
        pass

    for server, thread in reversed(servers):
        server.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def gateway_client(data_service_stack: dict[str, Any]):
    # [FUNCTION DOC]
    # Fixture yielding a synchronous httpx.Client with base_url set to the gateway and a
    # 30 s timeout.
    gateway_addr = str(data_service_stack["gateway_addr"])
    with httpx.Client(base_url=gateway_addr, timeout=30.0) as client:
        yield client


class TestServiceHealth:
    # [CLASS DOC]
    # Liveness of the three services.
    def test_all_services_healthy(self, data_service_stack: dict[str, Any]):
        # [FUNCTION DOC]
        # GET /health on worker, router and gateway all return 200.
        worker_addr = str(data_service_stack["worker_addr"])
        router_addr = str(data_service_stack["router_addr"])
        gateway_addr = str(data_service_stack["gateway_addr"])

        worker = httpx.get(f"{worker_addr}/health", timeout=3.0)
        router = httpx.get(f"{router_addr}/health", timeout=3.0)
        gateway = httpx.get(f"{gateway_addr}/health", timeout=3.0)

        assert worker.status_code == 200
        assert router.status_code == 200
        assert gateway.status_code == 200

    def test_router_shows_registered_worker(self, data_service_stack: dict[str, Any]):
        # [FUNCTION DOC]
        # The router's /workers lists exactly the registered worker as healthy.
        router_addr = str(data_service_stack["router_addr"])
        worker_addr = str(data_service_stack["worker_addr"])

        resp = httpx.get(
            f"{router_addr}/workers",
            headers=_admin_headers(),
            timeout=3.0,
        )
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert workers == [{"addr": worker_addr, "healthy": True}]


class TestDatasetRegistration:
    # [CLASS DOC]
    # Dataset registration through the gateway.
    def test_register_rl_dataset_returns_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Registering GSM8K as 'rl' returns a 'ds-' key and positive dataset_size.
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-rl"),
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type="rl",
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0

    def test_register_sft_dataset_returns_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Registering GSM8K as 'sft' with the Qwen3 tokenizer returns a key and positive
        # size.
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-sft"),
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type="sft",
            tokenizer_or_processor_path=str(data_service_stack["model_path"]),
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0

    def test_register_returns_dataset_size(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Registration response has positive dataset_size and no steps_per_epoch field.
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-steps"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        assert payload["dataset_size"] > 0
        assert "steps_per_epoch" not in payload

    def test_register_geometry3k_rl_dataset(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Registering Geometry3K as 'rl' with the Qwen2.5-VL processor returns a key and
        # positive size.
        payload = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("register-geo3k"),
            dataset_path=str(data_service_stack["geometry3k_path"]),
            dataset_type="rl",
            tokenizer_or_processor_path=str(data_service_stack["vlm_model_path"]),
        )
        assert str(payload["api_key"]).startswith("ds-")
        assert payload["dataset_size"] > 0


class TestBatchFetching:
    # [CLASS DOC]
    # Sample fetching through the gateway.
    def test_fetch_batch_returns_data(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Fetching index 0 returns one non-empty sample.
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-one"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        resp = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert resp.status_code == 200
        payload = resp.json()["samples"]
        assert len(payload) == 1
        assert payload[0]

    def test_fetch_multiple_batches(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Fetching indices 0, 1, 2 one at a time returns pairwise-different consecutive
        # samples.
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-multi"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])

        batches = []
        for idx in range(3):
            resp = gateway_client.post(
                "/v1/samples/fetch",
                headers=_dataset_headers(api_key),
                json={"indices": [idx]},
            )
            assert resp.status_code == 200
            batches.append(resp.json()["samples"][0])

        assert batches[0] != batches[1]
        assert batches[1] != batches[2]

    def test_fetch_after_epoch_advance(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # Fetch works before and after advancing to epoch 1, and the advance reports
        # workers_reset 1.
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("fetch-epoch"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        before = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert before.status_code == 200
        assert len(before.json()["samples"]) == 1

        reset = gateway_client.post(
            "/v1/epochs/advance",
            headers=_dataset_headers(api_key),
            json={"epoch": 1},
        )
        assert reset.status_code == 200
        assert reset.json()["workers_reset"] == 1

        after = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert after.status_code == 200
        body = after.json()["samples"]
        assert len(body) == 1
        assert body[0]


class TestStatePersistence:
    # [CLASS DOC]
    # Checkpoint persistence through the gateway.
    def test_state_save_and_load(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
    ):
        # [FUNCTION DOC]
        # /v1/state/save writes worker_0.pkl into the shared state_dir and
        # /v1/state/load from it returns status 'ok'.
        reg = _register_dataset(
            gateway_client,
            dataset_id=_unique_dataset_id("state"),
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])
        state_dir = Path(str(data_service_stack["state_dir"]))

        save = gateway_client.post(
            "/v1/state/save",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert save.status_code == 200
        assert save.json()["status"] == "ok"
        assert (state_dir / "worker_0.pkl").exists()

        load = gateway_client.post(
            "/v1/state/load",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"


class TestDatasetUnregistration:
    # [CLASS DOC]
    # Dataset unregistration.
    def test_unregister_revokes_key(
        self, gateway_client: httpx.Client, data_service_stack: dict[str, Any]
    ):
        # [FUNCTION DOC]
        # After admin unregister, fetching with the old dataset key returns 401.
        dataset_id = _unique_dataset_id("unregister")
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["dataset_path"]),
        )
        api_key = str(reg["api_key"])

        unreg = gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
        assert unreg.status_code == 200

        rejected = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert rejected.status_code == 401


class TestFullLifecycle:
    # [CLASS DOC]
    # Full register-fetch-advance-save-load-unregister flows.
    @pytest.mark.parametrize("dataset_type", ["rl", "sft"])
    def test_complete_lifecycle(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
        dataset_type: str,
    ):
        # [FUNCTION DOC]
        # Parametrized over 'rl' and 'sft' (sft passes the tokenizer path): registers,
        # fetches index 0 three times, advances to epoch 2 (workers_reset 1), fetches
        # again, saves and loads state in a per-dataset subdirectory, unregisters and
        # asserts the old key now gets 401.
        dataset_id = _unique_dataset_id(f"full-{dataset_type}")
        tokenizer_or_processor_path = (
            str(data_service_stack["model_path"]) if dataset_type == "sft" else ""
        )
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["dataset_path"]),
            dataset_type=dataset_type,
            tokenizer_or_processor_path=tokenizer_or_processor_path,
        )
        api_key = str(reg["api_key"])
        state_dir = Path(str(data_service_stack["state_dir"])) / f"{dataset_id}-state"

        for _ in range(3):
            resp = gateway_client.post(
                "/v1/samples/fetch",
                headers=_dataset_headers(api_key),
                json={"indices": [0]},
            )
            assert resp.status_code == 200
            assert resp.json()["samples"][0] is not None

        reset = gateway_client.post(
            "/v1/epochs/advance",
            headers=_dataset_headers(api_key),
            json={"epoch": 2},
        )
        assert reset.status_code == 200
        assert reset.json()["workers_reset"] == 1

        after_reset = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert after_reset.status_code == 200
        batch_item = after_reset.json()["samples"][0]
        assert batch_item is not None

        save = gateway_client.post(
            "/v1/state/save",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert save.status_code == 200

        load = gateway_client.post(
            "/v1/state/load",
            headers=_dataset_headers(api_key),
            json={"path": str(state_dir)},
        )
        assert load.status_code == 200

        unreg = gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
        assert unreg.status_code == 200

        rejected = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert rejected.status_code == 401

    def test_geometry3k_rl_lifecycle(
        self,
        gateway_client: httpx.Client,
        data_service_stack: dict[str, Any],
    ):
        # [FUNCTION DOC]
        # Registers Geometry3K with the VLM processor, asserts positive dataset_size,
        # fetches one non-empty sample and unregisters (unregister status not asserted).
        dataset_id = _unique_dataset_id("full-geo3k")
        reg = _register_dataset(
            gateway_client,
            dataset_id=dataset_id,
            dataset_path=str(data_service_stack["geometry3k_path"]),
            dataset_type="rl",
            tokenizer_or_processor_path=str(data_service_stack["vlm_model_path"]),
        )
        api_key = str(reg["api_key"])
        steps = int(reg["dataset_size"])
        assert steps > 0

        resp = gateway_client.post(
            "/v1/samples/fetch",
            headers=_dataset_headers(api_key),
            json={"indices": [0]},
        )
        assert resp.status_code == 200
        assert resp.json()["samples"][0]

        gateway_client.post(
            "/v1/datasets/unregister",
            headers=_admin_headers(),
            json={"dataset_id": dataset_id},
        )
