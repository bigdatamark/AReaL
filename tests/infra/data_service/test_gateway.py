# [MODULE DOC]
# Async tests for the data service gateway app (areal.infra.data_service.gateway.app),
# the client-facing HTTP front end that authenticates admin and per-dataset bearer keys,
# registers datasets by broadcasting load requests to all workers, routes sample fetches
# to a worker chosen by the router, and broadcasts epoch/state operations. The app runs
# in-process via httpx.ASGITransport; router and worker communication
# (_get_all_worker_addrs, _broadcast_to_workers, _query_router, aiohttp.ClientSession)
# is patched with AsyncMocks.
from __future__ import annotations

# pyright: reportMissingImports=false
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig

ADMIN_KEY = "test-admin-key"
WORKER_ADDR = "http://worker-1:8000"
WORKER_ADDR_2 = "http://worker-2:8000"
MODULE = "areal.infra.data_service.gateway.app"


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture returning a GatewayConfig on 127.0.0.1:18090 with the test admin key and a
    # mock router address.
    return GatewayConfig(
        host="127.0.0.1",
        port=18090,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8091",
    )


@pytest_asyncio.fixture
async def client(config):
    # [FUNCTION DOC]
    # Async fixture yielding an httpx.AsyncClient bound to a fresh gateway app.
    app = create_gateway_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns the Authorization header dict with the admin bearer token.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


async def _register_dataset(client, dataset_id: str = "train-sample") -> dict:
    # [FUNCTION DOC]
    # HELPER: Async; registers dataset_id (default 'train-sample') via POST
    # /v1/datasets/register with a single mocked worker that reports steps_per_epoch 10
    # and dataset_size 100. Asserts 200 and returns the JSON response (contains
    # api_key).
    with (
        patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers,
        patch(
            f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
        ) as mock_broadcast,
    ):
        mock_workers.return_value = [WORKER_ADDR]
        mock_broadcast.return_value = [
            {
                "addr": WORKER_ADDR,
                "status": 200,
                "data": {"steps_per_epoch": 10, "dataset_size": 100},
            }
        ]
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": dataset_id, "dataset_path": "/tmp/sample.jsonl"},
            headers=admin_headers(),
        )
    assert resp.status_code == 200
    return resp.json()


class TestGatewayHealth:
    # [CLASS DOC]
    # Tests GET /health.
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        # [FUNCTION DOC]
        # /health needs no auth and returns status 'ok'.
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        # [FUNCTION DOC]
        # /health reports the configured router_addr.
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


class TestGatewayAuth:
    # [CLASS DOC]
    # Tests authentication failures.
    @pytest.mark.asyncio
    async def test_fetch_samples_no_auth_401(self, client):
        # [FUNCTION DOC]
        # Fetching samples without a token returns 401.
        resp = await client.post("/v1/samples/fetch", json={"indices": [0]})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_fetch_samples_bad_key_401(self, client):
        # [FUNCTION DOC]
        # Fetching samples with an unknown token returns 401.
        resp = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": "Bearer unknown-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_endpoint_with_dataset_key_403(self, client):
        # [FUNCTION DOC]
        # Calling the admin-only register endpoint with a non-admin key returns 403.
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": "d1", "dataset_path": "/tmp/data.jsonl"},
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403


class TestDatasetRegistration:
    # [CLASS DOC]
    # Tests dataset registration lifecycle.
    @pytest.mark.asyncio
    async def test_register_dataset_returns_api_key(self, client):
        # [FUNCTION DOC]
        # Registering with two workers each reporting 120 rows returns an 'ds-' api_key,
        # the dataset_id and dataset_size summed across workers (240).
        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {
                    "addr": WORKER_ADDR,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
                {
                    "addr": WORKER_ADDR_2,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
            ]

            resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-a",
                    "dataset_path": "/tmp/a.jsonl",
                },
                headers=admin_headers(),
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["api_key"].startswith("ds-")
        assert payload["dataset_id"] == "dataset-a"
        assert payload["dataset_size"] == 240

    @pytest.mark.asyncio
    async def test_register_then_fetch_uses_dataset_key(self, client):
        # [FUNCTION DOC]
        # After registration, a fetch with the dataset key is forwarded to the router-
        # chosen worker's /v1/samples/fetch with the dataset_id injected into the JSON
        # body. Uses a hand-built aiohttp session mock that records post args.
        reg_payload = await _register_dataset(client, dataset_id="dataset-b")
        dataset_key = reg_payload["api_key"]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"samples": [{"text": "hello"}]})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = lambda *args, **kwargs: (
            setattr(mock_session, "_post_args", (args, kwargs)) or mock_resp
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(f"{MODULE}._query_router", new_callable=AsyncMock) as mock_route,
            patch(f"{MODULE}.aiohttp.ClientSession", return_value=mock_session),
        ):
            mock_route.return_value = WORKER_ADDR
            resp = await client.post(
                "/v1/samples/fetch",
                json={"indices": [0]},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        post_args, post_kwargs = mock_session._post_args
        assert post_args[0] == f"{WORKER_ADDR}/v1/samples/fetch"
        assert post_kwargs["json"]["dataset_id"] == "dataset-b"

    @pytest.mark.asyncio
    async def test_unregister_revokes_key(self, client):
        # [FUNCTION DOC]
        # Unregistering a dataset revokes its key: a later fetch with that key returns
        # 401.
        reg_payload = await _register_dataset(client, dataset_id="dataset-c")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}}
            ]
            resp = await client.post(
                "/v1/datasets/unregister",
                json={"dataset_id": "dataset-c"},
                headers=admin_headers(),
            )
        assert resp.status_code == 200

        after_revoke = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": f"Bearer {dataset_key}"},
        )
        assert after_revoke.status_code == 401

    @pytest.mark.asyncio
    async def test_register_failure_rolls_back_successful_workers(self, client):
        # [FUNCTION DOC]
        # If one worker fails to load (500), registration returns 502 and a second
        # broadcast unloads the dataset from only the successful worker (rollback).
        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.side_effect = [
                [
                    {"addr": WORKER_ADDR, "status": 200, "data": {}},
                    {"addr": WORKER_ADDR_2, "status": 500, "error": "boom"},
                ],
                [
                    {"addr": WORKER_ADDR, "status": 200, "data": {}},
                ],
            ]

            resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-rollback",
                    "dataset_path": "/tmp/a.jsonl",
                },
                headers=admin_headers(),
            )

        assert resp.status_code == 502
        assert mock_broadcast.await_count == 2
        rollback_call = mock_broadcast.await_args_list[1]
        assert rollback_call.args[0] == [WORKER_ADDR]
        assert rollback_call.args[1] == "/datasets/unload"
        assert rollback_call.args[2] == {"dataset_id": "dataset-rollback"}

    @pytest.mark.asyncio
    async def test_reregister_revokes_old_key(self, client):
        # [FUNCTION DOC]
        # Re-registering the same dataset id issues a new key and the old key is
        # rejected with 401.
        first = await _register_dataset(client, dataset_id="dataset-rekey")

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR]
            mock_broadcast.return_value = [
                {
                    "addr": WORKER_ADDR,
                    "status": 200,
                    "data": {"steps_per_epoch": 10, "dataset_size": 100},
                }
            ]
            second_resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-rekey",
                    "dataset_path": "/tmp/sample.jsonl",
                },
                headers=admin_headers(),
            )

        assert second_resp.status_code == 200
        second = second_resp.json()
        assert first["api_key"] != second["api_key"]

        old_key_resp = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": f"Bearer {first['api_key']}"},
        )
        assert old_key_resp.status_code == 401


class TestBroadcastEndpoints:
    # [CLASS DOC]
    # Tests dataset-key endpoints that broadcast to all workers.
    @pytest.mark.asyncio
    async def test_epoch_advance_broadcasts_to_all_workers(self, client):
        # [FUNCTION DOC]
        # POST /v1/epochs/advance with two successful workers returns workers_reset 2.
        reg_payload = await _register_dataset(client, dataset_id="dataset-f")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json()["workers_reset"] == 2

    @pytest.mark.asyncio
    async def test_epoch_advance_no_workers_returns_503(self, client):
        # [FUNCTION DOC]
        # Epoch advance with no workers returns 503.
        reg_payload = await _register_dataset(client, dataset_id="dataset-noworkers")
        dataset_key = reg_payload["api_key"]

        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = []
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_epoch_advance_worker_failure_returns_502(self, client):
        # [FUNCTION DOC]
        # Epoch advance where one worker returns 500 yields 502.
        reg_payload = await _register_dataset(client, dataset_id="dataset-partial")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 500, "error": "boom"},
            ]
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_state_save_broadcasts_to_all_workers(self, client):
        # [FUNCTION DOC]
        # POST /v1/state/save with all workers succeeding returns {'status': 'ok',
        # 'path': path}.
        reg_payload = await _register_dataset(client, dataset_id="dataset-g")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/save",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "path": "/tmp/ckpt"}

    @pytest.mark.asyncio
    async def test_state_load_broadcasts_to_all_workers(self, client):
        # [FUNCTION DOC]
        # POST /v1/state/load with all workers succeeding returns {'status': 'ok'}.
        reg_payload = await _register_dataset(client, dataset_id="dataset-h")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/load",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestStatusAndWorkers:
    # [CLASS DOC]
    # Tests worker listing and dataset status.
    @pytest.mark.asyncio
    async def test_workers_returns_router_workers(self, client):
        # [FUNCTION DOC]
        # Admin GET /v1/workers returns the router's worker addresses as a list of
        # {'addr': ...}.
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            resp = await client.get("/v1/workers", headers=admin_headers())

        assert resp.status_code == 200
        assert resp.json() == {
            "workers": [{"addr": WORKER_ADDR}, {"addr": WORKER_ADDR_2}]
        }

    @pytest.mark.asyncio
    async def test_status_returns_dataset_id(self, client):
        # [FUNCTION DOC]
        # GET /v1/status with a dataset key still returns status 'ok' and the dataset_id
        # even when the router query raises.
        reg_payload = await _register_dataset(client, dataset_id="dataset-status")
        dataset_key = reg_payload["api_key"]

        with patch(
            f"{MODULE}._query_router", new_callable=AsyncMock
        ) as mock_query_router:
            mock_query_router.side_effect = RuntimeError("router unavailable")
            resp = await client.get(
                "/v1/status",
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["dataset_id"] == "dataset-status"


class TestShutdown:
    # [CLASS DOC]
    # Tests POST /v1/shutdown.
    @pytest.mark.asyncio
    async def test_shutdown_requires_admin_key(self, client):
        # [FUNCTION DOC]
        # Shutdown with a non-admin key returns 403.
        resp = await client.post(
            "/v1/shutdown",
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_shutdown_returns_ok(self, client):
        # [FUNCTION DOC]
        # Admin shutdown with no workers returns {'status': 'ok'}.
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = []
            resp = await client.post("/v1/shutdown", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
