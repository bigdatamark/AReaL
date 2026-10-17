# [MODULE DOC]
# Async tests for the data service router app (areal.infra.data_service.router.app),
# which keeps a registry of data worker addresses with health flags and hands out
# workers round-robin. The FastAPI app is driven in-process via httpx.ASGITransport;
# poll_interval is huge so background health polling never runs. Admin endpoints require
# 'Authorization: Bearer <ADMIN_KEY>'.
from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from areal.infra.data_service.router.app import create_router_app
from areal.infra.data_service.router.config import RouterConfig

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:8000"
WORKER_2 = "http://worker-2:8000"
WORKER_3 = "http://worker-3:8000"


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture returning a RouterConfig on 127.0.0.1:18091 with the test admin key and
    # poll_interval=999.
    return RouterConfig(
        host="127.0.0.1",
        port=18091,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
    )


@pytest_asyncio.fixture
async def client(config):
    # [FUNCTION DOC]
    # Async fixture: creates the router app and yields an httpx.AsyncClient bound to it
    # through ASGITransport.
    app = create_router_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns the Authorization header dict carrying the admin bearer token.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestHealth:
    # [CLASS DOC]
    # Tests GET /health.
    @pytest.mark.asyncio
    async def test_health_returns_200_with_worker_count(self, client):
        # [FUNCTION DOC]
        # With no workers, /health returns 200, status 'ok' and workers 0.
        resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["workers"] == 0

    @pytest.mark.asyncio
    async def test_health_shows_healthy_count(self, client):
        # [FUNCTION DOC]
        # After registering two workers, /health reports workers 2 and healthy 2 (newly
        # registered workers count as healthy).
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        resp = await client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["workers"] == 2
        assert payload["healthy"] == 2


class TestWorkerRegistration:
    # [CLASS DOC]
    # Tests /register and /unregister.
    @pytest.mark.asyncio
    async def test_register_worker_success(self, client):
        # [FUNCTION DOC]
        # Registering a worker returns status 'ok' and the worker count becomes 1.
        resp = await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        health = await client.get("/health")
        assert health.json()["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self, client):
        # [FUNCTION DOC]
        # Registering the same address twice succeeds and keeps the count at 1.
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        resp = await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200

        health = await client.get("/health")
        assert health.json()["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /register without Authorization returns 401.
        resp = await client.post("/register", json={"worker_addr": WORKER_1})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_wrong_key_403(self, client):
        # [FUNCTION DOC]
        # /register with a wrong bearer key returns 403.
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_unregister_worker_removes(self, client):
        # [FUNCTION DOC]
        # /unregister removes a registered worker (count back to 0).
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        resp = await client.post(
            "/unregister", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        health = await client.get("/health")
        assert health.json()["workers"] == 0


class TestRouting:
    # [CLASS DOC]
    # Tests POST /route worker selection.
    @pytest.mark.asyncio
    async def test_route_round_robin_cycles(self, client):
        # [FUNCTION DOC]
        # With two workers, four routes alternate W1, W2, W1, W2.
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        picks = []
        for _ in range(4):
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            picks.append(resp.json()["worker_addr"])

        assert picks == [WORKER_1, WORKER_2, WORKER_1, WORKER_2]

    @pytest.mark.asyncio
    async def test_route_no_workers_503(self, client):
        # [FUNCTION DOC]
        # /route with no workers returns 503.
        resp = await client.post("/route", headers=admin_headers())
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_route_skips_unhealthy(self, client):
        # [FUNCTION DOC]
        # Marks WORKER_2 unhealthy directly in app.state.worker_healthy and asserts
        # every route returns WORKER_1.
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        app = client._transport.app
        app.state.worker_healthy[WORKER_2] = False

        for _ in range(3):
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            assert resp.json()["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_route_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /route without auth returns 401.
        resp = await client.post("/route")
        assert resp.status_code == 401


class TestWorkersList:
    # [CLASS DOC]
    # Tests GET /workers.
    @pytest.mark.asyncio
    async def test_workers_returns_all_registered(self, client):
        # [FUNCTION DOC]
        # Lists all three registered worker addresses.
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_3}, headers=admin_headers()
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert {w["addr"] for w in workers} == {WORKER_1, WORKER_2, WORKER_3}

    @pytest.mark.asyncio
    async def test_workers_shows_health_status(self, client):
        # [FUNCTION DOC]
        # A single registered worker is listed as {'addr': W1, 'healthy': True}.
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        workers = resp.json()["workers"]
        assert workers == [{"addr": WORKER_1, "healthy": True}]

    @pytest.mark.asyncio
    async def test_workers_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /workers without auth returns 401.
        resp = await client.get("/workers")
        assert resp.status_code == 401


class TestConcurrentRouting:
    # [CLASS DOC]
    # Round-robin under concurrency.
    @pytest.mark.asyncio
    async def test_concurrent_routes_distribute_evenly(self, client):
        # [FUNCTION DOC]
        # Ten concurrent /route calls over two workers are split exactly 5/5.
        await client.post(
            "/register", json={"worker_addr": WORKER_1}, headers=admin_headers()
        )
        await client.post(
            "/register", json={"worker_addr": WORKER_2}, headers=admin_headers()
        )

        async def route_once() -> str:
            # [FUNCTION DOC]
            # HELPER: Posts /route, asserts 200 and returns the chosen worker_addr.
            resp = await client.post("/route", headers=admin_headers())
            assert resp.status_code == 200
            return resp.json()["worker_addr"]

        routes = await asyncio.gather(*(route_once() for _ in range(10)))
        assert routes.count(WORKER_1) == 5
        assert routes.count(WORKER_2) == 5
