# [MODULE DOC]
# Async tests for the v2 agent-service Router ASGI app (created by create_router_app),
# exercised in-process via httpx.ASGITransport with admin auth headers: health,
# data-proxy register/unregister, sticky session routing, round-robin assignment and
# session removal. Skipped if httpx is not installed.
"""Tests for Agent Router HTTP server."""

from __future__ import annotations

import pytest

from areal.v2.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.v2.agent_service.router.app import create_router_app
from areal.v2.agent_service.router.config import RouterConfig

httpx = pytest.importorskip("httpx")

_AUTH = admin_headers(DEFAULT_ADMIN_API_KEY)


def _make_client():
    # [FUNCTION DOC]
    # HELPER: Builds a router app with the default admin key and returns an
    # httpx.AsyncClient bound to it via ASGI transport (base URL http://router).
    config = RouterConfig(admin_api_key=DEFAULT_ADMIN_API_KEY)
    app = create_router_app(config)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://router")


class TestRouterHealth:
    # [CLASS DOC]
    # Health endpoint tests.
    @pytest.mark.asyncio
    async def test_health(self):
        # [FUNCTION DOC]
        # Asserts GET /health returns 200 with status "ok" and zero registered proxies.
        async with _make_client() as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["registered_proxies"] == 0


class TestRegistration:
    # [CLASS DOC]
    # Proxy registration endpoint tests.
    @pytest.mark.asyncio
    async def test_register_and_health(self):
        # [FUNCTION DOC]
        # Registers one proxy and asserts health reports one registered proxy.
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 1

    @pytest.mark.asyncio
    async def test_unregister(self):
        # [FUNCTION DOC]
        # Registers then unregisters a proxy and asserts the count returns to zero.
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            await client.post(
                "/unregister", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.get("/health")
            assert resp.json()["registered_proxies"] == 0


class TestRouting:
    # [CLASS DOC]
    # Session routing endpoint tests.
    @pytest.mark.asyncio
    async def test_route_new_session(self):
        # [FUNCTION DOC]
        # Asserts POST /route for a new session returns the only registered proxy address.
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            resp = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp.json()["data_proxy_addr"] == "http://proxy1:9100"

    @pytest.mark.asyncio
    async def test_route_existing_session_returns_same(self):
        # [FUNCTION DOC]
        # With two proxies, asserts routing the same session key twice is sticky.
        async with _make_client() as client:
            await client.post(
                "/register", json={"addr": "http://proxy1:9100"}, headers=_AUTH
            )
            await client.post(
                "/register", json={"addr": "http://proxy2:9101"}, headers=_AUTH
            )
            r1 = await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            r2 = await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            assert r1.json()["data_proxy_addr"] == r2.json()["data_proxy_addr"]

    @pytest.mark.asyncio
    async def test_round_robin(self):
        # [FUNCTION DOC]
        # Asserts two distinct new sessions are spread over both proxies (round robin).
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"}, headers=_AUTH)
            await client.post("/register", json={"addr": "http://p2"}, headers=_AUTH)
            r1 = await client.post("/route", json={"session_key": "a"}, headers=_AUTH)
            r2 = await client.post("/route", json={"session_key": "b"}, headers=_AUTH)
            addrs = {r1.json()["data_proxy_addr"], r2.json()["data_proxy_addr"]}
            assert addrs == {"http://p1", "http://p2"}

    @pytest.mark.asyncio
    async def test_remove_session(self):
        # [FUNCTION DOC]
        # Routes a session, removes it via /remove_session and asserts active_sessions is 0.
        async with _make_client() as client:
            await client.post("/register", json={"addr": "http://p1"}, headers=_AUTH)
            await client.post("/route", json={"session_key": "s1"}, headers=_AUTH)
            await client.post(
                "/remove_session", json={"session_key": "s1"}, headers=_AUTH
            )
            health = await client.get("/health")
            assert health.json()["active_sessions"] == 0
