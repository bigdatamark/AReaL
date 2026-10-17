# [MODULE DOC]
# Unit tests for the v2 training-service gateway FastAPI app and its streaming helpers
# (query_router, forward_request). Router and upstream HTTP clients are MagicMocks with
# AsyncMock get/post injected into app.state; streaming helpers are patched per test to
# check auth handling, error mapping and request forwarding.
"""Unit tests for training-service gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.v2.training_service.gateway.app import create_app
from areal.v2.training_service.gateway.config import GatewayConfig
from areal.v2.training_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
    forward_request,
    query_router,
)

MODULE = "areal.v2.training_service.gateway.app"
ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key"
WORKER_ADDR = "http://mock-worker:18082"


@pytest.fixture
def config() -> GatewayConfig:
    # [FUNCTION DOC]
    # Fixture: GatewayConfig with mock router address, admin key, 2s router timeout and
    # 20s
    # forward timeout.
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        router_addr="http://mock-router:18081",
        admin_api_key=ADMIN_KEY,
        router_timeout=2.0,
        forward_timeout=20.0,
    )


@pytest_asyncio.fixture
async def client(config, router_client, upstream_client):
    # [FUNCTION DOC]
    # Async fixture: creates the gateway app, injects the mock router and upstream
    # clients
    # into app.state, and yields an httpx ASGI client.
    app = create_app(config)
    app.state.router_client = router_client
    app.state.upstream_client = upstream_client
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def router_client():
    # [FUNCTION DOC]
    # Fixture: MagicMock router HTTP client with AsyncMock get/post.
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


@pytest.fixture
def upstream_client():
    # [FUNCTION DOC]
    # Fixture: MagicMock upstream (model) HTTP client with AsyncMock get/post.
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


class TestGatewayHealth:
    # [CLASS DOC]
    # Health endpoint tests.
    @pytest.mark.asyncio
    async def test_health_reports_router(self, client, config):
        # [FUNCTION DOC]
        # /health returns ok and the configured router address.
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["router_addr"] == config.router_addr


class TestGatewayRoutingAndForwarding:
    # [CLASS DOC]
    # Tests for routing, forwarding and error mapping in the gateway.
    @pytest.mark.asyncio
    async def test_query_router_uses_provided_client(self, router_client, config):
        # [FUNCTION DOC]
        # query_router posts {"api_key": key} to router /route with the admin bearer
        # header and
        # router timeout using the supplied client, returning model_addr from the
        # response.
        router_client.post.return_value = httpx.Response(
            200,
            json={"model_addr": WORKER_ADDR},
            request=httpx.Request("POST", f"{config.router_addr}/route"),
        )

        model_addr = await query_router(
            config.router_addr,
            SESSION_KEY,
            config.router_timeout,
            admin_api_key=ADMIN_KEY,
            client=router_client,
        )

        assert model_addr == WORKER_ADDR
        router_client.post.assert_awaited_once_with(
            f"{config.router_addr}/route",
            json={"api_key": SESSION_KEY},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            timeout=config.router_timeout,
        )

    @pytest.mark.asyncio
    async def test_forward_request_uses_provided_client(self, upstream_client):
        # [FUNCTION DOC]
        # forward_request posts the body to the URL with the supplied client, drops the
        # Host
        # header, and uses a default 600s timeout.
        upstream_client.post.return_value = httpx.Response(
            200,
            json={"status": "success"},
            request=httpx.Request("POST", f"{WORKER_ADDR}/train_batch"),
        )

        resp = await forward_request(
            f"{WORKER_ADDR}/train_batch",
            b"{}",
            {"Authorization": f"Bearer {SESSION_KEY}", "Host": "ignored"},
            client=upstream_client,
        )

        assert resp.status_code == 200
        upstream_client.post.assert_awaited_once_with(
            f"{WORKER_ADDR}/train_batch",
            content=b"{}",
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
            timeout=600.0,
        )

    @pytest.mark.asyncio
    async def test_missing_bearer_token_returns_401(self, client):
        # [FUNCTION DOC]
        # Requests without a bearer token return 401.
        resp = await client.post("/train_batch", json={"args": [], "kwargs": {}})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_router_unreachable_maps_to_502(self, mock_query_router, client):
        # [FUNCTION DOC]
        # RouterUnreachableError from query_router maps to 502.
        mock_query_router.side_effect = RouterUnreachableError("router unavailable")

        resp = await client.post(
            "/train_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_router_404_key_rejected_maps_to_401(self, mock_query_router, client):
        # [FUNCTION DOC]
        # RouterKeyRejectedError 404 maps to 401.
        mock_query_router.side_effect = RouterKeyRejectedError("unknown key", 404)

        resp = await client.post(
            "/eval_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_forward_batch_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
        router_client,
        upstream_client,
        config,
    ):
        # [FUNCTION DOC]
        # /forward_batch queries the router with session key, timeout, admin key and the
        # shared
        # router client, forwards with the shared upstream client, and returns the
        # response.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/forward_batch",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        mock_query_router.assert_awaited_once_with(
            config.router_addr,
            SESSION_KEY,
            config.router_timeout,
            admin_api_key=ADMIN_KEY,
            client=router_client,
        )
        assert mock_forward_request.await_args.kwargs["client"] is upstream_client

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_ppo_actor_compute_logp_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        # [FUNCTION DOC]
        # /ppo/actor/compute_logp is routed and forwarded, returning success.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/ppo/actor/compute_logp",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_sft_train_forwards_response(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        # [FUNCTION DOC]
        # /sft/train is routed and forwarded, returning success.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward_request.return_value = httpx.Response(
            200,
            json={"status": "success", "result": {"ok": True}},
        )

        resp = await client.post(
            "/sft/train",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_offload_uses_admin_auth_upstream(
        self,
        mock_query_router,
        mock_forward_request,
        client,
    ):
        # [FUNCTION DOC]
        # /offload is forwarded upstream with the admin key in Authorization instead of
        # the
        # caller's session key.
        mock_query_router.return_value = WORKER_ADDR

        async def _check_forward(_url, _body, headers, _timeout, *, client):
            # [FUNCTION DOC]
            # HELPER: Forward stub asserting a client is passed and the admin bearer
            # header is used;
            # returns a success response.
            assert client is not None
            assert headers["Authorization"] == f"Bearer {ADMIN_KEY}"
            return httpx.Response(200, json={"status": "success", "result": None})

        mock_forward_request.side_effect = _check_forward

        resp = await client.post(
            "/offload",
            json={"args": [], "kwargs": {}},
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.streaming.query_router", new_callable=AsyncMock)
    async def test_get_version_uses_shared_clients(
        self,
        mock_query_router,
        client,
        router_client,
        upstream_client,
        config,
    ):
        # [FUNCTION DOC]
        # GET /get_version routes via the shared router client and performs an upstream
        # GET to
        # the model's /get_version with the forward timeout and the caller's
        # authorization
        # header.
        mock_query_router.return_value = WORKER_ADDR
        upstream_client.get.return_value = httpx.Response(
            200,
            json={"status": "success", "result": 11},
            request=httpx.Request("GET", f"{WORKER_ADDR}/get_version"),
        )

        resp = await client.get(
            "/get_version",
            headers={"Authorization": f"Bearer {SESSION_KEY}"},
        )

        assert resp.status_code == 200
        mock_query_router.assert_awaited_once_with(
            config.router_addr,
            SESSION_KEY,
            config.router_timeout,
            admin_api_key=ADMIN_KEY,
            client=router_client,
        )
        upstream_client.get.assert_awaited_once()
        upstream_call = upstream_client.get.await_args
        assert upstream_call.args == (f"{WORKER_ADDR}/get_version",)
        assert upstream_call.kwargs["timeout"] == config.forward_timeout
        assert (
            upstream_call.kwargs["headers"]["authorization"] == f"Bearer {SESSION_KEY}"
        )
