# [MODULE DOC]
# Unit tests for the v2 inference gateway FastAPI app (create_app). The gateway is
# driven
# in-process via an httpx ASGITransport client; the router/data-proxy helper functions
# in
# the gateway app module (query_router, forward_request, forward_sse_stream,
# register_session_in_router, revoke_session_in_router, resolve_worker_addr,
# broadcast_to_workers) are patched with mocks so no network is used. Covers health,
# admin vs session-key auth, Anthropic /v1/messages, admin/session endpoints, pause and
# continue broadcast, router error mapping, and start_session flow.
"""Unit tests for the inference gateway.

Tests gateway endpoints with mocked Router and data proxy workers.
Uses ``unittest.mock.patch`` to mock ``streaming.py`` functions at module level.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.gateway.app import create_app
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
)

# =============================================================================
# Constants & Config
# =============================================================================

ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key-abc123"
WORKER_ADDR = "http://worker-1:18082"
WORKER_ADDR_2 = "http://worker-2:18082"
WORKER_ADDR_3 = "http://worker-3:18082"

MODULE = "areal.v2.inference_service.gateway.app"


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: returns a GatewayConfig bound to 127.0.0.1:18080 with the test admin key,
    # a
    # fake router address, 2s router timeout and 30s forward timeout.
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8081",
        router_timeout=2.0,
        forward_timeout=30.0,
    )


@pytest_asyncio.fixture
async def client(config):
    # [FUNCTION DOC]
    # Async fixture: builds the gateway app from the config fixture and yields an
    # httpx.AsyncClient using an in-process ASGI transport (base URL http://test);
    # closes it
    # on teardown.
    """Create gateway app and yield an httpx async client."""
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization Bearer header dict carrying the admin API key.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization Bearer header dict carrying the session API key.
    return {"Authorization": f"Bearer {SESSION_KEY}"}


def anthropic_headers():
    # [FUNCTION DOC]
    # HELPER: Returns Anthropic-style headers: x-api-key set to the session key plus a
    # fixed
    # anthropic-version.
    return {
        "x-api-key": SESSION_KEY,
        "anthropic-version": "2023-06-01",
    }


# =============================================================================
# Health endpoint
# =============================================================================


class TestHealthEndpoint:
    # [CLASS DOC]
    # Tests for the unauthenticated /health endpoint.
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        # [FUNCTION DOC]
        # GET /health without auth returns 200 with status "ok".
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        # [FUNCTION DOC]
        # GET /health reports the configured router address in its JSON body.
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


# =============================================================================
# Auth rejection
# =============================================================================


class TestAuthRejection:
    # [CLASS DOC]
    # Tests that missing or insufficient credentials are rejected.
    @pytest.mark.asyncio
    async def test_missing_auth_chat_401(self, client):
        # [FUNCTION DOC]
        # POST /chat/completions with no Authorization header returns 401.
        """POST /chat/completions without auth → 401."""
        resp = await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_session_key_on_admin_endpoint_rejected(self, client):
        # [FUNCTION DOC]
        # A session key used on the admin-only /rl/start_session endpoint returns 403.
        """POST /rl/start_session with session key → 403."""
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t1"},
            headers=session_headers(),
        )
        assert resp.status_code == 403


class TestAnthropicMessages:
    # [CLASS DOC]
    # Tests for the Anthropic-compatible /v1/messages endpoint.
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_x_api_key_routes_non_streaming_messages(
        self,
        mock_query_router,
        mock_forward,
        client,
    ):
        # [FUNCTION DOC]
        # Non-streaming /v1/messages authenticated via x-api-key: router is queried with
        # the
        # session key and path, request is forwarded to the worker's /v1/messages, and
        # the
        # x-api-key and anthropic-version headers are passed through; response body
        # returned.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"id": "msg-1", "type": "message", "content": []},
        )

        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=anthropic_headers(),
        )

        assert response.status_code == 200
        assert response.json()["id"] == "msg-1"
        assert mock_query_router.await_args.args[1] == SESSION_KEY
        assert mock_query_router.await_args.args[2] == "/v1/messages"
        assert mock_forward.await_args.args[0] == f"{WORKER_ADDR}/v1/messages"
        forwarded_headers = mock_forward.await_args.args[2]
        assert forwarded_headers["x-api-key"] == SESSION_KEY
        assert forwarded_headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_sse_stream")
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_streaming_messages_forward_anthropic_sse(
        self,
        mock_query_router,
        mock_forward_sse,
        client,
    ):
        # [FUNCTION DOC]
        # Streaming /v1/messages: forward_sse_stream is mocked to yield Anthropic SSE
        # events; the
        # gateway returns a text/event-stream response containing them and targets the
        # worker's
        # /v1/messages URL.
        mock_query_router.return_value = WORKER_ADDR

        async def _stream():
            # [FUNCTION DOC]
            # HELPER: Async generator yielding two fake Anthropic SSE event chunks.
            yield b"event: message_start\ndata: {}\n\n"
            yield b"event: message_stop\ndata: {}\n\n"

        mock_forward_sse.return_value = _stream()

        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=anthropic_headers(),
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "event: message_start" in response.text
        assert mock_forward_sse.call_args.args[0] == f"{WORKER_ADDR}/v1/messages"


# =============================================================================
# Admin endpoints
# =============================================================================


class TestAdminEndpoints:
    # [CLASS DOC]
    # Tests for admin-key endpoints (chat, start_session, export_trajectories).
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_non_streaming(
        self, mock_query_router, mock_forward, client
    ):
        # [FUNCTION DOC]
        # Admin non-streaming /chat/completions is routed then forwarded; worker JSON
        # returned.
        """Admin key → /chat/completions (non-streaming) → response forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        # Simulate data proxy response
        mock_resp = httpx.Response(
            200,
            json={"id": "chatcmpl-1", "choices": [{"message": {"content": "Hi"}}]},
        )
        mock_forward.return_value = mock_resp

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-1"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_sse_stream")
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_admin_chat_completions_streaming(
        self, mock_query_router, mock_forward_sse, client
    ):
        # [FUNCTION DOC]
        # Admin streaming /chat/completions returns a text/event-stream response from
        # the mocked
        # SSE forwarder.
        """Admin key → /chat/completions (streaming) → SSE forwarded."""
        mock_query_router.return_value = WORKER_ADDR

        async def _stream():
            # [FUNCTION DOC]
            # HELPER: Async generator yielding one OpenAI delta chunk and the [DONE]
            # sentinel.
            yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_forward_sse.return_value = _stream()

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_creates_and_registers(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        # [FUNCTION DOC]
        # Admin /rl/start_session: router queried with no api_key, path
        # /rl/start_session and the
        # admin key; worker 201 response returned as-is; session list (key + id) is
        # registered in
        # the router against the chosen worker address.
        """Admin key → /rl/start_session → forwarded, response intercepted, session registered."""
        mock_query_router.return_value = WORKER_ADDR
        session_resp_data = {
            "group_id": "grp-test-1",
            "sessions": [{"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}],
        }
        mock_forward.return_value = httpx.Response(201, json=session_resp_data)

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "task-1"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["group_id"] == "grp-test-1"
        assert data["sessions"] == [
            {"session_id": "task-1-0", "session_api_key": "sess-key-xyz"}
        ]

        route_call = mock_query_router.call_args
        assert route_call.args == ("http://mock-router:8081",)
        assert route_call.kwargs["api_key"] is None
        assert route_call.kwargs["path"] == "/rl/start_session"
        assert route_call.kwargs["admin_api_key"] == ADMIN_KEY

        # Verify router registration
        mock_register.assert_called_once()
        reg_args = mock_register.call_args
        assert reg_args.args[0] == "http://mock-router:8081"  # router_addr
        assert reg_args.args[1] == [
            {"session_api_key": "sess-key-xyz", "session_id": "task-1-0"}
        ]  # sessions_list
        assert reg_args.args[2] == WORKER_ADDR  # worker_addr

    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_router_registration_fails(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        # [FUNCTION DOC]
        # When router registration raises RouterUnreachableError after session creation,
        # the
        # gateway returns 502 with "registration failed" in the error.
        """If router registration fails after session creation → 502."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-2",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )
        mock_register.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t"},
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "registration failed" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_admin_export_trajectories(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        # [FUNCTION DOC]
        # Admin /export_trajectories with a group_id routes by the first session id,
        # forwards,
        # and revokes the session in the router.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"interactions": []})

        resp = await client.post(
            "/export_trajectories",
            json={
                "session_ids": ["task-1-0"],
                "group_id": "grp-test",
                "discount": 1.0,
                "style": "sft",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_query_router.assert_called_once()
        assert mock_query_router.call_args.kwargs["session_id"] == "task-1-0"
        mock_revoke.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    async def test_online_export_without_group_id_skips_revoke(
        self, mock_forward, mock_query_router, mock_revoke, client
    ):
        # [FUNCTION DOC]
        # Online-mode export (no group_id, trajectory_id given) succeeds without
        # revoking any
        # session in the router.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"interactions": []})

        resp = await client.post(
            "/export_trajectories",
            json={
                "session_ids": ["__hitl__"],
                "trajectory_id": 0,
                "discount": 1.0,
                "style": "individual",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        mock_revoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_export_trajectories_missing_session_id(self, client):
        # [FUNCTION DOC]
        # /export_trajectories without session_ids returns 400 mentioning session_ids.
        """Admin key → /export_trajectories without session_id → 400."""
        resp = await client.post(
            "/export_trajectories",
            json={"discount": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 400
        assert "session_ids" in resp.json()["error"]


# =============================================================================
# Session endpoints
# =============================================================================


class TestSessionEndpoints:
    # [CLASS DOC]
    # Tests for session-key (and admin) calls routed to a pinned worker.
    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_chat_completions(
        self, mock_query_router, mock_forward, client
    ):
        # [FUNCTION DOC]
        # Session-key /chat/completions is forwarded to the routed worker and its JSON
        # returned.
        """Session key → /chat/completions → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"id": "chatcmpl-2", "choices": [{"message": {"content": "OK"}}]},
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "chatcmpl-2"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward_finish(
        self, mock_query_router, mock_forward, client
    ):
        # [FUNCTION DOC]
        # Session-key /rl/set_reward with finish=True is forwarded exactly once.
        """Session key → /rl/set_reward (finish=True) → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={"message": "success", "interaction_count": 5, "finished": True},
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 0.0, "finish": True},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_set_reward(self, mock_query_router, mock_forward, client):
        # [FUNCTION DOC]
        # Session-key /rl/set_reward is forwarded exactly once and returns 200.
        """Session key → /rl/set_reward → forwarded to pinned worker."""
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"message": "success"})

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_returns_ready_transition_without_router_notification(
        self, mock_query_router, mock_forward, client
    ):
        # [FUNCTION DOC]
        # /rl/set_reward passes through a worker response with ready_transition=True
        # unchanged
        # (no extra router notification is asserted).
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": True,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_set_reward_duplicate_ready_transition_is_forwarded_as_is(
        self, mock_query_router, mock_forward, client
    ):
        # [FUNCTION DOC]
        # /rl/set_reward passes through ready_transition=False (duplicate ready)
        # unchanged.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "message": "success",
                "session_id": "__hitl__",
                "trajectory_id": 0,
                "trajectory_ready": True,
                "ready_transition": False,
            },
        )

        resp = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["ready_transition"] is False


# =============================================================================
# Broadcast endpoints
# =============================================================================


class TestBroadcast:
    # [CLASS DOC]
    # Tests for per-worker pause/continue generation endpoints.
    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_generation_targets_worker(
        self, mock_resolve, mock_broadcast, client
    ):
        # [FUNCTION DOC]
        # /pause_generation/{worker_id} resolves the worker address and broadcasts to
        # that single
        # worker; returns one ok result.
        """Admin key → /pause_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {"worker_addr": WORKER_ADDR, "status": 200, "ok": True},
        ]

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["ok"] is True
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_continue_generation_targets_worker(
        self, mock_resolve, mock_broadcast, client
    ):
        # [FUNCTION DOC]
        # /continue_generation/{worker_id} resolves the worker and returns one result.
        """Admin key → /continue_generation/{worker_id} → resolves and targets single worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {"worker_addr": WORKER_ADDR, "status": 200, "ok": True},
        ]

        resp = await client.post(
            "/continue_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.broadcast_to_workers", new_callable=AsyncMock)
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_pause_worker_broadcast_failure(
        self, mock_resolve, mock_broadcast, client
    ):
        # [FUNCTION DOC]
        # A failing worker during pause is reported as ok=False in a 200 response.
        """Worker returns error → response shows ok=False for that worker."""
        mock_resolve.return_value = WORKER_ADDR
        mock_broadcast.return_value = [
            {
                "worker_addr": WORKER_ADDR,
                "status": 502,
                "ok": False,
                "error": "Connection refused",
            },
        ]

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["ok"] is False


# =============================================================================
# Router errors
# =============================================================================


class TestRouterErrors:
    # [CLASS DOC]
    # Tests mapping router failures to gateway HTTP errors.
    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_router_unreachable_502(self, mock_query_router, client):
        # [FUNCTION DOC]
        # RouterUnreachableError from query_router yields 502 with the message in
        # "error".
        """Router down → 502 JSON error."""
        mock_query_router.side_effect = RouterUnreachableError(
            "Router unreachable: connect timeout"
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 502
        assert "Router unreachable" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_no_healthy_workers_503(self, mock_query_router, client):
        # [FUNCTION DOC]
        # RouterKeyRejectedError with status 503 yields 503 with the message.
        """Router returns 503 (no healthy workers) → gateway returns 503."""
        mock_query_router.side_effect = RouterKeyRejectedError(
            "No healthy workers", 503
        )

        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No healthy workers" in resp.json()["error"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.resolve_worker_addr", new_callable=AsyncMock)
    async def test_router_unreachable_on_pause(self, mock_resolve, client):
        # [FUNCTION DOC]
        # RouterUnreachableError while resolving a worker for pause yields 502.
        """Router unreachable when resolving worker for pause → 502."""
        mock_resolve.side_effect = RouterUnreachableError("Router down")

        resp = await client.post(
            "/pause_generation/some-worker-id",
            content=b"{}",
            headers=admin_headers(),
        )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_session_not_found_for_export(self, mock_query_router, client):
        # [FUNCTION DOC]
        # RouterKeyRejectedError 404 (session not found) on export_trajectories yields
        # 401.
        """Session not found for export_trajectories → 401."""
        mock_query_router.side_effect = RouterKeyRejectedError("Session not found", 404)

        resp = await client.post(
            "/export_trajectories",
            json={"session_ids": ["nonexistent"], "discount": 1.0, "style": "sft"},
            headers=admin_headers(),
        )
        assert resp.status_code == 401
        assert "Session not found" in resp.json()["error"]


# =============================================================================
# Capacity enforcement — /rl/start_session with capacity checks
# =============================================================================


class TestStartSessionCapacity:
    # [CLASS DOC]
    # Tests for the start_session call sequence (capacity handled by router, not
    # gateway).
    @pytest.mark.asyncio
    @patch(f"{MODULE}.register_session_in_router", new_callable=AsyncMock)
    @patch(f"{MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{MODULE}.query_router", new_callable=AsyncMock)
    async def test_start_session_full_flow(
        self, mock_query_router, mock_forward, mock_register, client
    ):
        # [FUNCTION DOC]
        # Full start_session flow returns 201 and calls route, forward and register
        # exactly once
        # each.
        """Full flow: route → forward → register session.

        Gateway no longer manages capacity — that is handled by the
        router's ``/register_session`` endpoint.
        """
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            201,
            json={
                "group_id": "grp-test-3",
                "sessions": [{"session_id": "t-0", "session_api_key": "k"}],
            },
        )

        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "t"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201

        # Verify call order: route → forward → register
        mock_query_router.assert_called_once()
        mock_forward.assert_called_once()
        mock_register.assert_called_once()
