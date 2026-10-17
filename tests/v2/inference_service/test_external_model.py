# [MODULE DOC]
# Unit tests for external (third-party OpenAI-compatible) model support across the v2
# inference service: the router's ModelRegistry and /register_model, /route, /models
# endpoints; the gateway's /register_model, /chat/completions (external, streaming and
# internal fallback), /models and /export_trajectories with router/forward helpers
# patched in ROUTER_MODULE; the data proxy's external registration, chat proxying
# (streaming and not), human-in-the-loop "__hitl__" trajectory export and provider API
# key injection; and a gateway->router end-to-end registration flow. All apps run
# in-process via httpx.ASGITransport; no real network.
from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.data_proxy.app import (
    create_app as create_data_proxy_app,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionStore
from areal.v2.inference_service.gateway.app import (
    create_app as create_gateway_app,
)
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
)
from areal.v2.inference_service.router.app import (
    create_app as create_router_app,
)
from areal.v2.inference_service.router.config import RouterConfig
from areal.v2.inference_service.router.state import ModelRegistry

ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key-abc123"
WORKER_ADDR = "http://worker-1:18082"
ROUTER_MODULE = "areal.v2.inference_service.gateway.app"


def admin_headers() -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization bearer header dict for ADMIN_KEY.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers() -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization bearer header dict for SESSION_KEY.
    return {"Authorization": f"Bearer {SESSION_KEY}"}


@pytest.fixture
def router_config() -> RouterConfig:
    # [FUNCTION DOC]
    # Fixture: RouterConfig on 127.0.0.1:18081 with ADMIN_KEY, 999s poll interval (no
    # effective health polling) and round_robin routing.
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def router_client(router_config: RouterConfig):
    # [FUNCTION DOC]
    # Async fixture: yields an httpx AsyncClient over a fresh router app
    # ("http://test").
    app = create_router_app(router_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestModelRegistry:
    # [CLASS DOC]
    # Unit tests for the router's in-memory ModelRegistry.
    @pytest.mark.asyncio
    async def test_model_registry_register_get_list_remove(self):
        # [FUNCTION DOC]
        # Registers "ext-a" (url "http://api", no key, one data proxy) and asserts get()
        # returns matching name/url/data_proxy_addrs, list_names() is ["ext-a"],
        # remove() returns True and a later get() returns None.
        reg = ModelRegistry()
        await reg.register("ext-a", "http://api", None, [WORKER_ADDR])

        info = await reg.get("ext-a")
        assert info is not None
        assert info.name == "ext-a"
        assert info.url == "http://api"
        assert info.data_proxy_addrs == [WORKER_ADDR]

        names = await reg.list_names()
        assert names == ["ext-a"]

        removed = await reg.remove("ext-a")
        assert removed is True
        assert await reg.get("ext-a") is None


class TestRouterExternalEndpoints:
    # [CLASS DOC]
    # Tests for the router's external-model HTTP endpoints.
    @pytest.mark.asyncio
    async def test_register_model_success(self, router_client):
        # [FUNCTION DOC]
        # Registers a worker, then registers model "ext-1" bound to it; asserts 200 with
        # status "ok", model "ext-1" and data_proxy_addrs [WORKER_ADDR].
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )

        resp = await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "ext-1"
        assert data["data_proxy_addrs"] == [WORKER_ADDR]

    @pytest.mark.asyncio
    async def test_register_model_no_workers_503(self, router_client):
        # [FUNCTION DOC]
        # Registers a model with no workers registered and no data_proxy_addrs; asserts
        # 503.
        resp = await router_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
            headers=admin_headers(),
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_register_model_no_auth_401(self, router_client):
        # [FUNCTION DOC]
        # Posts /register_model without auth; asserts 401.
        resp = await router_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_route_model_success(self, router_client):
        # [FUNCTION DOC]
        # Registers a worker and model, then POST /route {"model": "ext-1"}; asserts 200
        # with worker_addr WORKER_ADDR and url "http://ext-api".
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )
        await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )

        resp = await router_client.post(
            "/route",
            json={"model": "ext-1"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_addr"] == WORKER_ADDR
        assert data["url"] == "http://ext-api"

    @pytest.mark.asyncio
    async def test_route_model_not_found_404(self, router_client):
        # [FUNCTION DOC]
        # Asserts routing an unknown model returns 404.
        resp = await router_client.post(
            "/route",
            json={"model": "nope"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_models_empty(self, router_client):
        # [FUNCTION DOC]
        # Asserts GET /models returns an empty "models" list initially.
        resp = await router_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    @pytest.mark.asyncio
    async def test_list_models_after_registration(self, router_client):
        # [FUNCTION DOC]
        # Registers a worker and model; asserts GET /models returns ["ext-1"].
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )
        await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )

        resp = await router_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == ["ext-1"]


@pytest.fixture
def gateway_config() -> GatewayConfig:
    # [FUNCTION DOC]
    # Fixture: GatewayConfig on 127.0.0.1:18080 with ADMIN_KEY, router
    # "http://mock-router:8081", 2s router timeout, 30s forward timeout.
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8081",
        router_timeout=2.0,
        forward_timeout=30.0,
    )


@pytest_asyncio.fixture
async def gateway_client(gateway_config: GatewayConfig):
    # [FUNCTION DOC]
    # Async fixture: yields an httpx AsyncClient over a fresh gateway app
    # ("http://test").
    app = create_gateway_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestGatewayExternalEndpoints:
    # [CLASS DOC]
    # Gateway external-model endpoint tests with router/forward helpers mocked.
    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.register_model_in_router", new_callable=AsyncMock)
    async def test_register_model_gateway_full_flow(
        self,
        mock_register_model,
        mock_forward,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # Mocks register_model_in_router (returns data_proxy_addrs [WORKER_ADDR]) and
        # forward_request (200); posts /register_model and asserts 200, the returned
        # data_proxy_addrs, and that both the router registration and one forward to the
        # data proxy occurred.
        mock_register_model.return_value = {
            "status": "ok",
            "model": "ext-1",
            "data_proxy_addrs": [WORKER_ADDR],
        }
        mock_forward.return_value = httpx.Response(200, json={"status": "ok"})

        resp = await gateway_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["data_proxy_addrs"] == [WORKER_ADDR]
        mock_register_model.assert_called_once()
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_external_model(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # Router returns WORKER_ADDR, forward returns {"id": "ext-chat-1"}; asserts a
        # chat request for model "ext-1" returns that body and was forwarded to a
        # /chat/completions URL.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "ext-chat-1"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "ext-chat-1"
        assert "/chat/completions" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_sse_stream")
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_external_model_streaming(
        self,
        mock_query_router,
        mock_forward_sse,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # Router returns WORKER_ADDR and forward_sse_stream returns a two-chunk async
        # generator; asserts a streaming chat request returns 200 with a
        # text/event-stream content type.
        mock_query_router.return_value = WORKER_ADDR

        async def _stream() -> AsyncGenerator[bytes, None]:
            # [FUNCTION DOC]
            # HELPER: Async generator yielding "data: hello" and "data: [DONE]" SSE
            # chunks.
            yield b"data: hello\n\n"
            yield b"data: [DONE]\n\n"

        mock_forward_sse.return_value = _stream()

        resp = await gateway_client.post(
            "/chat/completions",
            json={
                "model": "ext-1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_unregistered_model_falls_back(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # With a session key and unknown model "missing", asserts the request still
        # succeeds via the internal path (forwarded to /chat/completions, id
        # "internal-chat").
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "internal-chat"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "internal-chat"
        assert "/chat/completions" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_no_model_internal_path(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # With a session key and no model field, asserts the internal path returns 200
        # with id "internal-chat".
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "internal-chat"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "internal-chat"

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.list_models_from_router", new_callable=AsyncMock)
    async def test_list_models_gateway(self, mock_list_models, gateway_client):
        # [FUNCTION DOC]
        # Mocks list_models_from_router to return two names; asserts GET /models returns
        # them.
        mock_list_models.return_value = ["ext-1", "ext-2"]

        resp = await gateway_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == ["ext-1", "ext-2"]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_trajectories_routes_external_by_session_id(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        # [FUNCTION DOC]
        # Router returns WORKER_ADDR and forward returns a traj payload; asserts
        # exporting session_ids ["ext-1"] returns 200 and forwards to an
        # /export_trajectories URL.
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "traj": {
                    "interactions": [{"request": [], "response": "", "reward": 0.0}]
                },
            },
        )

        resp = await gateway_client.post(
            "/export_trajectories",
            json={"session_ids": ["ext-1"]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "/export_trajectories" in mock_forward.call_args.args[0]


@pytest.fixture
def data_proxy_config() -> DataProxyConfig:
    # [FUNCTION DOC]
    # Fixture: DataProxyConfig on 127.0.0.1:18082 with fake SGLang backend, tokenizer
    # "mock-tokenizer", 10s timeout, default admin key ("areal-admin-key", as used by
    # tests).
    return DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
    )


@pytest.fixture
def mock_tokenizer():
    # [FUNCTION DOC]
    # Fixture: minimal MagicMock tokenizer exposing only _tok.eos_token_id 2 and
    # _tok.pad_token_id 0.
    tok = MagicMock()
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    # [FUNCTION DOC]
    # Fixture: MagicMock client whose chat.completions.create is an AsyncMock returning
    # a fixed ChatCompletion (id "chatcmpl-mock", content "Hello!").
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=ChatCompletion(
            id="chatcmpl-mock",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
    )
    return mock_client


@pytest_asyncio.fixture
async def data_proxy_app(data_proxy_config, mock_tokenizer, mock_areal_client):
    # [FUNCTION DOC]
    # Async fixture: creates the data proxy app, a PauseState and a real InfBridge
    # (SGLangBridgeBackend, 5 retries, 0.01s wait), injects tokenizer, inf_bridge,
    # areal_client, pause_state, config, a SessionStore keyed with the config admin key,
    # version 0 and a real httpx.AsyncClient (10s timeout) as http_client; yields the
    # app and closes the http client afterwards.
    # KNOWN ISSUE: app.state.http_client is first set to a MagicMock and then
    # immediately overwritten by the real client (dead assignment).
    from areal.v2.inference_service.data_proxy.pause import PauseState
    from areal.v2.inference_service.inf_bridge import InfBridge
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_data_proxy_app(data_proxy_config)
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=data_proxy_config.backend_addr,
        pause_state=pause_state,
        request_timeout=data_proxy_config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = data_proxy_config
    store = SessionStore()
    store.set_admin_key(data_proxy_config.admin_api_key)
    app.state.session_store = store
    app.state.http_client = MagicMock()
    app.state.version = 0
    http_client = httpx.AsyncClient(timeout=10.0)
    app.state.http_client = http_client
    yield app
    await http_client.aclose()


@pytest_asyncio.fixture
async def data_proxy_client(data_proxy_app):
    # [FUNCTION DOC]
    # Async fixture: yields an httpx AsyncClient over the data proxy app
    # ("http://test").
    transport = httpx.ASGITransport(app=data_proxy_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestDataProxyExternalEndpoints:
    # [CLASS DOC]
    # Tests for the data proxy's external-model endpoints and HITL trajectory capture.
    @pytest.mark.asyncio
    async def test_register_external_model(self, data_proxy_client):
        # [FUNCTION DOC]
        # Registers external model "ext-1" (upstream model "gpt-4o") with the admin key;
        # asserts 200 and status "ok".
        resp = await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_register_external_model_without_admin_key_401(
        self, data_proxy_client
    ):
        # [FUNCTION DOC]
        # Registers without auth; asserts 401.
        resp = await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_external_chat_completions_non_streaming(
        self,
        data_proxy_client,
        data_proxy_app,
        monkeypatch,
    ):
        # [FUNCTION DOC]
        # Registers "ext-1", swaps app.state.http_client for _FakeClient, sends a
        # non-streaming chat; asserts 200 with id "ext-1-response". Then asserts
        # exporting "__hitl__" with remove_session False before a reward yields traj {},
        # that /rl/set_reward 1.0 reports trajectory_ready True, and that the export
        # then contains exactly one interaction.
        await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
            headers={"Authorization": "Bearer areal-admin-key"},
        )

        class _FakeClient:
            # [CLASS DOC]
            # HELPER: Fake shared HTTP client for non-streaming external calls.
            async def post(self, url, json=None, headers=None):
                # [FUNCTION DOC]
                # HELPER: Asserts the URL is "http://ext-api/chat/completions" and the
                # forwarded body's model was rewritten to "gpt-4o"; returns a 200
                # response {"id": "ext-1-response"}.
                assert url == "http://ext-api/chat/completions"
                assert json["model"] == "gpt-4o"
                return httpx.Response(200, json={"id": "ext-1-response"})

        data_proxy_app.state.http_client = _FakeClient()

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "ext-1-response"

        not_ready = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"], "remove_session": False},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert not_ready.status_code == 200
        assert not_ready.json()["traj"] == {}

        set_reward = await data_proxy_client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert set_reward.status_code == 200
        assert set_reward.json()["trajectory_ready"] is True

        exported = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported.status_code == 200
        payload = exported.json()
        assert len(payload["traj"]["interactions"]) == 1

    @pytest.mark.asyncio
    async def test_external_chat_completions_streaming(
        self,
        data_proxy_client,
        monkeypatch,
    ):
        # [FUNCTION DOC]
        # Registers "ext-1" and monkeypatches httpx.AsyncClient in the data proxy module
        # with _FakeClient; sends a streaming chat and asserts a 200 text/event-stream
        # response. Then sets reward 1.0 (trajectory_ready True), asserts the first
        # "__hitl__" export has one interaction and a second export returns traj {}
        # (exported session is consumed).
        await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
            headers={"Authorization": "Bearer areal-admin-key"},
        )

        class _FakeStreamResponse:
            # [CLASS DOC]
            # HELPER: Fake streamed upstream response: status 200, async context
            # manager, aread() returns empty bytes, aiter_bytes() yields "chunk-1" and
            # "[DONE]" SSE frames.
            status_code = 200

            async def __aenter__(self):
                # [FUNCTION DOC]
                # HELPER: Returns self.
                return self

            async def __aexit__(self, exc_type, exc, tb):
                # [FUNCTION DOC]
                # HELPER: Returns False (do not suppress exceptions).
                return False

            async def aread(self):
                # [FUNCTION DOC]
                # HELPER: Returns empty bytes.
                return b""

            async def aiter_bytes(self):
                # [FUNCTION DOC]
                # HELPER: Async generator of two SSE byte chunks.
                yield b"data: chunk-1\n\n"
                yield b"data: [DONE]\n\n"

        class _FakeClient:
            # [CLASS DOC]
            # HELPER: Fake httpx.AsyncClient class used for streaming; constructor args
            # ignored.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Ignores all arguments.
                pass

            async def __aenter__(self):
                # [FUNCTION DOC]
                # HELPER: Returns self.
                return self

            async def __aexit__(self, exc_type, exc, tb):
                # [FUNCTION DOC]
                # HELPER: Returns False (do not suppress exceptions).
                return False

            def stream(self, method, url, json=None, headers=None):
                # [FUNCTION DOC]
                # HELPER: Asserts a POST to "http://ext-api/chat/completions" with model
                # "gpt-4o" and returns a _FakeStreamResponse.
                assert method == "POST"
                assert url == "http://ext-api/chat/completions"
                assert json["model"] == "gpt-4o"
                return _FakeStreamResponse()

        monkeypatch.setattr(
            "areal.v2.inference_service.data_proxy.app.httpx.AsyncClient",
            _FakeClient,
        )

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={
                "model": "ext-1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        set_reward = await data_proxy_client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert set_reward.status_code == 200
        assert set_reward.json()["trajectory_ready"] is True

        exported = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported.status_code == 200
        payload = exported.json()
        assert len(payload["traj"]["interactions"]) == 1

        exported_again = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported_again.status_code == 200
        assert exported_again.json()["traj"] == {}

    @pytest.mark.asyncio
    async def test_unregistered_model_falls_through_to_internal(
        self, data_proxy_client
    ):
        # [FUNCTION DOC]
        # Sends a chat for unregistered model "missing" without auth; asserts 200, i.e.
        # it falls through to the internal (mocked areal_client) standalone path.
        resp = await data_proxy_client.post(
            "/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_external_chat_uses_stored_provider_api_key(
        self,
        data_proxy_client,
        data_proxy_app,
        monkeypatch,
    ):
        # [FUNCTION DOC]
        # Registers "ext-1" with provider api_key "sk-provider-key-99", swaps in a
        # header-capturing fake client, sends a chat with a session bearer key, and
        # asserts the upstream request carried "authorization: Bearer
        # sk-provider-key-99" (provider key replaces the caller's).
        await data_proxy_client.post(
            "/register_model",
            json={
                "name": "ext-1",
                "url": "http://ext-api",
                "model": "gpt-4o",
                "api_key": "sk-provider-key-99",
            },
            headers={"Authorization": "Bearer areal-admin-key"},
        )

        captured_headers: dict[str, str] = {}

        class _FakeClient:
            # [CLASS DOC]
            # HELPER: Fake client recording request headers into captured_headers.
            async def post(self, url, json=None, headers=None):
                # [FUNCTION DOC]
                # HELPER: Copies headers into captured_headers (closure mutation) and
                # returns 200 {"id": "ext-1-response"}.
                if headers:
                    captured_headers.update(headers)
                return httpx.Response(200, json={"id": "ext-1-response"})

        data_proxy_app.state.http_client = _FakeClient()

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-session-key"},
        )
        assert resp.status_code == 200
        assert captured_headers.get("authorization") == "Bearer sk-provider-key-99"


@pytest.mark.asyncio
async def test_external_model_end_to_end_register_then_chat(router_config):
    # [FUNCTION DOC]
    # End-to-end gateway + real router flow. Registers WORKER_ADDR in an in-process
    # router, builds a gateway app, and patches the gateway's register_model_in_router,
    # query_router and forward_request with fakes backed by the router client and a
    # proxy_state dict simulating the data proxy. Asserts gateway /register_model for
    # "ext-1" returns 200 and a subsequent chat for "ext-1" returns 200 with id
    # "ext-e2e".
    router_app = create_router_app(router_config)
    router_transport = httpx.ASGITransport(app=router_app)
    async with httpx.AsyncClient(
        transport=router_transport,
        base_url="http://router",
    ) as router_client:
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )

        gateway_config = GatewayConfig(
            host="127.0.0.1",
            port=18080,
            admin_api_key=ADMIN_KEY,
            router_addr="http://router",
            router_timeout=2.0,
            forward_timeout=30.0,
        )
        gateway_app = create_gateway_app(gateway_config)
        gateway_transport = httpx.ASGITransport(app=gateway_app)

        proxy_state: dict[str, dict[str, str | None]] = {}

        async def _register_model_in_router(
            router_addr: str,
            model: str,
            url: str,
            api_key: str | None,
            data_proxy_addrs: list[str],
            admin_api_key: str,
            timeout: float,
            *,
            client: httpx.AsyncClient | None = None,
        ) -> dict:
            # [FUNCTION DOC]
            # HELPER: Fake register_model_in_router; POSTs
            # model/url/api_key/data_proxy_addrs to the in-process router's
            # /register_model with admin headers, raises on HTTP error, returns the JSON
            # body. Ignores router_addr, admin key, timeout and client.
            del router_addr, admin_api_key, timeout, client
            resp = await router_client.post(
                "/register_model",
                json={
                    "model": model,
                    "url": url,
                    "api_key": api_key,
                    "data_proxy_addrs": data_proxy_addrs,
                },
                headers=admin_headers(),
            )
            resp.raise_for_status()
            return resp.json()

        async def _query_router(
            router_addr: str,
            api_key: str | None = None,
            path: str | None = None,
            timeout: float = 2.0,
            *,
            session_id: str | None = None,
            admin_api_key: str | None = None,
            model: str | None = None,
            client: httpx.AsyncClient | None = None,
        ) -> str:
            # [FUNCTION DOC]
            # HELPER: Fake query_router; builds a payload from model, api_key and
            # session_id (each if given), POSTs to the router's /route. For model
            # lookups maps 404/503 to RouterKeyRejectedError; otherwise raises on HTTP
            # error. Returns worker_addr.
            del router_addr, path, timeout, admin_api_key, client
            payload: dict[str, str] = {}
            if model is not None:
                payload["model"] = model
            if api_key is not None:
                payload["api_key"] = api_key
            if session_id is not None:
                payload["session_id"] = session_id
            resp = await router_client.post(
                "/route",
                json=payload,
                headers=admin_headers(),
            )
            if model is not None:
                if resp.status_code == 404:
                    raise RouterKeyRejectedError("not found", 404)
                if resp.status_code == 503:
                    raise RouterKeyRejectedError("no healthy workers", 503)
                resp.raise_for_status()
                return resp.json()["worker_addr"]
            resp.raise_for_status()
            return resp.json()["worker_addr"]

        async def _forward_request(
            upstream_url: str,
            body: bytes,
            headers: dict[str, str],
            timeout: float,
            *,
            client: httpx.AsyncClient | None = None,
        ) -> httpx.Response:
            # [FUNCTION DOC]
            # HELPER: Fake forward_request simulating the data proxy: a /register_model
            # POST records name -> {url, model} in proxy_state and returns 200 ok; a
            # /chat/completions POST asserts the model is registered and returns 200
            # {"id": "ext-e2e"}; anything else returns 500.
            del timeout, client
            if upstream_url == f"{WORKER_ADDR}/register_model":
                data = json.loads(body)
                proxy_state[data["name"]] = {
                    "url": data["url"],
                    "model": data.get("model"),
                }
                return httpx.Response(200, json={"status": "ok"})
            if upstream_url == f"{WORKER_ADDR}/chat/completions":
                data = json.loads(body)
                assert data["model"] in proxy_state
                return httpx.Response(200, json={"id": "ext-e2e"})
            return httpx.Response(500, json={"error": "unexpected"})

        with (
            patch(
                f"{ROUTER_MODULE}.register_model_in_router",
                new=AsyncMock(side_effect=_register_model_in_router),
            ),
            patch(
                f"{ROUTER_MODULE}.query_router",
                new=AsyncMock(side_effect=_query_router),
            ),
            patch(
                f"{ROUTER_MODULE}.forward_request",
                new=AsyncMock(side_effect=_forward_request),
            ),
        ):
            async with httpx.AsyncClient(
                transport=gateway_transport,
                base_url="http://gateway",
            ) as gateway_client:
                reg = await gateway_client.post(
                    "/register_model",
                    json={
                        "model": "ext-1",
                        "url": "http://ext-api",
                    },
                    headers=admin_headers(),
                )
                assert reg.status_code == 200

                chat = await gateway_client.post(
                    "/chat/completions",
                    json={
                        "model": "ext-1",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=admin_headers(),
                )
                assert chat.status_code == 200
                assert chat.json()["id"] == "ext-e2e"
