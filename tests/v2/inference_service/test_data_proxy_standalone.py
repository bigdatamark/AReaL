# [MODULE DOC]
# Unit tests for data-proxy standalone mode: /chat/completions requests whose bearer
# token is absent, the admin key, or an unknown key are served without a session and
# without caching (areal_cache None); a valid session key still uses the session flow.
# Uses an in-process ASGI client with a mocked tokenizer and OpenAI-style client.
"""Unit tests for data proxy standalone mode (no-session requests).

Tests that /chat/completions works without a session.
If the bearer token is a known session key, use session cache;
otherwise (no auth, admin key, unknown key) → standalone mode (no caching).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionStore

# =============================================================================
# Fixtures
# =============================================================================

ADMIN_KEY = "areal-admin-key"


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: DataProxyConfig on 127.0.0.1:18082 with fake backend
    # "http://mock-sglang:30000", tokenizer "mock-tokenizer" and 10s timeout; other
    # fields (including the admin key, assumed to equal ADMIN_KEY) use defaults.
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
    # Fixture: MagicMock tokenizer with async tokenize -> [101, 102, 103], decode_token
    # -> "tok_<id>", decode_tokens -> "hello world", async apply_chat_template -> [100,
    # 200, 300], eos 2 / pad 0 (also on inner _tok).
    tok = MagicMock()
    tok.tokenize = AsyncMock(return_value=[101, 102, 103])
    tok.decode_token = MagicMock(side_effect=lambda tid: f"tok_{tid}")
    tok.decode_tokens = MagicMock(return_value="hello world")
    tok.apply_chat_template = AsyncMock(return_value=[100, 200, 300])
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    # [FUNCTION DOC]
    # Fixture: MagicMock client whose chat.completions.create is an AsyncMock always
    # returning one prebuilt ChatCompletion (id "chatcmpl-standalone-test", content
    # "Hello!").
    """Mock ArealOpenAI client that returns a valid ChatCompletion."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    mock_client = MagicMock()

    completion = ChatCompletion(
        id="chatcmpl-standalone-test",
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

    async def _mock_create(*, areal_cache=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake create(); ignores areal_cache and kwargs and returns the fixed
        # completion.
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def client(config, mock_tokenizer, mock_areal_client):
    # [FUNCTION DOC]
    # Async fixture: creates the app, a PauseState and a real InfBridge
    # (SGLangBridgeBackend, 0.01s resubmit wait), injects tokenizer, inf_bridge,
    # areal_client, pause_state, config and a fresh SessionStore into app.state, and
    # yields an httpx AsyncClient on "http://test" with no default auth header. App
    # exceptions propagate (raise_app_exceptions default).
    """Create app with mocked deps and yield an httpx async client (no auth header)."""
    from areal.v2.inference_service.data_proxy.pause import PauseState
    from areal.v2.inference_service.inf_bridge import InfBridge
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_app(config)
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    transport = httpx.ASGITransport(app=app)
    # No default auth header — tests supply their own
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization header dict carrying the admin bearer key
    # ADMIN_KEY.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


# =============================================================================
# Standalone /chat/completions (no session)
# =============================================================================


class TestStandaloneChat:
    # [CLASS DOC]
    # Tests for session-less /chat/completions requests and backend selection.
    def test_config_can_select_vllm_backend(self):
        # [FUNCTION DOC]
        # Builds a DataProxyConfig with backend_type "vllm" and asserts the private
        # _create_inf_bridge factory returns a bridge whose backend is a
        # VLLMBridgeBackend.
        """backend_type=vllm creates a vLLM-backed InfBridge."""
        from areal.v2.inference_service.data_proxy.app import (
            _create_inf_bridge,
        )
        from areal.v2.inference_service.data_proxy.pause import PauseState
        from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend

        config = DataProxyConfig(
            host="127.0.0.1",
            port=18082,
            backend_addr="http://mock-vllm:30000",
            backend_type="vllm",
            tokenizer_path="mock-tokenizer",
        )

        bridge = _create_inf_bridge(config.backend_addr, PauseState(), config)

        assert isinstance(bridge.backend, VLLMBridgeBackend)

    @pytest.mark.asyncio
    async def test_no_auth_chat_completions_returns_valid_response(self, client):
        # [FUNCTION DOC]
        # Posts a chat request with no auth header; asserts 200 and the mocked
        # id/content.
        """No auth header → standalone mode, returns valid response."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_admin_key_chat_completions_returns_valid_response(self, client):
        # [FUNCTION DOC]
        # Posts a chat request with the admin key; asserts 200 and the mocked
        # id/content.
        """Admin key → standalone mode, returns valid response."""
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
        assert data["id"] == "chatcmpl-standalone-test"
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_unknown_key_chat_completions_returns_valid_response(self, client):
        # [FUNCTION DOC]
        # Posts with an unknown bearer key; asserts it is served (200), not rejected.
        """Unknown key → standalone mode (not rejected)."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer unknown-key-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-standalone-test"

    @pytest.mark.asyncio
    async def test_standalone_chat_no_session_created(self, client):
        # [FUNCTION DOC]
        # Posts without auth and asserts the app's SessionStore still has session_count
        # 0 (reaches into the ASGI transport's app via private attributes).
        """Standalone mode does NOT create a session."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        store: SessionStore = client._transport.app.state.session_store  # type: ignore[attr-defined]
        assert store.session_count == 0

    @pytest.mark.asyncio
    async def test_standalone_chat_passes_none_cache(self, client, mock_areal_client):
        # [FUNCTION DOC]
        # Posts without auth and asserts create was called exactly once with areal_cache
        # None.
        """Standalone mode passes areal_cache=None (no caching)."""
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        # Verify create was called with areal_cache=None
        mock_areal_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_areal_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("areal_cache") is None


# =============================================================================
# Session-key flows unchanged
# =============================================================================


class TestSessionKeyUnchanged:
    # [CLASS DOC]
    # Tests that the session-key flow still works alongside standalone mode.
    @pytest.mark.asyncio
    async def test_session_chat_completions_still_works(
        self, client, mock_areal_client
    ):
        # [FUNCTION DOC]
        # Starts a session via POST /rl/start_session (admin key, task_id "test-task"),
        # expects 201, takes sessions[0].session_api_key, then posts a chat request with
        # that key and asserts 200 with content "Hello!". Does not assert that caching
        # was actually used.
        """Session key callers still use the session-based flow."""
        # Start a session first
        resp = await client.post(
            "/rl/start_session",
            json={"task_id": "test-task"},
            headers=admin_headers(),
        )
        assert resp.status_code == 201
        session_api_key = resp.json()["sessions"][0]["session_api_key"]

        # Now use session key for chat completions
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": f"Bearer {session_api_key}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
