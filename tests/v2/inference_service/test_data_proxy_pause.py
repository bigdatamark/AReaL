# [MODULE DOC]
# Unit tests for data-proxy pause/resume: the PauseState async flag and the FastAPI
# endpoints /pause_generation, /continue_generation and the "paused" field of /health.
# The app is driven in-process through httpx.ASGITransport with mocked tokenizer,
# OpenAI-style client and InfBridge pause/resume (no real backend calls).
"""Unit tests for data proxy pause/resume (Plan 3c)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import SessionStore

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: returns a DataProxyConfig on 127.0.0.1:18083 pointing at a fake SGLang
    # backend "http://mock-sglang:30000" with tokenizer "mock-tokenizer", 10s request
    # timeout, 5 resubmit retries and a 0.01s resubmit wait (fast tests). No I/O.
    return DataProxyConfig(
        host="127.0.0.1",
        port=18083,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
        max_resubmit_retries=5,
        resubmit_wait=0.01,  # fast for tests
    )


@pytest.fixture
def mock_tokenizer():
    # [FUNCTION DOC]
    # Fixture: returns a MagicMock tokenizer: async tokenize -> [101, 102, 103],
    # decode_token -> "tok_<id>", decode_tokens -> "hello world", async
    # apply_chat_template -> [100, 200, 300], eos_token_id 2 and pad_token_id 0 (also on
    # the inner _tok mock).
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
    # Fixture: returns a MagicMock ArealOpenAI-like client whose chat.completions.create
    # is an AsyncMock delegating to the nested _mock_create.
    """Mock ArealOpenAI client that returns a valid ChatCompletion."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    mock_client = MagicMock()

    async def _mock_create(*, areal_cache=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake create(); builds a fixed ChatCompletion (id "chatcmpl-mock", one
        # "stop" choice with content "mocked response", usage 3/5/8). If areal_cache (a
        # dict) is given, also builds an InteractionWithTokenLogpReward with a cached
        # input_ids [100, 200, 300] and output_tokens [1234, 5678] tensors and the
        # completion, stored in areal_cache under the completion id (mutates the passed
        # dict). Extra kwargs ignored. Returns the completion.
        import torch

        completion = ChatCompletion(
            id="chatcmpl-mock",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content="mocked response", role="assistant"
                    ),
                )
            ],
            created=1700000000,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
        if areal_cache is not None:
            from areal.experimental.openai.types import (
                InteractionWithTokenLogpReward,
            )

            interaction = InteractionWithTokenLogpReward(
                messages=[{"role": "user", "content": "test"}],
            )
            interaction._cache = {
                "input_ids": torch.tensor([100, 200, 300]),
                "output_tokens": torch.tensor([1234, 5678]),
            }
            interaction.completion = completion
            cid = completion.id
            areal_cache[cid] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def app_client(config, mock_tokenizer, mock_areal_client):
    # [FUNCTION DOC]
    # Async fixture: creates the data proxy app via create_app(config), a fresh
    # PauseState and a real InfBridge (SGLangBridgeBackend, settings from config) whose
    # pause/resume are replaced by AsyncMocks that flip the shared PauseState to
    # True/False. Injects tokenizer, inf_bridge, areal_client, pause_state, config, a
    # new SessionStore and version 0 into app.state, then yields (httpx AsyncClient on
    # base "http://test", app, pause_state); app exceptions become 500 responses instead
    # of raising. Closes the client afterwards.
    """Create an ASGI test client with all app.state attributes injected."""
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
        resubmit_wait=config.resubmit_wait,
    )

    # Mock pause/resume so they don't make real HTTP calls to SGLang.
    # side_effect must be an async function (not a lambda returning a coroutine)
    # so that AsyncMock awaits the state change before the endpoint returns.
    async def _mock_pause():
        # [FUNCTION DOC]
        # HELPER: Pause side effect; awaits _set_paused(pause_state, True).
        await _set_paused(pause_state, True)

    async def _mock_resume():
        # [FUNCTION DOC]
        # HELPER: Resume side effect; awaits _set_paused(pause_state, False).
        await _set_paused(pause_state, False)

    inf_bridge.pause = AsyncMock(side_effect=_mock_pause)
    inf_bridge.resume = AsyncMock(side_effect=_mock_resume)

    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.version = 0

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app, pause_state


async def _set_paused(pause_state: PauseState, value: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Awaits pause_state.set_paused(value). Used by the mocked pause/resume.
    """Helper to set pause state in async mock side effects."""
    await pause_state.set_paused(value)


# =============================================================================
# PauseState unit tests
# =============================================================================


class TestPauseState:
    # [CLASS DOC]
    # Unit tests for PauseState flag transitions (no app involved).
    """Test PauseState flag transitions."""

    @pytest.mark.asyncio
    async def test_default_not_paused(self):
        # [FUNCTION DOC]
        # Asserts a new PauseState reports not paused.
        state = PauseState()
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_true(self):
        # [FUNCTION DOC]
        # Asserts set_paused(True) makes is_paused() True.
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_false(self):
        # [FUNCTION DOC]
        # Asserts pausing then unpausing returns is_paused() to False.
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()
        await state.set_paused(False)
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_multiple_transitions(self):
        # [FUNCTION DOC]
        # Asserts three consecutive pause/unpause cycles each report the expected state.
        state = PauseState()
        for _ in range(3):
            await state.set_paused(True)
            assert await state.is_paused()
            await state.set_paused(False)
            assert not await state.is_paused()


# =============================================================================
# Endpoint tests — /pause_generation and /continue_generation
# =============================================================================


class TestPauseResumeEndpoints:
    # [CLASS DOC]
    # Endpoint tests for /pause_generation, /continue_generation and /health's paused
    # flag.
    """Test POST /pause_generation and POST /continue_generation endpoints."""

    @pytest.mark.asyncio
    async def test_health_includes_paused_false(self, app_client):
        # [FUNCTION DOC]
        # Asserts GET /health returns 200 with "paused" present and False.
        client, app, pause_state = app_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "paused" in data
        assert data["paused"] is False

    @pytest.mark.asyncio
    async def test_pause_endpoint(self, app_client):
        # [FUNCTION DOC]
        # Asserts POST /pause_generation returns 200 {"status": "ok", "paused": True}
        # and that inf_bridge.pause was awaited exactly once.
        client, app, pause_state = app_client

        resp = await client.post("/pause_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is True

        # Verify inf_bridge.pause() was called
        app.state.inf_bridge.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_endpoint(self, app_client):
        # [FUNCTION DOC]
        # Pre-pauses the state, then asserts POST /continue_generation returns 200 with
        # status "ok" and paused False, and inf_bridge.resume was called once.
        client, app, pause_state = app_client

        # First pause
        await pause_state.set_paused(True)

        resp = await client.post("/continue_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is False

        # Verify inf_bridge.resume() was called
        app.state.inf_bridge.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_paused_true_after_pause(self, app_client):
        # [FUNCTION DOC]
        # Asserts /health reports paused True after POST /pause_generation.
        client, app, pause_state = app_client

        await client.post("/pause_generation")

        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["paused"] is True
