# [MODULE DOC]
# Unit tests for the data proxy's session and chat layer: SessionStore/SessionData
# semantics (session ids, api-key lookup, reward finalization with timeouts, HITL
# sessions, prefix matcher retention), trajectory remotization/flushing helpers, and the
# FastAPI endpoints /rl/start_session, /chat/completions, Anthropic-style /v1/messages,
# /rl/set_reward, /export_trajectories and related behaviours. Endpoints are driven
# in-process via httpx.ASGITransport with a mocked tokenizer and a mocked ArealOpenAI
# client that records interactions into the session cache; no real backend is contacted.
"""Unit tests for data proxy chat/session endpoints (Plan 3b)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.utils.seeding import derive_deterministic_seed
from areal.v2.inference_service.data_proxy.app import (
    _flush_ready_trajectories,
    _remotize_trajectory,
    create_app,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import (
    SessionData,
    SessionStore,
)

# =============================================================================
# Fixtures
# =============================================================================

ADMIN_KEY = "areal-admin-key"


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: DataProxyConfig on 127.0.0.1:18082 with fake SGLang backend, tokenizer
    # "mock-tokenizer", 10s timeout; admin key left at its default (ADMIN_KEY).
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
    # 200, 300], eos 2 / pad 0, also exposed on the inner _tok mock.
    tok = MagicMock()
    tok.tokenize = AsyncMock(return_value=[101, 102, 103])
    tok.decode_token = MagicMock(side_effect=lambda tid: f"tok_{tid}")
    tok.decode_tokens = MagicMock(return_value="hello world")
    tok.apply_chat_template = AsyncMock(return_value=[100, 200, 300])
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    # Expose underlying _tok for ModelResponse.output_tokens_without_stop
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    # [FUNCTION DOC]
    # Fixture: MagicMock ArealOpenAI-like client whose chat.completions.create is an
    # AsyncMock delegating to _mock_create (call counter kept in a closure).
    """Mock ArealOpenAI client that returns a valid ChatCompletion.
    Also stores the interaction in the session's InteractionCache.

    The mock has `.chat.completions.create()` as an AsyncMock to match
    the ArealOpenAI interface used by the data proxy app.
    """
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    mock_client = MagicMock()

    call_index = 0

    async def _mock_create(*, areal_cache=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake create(); builds a ChatCompletion with id "chatcmpl-test<N>" (N
        # increments per call), content "Hello!", created 1234567890+N and usage 3/5/8.
        # Builds an InteractionWithTokenLogpReward from kwargs["messages"] with that
        # completion and an assistant output message, pre-populating _cache with a
        # 6-token input_ids, loss_mask, logprobs, versions, attention_mask and a zero
        # rewards tensor so to_tensor_dict works. If areal_cache is given, stores the
        # interaction there under the completion id (mutation). Returns the completion.
        """Mock create that stores the interaction in session cache via areal_cache."""
        import torch

        nonlocal call_index

        completion = ChatCompletion(
            id=f"chatcmpl-test{call_index}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890 + call_index,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
        call_index += 1

        messages = kwargs.get("messages", [])

        interaction = InteractionWithTokenLogpReward(
            messages=messages if isinstance(messages, list) else list(messages),
            completion=completion,
            output_message_list=[{"role": "assistant", "content": "Hello!"}],
        )
        # Pre-populate _cache so to_tensor_dict() works without ModelResponse
        interaction._cache = {
            "input_ids": torch.tensor([[100, 200, 300, 1234, 5678, 2]]),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "logprobs": torch.tensor([[0.0, 0.0, 0.0, -0.5, -0.3, -0.1]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.ones(6, dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([0.0]),
        }
        if areal_cache is not None:
            areal_cache[completion.id] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def client(config, mock_tokenizer, mock_areal_client):
    # [FUNCTION DOC]
    # Async fixture: creates the app, a PauseState and a real InfBridge
    # (SGLangBridgeBackend, 5 retries, 0.01s wait), injects the mocks, config, a
    # SessionStore keyed with the config admin key and version 0 into app.state
    # (bypassing lifespan), and yields an httpx AsyncClient on "http://test". App
    # exceptions propagate.
    """Create app with mocked deps and yield an httpx async client."""
    from areal.v2.inference_service.data_proxy.pause import PauseState
    from areal.v2.inference_service.inf_bridge import InfBridge
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_app(config)
    # Bypass lifespan — inject mocks directly into app.state
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.session_store = store
    app.state.version = 0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization bearer header dict for ADMIN_KEY.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers(api_key: str):
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization bearer header dict for the given api_key.
    return {"Authorization": f"Bearer {api_key}"}


def anthropic_headers(api_key: str):
    # [FUNCTION DOC]
    # HELPER: Returns Anthropic-style headers: x-api-key api_key and anthropic-version
    # "2023-06-01".
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


# =============================================================================
# SessionStore unit tests
# =============================================================================


class TestSessionStore:
    # [CLASS DOC]
    # Direct unit tests of SessionStore and SessionData (no HTTP).
    def test_start_session_returns_ids(self):
        # [FUNCTION DOC]
        # Asserts start_session("task-1") returns session id "task-1-0" and a non-empty
        # string api key.
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        assert session_id == "task-1-0"
        assert isinstance(api_key, str)
        assert len(api_key) > 0

    def test_get_session_by_api_key(self):
        # [FUNCTION DOC]
        # Asserts get_session_by_api_key returns the session with the matching id.
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        session = store.get_session_by_api_key(api_key)
        assert session is not None
        assert session.session_id == session_id

    def test_get_session_by_api_key_not_found(self):
        # [FUNCTION DOC]
        # Asserts an unknown api key yields None.
        store = SessionStore()
        assert store.get_session_by_api_key("nonexistent") is None

    def test_set_reward_marks_trajectory_ready(self):
        # [FUNCTION DOC]
        # Adds a fake interaction to a session's active_completions and asserts
        # set_reward marks a trajectory ready (ready_transition True,
        # has_ready_trajectories True).
        store = SessionStore()
        session_id, _ = store.start_session("task-1")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)
        assert not session.has_ready_trajectories
        # Populate with a fake interaction so set_reward succeeds
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction
        result = session.set_reward(interaction_id="fake-id", reward=1.0)
        assert session.has_ready_trajectories
        assert result.ready_transition is True

    def test_set_reward_waits_for_timeout_before_ready(self):
        # [FUNCTION DOC]
        # With set_reward_finish_timeout 5.0, asserts set_reward is pending (no ready
        # transition, no trajectory id), finalize_if_reward_timeout_elapsed() returns
        # None immediately, and returns a ready result with trajectory_id 0 when called
        # with now = last access + 6s.
        store = SessionStore(set_reward_finish_timeout=5.0)
        session_id, _ = store.start_session("task-1")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction

        pending = session.set_reward(interaction_id="fake-id", reward=1.0)
        assert pending.ready_transition is False
        assert pending.trajectory_id is None
        assert not session.has_ready_trajectories

        not_ready = session.finalize_if_reward_timeout_elapsed()
        assert not_ready is None

        ready = session.finalize_if_reward_timeout_elapsed(
            now=session._last_access_time + 6.0
        )
        assert ready is not None
        assert ready.ready_transition is True
        assert ready.trajectory_id == 0
        assert session.has_ready_trajectories

    def test_multiple_set_reward_calls_update_same_trajectory_before_timeout(self):
        # [FUNCTION DOC]
        # On a SessionData with a 5s finish timeout, two set_reward calls (1.0 then 2.0)
        # stay pending; after forced finalization (trajectory 0) the exported
        # individual-style interaction carries the latest reward 2.0.
        session = SessionData("task-1", set_reward_finish_timeout=5.0)

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction

        first = session.set_reward(interaction_id="fake-id", reward=1.0)
        second = session.set_reward(interaction_id="fake-id", reward=2.0)

        assert first.ready_transition is False
        assert second.ready_transition is False
        assert second.trajectory_id is None

        ready = session.finalize_if_reward_timeout_elapsed(
            now=session._last_access_time + 6.0
        )
        assert ready is not None
        assert ready.trajectory_id == 0

        _, interactions = session.export_trajectory(discount=1.0, style="individual")
        assert interactions["fake-id"].reward == 2.0

    def test_finish_session_not_found(self):
        # [FUNCTION DOC]
        # Asserts get_session for an unknown id returns None.
        # KNOWN ISSUE: the name refers to finishing a session, but only get_session is
        # exercised.
        store = SessionStore()
        session = store.get_session("nonexistent")
        assert session is None

    def test_session_count(self):
        # [FUNCTION DOC]
        # Asserts session_count increases 0 -> 1 -> 2 as sessions start.
        store = SessionStore()
        assert store.session_count == 0
        store.start_session("task-1")
        assert store.session_count == 1
        store.start_session("task-2")
        assert store.session_count == 2

    def test_remove_session(self):
        # [FUNCTION DOC]
        # Asserts remove_session removes both id and api-key lookups.
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        store.remove_session(session_id)
        assert store.get_session(session_id) is None
        assert store.get_session_by_api_key(api_key) is None

    def test_duplicate_session_ids_increment(self):
        # [FUNCTION DOC]
        # Asserts repeated task ids get suffixes -0 and -1.
        store = SessionStore()
        sid1, _ = store.start_session("task-1")
        sid2, _ = store.start_session("task-1")
        assert sid1 == "task-1-0"
        assert sid2 == "task-1-1"

    def test_get_or_create_hitl_session_reuses_same_session(self):
        # [FUNCTION DOC]
        # Asserts get_or_create_hitl_session returns the same "__hitl__" session object
        # twice.
        store = SessionStore()
        store.set_admin_key(ADMIN_KEY)
        online_session1 = store.get_or_create_hitl_session()
        online_session2 = store.get_or_create_hitl_session()
        assert online_session1 is online_session2
        assert online_session1.session_id == "__hitl__"

    def test_prefix_matcher_is_preserved_when_active_cache_rotates(self):
        # [FUNCTION DOC]
        # Creates a store with swe_prefix_matcher; asserts the session's active
        # completion cache keeps that prefix matcher both initially and after set_reward
        # rotates the active cache.
        from examples.swe.prefix_matchers import swe_prefix_matcher

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        store = SessionStore(prefix_matcher=swe_prefix_matcher)
        session_id, _ = store.start_session("task-prefix")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)
        assert session.active_completions._prefix_matcher is swe_prefix_matcher

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}],
        )
        interaction.interaction_id = "interaction-1"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["interaction-1"] = interaction

        session.set_reward("interaction-1", 1.0)

        assert session.active_completions._prefix_matcher is swe_prefix_matcher

    def test_online_session_count(self):
        # [FUNCTION DOC]
        # Asserts creating the HITL session counts toward session_count (1).
        store = SessionStore()
        store.set_admin_key(ADMIN_KEY)
        store.get_or_create_hitl_session()
        assert store.session_count == 1


def test_remotize_trajectory_keeps_reward_metadata_inline(monkeypatch):
    # [FUNCTION DOC]
    # Monkeypatches RTensor.remotize in the data proxy app to capture its input; asserts
    # _remotize_trajectory sends only input_ids to remotize with the given node address,
    # and returns the remote handle for input_ids while keeping rewards and
    # original_rewards tensors inline (same objects).
    import torch

    rewards = torch.tensor([1.0])
    original_rewards = torch.tensor([3.0])
    captured = {}

    def _remotize(payload, node_addr):
        # [FUNCTION DOC]
        # HELPER: Fake remotize; records payload and node_addr and returns {"input_ids":
        # "remote-input-ids"}.
        captured["payload"] = payload
        captured["node_addr"] = node_addr
        return {"input_ids": "remote-input-ids"}

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        _remotize,
    )

    result = _remotize_trajectory(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "rewards": rewards,
            "original_rewards": original_rewards,
        },
        node_addr="127.0.0.1:1234",
    )

    assert set(captured["payload"]) == {"input_ids"}
    assert captured["node_addr"] == "127.0.0.1:1234"
    assert result["input_ids"] == "remote-input-ids"
    assert result["rewards"] is rewards
    assert result["original_rewards"] is original_rewards


# =============================================================================
# Endpoint tests: /rl/start_session
# =============================================================================


@pytest.mark.asyncio
async def test_start_session_with_admin_key(client):
    # [FUNCTION DOC]
    # Asserts POST /rl/start_session with the admin key returns 201 with group_id and
    # one session whose id starts with "test-task-" and has a session_api_key.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "group_id" in data
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"].startswith("test-task-")
    assert "session_api_key" in data["sessions"][0]


@pytest.mark.asyncio
async def test_start_session_without_admin_key(client):
    # [FUNCTION DOC]
    # Asserts /rl/start_session without auth returns 401.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_anthropic_messages_applies_preprocessors_and_uses_session_cache(
    config,
    mock_areal_client,
):
    # [FUNCTION DOC]
    # Builds a config with the SWE message preprocessors (strip Anthropic billing header
    # and system reminders) and swe_prefix_matcher, a fresh app with a capacity-1
    # SessionStore and the mocked client; starts a session and posts an Anthropic-format
    # /v1/messages request (x-api-key session key) with a system prompt containing a
    # billing header and a system-reminder block. Asserts 200 with type "message", that
    # the messages passed to create no longer contain the billing header or
    # system-reminder text but keep "stable", and that the session's cache uses the
    # prefix matcher and holds one completion.
    from examples.swe.prefix_matchers import swe_prefix_matcher

    cc_config = DataProxyConfig(
        **{
            **config.__dict__,
            "message_preprocessors": (
                "examples.swe.preprocessors.StripAnthropicBillingHeader",
                "examples.swe.preprocessors.StripAllSystemReminders",
            ),
            "prefix_matcher": "examples.swe.prefix_matchers.swe_prefix_matcher",
        }
    )
    app = create_app(cc_config)
    store = SessionStore(prefix_matcher=swe_prefix_matcher)
    store.set_admin_key(cc_config.admin_api_key)
    store.set_capacity(1)
    app.state.session_store = store
    app.state.areal_client = mock_areal_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        start = await c.post(
            "/rl/start_session",
            json={"task_id": "cc-task"},
            headers=admin_headers(),
        )
        session_key = start.json()["sessions"][0]["session_api_key"]
        response = await c.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "system": (
                    "x-anthropic-billing-header: volatile\n"
                    "stable<system-reminder>volatile</system-reminder>"
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            },
            headers=anthropic_headers(session_key),
        )

    assert response.status_code == 200
    assert response.json()["type"] == "message"
    call_kwargs = mock_areal_client.chat.completions.create.await_args.kwargs
    messages = call_kwargs["messages"]
    serialized_messages = str(messages)
    assert "x-anthropic-billing-header" not in serialized_messages
    assert "system-reminder" not in serialized_messages
    assert "stable" in serialized_messages

    session = store.get_session_by_api_key(session_key)
    assert session is not None
    assert session.active_completions._prefix_matcher is swe_prefix_matcher
    assert len(session.active_completions) == 1


@pytest.mark.asyncio
async def test_anthropic_messages_streams_translated_events(
    config,
    mock_areal_client,
    monkeypatch,
):
    # [FUNCTION DOC]
    # Monkeypatches translate_anthropic_stream to yield message_start/message_stop SSE
    # events; starts a session and posts a streaming /v1/messages request. Asserts a 200
    # text/event-stream response containing both events and that create was awaited with
    # stream True.
    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    store.set_capacity(1)
    app.state.session_store = store
    app.state.areal_client = mock_areal_client

    async def _translated_stream():
        # [FUNCTION DOC]
        # HELPER: Async generator yielding fixed Anthropic message_start and
        # message_stop SSE strings.
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.translate_anthropic_stream",
        lambda _stream, model: _translated_stream(),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        start = await c.post(
            "/rl/start_session",
            json={"task_id": "cc-stream"},
            headers=admin_headers(),
        )
        session_key = start.json()["sessions"][0]["session_api_key"]
        response = await c.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=anthropic_headers(session_key),
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: message_start" in response.text
    assert "event: message_stop" in response.text
    assert mock_areal_client.chat.completions.create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_start_session_wrong_admin_key(client):
    # [FUNCTION DOC]
    # Asserts /rl/start_session with a wrong admin key returns 403.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


# =============================================================================
# Endpoint tests: /chat/completions
# =============================================================================


@pytest.mark.asyncio
async def test_chat_completions_with_session_key(client, mock_areal_client):
    # [FUNCTION DOC]
    # Starts a session and sends an OpenAI-format chat with the session key; asserts
    # 200, a chat.completion object with content "Hello!" and exactly one create call.
    # Start session first
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "chat-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    # Call chat/completions (OpenAI-compatible format)
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
    mock_areal_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completions_without_session_key(client):
    # [FUNCTION DOC]
    # Asserts a chat without any auth header is served (200, standalone).
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_with_invalid_key(client):
    # [FUNCTION DOC]
    # Asserts a chat with an unknown bearer key is served (200, standalone).
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer fake-key"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_offline_chat_unknown_token_falls_through_to_standalone(client):
    # [FUNCTION DOC]
    # Asserts a bearer token of the form "<admin key>:agent-online" is not treated as a
    # session and still gets a 200 standalone response.
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(f"{ADMIN_KEY}:agent-online"),
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_passes_sampling_params(client, mock_areal_client):
    # [FUNCTION DOC]
    # Asserts temperature, top_p and max_tokens from the request are forwarded to create
    # and no seed is injected when deterministic sampling is off.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "sp-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    call_kwargs = mock_areal_client.chat.completions.create.call_args
    kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kw["temperature"] == 0.5
    assert kw["top_p"] == 0.9
    assert kw["max_tokens"] == 100
    assert "seed" not in kw


@pytest.mark.asyncio
async def test_chat_completions_deterministic_seed_distinguishes_group_sessions(
    client, config, mock_areal_client
):
    # [FUNCTION DOC]
    # Enables deterministic_sampling on the shared config, starts a group of 2 sessions
    # and chats once in each; asserts the injected seeds equal derive_deterministic_seed
    # for "seed-test:0" and "seed-test:1" at turn 0 and differ.
    config.deterministic_sampling = True
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "seed-test", "group_size": 2},
        headers=admin_headers(),
    )
    sessions = resp.json()["sessions"]

    for session in sessions:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(session["session_api_key"]),
        )
        assert resp.status_code == 200

    seeds = [
        call.kwargs["seed"]
        for call in mock_areal_client.chat.completions.create.call_args_list
    ]
    assert seeds == [
        derive_deterministic_seed("seed-test:0", 0),
        derive_deterministic_seed("seed-test:1", 0),
    ]
    assert seeds[0] != seeds[1]


@pytest.mark.asyncio
async def test_chat_completions_preserves_explicit_seed(
    client, config, mock_areal_client
):
    # [FUNCTION DOC]
    # With deterministic sampling, asserts an explicit request seed 12345 is passed
    # through unchanged, and that the next turn without a seed gets
    # derive_deterministic_seed("explicit-seed", 1) (single-session key has no group
    # suffix).
    config.deterministic_sampling = True
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "explicit-seed"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 12345,
        },
        headers=session_headers(api_key),
    )

    assert resp.status_code == 200
    assert mock_areal_client.chat.completions.create.call_args.kwargs["seed"] == 12345

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "again"}],
        },
        headers=session_headers(api_key),
    )

    assert resp.status_code == 200
    assert mock_areal_client.chat.completions.create.call_args.kwargs[
        "seed"
    ] == derive_deterministic_seed("explicit-seed", 1)


# =============================================================================
# Endpoint tests: /rl/set_reward
# =============================================================================


@pytest.mark.asyncio
async def test_set_reward_success(client):
    # [FUNCTION DOC]
    # Starts a session, chats once, then set_reward 1.0; asserts 200 with message
    # "success", interaction_count 1, session id prefix, trajectory_id 0,
    # trajectory_ready and ready_transition True.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["message"] == "success"
    data = resp.json()
    assert data["interaction_count"] == 1
    assert data["session_id"].startswith("reward-test-")
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_set_reward_no_interactions(client):
    # [FUNCTION DOC]
    # Asserts set_reward on a session with no interactions returns 400 with "No
    # interactions" in the detail.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-empty"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 400
    assert "No interactions" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_set_reward_without_session_key(client):
    # [FUNCTION DOC]
    # Asserts set_reward without auth returns 401.
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
    )
    assert resp.status_code == 401


# =============================================================================
# Endpoint tests: /rl/set_reward
# =============================================================================


@pytest.mark.asyncio
async def test_set_reward_auto_finishes(client):
    # [FUNCTION DOC]
    # Asserts set_reward 0.0 after one chat immediately finishes the trajectory
    # (success, trajectory 0, ready and ready_transition True).
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "end-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 0.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "success"
    assert "interaction_count" in data
    assert data["session_id"].startswith("end-test-")
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_set_reward_only_once_allowed(client):
    # [FUNCTION DOC]
    # Asserts a second set_reward on a finished trajectory still returns 200 but with
    # ready_transition False and trajectory_ready True.
    # KNOWN ISSUE: the name says "only once allowed", yet the second call succeeds
    # (200); the test checks it is a no-op transition rather than rejected.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "twice-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["ready_transition"] is True

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["ready_transition"] is False
    assert resp.json()["trajectory_ready"] is True


@pytest.mark.asyncio
async def test_set_reward_timeout_delays_readiness_and_direct_callback(
    client, monkeypatch
):
    # [FUNCTION DOC]
    # Reconfigures app.state for a 5s set_reward finish timeout and callback server
    # "http://controller" with a fresh SessionStore, and monkeypatches
    # _post_online_ready_callback to record calls. Chats and sets reward using the admin
    # key (HITL session); asserts the reward is pending and that
    # _flush_ready_trajectories sends no callback. After forcing the HITL session's
    # finalization 6s in the future, a flush sends exactly one callback
    # ("http://controller", ADMIN_KEY, "__hitl__", 0).
    app = client._transport.app
    app.state.config.set_reward_finish_timeout = 5.0
    app.state.config.callback_server_addr = "http://controller"
    app.state.session_store = SessionStore(set_reward_finish_timeout=5.0)
    app.state.session_store.set_admin_key(ADMIN_KEY)

    callback_calls = []

    async def _mock_post_callback(
        callback_server_addr, admin_api_key, notification, timeout, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: Fake callback poster; appends (addr, admin key, session id, trajectory
        # id) to callback_calls and returns True.
        callback_calls.append(
            (
                callback_server_addr,
                admin_api_key,
                notification.session_id,
                notification.trajectory_id,
            )
        )
        return True

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app._post_online_ready_callback",
        _mock_post_callback,
    )

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(ADMIN_KEY),
    )

    reward_resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(ADMIN_KEY),
    )
    assert reward_resp.status_code == 200
    assert reward_resp.json()["ready_transition"] is False
    assert reward_resp.json()["trajectory_ready"] is False

    await _flush_ready_trajectories(app)
    assert callback_calls == []

    hitl_session = app.state.session_store.get_session("__hitl__")
    assert hitl_session is not None
    hitl_session.finalize_if_reward_timeout_elapsed(now=time.time() + 6.0)
    await _flush_ready_trajectories(app)

    assert callback_calls == [("http://controller", ADMIN_KEY, "__hitl__", 0)]


@pytest.mark.asyncio
async def test_set_reward_without_key(client):
    # [FUNCTION DOC]
    # Asserts set_reward without auth returns 401.
    # KNOWN ISSUE: duplicates test_set_reward_without_session_key.
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 0.0},
    )
    assert resp.status_code == 401


# =============================================================================
# HITL tests — admin key → single persistent session
# =============================================================================


@pytest.mark.asyncio
async def test_hitl_admin_key_creates_single_persistent_session(client):
    # [FUNCTION DOC]
    # Sends two chats with the admin key as bearer; asserts both succeed and /health
    # reports exactly one session (the persistent HITL session).
    resp = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(ADMIN_KEY),
    )
    assert resp.status_code == 200

    resp2 = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "bye"}]},
        headers=session_headers(ADMIN_KEY),
    )
    assert resp2.status_code == 200

    health = await client.get("/health")
    assert health.json()["sessions"] == 1


@pytest.mark.asyncio
async def test_hitl_reuses_same_session_across_multiple_chat_requests(client):
    # [FUNCTION DOC]
    # Asserts two admin-key chats resolve to the same "__hitl__" SessionData object in
    # the store.
    store: SessionStore = client._transport.app.state.session_store

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "a"}]},
        headers=session_headers(ADMIN_KEY),
    )
    session1 = store.get_session("__hitl__")
    assert session1 is not None

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "b"}]},
        headers=session_headers(ADMIN_KEY),
    )
    session2 = store.get_session("__hitl__")
    assert session1 is session2


@pytest.mark.asyncio
async def test_hitl_online_set_reward_separates_trajectories(client):
    # [FUNCTION DOC]
    # On the HITL session, chat + set_reward 1.0 yields trajectory_id 0, and a second
    # chat + set_reward 2.0 yields trajectory_id 1 (each reward closes a separate
    # trajectory).
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "t0"}]},
        headers=session_headers(token),
    )
    r0 = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert r0.status_code == 200
    assert r0.json()["trajectory_id"] == 0

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "t1"}]},
        headers=session_headers(token),
    )
    r1 = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert r1.status_code == 200
    assert r1.json()["trajectory_id"] == 1


@pytest.mark.asyncio
async def test_hitl_ready_transition_after_each_online_set_reward(client):
    # [FUNCTION DOC]
    # Asserts an HITL chat followed by set_reward reports ready_transition and
    # trajectory_ready True.
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(token),
    )
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert resp.json()["ready_transition"] is True
    assert resp.json()["trajectory_ready"] is True


# =============================================================================
# =============================================================================
@pytest.mark.asyncio
async def test_online_start_session_returns_generated_session_key(client):
    # [FUNCTION DOC]
    # Asserts /rl/start_session returns 201 with group_id and one session whose id
    # starts with "batch-1-" and has a non-empty generated api key.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-1"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "group_id" in data
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"].startswith("batch-1-")
    assert len(data["sessions"][0]["session_api_key"]) > 0


@pytest.mark.asyncio
async def test_batch_session_produces_single_trajectory(client):
    # [FUNCTION DOC]
    # Starts a session, chats, sets reward 1.0; asserts trajectory_id 0, matching
    # session_id and ready_transition True.
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-traj"},
        headers=admin_headers(),
    )
    session_api_key = start.json()["sessions"][0]["session_api_key"]
    session_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(session_api_key),
    )
    reward_resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session_api_key),
    )
    assert reward_resp.status_code == 200
    assert reward_resp.json()["trajectory_id"] == 0
    assert reward_resp.json()["session_id"] == session_id
    assert reward_resp.json()["ready_transition"] is True


@pytest.mark.asyncio
async def test_batch_online_set_reward_completes_that_session(client):
    # [FUNCTION DOC]
    # Starts a session, chats, sets reward, then exports it (trajectory 0, discount 1.0,
    # individual style); asserts 200 and a "traj" key in the response.
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-complete"},
        headers=admin_headers(),
    )
    session_api_key = start.json()["sessions"][0]["session_api_key"]
    session_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
        headers=session_headers(session_api_key),
    )
    await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session_api_key),
    )

    export_resp = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session_id],
            "trajectory_id": 0,
            "discount": 1.0,
            "style": "individual",
        },
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    assert "traj" in export_resp.json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_rewards", "normalized_rewards"),
    [
        ((1.0, 3.0), (-1.0, 1.0)),
        ((1.0, 1.0), (0.0, 0.0)),
        ((3.0,), (3.0,)),
    ],
)
async def test_export_trajectories_normalizes_rewards_within_group(
    client, monkeypatch, raw_rewards, normalized_rewards
):
    # [FUNCTION DOC]
    # Parametrized over raw/normalized reward pairs. Monkeypatches RTensor.remotize to
    # the identity, starts a group sized to the rewards, chats and sets each reward,
    # then exports all sessions with reward_normalization. Deserializes "traj" and
    # asserts rewards equal the group-normalized values (mean-centred and std-scaled; a
    # single-member group is left unchanged and equal rewards become 0) and
    # original_rewards keep the raw values.
    import torch

    from areal.infra.rpc.serialization import deserialize_value

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "normalized-group", "group_size": len(raw_rewards)},
        headers=admin_headers(),
    )
    sessions = start.json()["sessions"]
    for session, reward in zip(sessions, raw_rewards):
        await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
            headers=session_headers(session["session_api_key"]),
        )
        reward_response = await client.post(
            "/rl/set_reward",
            json={"reward": reward},
            headers=session_headers(session["session_api_key"]),
        )
        assert reward_response.status_code == 200

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"] for session in sessions],
            "discount": 1.0,
            "style": "individual",
            "reward_normalization": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    trajectory = deserialize_value(export_response.json()["traj"])
    torch.testing.assert_close(
        trajectory["rewards"].flatten(),
        torch.tensor(normalized_rewards),
    )
    torch.testing.assert_close(
        trajectory["original_rewards"].flatten(),
        torch.tensor(raw_rewards),
    )


@pytest.mark.asyncio
async def test_export_trajectories_drops_incomplete_normalization_group(
    client, monkeypatch
):
    # [FUNCTION DOC]
    # With identity remotize, completes one session and exports it together with a
    # missing session id under reward_normalization; asserts 200 with traj {}
    # (incomplete groups are dropped).
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "incomplete-group"},
        headers=admin_headers(),
    )
    session = start.json()["sessions"][0]
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
        headers=session_headers(session["session_api_key"]),
    )
    await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session["session_api_key"]),
    )

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"], "missing-session"],
            "reward_normalization": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    assert export_response.json()["traj"] == {}


# =============================================================================
# Coexistence tests — HITL and batch running simultaneously
# =============================================================================


@pytest.mark.asyncio
async def test_hitl_and_batch_can_run_simultaneously(client):
    # [FUNCTION DOC]
    # Runs a batch session and the HITL session side by side (chat + set_reward each);
    # asserts both succeed, the HITL reward reports "__hitl__", the batch reward reports
    # its own session id, and /health counts 2 sessions.
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "coexist-batch"},
        headers=admin_headers(),
    )
    batch_key = start.json()["sessions"][0]["session_api_key"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hitl"}]},
        headers=session_headers(ADMIN_KEY),
    )
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "batch"}]},
        headers=session_headers(batch_key),
    )

    hitl_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(ADMIN_KEY),
    )
    batch_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(batch_key),
    )

    assert hitl_reward.status_code == 200
    assert batch_reward.status_code == 200
    assert hitl_reward.json()["session_id"] == "__hitl__"
    assert (
        batch_reward.json()["session_id"] == start.json()["sessions"][0]["session_id"]
    )

    health = await client.get("/health")
    assert health.json()["sessions"] == 2


@pytest.mark.asyncio
async def test_admin_key_still_only_maps_to_hitl_session_while_batch_uses_session_key(
    client,
):
    # [FUNCTION DOC]
    # Starts a batch session and sends an HITL chat with the admin key; asserts the HITL
    # and batch sessions are distinct objects and the batch api key maps to the batch
    # session.
    store: SessionStore = client._transport.app.state.session_store

    start = await client.post(
        "/rl/start_session",
        json={"task_id": "mapping"},
        headers=admin_headers(),
    )
    batch_key = start.json()["sessions"][0]["session_api_key"]
    batch_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hitl"}]},
        headers=session_headers(ADMIN_KEY),
    )

    hitl_session = store.get_session("__hitl__")
    batch_session = store.get_session(batch_id)
    assert hitl_session is not None
    assert batch_session is not None
    assert hitl_session is not batch_session

    batch_by_key = store.get_session_by_api_key(batch_key)
    assert batch_by_key is batch_session


# =============================================================================
# Negative tests
# =============================================================================


@pytest.mark.asyncio
async def test_online_start_session_rejects_wrong_admin_key(client):
    # [FUNCTION DOC]
    # Asserts /rl/start_session with a wrong admin key returns 403.
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reject"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_online_set_reward_rejects_unknown_session_key(client):
    # [FUNCTION DOC]
    # Asserts /rl/set_reward with an unknown bearer token returns 401.
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers={"Authorization": "Bearer unknown-token-xyz"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_completion_without_valid_token_falls_through_to_standalone(
    client,
):
    # [FUNCTION DOC]
    # Asserts /chat/completions with an unknown bearer token is still served (200).
    resp = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer unknown-token-xyz"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_export_trajectories_not_found(client):
    # [FUNCTION DOC]
    # Asserts exporting a nonexistent session returns 200 with traj {}.
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["nonexistent"], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["traj"] == {}


@pytest.mark.asyncio
async def test_export_trajectories_without_admin_key(client):
    # [FUNCTION DOC]
    # Asserts /export_trajectories without the admin key returns 401.
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["x"], "discount": 1.0, "style": "individual"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_export_trajectories_drops_retry_orphans_before_discount(
    client, monkeypatch
):
    # [FUNCTION DOC]
    # With identity remotize, sends the same first-turn chat twice (a retry) and then a
    # follow-up turn extending the first answer, sets reward, marks each ready
    # interaction's chat_template_type as "concat" via private session state, and
    # exports with style "concat" and drop_retry_orphans True; asserts the deserialized
    # trajectory has exactly one input_ids row (the orphaned retry was dropped before
    # merging).
    from areal.infra.rpc.serialization import deserialize_value

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "retry-orphan"},
        headers=admin_headers(),
    )
    session = start.json()["sessions"][0]
    headers = session_headers(session["session_api_key"])
    initial_messages = [{"role": "user", "content": "hi"}]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": initial_messages},
        headers=headers,
    )
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": initial_messages},
        headers=headers,
    )
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": initial_messages
            + [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "more"},
            ],
        },
        headers=headers,
    )
    reward_response = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=headers,
    )
    assert reward_response.status_code == 200

    app = client._transport.app
    session_data = app.state.session_store.get_session(session["session_id"])
    assert session_data is not None
    ready = next(iter(session_data._ready_trajectories.values()))
    for interaction in ready.completions.values():
        interaction.chat_template_type = "concat"

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"]],
            "discount": 1.0,
            "style": "concat",
            "drop_retry_orphans": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    trajectory = deserialize_value(export_response.json()["traj"])
    assert trajectory["input_ids"].shape[0] == 1


# =============================================================================
# =============================================================================
@pytest.mark.asyncio
async def test_online_chat_completion_implicitly_binds_session(client):
    # [FUNCTION DOC]
    # Asserts an admin-key chat implicitly creates/binds the HITL session (/health
    # sessions 1).
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(ADMIN_KEY),
    )
    assert resp.status_code == 200

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["sessions"] == 1


@pytest.mark.asyncio
async def test_online_set_reward_returns_trajectory_metadata(client):
    # [FUNCTION DOC]
    # Asserts HITL set_reward returns session_id "__hitl__", trajectory_id 0 and both
    # ready flags True.
    token = ADMIN_KEY
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(token),
    )

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "__hitl__"
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_online_set_reward_duplicate_is_idempotent(client):
    # [FUNCTION DOC]
    # Asserts two consecutive HITL set_reward calls without a new chat both report
    # trajectory 0, with only the first making a ready transition (second is
    # idempotent).
    token = ADMIN_KEY
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(token),
    )

    first = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    second = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["trajectory_id"] == second.json()["trajectory_id"] == 0
    assert first.json()["ready_transition"] is True
    assert second.json()["ready_transition"] is False


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_online_export_latest_ready_without_trajectory_id(client):
    # [FUNCTION DOC]
    # Skipped (pending traj schema migration). Two HITL chat+reward rounds; asserts
    # exporting "__hitl__" without trajectory_id returns only the latest trajectory's
    # interaction ("chatcmpl-test1").
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "first"}],
        },
        headers=session_headers(token),
    )
    first_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert first_reward.status_code == 200

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "second"}],
        },
        headers=session_headers(token),
    )
    second_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert second_reward.status_code == 200
    assert second_reward.json()["trajectory_id"] == 1

    export_resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["__hitl__"], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    interactions = export_resp.json()["interactions"]
    assert list(interactions) == ["chatcmpl-test1"]


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_online_export_explicit_trajectory_id(client):
    # [FUNCTION DOC]
    # Skipped (pending traj schema migration). Two HITL rounds; exporting trajectory 0
    # with remove_session False returns only "chatcmpl-test0" and the HITL session
    # remains (/health sessions 1).
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "first"}],
        },
        headers=session_headers(token),
    )
    first_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert first_reward.status_code == 200

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "second"}],
        },
        headers=session_headers(token),
    )
    second_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert second_reward.status_code == 200

    export_resp = await client.post(
        "/export_trajectories",
        json={
            "session_ids": ["__hitl__"],
            "trajectory_id": 0,
            "discount": 1.0,
            "style": "individual",
            "remove_session": False,
        },
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    interactions = export_resp.json()["interactions"]
    assert list(interactions) == ["chatcmpl-test0"]

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["sessions"] == 1


# =============================================================================
# Endpoint tests: /health (updated with sessions count)
# =============================================================================


@pytest.mark.asyncio
async def test_health_includes_sessions(client):
    # [FUNCTION DOC]
    # Asserts /health returns status "ok" with sessions 0 initially.
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "sessions" in data
    assert data["sessions"] == 0


@pytest.mark.asyncio
async def test_health_sessions_count_after_start(client):
    # [FUNCTION DOC]
    # Asserts /health sessions becomes 1 after one start_session.
    # Start a session
    await client.post(
        "/rl/start_session",
        json={"task_id": "health-test"},
        headers=admin_headers(),
    )
    resp = await client.get("/health")
    assert resp.json()["sessions"] == 1


# =============================================================================
# Full lifecycle test
# =============================================================================


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_full_session_lifecycle(client, mock_areal_client):
    # [FUNCTION DOC]
    # Skipped (pending traj schema migration). Start session, chat, set_reward (one
    # interaction, ready), export, and assert an "interactions" key.
    """Test the complete flow: start → chat → set_reward → export."""
    # 1. Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "lifecycle"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    session_id = resp.json()["sessions"][0]["session_id"]
    api_key = resp.json()["sessions"][0]["session_api_key"]

    # 2. Chat completion
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["object"] == "chat.completion"

    # 3. Set reward (auto-finishes session)
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["interaction_count"] == 1
    assert resp.json()["ready_transition"] is True

    # 4. Export trajectories
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": [session_id], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "interactions" in data
