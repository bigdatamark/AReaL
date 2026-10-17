# [MODULE DOC]
# Tests that the proxy rollout server's /chat/completions handler returns an SSE
# StreamingResponse (data lines terminated by a [DONE] event) when stream is true and a
# plain JSON ChatCompletion otherwise. The server's module-level OpenAI client is
# replaced by a fake whose create returns canned objects; the app runs in-process via
# httpx ASGITransport. Regression test for AReaL issue 1046. Skipped if httpx is
# missing.
"""Tests for streaming and non-streaming behaviour of the ``/chat/completions`` endpoint.

The proxy rollout server's ``chat_completions`` handler must correctly return a
``StreamingResponse`` (SSE) when ``stream=True`` is requested, and a plain JSON
``ChatCompletion`` otherwise.

Ref: https://github.com/areal-project/AReaL/issues/1046
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from areal.experimental.openai.proxy import proxy_rollout_server as srv

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_proxy_rollout_server.py)
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"

httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    # [FUNCTION DOC]
    # HELPER: Returns an httpx.AsyncClient bound to the server app's ASGI transport.
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization Bearer header for the test admin key.
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


def _session_headers(api_key: str):
    # [FUNCTION DOC]
    # HELPER: Returns Authorization Bearer header for a session key plus JSON content
    # type.
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture resetting the server module's globals (session cache, key maps,
    # capacity 0, admin key, lock, last cleanup time) via monkeypatch before each test.
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)


# ---------------------------------------------------------------------------
# Fake create function (replaces _openai_client.chat.completions.create)
# ---------------------------------------------------------------------------


async def _fake_create(
    *,
    messages=None,
    stream=None,
    temperature=None,
    top_p=None,
    areal_cache=None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Async stand-in for the reward-tracking completions create. Accepts
    # keyword-only messages, stream, temperature, top_p, areal_cache and extra kwargs
    # (explicit names keep them through the server's signature-based argument
    # filtering). If stream is truthy returns an async generator yielding one
    # ChatCompletionChunk with assistant content "hello"; otherwise returns a
    # ChatCompletion with one "hello" message and finish_reason stop. All inputs are
    # ignored.
    """Minimal stand-in for ``AsyncCompletionsWithReward.create``.

    Returns an ``AsyncGenerator[ChatCompletionChunk]`` when *stream* is truthy,
    or a ``ChatCompletion`` otherwise — mirroring the real client's behaviour.

    The explicit keyword parameters (``messages``, ``stream``, etc.) are
    required so that ``_call_client_create`` keeps them after its
    ``inspect.signature``-based filtering of request-body fields.
    """
    if stream:

        async def _gen():
            # [FUNCTION DOC]
            # HELPER: Async generator yielding a single chunk with delta role assistant
            # and content "hello".
            yield ChatCompletionChunk(
                id="chatcmpl-test",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(role="assistant", content="hello"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                created=0,
                model="test",
                object="chat.completion.chunk",
            )

        return _gen()

    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        created=0,
        model="test",
        object="chat.completion",
    )


@pytest.fixture()
def _mock_openai_client(monkeypatch):
    # [FUNCTION DOC]
    # Fixture installing a MagicMock as the server's _openai_client with
    # chat.completions.create replaced by the fake create function (via monkeypatch).
    """Inject a fake OpenAI client so no real inference engine is needed."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _fake_create
    monkeypatch.setattr(srv, "_openai_client", mock_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    # [CLASS DOC]
    # Tests for /chat/completions streaming vs non-streaming responses.
    """Verify streaming and non-streaming responses from ``/chat/completions``."""

    @pytest.mark.asyncio
    async def test_streaming_returns_sse_response(
        self, monkeypatch, _mock_openai_client
    ):
        # [FUNCTION DOC]
        # Starts a session (capacity 1), posts a streaming chat request with the session
        # key and asserts 200, an event-stream content type, at least two data events
        # with the last being the [DONE] sentinel, and that the first event parses as a
        # chat.completion.chunk whose delta content is "hello".
        """``stream=True`` returns an SSE stream (``text/event-stream``)."""
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            resp = await client.post(
                "/chat/completions",
                headers=_session_headers(api_key),
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                    "stream": True,
                },
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE body: data lines separated by blank lines
            events = [
                line
                for line in resp.text.strip().split("\n\n")
                if line.startswith("data: ")
            ]
            assert len(events) >= 2  # at least one chunk + [DONE]
            assert events[-1] == "data: [DONE]"

            # Verify the data payload is a valid ChatCompletionChunk
            chunk = json.loads(events[0].removeprefix("data: "))
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["choices"][0]["delta"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_non_streaming_returns_json(self, monkeypatch, _mock_openai_client):
        # [FUNCTION DOC]
        # Starts a session and posts a non-streaming chat request; asserts 200 and a
        # JSON body of object chat.completion with message content "hello".
        """Without ``stream``, returns a JSON ``ChatCompletion``."""
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            resp = await client.post(
                "/chat/completions",
                headers=_session_headers(api_key),
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "hello"
