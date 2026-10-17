# [MODULE DOC]
# Async tests for the v2 Agent Worker ASGI app (create_worker_app), which imports an
# agent class by path and serves /health and /run, returning the agent's summary,
# metadata and collected stream events. The import is patched to stub agents; requests
# go through httpx.ASGITransport. Also tests _CollectingEmitter and the AgentRunnable
# runtime protocol. Skipped if httpx is missing.
"""Tests for Agent Worker HTTP server."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
)
from areal.v2.agent_service.worker.app import (
    _CollectingEmitter,
    create_worker_app,
)

httpx = pytest.importorskip("httpx")


class _EchoAgent:
    # [CLASS DOC]
    # HELPER: Stub agent echoing the message as a delta and summary; reports history length.
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        # [FUNCTION DOC]
        # Emits delta "echo: <message>" and returns AgentResponse with the same summary and
        # metadata history_len = number of history entries.
        await emitter.emit_delta(f"echo: {request.message}")
        return AgentResponse(
            summary=f"echo: {request.message}",
            metadata={"history_len": len(request.history)},
        )


class _ToolAgent:
    # [CLASS DOC]
    # HELPER: Stub agent emitting a tool call, tool result and a delta.
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        # [FUNCTION DOC]
        # Emits tool_call("search", JSON args), tool_result("search", "found it"), delta "Done";
        # returns summary "Done".
        await emitter.emit_tool_call("search", '{"q": "test"}')
        await emitter.emit_tool_result("search", "found it")
        await emitter.emit_delta("Done")
        return AgentResponse(summary="Done")


class _FailAgent:
    # [CLASS DOC]
    # HELPER: Stub agent that always fails.
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        # [FUNCTION DOC]
        # Raises RuntimeError("boom").
        raise RuntimeError("boom")


def _make_client(agent_cls):
    # [FUNCTION DOC]
    # HELPER: Creates the worker app with import_from_string patched to return agent_cls,
    # and returns an httpx.AsyncClient on it (base URL http://worker).
    with patch(
        "areal.v2.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        app = create_worker_app("mock.path")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://worker")


class TestWorkerHealth:
    # [CLASS DOC]
    # Worker health endpoint tests.
    @pytest.mark.asyncio
    async def test_health(self):
        # [FUNCTION DOC]
        # Asserts GET /health returns 200 with status "ok".
        async with _make_client(_EchoAgent) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestWorkerRun:
    # [CLASS DOC]
    # Worker /run endpoint tests.
    @pytest.mark.asyncio
    async def test_echo(self):
        # [FUNCTION DOC]
        # Asserts /run with the echo agent returns 200, the echo summary and a delta event.
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "hello", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["summary"] == "echo: hello"
            assert any(e["type"] == "delta" for e in data["events"])

    @pytest.mark.asyncio
    async def test_history_forwarded(self):
        # [FUNCTION DOC]
        # Asserts request history is forwarded to the agent (history_len == 1).
        async with _make_client(_EchoAgent) as client:
            resp = await client.post(
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [{"role": "user", "content": "prev"}],
                },
            )
            assert resp.json()["metadata"]["history_len"] == 1

    @pytest.mark.asyncio
    async def test_tool_events(self):
        # [FUNCTION DOC]
        # Asserts tool_call, tool_result and delta event types appear in the response.
        async with _make_client(_ToolAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "go", "session_key": "s1", "run_id": "r1"},
            )
            types = [e["type"] for e in resp.json()["events"]]
            assert "tool_call" in types
            assert "tool_result" in types
            assert "delta" in types

    @pytest.mark.asyncio
    async def test_agent_failure(self):
        # [FUNCTION DOC]
        # Asserts an agent exception yields HTTP 500.
        async with _make_client(_FailAgent) as client:
            resp = await client.post(
                "/run",
                json={"message": "x", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 500


class TestCollectingEmitter:
    # [CLASS DOC]
    # Tests for the worker-side event collector.
    @pytest.mark.asyncio
    async def test_collects_all_event_types(self):
        # [FUNCTION DOC]
        # Emits one delta, tool call and tool result and asserts three events were collected.
        e = _CollectingEmitter()
        await e.emit_delta("hi")
        await e.emit_tool_call("fn", "{}")
        await e.emit_tool_result("fn", "ok")
        assert len(e.events) == 3


class TestAgentRunnableProtocol:
    # [CLASS DOC]
    # Tests for the runtime-checkable AgentRunnable protocol.
    def test_echo_satisfies(self):
        # [FUNCTION DOC]
        # Asserts an object with an async run method satisfies AgentRunnable.
        assert isinstance(_EchoAgent(), AgentRunnable)

    def test_plain_object_does_not(self):
        # [FUNCTION DOC]
        # Asserts a plain object does not satisfy AgentRunnable.
        assert not isinstance(object(), AgentRunnable)
