# [MODULE DOC]
# In-process integration tests for the v2 agent service HTTP stack: Worker, DataProxy
# (session history, passthrough relaying, inference-routing injection), Router, and the
# Gateway with its OpenAI-style ChatCompletions / OpenResponses bridges. Apps are wired
# together with httpx.ASGITransport and by monkeypatching httpx.AsyncClient.post/send
# to forward "http://worker" calls into the worker app. No real network. Skipped
# without httpx.
"""Integration tests for the Agent Service.

Tests the full HTTP microservice stack: Worker → DataProxy → Router,
plus utility functions from the Bridge and Gateway health endpoints.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from areal.v2.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.v2.agent_service.data_proxy.app import create_data_proxy_app
from areal.v2.agent_service.data_proxy.config import DataProxyConfig
from areal.v2.agent_service.gateway.app import create_gateway_app
from areal.v2.agent_service.gateway.bridge import (
    SESSION_KEY_HEADER,
    ChatCompletionsBridge,
    OpenResponsesBridge,
)
from areal.v2.agent_service.gateway.config import GatewayConfig
from areal.v2.agent_service.protocol import PASSTHROUGH_HEADER
from areal.v2.agent_service.router.app import create_router_app
from areal.v2.agent_service.router.config import RouterConfig
from areal.v2.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    StreamResponse,
)
from areal.v2.agent_service.worker.app import create_worker_app

httpx = pytest.importorskip("httpx")

_AUTH = admin_headers(DEFAULT_ADMIN_API_KEY)


class _EchoAgent:
    # [CLASS DOC]
    # HELPER: Stub agent echoing the message and history size.
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        # [FUNCTION DOC]
        # Emits delta "echo: <msg> (history=<n>)" and returns summary "echo: <msg>".
        history_summary = f"history={len(request.history)}"
        await emitter.emit_delta(f"echo: {request.message} ({history_summary})")
        return AgentResponse(summary=f"echo: {request.message}")


class _ToolAgent:
    # [CLASS DOC]
    # HELPER: Stub agent emitting a lookup tool call/result and a final delta.
    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse:
        # [FUNCTION DOC]
        # Emits tool_call/tool_result for "lookup" and delta "Lookup complete"; returns that
        # summary with tool_calls metadata.
        await emitter.emit_tool_call("lookup", '{"id": "123"}')
        await emitter.emit_tool_result("lookup", '{"status": "ok"}')
        await emitter.emit_delta("Lookup complete")
        return AgentResponse(
            summary="Lookup complete",
            metadata={"tool_calls": [{"name": "lookup", "arguments": {"id": "123"}}]},
        )


class _StreamAgent:
    # [CLASS DOC]
    # HELPER: Agent returning a raw StreamResponse for passthrough turns.
    """Returns a :class:`StreamResponse` from ``run`` when the turn carries a
    ``chat_request`` (the raw-passthrough channel), echoing the request back as
    an SSE body so tests can assert the worker relays it verbatim.  Otherwise it
    falls back to a structured :class:`AgentResponse`."""

    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> AgentResponse | StreamResponse:
        # [FUNCTION DOC]
        # If request.metadata has no "chat_request", returns AgentResponse("structured");
        # otherwise returns a 200 text/event-stream StreamResponse whose body yields SSE lines
        # with the message, the JSON chat_request and a [DONE] marker.
        chat_request = (request.metadata or {}).get("chat_request")
        if chat_request is None:
            return AgentResponse(summary="structured")

        async def _body():
            # [FUNCTION DOC]
            # HELPER: Async generator producing the three SSE byte chunks.
            yield f"data: msg={request.message}\n\n".encode()
            yield f"data: chat_request={json.dumps(chat_request)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamResponse(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            body=_body(),
        )


class _JsonPassthroughAgent:
    # [CLASS DOC]
    # HELPER: Agent returning a non-streaming JSON StreamResponse (passthrough).
    """Returns a non-streaming :class:`StreamResponse` whose body is
    ``application/json``.  Used to prove the DataProxy keys its relay-vs-parse
    decision on the passthrough marker, not on ``Content-Type``: without the
    marker this body would be mistaken for a structured turn and parsed."""

    async def run(
        self, request: AgentRequest, *, emitter: EventEmitter
    ) -> StreamResponse:
        # [FUNCTION DOC]
        # Returns a 200 application/json StreamResponse whose body is JSON
        # {"verbatim": message, "events": []}.
        async def _body():
            # [FUNCTION DOC]
            # HELPER: Async generator yielding the single JSON body chunk.
            yield json.dumps({"verbatim": request.message, "events": []}).encode()

        return StreamResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=_body(),
        )


def _make_worker_app(agent_cls):
    # [FUNCTION DOC]
    # HELPER: Creates a worker app with import_from_string patched to return agent_cls.
    with patch(
        "areal.v2.agent_service.worker.app.import_from_string",
        return_value=agent_cls,
    ):
        return create_worker_app("mock.path")


def _worker_patches(worker_transport):
    # [FUNCTION DOC]
    # HELPER: Builds replacement AsyncClient.post/send functions that route any URL
    # containing "http://worker" into worker_transport (via the unpatched send, avoiding
    # recursion) and delegate other URLs to the originals. The send replacement re-wraps
    # the worker response as a streaming httpx.Response, keeping content-type and the
    # passthrough marker header. Returns (patched_post, patched_send).
    """Route the DataProxy's worker calls through the worker ASGI app.

    The DataProxy opens the worker ``/run`` stream via ``client.send`` (so it
    can sniff the response ``Content-Type`` before reading) and closes worker
    sessions via ``client.post``; patch both to forward to ``worker_transport``.
    """
    original_post = httpx.AsyncClient.post
    original_send = httpx.AsyncClient.send

    async def _forward(path, body):
        # [FUNCTION DOC]
        # HELPER: POSTs JSON body to path on a fresh client bound to the worker transport and
        # returns the response.
        # Use the unbound original ``send`` so the inner ASGI call does not
        # re-enter the patched ``post``/``send`` (which would recurse forever).
        async with httpx.AsyncClient(
            transport=worker_transport, base_url="http://worker"
        ) as wc:
            req = wc.build_request("POST", path, json=body)
            return await original_send(wc, req)

    async def patched_post(self, url, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Replacement for AsyncClient.post forwarding worker URLs.
        if "http://worker" in str(url):
            path = str(url).split("http://worker")[-1]
            return await _forward(path, kwargs.get("json"))
        return await original_post(self, url, **kwargs)

    async def patched_send(self, request, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Replacement for AsyncClient.send forwarding worker URLs (JSON-decoding the
        # request body) and returning a streamed response.
        if "http://worker" in str(request.url):
            path = str(request.url).split("http://worker")[-1]
            body = json.loads(request.content) if request.content else None
            r = await _forward(path, body)
            headers = {
                "content-type": r.headers.get("content-type", "application/json")
            }
            # Preserve the passthrough marker so the DataProxy's relay-vs-parse
            # decision is exercised faithfully end-to-end.
            if PASSTHROUGH_HEADER in r.headers:
                headers[PASSTHROUGH_HEADER] = r.headers[PASSTHROUGH_HEADER]
            # Return the body as a stream (the real DataProxy opens the worker
            # with ``stream=True``); a bytes payload would refuse ``aiter_raw``.
            raw = r.content

            async def _stream():
                # [FUNCTION DOC]
                # HELPER: Yields the buffered worker body once.
                yield raw

            return httpx.Response(
                r.status_code,
                headers=headers,
                content=_stream(),
                request=request,
            )
        return await original_send(self, request, **kwargs)

    return patched_post, patched_send


class TestWorkerDataProxyIntegration:
    # [CLASS DOC]
    # DataProxy <-> Worker chain tests (worker reached via patched httpx).
    """Test DataProxy → Worker chain using ASGITransport for the Worker."""

    @pytest.mark.asyncio
    async def test_single_turn(self):
        # [FUNCTION DOC]
        # Calls the worker /run directly and asserts 200 and the echo summary.
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        async with httpx.AsyncClient(
            transport=worker_transport, base_url="http://worker"
        ) as worker_client:
            # DataProxy forwards to worker — test worker directly first
            resp = await worker_client.post(
                "/run",
                json={
                    "message": "hello",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "echo: hello" in data["summary"]

    @pytest.mark.asyncio
    async def test_data_proxy_manages_history(self):
        # [FUNCTION DOC]
        # Sends two turns through the DataProxy for session s1 and asserts the stored history
        # has at least two entries afterwards.
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        patched_post, patched_send = _worker_patches(worker_transport)
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with (
            patch.object(httpx.AsyncClient, "post", patched_post),
            patch.object(httpx.AsyncClient, "send", patched_send),
        ):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                # Turn 1
                r1 = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "hello", "run_id": "r1"},
                )
                assert r1.status_code == 200
                assert "echo: hello" in r1.json()["summary"]

                # Turn 2 — history should have turn 1
                r2 = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "world", "run_id": "r2"},
                )
                assert r2.status_code == 200

                # Check history grew
                h = await proxy_client.get("/session/s1/history")
                history = h.json()["history"]
                assert len(history) >= 2  # at least user+assistant from turn 1

    @pytest.mark.asyncio
    async def test_close_session_clears_history(self):
        # [FUNCTION DOC]
        # After one turn, POST /session/s1/close and asserts history is empty.
        worker_app = _make_worker_app(_EchoAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        patched_post, patched_send = _worker_patches(worker_transport)
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with (
            patch.object(httpx.AsyncClient, "post", patched_post),
            patch.object(httpx.AsyncClient, "send", patched_send),
        ):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "hi", "run_id": "r1"},
                )
                await proxy_client.post("/session/s1/close")
                h = await proxy_client.get("/session/s1/history")
                assert h.json()["history"] == []

    @pytest.mark.asyncio
    async def test_json_passthrough_relayed_verbatim_not_parsed(self):
        # [FUNCTION DOC]
        # With the JSON passthrough agent, asserts the DataProxy relays the body byte-for-byte
        # and records no history.
        """A non-streaming passthrough body is itself ``application/json``; the
        DataProxy must relay it verbatim on the marker, never mistaking it for a
        structured turn (which would rebuild history and re-serialise)."""
        worker_app = _make_worker_app(_JsonPassthroughAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        patched_post, patched_send = _worker_patches(worker_transport)
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with (
            patch.object(httpx.AsyncClient, "post", patched_post),
            patch.object(httpx.AsyncClient, "send", patched_send),
        ):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                r = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "hi", "run_id": "r1"},
                )
                assert r.status_code == 200
                # Body relayed byte-for-byte (the agent's exact JSON).
                assert r.json() == {"verbatim": "hi", "events": []}
                # No history was kept on the passthrough path.
                h = await proxy_client.get("/session/s1/history")
                assert h.json()["history"] == []


class TestRouterIntegration:
    # [CLASS DOC]
    # Router registration/routing in isolation.
    @pytest.mark.asyncio
    async def test_register_and_route(self):
        # [FUNCTION DOC]
        # Registers one proxy and asserts the same session routes to it twice.
        router_app = create_router_app(
            RouterConfig(admin_api_key=DEFAULT_ADMIN_API_KEY)
        )
        transport = httpx.ASGITransport(app=router_app)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://router"
        ) as client:
            await client.post(
                "/register",
                json={"addr": "http://proxy1:9100"},
                headers=_AUTH,
            )
            resp = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp.json()["data_proxy_addr"] == "http://proxy1:9100"

            resp2 = await client.post(
                "/route", json={"session_key": "s1"}, headers=_AUTH
            )
            assert resp2.json()["data_proxy_addr"] == "http://proxy1:9100"


class TestToolCallFlow:
    # [CLASS DOC]
    # Tool-call event propagation through the DataProxy.
    @pytest.mark.asyncio
    async def test_tool_events_through_proxy(self):
        # [FUNCTION DOC]
        # With the tool agent, asserts the turn response has tool_call/tool_result events and
        # history contains a "tool" role message with a tool_call_id.
        worker_app = _make_worker_app(_ToolAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        patched_post, patched_send = _worker_patches(worker_transport)
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with (
            patch.object(httpx.AsyncClient, "post", patched_post),
            patch.object(httpx.AsyncClient, "send", patched_send),
        ):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as proxy_client:
                resp = await proxy_client.post(
                    "/session/s1/turn",
                    json={"message": "lookup 123", "run_id": "r1"},
                )
                data = resp.json()
                assert data["summary"] == "Lookup complete"
                events = data["events"]
                types = {e["type"] for e in events}
                assert "tool_call" in types
                assert "tool_result" in types

                # History should include tool call records
                h = await proxy_client.get("/session/s1/history")
                history = h.json()["history"]
                tool_msgs = [m for m in history if m.get("role") == "tool"]
                assert len(tool_msgs) > 0
                assert "tool_call_id" in tool_msgs[0]


class TestArealInference:
    # [CLASS DOC]
    # Tests that the DataProxy injects caller-supplied inference routing into worker
    # request metadata (areal_inference), caching it on the session.
    """Self-evolution (decoupled from the training side): the **caller** mints
    its own per-session ``sk-sess-*`` and passes it on the turn body as
    ``session_api_key``.  The DataProxy never contacts ``/rl/start_session`` —
    it caches the routing handle on the session and injects ``areal_inference``
    into the worker request metadata.  A multi-turn session may send the fields
    on the first turn and omit them afterwards.  Without the flag the worker
    metadata stays clean.
    """

    @staticmethod
    def _patched_send(worker_bodies, *, stream=False):
        # [FUNCTION DOC]
        # HELPER: Returns a replacement AsyncClient.send that, for URLs ending in "/run",
        # appends the decoded JSON body to worker_bodies and returns either a passthrough SSE
        # stream (stream=True; marked with the passthrough header) or a minimal JSON structured
        # response; other requests go to the original send.
        original_send = httpx.AsyncClient.send

        async def patched_send(self, request, **kwargs):
            # [FUNCTION DOC]
            # HELPER: The patched send closure described above.
            if str(request.url).endswith("/run"):
                worker_bodies.append(json.loads(request.content))
                if stream:

                    async def _s():
                        # [FUNCTION DOC]
                        # HELPER: Yields one SSE "data: ok" chunk.
                        yield b"data: ok\n\n"

                    return httpx.Response(
                        200,
                        headers={
                            "content-type": "text/event-stream",
                            PASSTHROUGH_HEADER: "1",
                        },
                        content=_s(),
                        request=request,
                    )
                return httpx.Response(
                    200,
                    json={"summary": "ok", "events": [], "metadata": {}},
                    request=request,
                )
            return await original_send(self, request, **kwargs)

        return patched_send

    @pytest.mark.asyncio
    async def test_turn_caches_inference_routing_once(self):
        # [FUNCTION DOC]
        # First turn carries inf_base_url/session_api_key/inf_model, second omits them; asserts
        # both worker requests carry metadata.areal_inference with those values and that the
        # response metadata does not expose areal_inference.
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        worker_bodies: list[dict] = []
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "send", self._patched_send(worker_bodies)):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as client:
                turn_body = {
                    "message": "hi",
                    "run_id": "r1",
                    "inf_base_url": "http://inf-gateway",
                    "session_api_key": "sk-sess-xyz",
                    "inf_model": "Qwen",
                }
                r1 = await client.post("/session/s1/turn", json=turn_body)
                assert r1.status_code == 200
                # The second turn omits the inference fields; the proxy reuses
                # the handle cached on the session.
                r2 = await client.post(
                    "/session/s1/turn", json={"message": "again", "run_id": "r2"}
                )
                assert r2.status_code == 200

        # The worker received the inference routing handle on both turns.
        assert len(worker_bodies) == 2
        for body in worker_bodies:
            inf = body["metadata"]["areal_inference"]
            assert inf["base_url"] == "http://inf-gateway"
            assert inf["api_key"] == "sk-sess-xyz"
            assert inf["model"] == "Qwen"

        # The Agent Service never surfaces a training-side session id back.
        assert "areal_inference" not in r2.json().get("metadata", {})

    @pytest.mark.asyncio
    async def test_turn_without_inference_fields_skips_inference(self):
        # [FUNCTION DOC]
        # Asserts a turn without inference fields sends no areal_inference metadata.
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        worker_bodies: list[dict] = []
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "send", self._patched_send(worker_bodies)):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as client:
                r = await client.post(
                    "/session/s2/turn", json={"message": "hi", "run_id": "r1"}
                )
                assert r.status_code == 200

        assert len(worker_bodies) == 1
        assert "areal_inference" not in worker_bodies[0]["metadata"]

    @pytest.mark.asyncio
    async def test_turn_requires_both_inference_fields(self):
        # [FUNCTION DOC]
        # Asserts a turn with inf_base_url but no session_api_key returns 400 and never
        # reaches the worker.
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        worker_bodies: list[dict] = []
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(httpx.AsyncClient, "send", self._patched_send(worker_bodies)):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as client:
                # inf_base_url present but session_api_key missing → 400 before
                # any worker call (the pair is required once either is sent).
                r = await client.post(
                    "/session/s3/turn",
                    json={"message": "hi", "inf_base_url": "http://inf-gateway"},
                )
                assert r.status_code == 400

        assert worker_bodies == []

    @pytest.mark.asyncio
    async def test_stream_turn_injects_inference(self):
        # [FUNCTION DOC]
        # With a streaming passthrough worker response, asserts the proxy relays the SSE body
        # verbatim and still injected areal_inference into the worker request.
        """The same routing injection applies to the raw-passthrough channel:
        when the worker returns a stream, the proxy still injects
        ``areal_inference`` into the worker request and relays the body."""
        proxy_app = create_data_proxy_app(DataProxyConfig(worker_addr="http://worker"))
        worker_bodies: list[dict] = []
        proxy_transport = httpx.ASGITransport(app=proxy_app)

        with patch.object(
            httpx.AsyncClient, "send", self._patched_send(worker_bodies, stream=True)
        ):
            async with httpx.AsyncClient(
                transport=proxy_transport, base_url="http://proxy"
            ) as client:
                r = await client.post(
                    "/session/s1/turn",
                    json={
                        "message": "hi",
                        "run_id": "r1",
                        "inf_base_url": "http://inf-gateway",
                        "session_api_key": "sk-sess-xyz",
                        "inf_model": "Qwen",
                    },
                )
                assert r.status_code == 200
                assert r.text == "data: ok\n\n"

        assert len(worker_bodies) == 1
        inf = worker_bodies[0]["metadata"]["areal_inference"]
        assert inf["base_url"] == "http://inf-gateway"
        assert inf["api_key"] == "sk-sess-xyz"
        assert inf["model"] == "Qwen"


class TestWorkerRunStream:
    # [CLASS DOC]
    # Worker /run behaviour for StreamResponse vs AgentResponse agents.
    """The single ``/run`` endpoint relays an agent's :class:`StreamResponse`
    verbatim (tagged with the passthrough marker header so the DataProxy relays
    rather than parses), and serialises an :class:`AgentResponse` to JSON
    otherwise."""

    @pytest.mark.asyncio
    async def test_run_relays_stream_verbatim_with_metadata(self):
        # [FUNCTION DOC]
        # Streams /run with a chat_request in metadata and asserts SSE content-type, the
        # passthrough marker header "1", and that all three SSE chunks arrive in order.
        worker_app = _make_worker_app(_StreamAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        async with httpx.AsyncClient(
            transport=worker_transport, base_url="http://worker"
        ) as client:
            async with client.stream(
                "POST",
                "/run",
                json={
                    "message": "hi",
                    "session_key": "s1",
                    "run_id": "r1",
                    "history": [],
                    "metadata": {"chat_request": {"model": "m", "stream": True}},
                },
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                # The worker tags raw-passthrough turns for the DataProxy.
                assert resp.headers.get(PASSTHROUGH_HEADER) == "1"
                body = b""
                async for chunk in resp.aiter_bytes():
                    body += chunk

        text = body.decode()
        assert "data: msg=hi\n\n" in text
        assert 'data: chat_request={"model": "m", "stream": true}\n\n' in text
        assert text.endswith("data: [DONE]\n\n")

    @pytest.mark.asyncio
    async def test_run_returns_json_for_structured_response(self):
        # [FUNCTION DOC]
        # Without chat_request asserts a JSON response, no passthrough marker, and summary
        # "structured".
        worker_app = _make_worker_app(_StreamAgent)
        worker_transport = httpx.ASGITransport(app=worker_app)

        async with httpx.AsyncClient(
            transport=worker_transport, base_url="http://worker"
        ) as client:
            resp = await client.post(
                "/run",
                json={"message": "hi", "session_key": "s1", "run_id": "r1"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/json")
            # No marker on a structured turn → the DataProxy parses it.
            assert PASSTHROUGH_HEADER not in resp.headers
            assert resp.json()["summary"] == "structured"


class TestGatewayHealth:
    # [CLASS DOC]
    # Gateway health endpoint test.
    @pytest.mark.asyncio
    async def test_health(self):
        # [FUNCTION DOC]
        # Asserts the gateway's GET /health returns 200 with status "ok".
        app = create_gateway_app(GatewayConfig(router_addr="http://fake-router"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gw"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


class TestGatewayCloseSession:
    # [CLASS DOC]
    # Gateway POST /sessions/close forwarding tests.
    """The gateway's ``POST /sessions/close`` (session_key in the body) resolves
    the owning DataProxy via the Router and forwards the close, mirroring the
    internal ``/session/{key}/close`` endpoint shape."""

    @staticmethod
    def _patched_post(calls):
        # [FUNCTION DOC]
        # HELPER: Returns a replacement AsyncClient.post: "/route" URLs answer with data proxy
        # http://proxy1; http://proxy1 ".../close" URLs are recorded in calls and answer ok;
        # others delegate to the original post.
        original_post = httpx.AsyncClient.post

        async def patched_post(self, url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: The patched post closure described above.
            url = str(url)
            if url.endswith("/route"):
                return httpx.Response(
                    200,
                    json={"data_proxy_addr": "http://proxy1"},
                    request=httpx.Request("POST", url),
                )
            if url.startswith("http://proxy1") and url.endswith("/close"):
                calls.append(url)
                return httpx.Response(
                    200, json={"status": "ok"}, request=httpx.Request("POST", url)
                )
            return await original_post(self, url, **kwargs)

        return patched_post

    @pytest.mark.asyncio
    async def test_close_forwards_to_data_proxy(self):
        # [FUNCTION DOC]
        # Asserts closing session s1 via the gateway returns ok and forwards exactly one close
        # to http://proxy1/session/s1/close.
        app = create_gateway_app(GatewayConfig(router_addr="http://fake-router"))
        transport = httpx.ASGITransport(app=app)
        calls: list[str] = []

        with patch.object(httpx.AsyncClient, "post", self._patched_post(calls)):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gw"
            ) as client:
                resp = await client.post(
                    "/sessions/close", json={"session_key": "s1"}, headers=_AUTH
                )
                assert resp.status_code == 200
                assert resp.json()["status"] == "ok"

        assert calls == ["http://proxy1/session/s1/close"]

    @pytest.mark.asyncio
    async def test_close_requires_session_key(self):
        # [FUNCTION DOC]
        # Asserts a close request without session_key returns 400.
        app = create_gateway_app(GatewayConfig(router_addr="http://fake-router"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gw"
        ) as client:
            resp = await client.post("/sessions/close", json={}, headers=_AUTH)
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_close_requires_admin_key(self):
        # [FUNCTION DOC]
        # Asserts a close request without admin auth returns 401 or 422.
        app = create_gateway_app(GatewayConfig(router_addr="http://fake-router"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gw"
        ) as client:
            resp = await client.post("/sessions/close", json={"session_key": "s1"})
            assert resp.status_code in (401, 422)


class TestBridgeExtractMessage:
    # [CLASS DOC]
    # Tests for OpenResponsesBridge._extract_message (Responses input items -> text).
    def test_text_message(self):
        # [FUNCTION DOC]
        # Asserts input_text content parts are extracted.
        items = [
            {
                "type": "message",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        assert OpenResponsesBridge._extract_message(items, "") == "Hello"

    def test_string_content(self):
        # [FUNCTION DOC]
        # Asserts plain string content is returned as is.
        items = [{"type": "message", "content": "Simple"}]
        assert OpenResponsesBridge._extract_message(items, "") == "Simple"

    def test_instructions_prepended(self):
        # [FUNCTION DOC]
        # Asserts instructions are prepended before the message text.
        items = [{"type": "message", "content": "Hi"}]
        result = OpenResponsesBridge._extract_message(items, "Be helpful")
        assert result.startswith("Be helpful")
        assert "Hi" in result

    def test_function_call_output(self):
        # [FUNCTION DOC]
        # Asserts function_call_output items render as "[tool result] <output>".
        items = [{"type": "function_call_output", "output": "42"}]
        result = OpenResponsesBridge._extract_message(items, "")
        assert "[tool result] 42" in result


class TestBridgeDeriveSessionKey:
    # [CLASS DOC]
    # Tests for OpenResponsesBridge._derive_session_key.
    def test_with_user(self):
        # [FUNCTION DOC]
        # Asserts key "agent:<model>:<user>" when a user is given.
        key = OpenResponsesBridge._derive_session_key("user1", "model1")
        assert key == "agent:model1:user1"

    def test_without_user_is_unique(self):
        # [FUNCTION DOC]
        # Asserts keys without a user are unique per call and prefixed "agent:<model>:".
        k1 = OpenResponsesBridge._derive_session_key("", "m")
        k2 = OpenResponsesBridge._derive_session_key("", "m")
        assert k1 != k2
        assert k1.startswith("agent:m:")

    def test_default_model(self):
        # [FUNCTION DOC]
        # Asserts an empty model becomes "default".
        key = OpenResponsesBridge._derive_session_key("u1", "")
        assert key == "agent:default:u1"


class TestChatBridgeResolveSessionKey:
    # [CLASS DOC]
    # Tests for ChatCompletionsBridge._resolve_session_key.
    """``ChatCompletionsBridge`` requires an explicit session key: the
    ``X-AReaL-Session-Key`` header wins, else it is derived from the OpenAI
    ``user`` field; with neither, the resolver returns ``None`` so the request
    is rejected with ``400`` instead of being minted a random key."""

    @staticmethod
    def _request(header: str | None):
        # [FUNCTION DOC]
        # HELPER: Returns a minimal request stub whose headers mapping contains the session-key
        # header only when header is truthy.
        """Minimal stand-in for a Starlette ``Request`` exposing ``headers.get``."""

        class _Headers(dict):
            # [CLASS DOC]
            # HELPER: dict subclass exposing get(key, default).
            def get(self, key, default=None):
                # [FUNCTION DOC]
                # Delegates to dict.get.
                return super().get(key, default)

        class _Req:
            # [CLASS DOC]
            # HELPER: Request stub carrying a headers attribute.
            def __init__(self, key):
                # [FUNCTION DOC]
                # Builds headers with the session-key header set to key if key is truthy.
                self.headers = _Headers({SESSION_KEY_HEADER: key} if key else {})

        return _Req(header)

    def test_explicit_header_wins(self):
        # [FUNCTION DOC]
        # Asserts the explicit header value wins over the user field.
        req = self._request("sess-1")
        key = ChatCompletionsBridge._resolve_session_key(
            req, {"user": "u1", "model": "m"}
        )
        assert key == "sess-1"

    def test_derived_from_user(self):
        # [FUNCTION DOC]
        # Asserts fallback key "chat:<model>:<user>".
        req = self._request(None)
        key = ChatCompletionsBridge._resolve_session_key(
            req, {"user": "u1", "model": "m"}
        )
        assert key == "chat:m:u1"

    def test_user_default_model(self):
        # [FUNCTION DOC]
        # Asserts missing model maps to "chat:default:<user>".
        req = self._request(None)
        key = ChatCompletionsBridge._resolve_session_key(req, {"user": "u1"})
        assert key == "chat:default:u1"

    def test_neither_returns_none(self):
        # [FUNCTION DOC]
        # Asserts None is returned with neither header nor user.
        req = self._request(None)
        key = ChatCompletionsBridge._resolve_session_key(req, {"model": "m"})
        assert key is None


class TestBridgeBuildOutputItems:
    # [CLASS DOC]
    # Tests for OpenResponsesBridge._build_output_items.
    """The bridge translates a structured turn result into OpenAI Responses
    output items: a non-empty summary → assistant ``message``; each
    ``tool_call`` event → a ``function_call`` item."""

    def test_summary_becomes_message_item(self):
        # [FUNCTION DOC]
        # Asserts a summary becomes a single assistant message item with output_text.
        items = OpenResponsesBridge._build_output_items({"summary": "hello"})
        assert items == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]

    def test_tool_call_events_become_function_calls(self):
        # [FUNCTION DOC]
        # Asserts only tool_call events become function_call items (results/deltas ignored)
        # and an empty summary yields no message item.
        result = {
            "summary": "",
            "events": [
                {"type": "tool_call", "name": "lookup", "args": '{"id": "1"}'},
                {"type": "tool_result", "name": "lookup", "result": "ok"},
                {"type": "delta", "text": "ignored"},
            ],
        }
        items = OpenResponsesBridge._build_output_items(result)
        assert items == [
            {"type": "function_call", "name": "lookup", "arguments": '{"id": "1"}'}
        ]

    def test_empty_result_yields_no_items(self):
        # [FUNCTION DOC]
        # Asserts an empty result yields no items.
        assert OpenResponsesBridge._build_output_items({}) == []


def _parse_sse(raw: bytes) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Splits raw SSE bytes on blank lines and JSON-decodes each "data: " block,
    # returning the list of event dicts (non-data blocks ignored).
    events = []
    for block in raw.decode().split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: ") :]))
    return events


class TestBridgeResponsesSSE:
    # [CLASS DOC]
    # Tests for OpenResponsesBridge._responses_sse SSE re-encoding.
    """``/v1/responses`` streaming (A): the bridge re-encodes the collected
    structured output into an OpenAI Responses-format SSE event stream
    (``response.created`` → ``output_text`` deltas → ``response.completed``)."""

    @pytest.mark.asyncio
    async def test_sse_reencodes_collected_output(self):
        # [FUNCTION DOC]
        # Collects OpenResponsesBridge._responses_sse output for a summary item and asserts
        # event order (created first, completed last), the text delta, and the completed
        # response's status, model, output and metadata.
        output_items = OpenResponsesBridge._build_output_items({"summary": "hi there"})
        raw = b""
        async for chunk in OpenResponsesBridge._responses_sse(
            "resp-1", "GLM-5", output_items, {"k": "v"}
        ):
            raw += chunk
        events = _parse_sse(raw)

        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert "response.output_text.delta" in types
        assert "response.output_text.done" in types
        assert types[-1] == "response.completed"

        delta = next(e for e in events if e["type"] == "response.output_text.delta")
        assert delta["delta"] == "hi there"

        completed = events[-1]["response"]
        assert completed["status"] == "completed"
        assert completed["model"] == "GLM-5"
        assert completed["output"] == output_items
        assert completed["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_sse_without_message_still_completes(self):
        # [FUNCTION DOC]
        # For tool-call-only output asserts only created and completed events are emitted.
        # Tool-call-only output (no assistant message) emits no text deltas but
        # still brackets the stream with created/completed.
        output_items = OpenResponsesBridge._build_output_items(
            {"events": [{"type": "tool_call", "name": "f", "args": "{}"}]}
        )
        raw = b""
        async for chunk in OpenResponsesBridge._responses_sse(
            "resp-2", "GLM-5", output_items, {}
        ):
            raw += chunk
        events = _parse_sse(raw)
        types = [e["type"] for e in events]
        assert types == ["response.created", "response.completed"]
        assert events[-1]["response"]["output"] == output_items


class TestBridgeStreamingEndToEnd:
    # [CLASS DOC]
    # End-to-end bridge tests via mount_bridge on a FastAPI app with patched httpx.
    """``handle_request`` with ``stream=True`` runs a structured collect turn,
    then returns a ``text/event-stream`` re-encoding it (A), echoing the
    resolved session key on the response header."""

    @staticmethod
    def _patched_post(turn_result):
        # [FUNCTION DOC]
        # HELPER: Returns a replacement AsyncClient.post answering "/route" with http://proxy1
        # and "/turn" with turn_result; other URLs delegate to the original.
        original_post = httpx.AsyncClient.post

        async def patched_post(self, url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: The patched post closure described above.
            url = str(url)
            if url.endswith("/route"):
                return httpx.Response(
                    200,
                    json={"data_proxy_addr": "http://proxy1"},
                    request=httpx.Request("POST", url),
                )
            if url.endswith("/turn"):
                return httpx.Response(
                    200, json=turn_result, request=httpx.Request("POST", url)
                )
            return await original_post(self, url, **kwargs)

        return patched_post

    @pytest.mark.asyncio
    async def test_stream_response_is_sse(self):
        # [FUNCTION DOC]
        # Mounts an OpenResponsesBridge on a FastAPI app and posts a streaming /v1/responses
        # request; asserts SSE content-type, the echoed session-key header, created/completed
        # bracketing and the delta text.
        from fastapi import FastAPI

        from areal.v2.agent_service.gateway.bridge import (
            mount_bridge,
        )

        bridge = OpenResponsesBridge(router_addr="http://router")
        app = FastAPI()
        mount_bridge(app, bridge)
        turn_result = {"summary": "streamed reply", "events": [], "metadata": {}}

        with patch.object(httpx.AsyncClient, "post", self._patched_post(turn_result)):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gw"
            ) as client:
                resp = await client.post(
                    "/v1/responses",
                    headers={**_AUTH, "X-AReaL-Session-Key": "sess-1"},
                    json={
                        "model": "GLM-5",
                        "stream": True,
                        "input": [{"type": "message", "content": "hi"}],
                    },
                )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers.get("X-AReaL-Session-Key") == "sess-1"
        events = _parse_sse(resp.content)
        types = [e["type"] for e in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        delta = next(e for e in events if e["type"] == "response.output_text.delta")
        assert delta["delta"] == "streamed reply"

    @pytest.mark.asyncio
    async def test_non_stream_response_is_json(self):
        # [FUNCTION DOC]
        # Non-streaming /v1/responses: asserts a JSON "response" object with status completed
        # and the reply text in the first output item.
        from fastapi import FastAPI

        from areal.v2.agent_service.gateway.bridge import (
            mount_bridge,
        )

        bridge = OpenResponsesBridge(router_addr="http://router")
        app = FastAPI()
        mount_bridge(app, bridge)
        turn_result = {"summary": "plain reply", "events": [], "metadata": {}}

        with patch.object(httpx.AsyncClient, "post", self._patched_post(turn_result)):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://gw"
            ) as client:
                resp = await client.post(
                    "/v1/responses",
                    headers={**_AUTH, "X-AReaL-Session-Key": "sess-2"},
                    json={
                        "model": "GLM-5",
                        "input": [{"type": "message", "content": "hi"}],
                    },
                )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output"][0]["content"][0]["text"] == "plain reply"
