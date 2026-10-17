# [MODULE DOC]
# FastAPI Gateway app: public entry point exposing a health check, an admin
# session-close endpoint and an authenticated WebSocket that accepts protocol req
# frames, routes each session via the Router, forwards the turn to the owning
# DataProxy over HTTP and replays the resulting events as protocol frames.
# SPDX-License-Identifier: Apache-2.0

"""Agent Gateway — public-facing HTTP/WebSocket server."""

from __future__ import annotations

import hmac
import json
import traceback
from typing import Any

import httpx
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from areal.utils import logging

from ..auth import admin_headers, make_admin_dependency
from ..protocol import (
    FrameType,
    RequestFrame,
    RequestMethod,
    RunStatus,
    generate_run_id,
    make_complete_response,
    make_delta_event,
    make_failed_response,
    make_tool_call_event,
    parse_frame,
    serialize_frame,
)
from .config import GatewayConfig

logger = logging.getLogger("AgentGateway")


def _make_accepted_json(request_id: str, run_id: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the JSON string of an ok "res" frame with status accepted for the
    # given request id and run id (equivalent to serializing make_accepted_response).
    return json.dumps(
        {
            "type": FrameType.RES,
            "id": request_id,
            "ok": True,
            "payload": {"runId": run_id, "status": RunStatus.ACCEPTED},
        }
    )


def create_gateway_app(config: GatewayConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Builds the Gateway app. Creates one shared httpx.AsyncClient (forward_timeout),
    # admin auth headers for calls to the Router, and the admin auth dependency for
    # inbound admin routes. Registers the routes below and returns the app.
    app = FastAPI(title="AReaL Agent Gateway")
    http_client = httpx.AsyncClient(timeout=config.forward_timeout)
    _auth_headers = admin_headers(config.admin_api_key)
    _admin = make_admin_dependency(config.admin_api_key)

    async def _route(session_key: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Asks the Router (POST /route, admin headers, router_timeout) which DataProxy
        # owns session_key; returns its address. Raises httpx errors on failure.
        resp = await http_client.post(
            f"{config.router_addr}/route",
            json={"session_key": session_key},
            headers=_auth_headers,
            timeout=config.router_timeout,
        )
        resp.raise_for_status()
        return resp.json()["data_proxy_addr"]

    async def _send_turn(
        data_proxy_addr: str,
        session_key: str,
        message: str,
        run_id: str,
        queue_mode: str,
        metadata: dict,
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: POSTs {message, run_id, queue_mode, metadata} to
        # <data_proxy>/session/<key>/turn and returns the parsed JSON result (expected to
        # contain "events" and "summary"). Raises on HTTP error.
        resp = await http_client.post(
            f"{data_proxy_addr}/session/{session_key}/turn",
            json={
                "message": message,
                "run_id": run_id,
                "queue_mode": queue_mode,
                "metadata": metadata,
            },
        )
        resp.raise_for_status()
        return resp.json()

    @app.get("/health")
    async def health():
        # [FUNCTION DOC]
        # GET /health: returns {"status": "ok"}.
        return {"status": "ok"}

    @app.post("/sessions/close", dependencies=[Depends(_admin)])
    async def close_session(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /sessions/close (admin): requires body "session_key" (HTTP 400 if empty),
        # routes it to its DataProxy and POSTs /session/<key>/close there. Returns status ok;
        # HTTP errors from Router/DataProxy propagate.
        # KNOWN ISSUE: routing an unknown key makes the Router assign it a new proxy as a
        # side effect before closing.
        """Close a session, forwarding to the DataProxy that owns it.

        The ``session_key`` is supplied in the request body.  Session lifecycle
        is keyed solely by ``session_key`` and is identical across every gateway
        surface (WebSocket, ``/v1/responses``, ``/v1/chat/completions``): the
        Router pins a key to one DataProxy via route affinity, so resolving the
        key here lands on the DataProxy holding that session's state.  The
        DataProxy close is idempotent and propagates down to the Worker, so an
        unknown key is a no-op success.
        """
        session_key = body.get("session_key", "")
        if not session_key:
            raise HTTPException(status_code=400, detail="'session_key' is required")
        data_proxy_addr = await _route(session_key)
        resp = await http_client.post(f"{data_proxy_addr}/session/{session_key}/close")
        resp.raise_for_status()
        return {"status": "ok"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
        # [FUNCTION DOC]
        # WebSocket /ws: the "token" query param must equal the admin key (constant-time),
        # else closes with code 4001. Then loops reading text frames: non-req frames,
        # non-agent methods and missing sessionKey get a failed res frame. Valid requests
        # get an accepted frame with a fresh run id, then the turn is routed and sent
        # synchronously (metadata is the full params dict); returned delta and tool_call
        # events are sent as event frames (tool_result events are dropped), followed by a
        # complete frame with the summary. Turn errors are logged and returned as a failed
        # frame. Disconnect ends the loop (logged).
        # KNOWN ISSUE: a malformed frame makes parse_frame raise, which escapes to the outer
        # handler and silently ends the connection without closing it or sending an error.
        if not hmac.compare_digest(token, config.admin_api_key):
            await websocket.close(code=4001, reason="Invalid admin key")
            return
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        try:
            while True:
                raw = await websocket.receive_text()
                frame = parse_frame(raw)

                if not isinstance(frame, RequestFrame):
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response("unknown", "", "Expected req frame")
                        )
                    )
                    continue

                if frame.method != RequestMethod.AGENT:
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(
                                frame.id, "", f"Unsupported method: {frame.method}"
                            )
                        )
                    )
                    continue

                session_key = frame.session_key
                if not session_key:
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(
                                frame.id, "", "Missing sessionKey in params"
                            )
                        )
                    )
                    continue

                run_id = generate_run_id()
                await websocket.send_text(_make_accepted_json(frame.id, run_id))

                try:
                    data_proxy_addr = await _route(session_key)
                    result = await _send_turn(
                        data_proxy_addr=data_proxy_addr,
                        session_key=session_key,
                        message=frame.message,
                        run_id=run_id,
                        queue_mode=frame.queue_mode.value,
                        metadata=frame.params,
                    )

                    for evt in result.get("events", []):
                        if evt.get("type") == "delta":
                            await websocket.send_text(
                                serialize_frame(
                                    make_delta_event(run_id, evt.get("text", ""))
                                )
                            )
                        elif evt.get("type") == "tool_call":
                            await websocket.send_text(
                                serialize_frame(
                                    make_tool_call_event(
                                        run_id,
                                        evt.get("name", ""),
                                        evt.get("args", ""),
                                    )
                                )
                            )

                    await websocket.send_text(
                        serialize_frame(
                            make_complete_response(
                                frame.id, run_id, result.get("summary", "")
                            )
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Run %s failed: %s\n%s", run_id, exc, traceback.format_exc()
                    )
                    await websocket.send_text(
                        serialize_frame(
                            make_failed_response(frame.id, run_id, str(exc))
                        )
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception:
            logger.exception("Unexpected error in WebSocket handler")

    @app.on_event("shutdown")
    async def shutdown():
        # [FUNCTION DOC]
        # Shutdown hook: closes the shared HTTP client.
        await http_client.aclose()

    return app
