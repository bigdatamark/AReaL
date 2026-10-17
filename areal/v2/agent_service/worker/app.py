# [MODULE DOC]
# FastAPI Worker app: loads one AgentRunnable implementation and exposes /run for
# executing turns, plus health, per-session close and shutdown hooks. Holds no
# session state itself.
# SPDX-License-Identifier: Apache-2.0

"""Agent Worker — stateless HTTP server for agent execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

from ..protocol import PASSTHROUGH_HEADER, QueueMode
from ..types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    StreamResponse,
)

logger = logging.getLogger("AgentWorker")


class _CollectingEmitter:
    # [CLASS DOC]
    # HELPER: In-memory EventEmitter implementation that appends each event as a dict
    # to self.events so it can be returned with the structured response.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes the empty events list.
        self.events: list[dict[str, Any]] = []

    async def emit_delta(self, text: str) -> None:
        # [FUNCTION DOC]
        # Appends {"type": "delta", "text"}.
        self.events.append({"type": "delta", "text": text})

    async def emit_tool_call(self, name: str, args: str) -> None:
        # [FUNCTION DOC]
        # Appends {"type": "tool_call", "name", "args"}.
        self.events.append({"type": "tool_call", "name": name, "args": args})

    async def emit_tool_result(self, name: str, result: str) -> None:
        # [FUNCTION DOC]
        # Appends {"type": "tool_result", "name", "result"}.
        self.events.append({"type": "tool_result", "name": name, "result": result})


def create_worker_app(
    agent_cls_path: str,
    **agent_kwargs: Any,
) -> FastAPI:
    # [FUNCTION DOC]
    # Imports the class at agent_cls_path, instantiates it with agent_kwargs, verifies
    # it satisfies AgentRunnable (raises TypeError otherwise), logs, and returns a
    # FastAPI app with the routes below.
    app = FastAPI(title="AReaL Agent Worker")

    cls = import_from_string(agent_cls_path)
    agent: AgentRunnable = cls(**agent_kwargs)
    if not isinstance(agent, AgentRunnable):
        raise TypeError(
            f"Loaded class {agent_cls_path} does not satisfy AgentRunnable protocol "
            f"(missing async def run(request, *, emitter) method)"
        )
    logger.info("Agent loaded: %s", agent_cls_path)

    @app.get("/health")
    async def health():
        # [FUNCTION DOC]
        # GET /health: returns {"status": "ok"}.
        return {"status": "ok"}

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        # [FUNCTION DOC]
        # POST /session/<key>/close: awaits the agent's optional close_session(key) if
        # defined. Returns status ok.
        close_fn = getattr(agent, "close_session", None)
        if close_fn is not None:
            await close_fn(session_key)
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def shutdown():
        # [FUNCTION DOC]
        # App shutdown hook: awaits the agent's optional close_all_sessions() if defined.
        close_all_fn = getattr(agent, "close_all_sessions", None)
        if close_all_fn is not None:
            await close_all_fn()

    @app.post("/run")
    async def run(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /run: builds an AgentRequest from the JSON body (missing fields default to
        # empty; queue_mode parsed into QueueMode, invalid value raises ValueError -> 500),
        # runs the agent with a collecting emitter. On agent exception logs and returns a
        # 500 JSON {"error": {message, type}}. For StreamResponse, strips content-length,
        # transfer-encoding and connection headers, adds the passthrough marker header and
        # streams the body with the agent's status/content-type. Otherwise returns the
        # AgentResponse fields as JSON plus "events" from the emitter.
        """Single agent entry point for every protocol and streaming mode.

        Calls the agent's ``run`` and relays whichever shape it returns:

        - :class:`StreamResponse` — raw passthrough; ``status_code`` /
          ``headers`` / ``body`` are forwarded untouched so the caller gets the
          upstream's exact wire format (e.g. SSE chat completions).  The
          response carries the :data:`PASSTHROUGH_HEADER` marker so the
          DataProxy relays it verbatim instead of parsing it — this works even
          for a *non-streaming* passthrough whose body is ``application/json``.
        - :class:`AgentResponse` — structured JSON ``{summary, metadata,
          events}`` (``application/json``); the DataProxy rebuilds history from
          ``events``.
        """
        request = AgentRequest(
            message=body.get("message", ""),
            session_key=body.get("session_key", ""),
            run_id=body.get("run_id", ""),
            history=body.get("history", []),
            queue_mode=QueueMode(body.get("queue_mode", "collect")),
            metadata=body.get("metadata", {}),
        )

        emitter = _CollectingEmitter()

        try:
            response: AgentResponse | StreamResponse = await agent.run(
                request, emitter=emitter
            )
        except Exception as exc:
            logger.exception("Agent run failed (session=%s)", request.session_key)
            return JSONResponse(
                {"error": {"message": str(exc), "type": type(exc).__name__}},
                status_code=500,
            )

        if isinstance(response, StreamResponse):
            # Drop hop-by-hop / length headers that would conflict with chunked
            # relaying; FastAPI/uvicorn set framing headers themselves.
            headers = {
                k: v
                for k, v in response.headers.items()
                if k.lower()
                not in ("content-length", "transfer-encoding", "connection")
            }
            # Mark the turn as raw-passthrough so the DataProxy relays it
            # verbatim regardless of its Content-Type.
            headers[PASSTHROUGH_HEADER] = "1"
            return StreamingResponse(
                response.body,
                status_code=response.status_code,
                headers=headers,
                media_type=response.headers.get("content-type"),
            )

        return {**asdict(response), "events": emitter.events}

    return app
