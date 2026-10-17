# [MODULE DOC]
# FastAPI Router app that maps session keys to registered DataProxy addresses with
# round-robin assignment and sticky (session-affine) reuse. All state is in-memory
# inside the app factory closure.
# SPDX-License-Identifier: Apache-2.0

"""Agent Router — session-affine routing service."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from areal.utils import logging

from ..auth import make_admin_dependency
from .config import RouterConfig

logger = logging.getLogger("AgentRouter")


def create_router_app(config: RouterConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Builds the Router FastAPI app. State: list of registered proxy addresses, a
    # session_key -> proxy map, a round-robin counter and an asyncio.Lock guarding both.
    # Admin endpoints require the bearer admin key from config. Returns the app.
    # KNOWN ISSUE: config.poll_interval and worker_health_timeout are never used (no
    # health polling of proxies is implemented).
    app = FastAPI(title="AReaL Agent Router")
    auth = make_admin_dependency(config.admin_api_key)

    registered_proxies: list[str] = []
    session_map: dict[str, str] = {}
    rr_idx = 0
    lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        # [FUNCTION DOC]
        # GET /health (no auth): returns status "ok" plus counts of registered proxies and
        # active sessions.
        return {
            "status": "ok",
            "registered_proxies": len(registered_proxies),
            "active_sessions": len(session_map),
        }

    @app.post("/register", dependencies=[Depends(auth)])
    async def register(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /register (admin): adds body["addr"] to the proxy list if absent, logs it.
        # Returns {"status": "ok"}. Missing "addr" raises KeyError (500).
        addr = body["addr"]
        async with lock:
            if addr not in registered_proxies:
                registered_proxies.append(addr)
                logger.info(
                    "Registered DataProxy: %s (total=%d)", addr, len(registered_proxies)
                )
        return {"status": "ok"}

    @app.post("/unregister", dependencies=[Depends(auth)])
    async def unregister(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /unregister (admin): removes body["addr"] from the proxy list and drops all
        # sessions mapped to it; logs count removed. No-op if unknown. Returns status ok.
        addr = body["addr"]
        async with lock:
            if addr in registered_proxies:
                registered_proxies.remove(addr)
                stale = [k for k, v in session_map.items() if v == addr]
                for k in stale:
                    del session_map[k]
                logger.info(
                    "Unregistered DataProxy: %s (removed %d sessions)", addr, len(stale)
                )
        return {"status": "ok"}

    @app.post("/route", dependencies=[Depends(auth)])
    async def route(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /route (admin): returns {"data_proxy_addr"} for body["session_key"]. Existing
        # mapping is reused; otherwise picks the next proxy round-robin, records the
        # mapping and logs it. Returns 503 JSON error if no proxy is registered.
        nonlocal rr_idx
        session_key = body["session_key"]

        async with lock:
            if session_key in session_map:
                return {"data_proxy_addr": session_map[session_key]}

            if not registered_proxies:
                return JSONResponse(
                    {"error": "No DataProxy registered"}, status_code=503
                )

            addr = registered_proxies[rr_idx % len(registered_proxies)]
            rr_idx += 1
            session_map[session_key] = addr
            logger.info("Routed session %s → %s", session_key, addr)

        return {"data_proxy_addr": addr}

    @app.post("/remove_session", dependencies=[Depends(auth)])
    async def remove_session(body: dict[str, Any]):
        # [FUNCTION DOC]
        # POST /remove_session (admin): forgets the mapping for body["session_key"]
        # (no-op if absent). Returns status ok.
        session_key = body["session_key"]
        async with lock:
            session_map.pop(session_key, None)
        return {"status": "ok"}

    return app
