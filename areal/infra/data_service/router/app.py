# [MODULE DOC]
# FastAPI application for the data service router: keeps an in-memory list of registered
# data worker URLs, polls their /health in the background, and hands out healthy workers
# round-robin. All endpoints except /health require the admin bearer key. FastAPI and
# pydantic are imported dynamically via importlib.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import importlib
from contextlib import asynccontextmanager

import aiohttp

from areal.infra.data_service.router.config import RouterConfig
from areal.utils import logging

_fastapi = importlib.import_module("fastapi")
FastAPI = _fastapi.FastAPI
HTTPException = _fastapi.HTTPException
Request = _fastapi.Request
BaseModel = importlib.import_module("pydantic").BaseModel

logger = logging.getLogger("DataRouter")


def _extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the token from an "Authorization: Bearer ..." header (case-
    # insensitive prefix); raises HTTPException 401 otherwise.
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header.",
    )


def _require_admin_key(request: Request, admin_key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Extracts the bearer token and compares it to admin_key in constant time;
    # raises HTTPException 403 if different. Returns the token.
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


class RegisterWorkerRequest(BaseModel):
    # [CLASS DOC]
    # Pydantic body for /register: worker_addr (base URL of a data worker).
    worker_addr: str


class UnregisterWorkerRequest(BaseModel):
    # [CLASS DOC]
    # Pydantic body for /unregister: worker_addr.
    worker_addr: str


def create_router_app(config: RouterConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Builds the router app for config. Closure state: registered worker list (insertion
    # order), per-worker health flags, a round-robin counter and an asyncio lock. A
    # lifespan context starts the background health poller on startup and cancels it on
    # shutdown. Returns the FastAPI app (health map also exposed on app.state).
    registered_workers: list[str] = []
    worker_healthy: dict[str, bool] = {}
    rr_idx: int = 0
    lock = asyncio.Lock()

    async def _poll_workers() -> None:
        # [FUNCTION DOC]
        # HELPER: Forever loop: concurrently GETs /health on every registered worker
        # (per-request timeout worker_health_timeout), marks each healthy iff status is
        # 200 (False on error), then sleeps poll_interval.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=config.worker_health_timeout),
            trust_env=False,
        ) as session:

            async def _check_health(addr: str) -> None:
                # [FUNCTION DOC]
                # HELPER: Health check for one worker updating its flag.
                try:
                    async with session.get(f"{addr}/health") as resp:
                        worker_healthy[addr] = resp.status == 200
                except Exception:
                    worker_healthy[addr] = False

            while True:
                await asyncio.gather(
                    *(_check_health(addr) for addr in list(registered_workers))
                )
                await asyncio.sleep(config.poll_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # [FUNCTION DOC]
        # HELPER: Lifespan context: logs startup, runs the poll task while the app is
        # alive, then cancels and awaits it and logs shutdown.
        logger.info("DataRouter starting")
        poll_task = asyncio.create_task(_poll_workers())
        yield
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass
        logger.info("DataRouter shutting down")

    app = FastAPI(title="AReaL Data Router", lifespan=lifespan)
    app.state.worker_healthy = worker_healthy

    @app.get("/health")
    async def health():
        # [FUNCTION DOC]
        # GET /health (no auth): returns status ok, number of registered workers and
        # number currently healthy.
        return {
            "status": "ok",
            "workers": len(registered_workers),
            "healthy": sum(1 for h in worker_healthy.values() if h),
        }

    @app.post("/register")
    async def register(body: RegisterWorkerRequest, request: Request):
        # [FUNCTION DOC]
        # POST /register (admin): adds the worker under the lock if new, marking it
        # healthy immediately, and logs INFO. Idempotent; returns status ok.
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            if body.worker_addr not in registered_workers:
                registered_workers.append(body.worker_addr)
                worker_healthy[body.worker_addr] = True
                logger.info(
                    "Worker registered: %s (total=%d)",
                    body.worker_addr,
                    len(registered_workers),
                )
        return {"status": "ok"}

    @app.post("/unregister")
    async def unregister(body: UnregisterWorkerRequest, request: Request):
        # [FUNCTION DOC]
        # POST /unregister (admin): removes the worker and its health flag if present,
        # logs INFO; returns status ok.
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            if body.worker_addr in registered_workers:
                registered_workers.remove(body.worker_addr)
                worker_healthy.pop(body.worker_addr, None)
                logger.info("Worker unregistered: %s", body.worker_addr)
        return {"status": "ok"}

    @app.post("/route")
    async def route(request: Request):
        # [FUNCTION DOC]
        # POST /route (admin): under the lock, picks the next healthy worker round-robin
        # (counter modulo the number of healthy workers, so the order shifts when health
        # changes) and returns {"worker_addr": ...}. Raises 503 if none are healthy.
        nonlocal rr_idx
        _require_admin_key(request, config.admin_api_key)
        async with lock:
            healthy = [
                addr for addr in registered_workers if worker_healthy.get(addr, False)
            ]
            if not healthy:
                raise HTTPException(
                    status_code=503,
                    detail="No healthy workers available",
                )
            addr = healthy[rr_idx % len(healthy)]
            rr_idx += 1
        return {"worker_addr": addr}

    @app.get("/workers")
    async def list_workers(request: Request):
        # [FUNCTION DOC]
        # GET /workers (admin): returns every registered worker with its addr and
        # healthy flag.
        _require_admin_key(request, config.admin_api_key)
        return {
            "workers": [
                {"addr": addr, "healthy": worker_healthy.get(addr, False)}
                for addr in registered_workers
            ]
        }

    return app
