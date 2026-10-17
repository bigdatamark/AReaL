# [MODULE DOC]
# FastAPI app factory and forwarding helpers for the training gateway. Each request's
# bearer token is resolved by the router to a data proxy address ("model_addr") and the
# request is forwarded there with shared httpx clients created in the lifespan.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from areal.utils import logging
from areal.v2.training_service.gateway import streaming
from areal.v2.training_service.gateway.auth import extract_bearer_token
from areal.v2.training_service.gateway.config import GatewayConfig
from areal.v2.training_service.gateway.engine import register_engine_routes

logger = logging.getLogger("TrainGateway")


def _router_error_response(exc: Exception) -> JSONResponse:
    # [FUNCTION DOC]
    # HELPER: Maps router errors to JSON responses: unreachable -> 502; key rejected ->
    # its status, with 404 converted to 401; anything else -> 500.
    if isinstance(exc, streaming.RouterUnreachableError):
        return JSONResponse({"error": str(exc)}, status_code=502)
    if isinstance(exc, streaming.RouterKeyRejectedError):
        status = 401 if exc.status_code == 404 else exc.status_code
        return JSONResponse({"error": exc.detail}, status_code=status)
    return JSONResponse({"error": str(exc)}, status_code=500)


async def _forward_post(
    request: Request,
    path: str,
    config: GatewayConfig,
    *,
    use_admin_auth_for_upstream: bool = False,
) -> Response:
    # [FUNCTION DOC]
    # HELPER: Extracts the bearer token (401 if missing), asks the router for the data
    # proxy address of that key (router errors mapped), and POSTs the raw body with
    # filtered headers to address+path using forward_timeout. With
    # use_admin_auth_for_upstream, replaces any Authorization header with the admin key.
    # Returns the upstream content/status/content-type, or 502 on forwarding exceptions.
    # Relies on router_client/upstream_client on app.state (requires the lifespan).
    token = extract_bearer_token(request)
    try:
        model_addr = await streaming.query_router(
            config.router_addr,
            token,
            config.router_timeout,
            admin_api_key=config.admin_api_key,
            client=request.app.state.router_client,
        )
    except (streaming.RouterUnreachableError, streaming.RouterKeyRejectedError) as exc:
        return _router_error_response(exc)

    body = await request.body()
    headers = streaming._forwarding_headers(dict(request.headers))
    if use_admin_auth_for_upstream:
        for key in list(headers.keys()):
            if key.lower() == "authorization":
                headers.pop(key)
        headers["Authorization"] = f"Bearer {config.admin_api_key}"
    try:
        resp = await streaming.forward_request(
            f"{model_addr}{path}",
            body,
            headers,
            config.forward_timeout,
            client=request.app.state.upstream_client,
        )
    except Exception as exc:
        logger.error("Forwarding POST failed for %s: %s", path, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


async def _forward_get(request: Request, path: str, config: GatewayConfig) -> Response:
    # [FUNCTION DOC]
    # HELPER: Same routing as _forward_post, then GETs address+path with filtered
    # headers and passes the response through; 502 on exceptions.
    token = extract_bearer_token(request)
    try:
        model_addr = await streaming.query_router(
            config.router_addr,
            token,
            config.router_timeout,
            admin_api_key=config.admin_api_key,
            client=request.app.state.router_client,
        )
    except (streaming.RouterUnreachableError, streaming.RouterKeyRejectedError) as exc:
        return _router_error_response(exc)

    try:
        resp = await request.app.state.upstream_client.get(
            f"{model_addr}{path}",
            headers=streaming._forwarding_headers(dict(request.headers)),
            timeout=config.forward_timeout,
        )
    except Exception as exc:
        logger.error("Forwarding GET failed for %s: %s", path, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
    )


def create_app(config: GatewayConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Creates the app whose lifespan builds a router httpx client (router_timeout) and
    # an upstream client (forward_timeout) on app.state and closes both on shutdown;
    # registers the engine routes with the forwarding helpers and returns the app.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # [FUNCTION DOC]
        # HELPER: Creates and later closes the two shared httpx clients.
        router_client = httpx.AsyncClient(timeout=config.router_timeout)
        upstream_client = httpx.AsyncClient(timeout=config.forward_timeout)
        app.state.router_client = router_client
        app.state.upstream_client = upstream_client
        try:
            yield
        finally:
            await upstream_client.aclose()
            await router_client.aclose()

    app = FastAPI(title="AReaL Training Gateway", lifespan=lifespan)

    register_engine_routes(
        app,
        config,
        _forward_post=_forward_post,
        _forward_get=_forward_get,
    )

    return app
