# [MODULE DOC]
# FastAPI application for the data proxy: sits between the router/gateway and one
# inference backend (SGLang/vLLM via InfBridge) or external OpenAI-compatible providers.
# Serves OpenAI chat completions and Anthropic messages while recording per-session
# interactions, manages RL sessions and rewards, exports trajectories (large tensors as
# RTensor shards stored on this node), forwards pause/resume/offload/version control,
# pushes "trajectory ready" callbacks to the controller, and mounts the RTensor data
# blueprint.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from anthropic.types.message import Message
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import Response as RawResponse
from fastapi.responses import StreamingResponse
from flask import Flask
from pydantic import BaseModel

from areal.experimental.openai.anthropic import (
    MessagePreprocessor,
    translate_anthropic_request,
    translate_anthropic_response,
    translate_anthropic_stream,
)
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    concat_string_interactions,
    normalize_group_rewards,
)
from areal.infra.rpc.guard.data_blueprint import (
    data_bp,
)
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import serialize_value
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.seeding import derive_deterministic_seed
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import (
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    ReadyNotification,
    SessionCredentials,
    SessionData,
    SessionStore,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
)
from areal.v2.inference_service.data_proxy.tokenizer_proxy import (
    TokenizerProxy,
)
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend
from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend

logger = logging.getLogger("InferenceDataProxy")

_INLINE_TRAJECTORY_FIELDS = frozenset(("rewards", "original_rewards"))


def _remotize_trajectory(traj: dict[str, Any], node_addr: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Splits traj into inline reward fields ("rewards", "original_rewards") kept
    # as-is and the rest, which is converted to RTensor references stored at node_addr.
    # Returns the merged dict.
    """Keep scalar reward metadata local while remotizing large tensors."""
    inline = {key: traj[key] for key in _INLINE_TRAJECTORY_FIELDS if key in traj}
    remote_payload = {key: value for key, value in traj.items() if key not in inline}
    remotized = RTensor.remotize(remote_payload, node_addr=node_addr)
    return {**remotized, **inline}


async def _safe_stream_wrapper(stream: AsyncGenerator[Any, None]):
    # [FUNCTION DOC]
    # HELPER: Async generator re-yielding stream items and always calling the stream's
    # aclose (if present) when iteration ends or the client disconnects.
    """Close an upstream stream when forwarding ends or the client disconnects."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()


# =============================================================================
# Response models
# =============================================================================


class DataProxyHealthResponse(BaseModel):
    # [CLASS DOC]
    # Health payload: status, backend address, session count, paused flag, weight
    # version.
    status: str
    backend: str | None
    sessions: int
    paused: bool
    version: int


class DataProxyStatusResponse(BaseModel):
    # [CLASS DOC]
    # Generic {status} response.
    status: str


class PauseGenerationResponse(BaseModel):
    # [CLASS DOC]
    # {status, paused} response for pause/continue endpoints.
    status: str
    paused: bool


class SetVersionResponse(BaseModel):
    # [CLASS DOC]
    # {status, version} response.
    status: str
    version: int


class GetVersionResponse(BaseModel):
    # [CLASS DOC]
    # {version} response.
    version: int


class SetRewardResponse(BaseModel):
    # [CLASS DOC]
    # Reward response: message, interaction count, session id, trajectory id (if a
    # trajectory became/was ready), trajectory_ready flag, and whether this call caused
    # the ready transition.
    message: str
    interaction_count: int
    session_id: str
    trajectory_id: int | None
    trajectory_ready: bool
    ready_transition: bool


class RegisterModelResponse(BaseModel):
    # [CLASS DOC]
    # {status, name} response.
    status: str
    name: str


class ConfigureBackendResponse(BaseModel):
    # [CLASS DOC]
    # {status, backend_addr} response.
    status: str
    backend_addr: str


# =============================================================================
# API Key helpers (for RL control-plane endpoints only)
# =============================================================================


def _extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the token from "Authorization: Bearer" (case-insensitive prefix)
    # or the x-api-key header; raises HTTPException 401 if neither is present.
    """Extract API token from an OpenAI or Anthropic authentication header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key:
        return x_api_key
    raise HTTPException(
        status_code=401,
        detail=(
            "Missing or malformed authentication header. Expected "
            "'Bearer <token>' or 'x-api-key: <token>'."
        ),
    )


def _require_admin_key(request: Request, store: SessionStore) -> str:
    # [FUNCTION DOC]
    # HELPER: Extracts the token and constant-time compares it with the store's admin
    # key; 403 on mismatch. Returns the token.
    """Validate that the request carries the admin API key."""
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, store.admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request, store: SessionStore) -> str:
    # [FUNCTION DOC]
    # HELPER: Extracts the token and returns the session id it maps to; 401 if unknown.
    """Resolve session_id from an OpenAI or Anthropic session API key."""
    token = _extract_bearer_token(request)
    session = store.get_session_by_api_key(token)
    if session is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session API key."
        )
    return session.session_id


def _resolve_session_from_token(
    token: str | None,
    store: SessionStore,
) -> SessionData | None:
    # [FUNCTION DOC]
    # HELPER: Returns the session for a session key, the persistent human-in-the-loop
    # session for the admin key (created on demand), or None for missing/unknown tokens.
    """Resolve a session from an API token.

    Session key → lookup by API key.
    Admin key → persistent HITL session.
    """
    if token is None:
        return None
    session = store.get_session_by_api_key(token)
    if session is not None:
        return session
    if hmac.compare_digest(token, store.admin_api_key):
        return store.get_or_create_hitl_session()
    return None


def _try_extract_bearer_token(request: Request) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Like _extract_bearer_token but returns None instead of raising.
    """Extract an OpenAI or Anthropic token if present.

    Unlike _extract_bearer_token, this never raises — it's for endpoints
    that accept requests with or without auth.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key:
        return x_api_key
    return None


def _create_inf_bridge(
    backend_addr: str,
    pause_state: PauseState,
    config: DataProxyConfig,
) -> InfBridge:
    # [FUNCTION DOC]
    # HELPER: Builds an InfBridge with an SGLang or vLLM backend adapter (ValueError for
    # other types), the pause state, request timeout and resubmit settings from config.
    """Create an InfBridge instance from proxy config."""
    if config.backend_type == "sglang":
        backend = SGLangBridgeBackend()
    elif config.backend_type == "vllm":
        backend = VLLMBridgeBackend()
    else:
        raise ValueError(f"Unsupported backend_type: {config.backend_type!r}")

    return InfBridge(
        backend=backend,
        backend_addr=backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )


def _create_areal_client(
    inf_bridge: InfBridge,
    tok: TokenizerProxy,
    config: DataProxyConfig,
) -> ArealOpenAI:
    # [FUNCTION DOC]
    # HELPER: Builds an ArealOpenAI client using the bridge as engine, the proxy's
    # underlying tokenizer, and parser/template/max-token options from config.
    """Create an ArealOpenAI client backed by the given InfBridge."""
    return ArealOpenAI(
        engine=inf_bridge,
        tokenizer=tok._tok,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        engine_max_tokens=config.engine_max_tokens,
        chat_template_type=config.chat_template_type,
    )


async def _post_online_ready_callback(
    callback_server_addr: str,
    admin_api_key: str,
    notification: ReadyNotification,
    timeout: float,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: POSTs {session_id, trajectory_id} to <callback>/callback/online_ready with
    # admin auth using the given or a temporary client. Returns True on status < 400;
    # False when no callback address is configured, on error status, or on any exception
    # (all logged).
    if not callback_server_addr:
        return False

    callback_base = callback_server_addr.rstrip("/")
    try:

        async def _do(c: httpx.AsyncClient) -> httpx.Response:
            # [FUNCTION DOC]
            # HELPER: Performs the callback POST with the given client.
            return await c.post(
                f"{callback_base}/callback/online_ready",
                json={
                    "session_id": notification.session_id,
                    "trajectory_id": notification.trajectory_id,
                },
                headers={"Authorization": f"Bearer {admin_api_key}"},
                timeout=timeout,
            )

        if client is not None:
            resp = await _do(client)
        else:
            async with create_httpx_client(timeout=timeout) as c:
                resp = await _do(c)

        if resp.status_code >= 400:
            logger.warning(
                "Online ready callback failed for %s/%s with %d: %s",
                notification.session_id,
                notification.trajectory_id,
                resp.status_code,
                resp.text,
            )
            return False
        return True
    except Exception as exc:
        logger.warning(
            "Online ready callback unreachable for %s/%s: %s",
            notification.session_id,
            notification.trajectory_id,
            exc,
        )
        return False


async def _flush_ready_trajectories(app: FastAPI) -> None:
    # [FUNCTION DOC]
    # HELPER: Finalizes rewarded trajectories in the session store (logging each ready
    # one), then concurrently delivers all pending online-ready notifications and marks
    # the delivered ones in the store so they are not resent.
    store: SessionStore = app.state.session_store
    config: DataProxyConfig = app.state.config
    http_client: httpx.AsyncClient | None = getattr(app.state, "http_client", None)

    for ready_result in store.finalize_rewarded_trajectories():
        logger.info(
            "Trajectory ready: session=%s trajectory=%s interactions=%s",
            ready_result.session_id,
            ready_result.trajectory_id,
            ready_result.interaction_count,
        )

    pending_notifications = store.pending_online_callbacks()

    async def _deliver(
        notification: ReadyNotification,
    ) -> tuple[ReadyNotification, bool]:
        # [FUNCTION DOC]
        # HELPER: Delivers one notification, treating any exception as not delivered.
        # Returns (notification, delivered).
        try:
            delivered = await _post_online_ready_callback(
                config.callback_server_addr,
                config.admin_api_key,
                notification,
                config.request_timeout,
                client=http_client,
            )
        except BaseException as exc:
            logger.warning(
                "Callback delivery failed for %s/%s: %s",
                notification.session_id,
                notification.trajectory_id,
                exc,
            )
            delivered = False
        return notification, delivered

    results = await asyncio.gather(*[_deliver(n) for n in pending_notifications])
    for notification, delivered in results:
        if delivered:
            store.mark_online_callback_delivered(
                notification.session_id,
                notification.trajectory_id,
            )


async def _ready_trajectory_loop(app: FastAPI) -> None:
    # [FUNCTION DOC]
    # HELPER: Background task: flushes ready trajectories every 0.1s forever.
    while True:
        await _flush_ready_trajectories(app)
        await asyncio.sleep(0.1)


def create_app(config: DataProxyConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Builds the FastAPI app. Eagerly imports and instantiates configured message
    # preprocessor classes and imports the optional prefix matcher. The lifespan sets up
    # app.state (pause state, config, SessionStore with admin key, version 0, shared
    # httpx client, and when a backend address is configured a TokenizerProxy, InfBridge
    # and ArealOpenAI client; otherwise these are None) and runs the ready-trajectory
    # loop, closing everything on shutdown. Registers the endpoints below plus a WSGI
    # mount of the RTensor data blueprint at "/". Keeps a closure-local registry of
    # external models. Returns the app.
    """Factory that creates the FastAPI app with lifespan-managed resources."""

    message_preprocessors: list[MessagePreprocessor] = []
    for path in config.message_preprocessors:
        preprocessor_cls = import_from_string(path)
        message_preprocessors.append(preprocessor_cls())
        logger.info("Loaded message preprocessor: %s", path)

    prefix_matcher = None
    if config.prefix_matcher:
        prefix_matcher = import_from_string(config.prefix_matcher)
        logger.info("Loaded prefix matcher: %s", config.prefix_matcher)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # [FUNCTION DOC]
        # HELPER: Async context manager creating and tearing down the app.state
        # resources and the background ready-trajectory task described above.
        logger.info(
            "Data proxy starting — backend=%s, tokenizer=%s",
            config.backend_addr or "(none)",
            config.tokenizer_path,
        )

        pause_state = PauseState()
        app.state.pause_state = pause_state
        app.state.config = config
        app.state.session_store = SessionStore(
            set_reward_finish_timeout=config.set_reward_finish_timeout,
            prefix_matcher=prefix_matcher,
        )
        app.state.session_store.set_admin_key(config.admin_api_key)
        app.state.version = 0
        app.state.http_client = create_httpx_client(timeout=config.request_timeout)

        if not config.backend_addr:
            app.state.tokenizer = None
            app.state.inf_bridge = None
            app.state.areal_client = None
        else:
            tok = TokenizerProxy(config.tokenizer_path)
            inf_bridge = _create_inf_bridge(config.backend_addr, pause_state, config)
            areal_client = _create_areal_client(inf_bridge, tok, config)
            app.state.tokenizer = tok
            app.state.inf_bridge = inf_bridge
            app.state.areal_client = areal_client

        ready_task = asyncio.create_task(_ready_trajectory_loop(app))
        try:
            yield
        finally:
            ready_task.cancel()
            try:
                await ready_task
            except asyncio.CancelledError:
                pass
            if app.state.inf_bridge is not None:
                await app.state.inf_bridge.aclose()
            await app.state.http_client.aclose()
        logger.info("Data proxy shutting down")

    app = FastAPI(title="AReaL Data Proxy", lifespan=lifespan)
    _registered_models: dict[str, dict[str, str | None]] = {}

    async def _create_internal_chat_result(
        body_json: dict[str, Any],
        session: SessionData | None,
    ) -> tuple[Any, bool]:
        # [FUNCTION DOC]
        # HELPER: Runs a chat completion through the ArealOpenAI client (503 if no
        # backend). Drops "model", defaults temperature/top_p to 1.0, uses the session's
        # completion cache when a session is given, and with deterministic sampling
        # assigns a seed derived from the session's seed identity and request index if
        # none was given. Any client error becomes HTTPException 500. Returns (result,
        # is_streaming).
        areal_client: ArealOpenAI | None = app.state.areal_client
        if areal_client is None:
            raise HTTPException(status_code=503, detail="Inference backend unavailable")

        kwargs = dict(body_json)
        kwargs.pop("model", None)
        is_streaming = bool(kwargs.get("stream", False))
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("top_p", 1.0)
        areal_cache = session.active_completions if session is not None else None

        deterministic_sampling = session is not None and config.deterministic_sampling
        request_index = (
            session.next_sampling_request_index() if deterministic_sampling else None
        )
        if deterministic_sampling and kwargs.get("seed") is None:
            assert session is not None
            assert request_index is not None
            kwargs["seed"] = derive_deterministic_seed(
                session.sampling_seed_identity,
                request_index,
            )

        try:
            result = await areal_client.chat.completions.create(
                areal_cache=areal_cache,
                **kwargs,
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return result, is_streaming

    # =========================================================================
    # Health
    # =========================================================================

    @app.get("/health", response_model=DataProxyHealthResponse)
    async def health():
        # [FUNCTION DOC]
        # GET /health: returns status, configured backend address, session count, paused
        # flag and version.
        # KNOWN ISSUE: reports the backend address from the startup config, not the one
        # set via /configure_backend.
        store: SessionStore = app.state.session_store
        pause_state: PauseState = app.state.pause_state
        return DataProxyHealthResponse(
            status="ok",
            backend=config.backend_addr,
            sessions=store.session_count,
            paused=await pause_state.is_paused(),
            version=app.state.version,
        )

    @app.post("/configure", response_model=DataProxyStatusResponse)
    async def configure():
        # [FUNCTION DOC]
        # POST /configure: no-op returning status ok.
        return DataProxyStatusResponse(status="ok")

    # =========================================================================
    # Pause/Resume — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/pause_generation", response_model=PauseGenerationResponse)
    async def pause_generation():
        # [FUNCTION DOC]
        # POST /pause_generation (no auth): pauses the bridge (503 if no backend).
        # Returns paused.
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.pause()
        return PauseGenerationResponse(status="ok", paused=True)

    @app.post("/continue_generation", response_model=PauseGenerationResponse)
    async def continue_generation():
        # [FUNCTION DOC]
        # POST /continue_generation (no auth): resumes the bridge (503 if no backend).
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.resume()
        return PauseGenerationResponse(status="ok", paused=False)

    @app.post("/release_memory_occupation")
    async def release_memory_occupation():
        # [FUNCTION DOC]
        # POST /release_memory_occupation: offloads backend memory via the bridge (503
        # if none).
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        await inf_bridge.offload()
        return {"status": "ok"}

    @app.post("/resume_memory_occupation")
    async def resume_memory_occupation(request: Request):
        # [FUNCTION DOC]
        # POST /resume_memory_occupation: onloads backend memory with optional body
        # "tags" (503 if no backend).
        inf_bridge: InfBridge | None = app.state.inf_bridge
        if inf_bridge is None:
            raise HTTPException(
                status_code=503,
                detail="No inference backend configured (external model mode).",
            )
        body = await request.json() if await request.body() else {}
        tags = body.get("tags")
        await inf_bridge.onload(tags=tags)
        return {"status": "ok"}

    # =========================================================================
    # Version management — internal control plane (no auth at data proxy level)
    # =========================================================================

    @app.post("/set_version", response_model=SetVersionResponse)
    async def set_version(request: Request):
        # [FUNCTION DOC]
        # POST /set_version: requires int "version" (400 otherwise), stores it and
        # propagates it to the bridge for token version stamping.
        body = await request.json()
        version = body.get("version")
        if version is None or not isinstance(version, int):
            raise HTTPException(status_code=400, detail="'version' (int) is required")
        app.state.version = version

        # Propagate version to InfBridge so it stamps correct versions on generated tokens
        if app.state.inf_bridge is not None:
            app.state.inf_bridge.set_version(version)

        return SetVersionResponse(status="ok", version=version)

    @app.get("/get_version", response_model=GetVersionResponse)
    async def get_version():
        # [FUNCTION DOC]
        # GET /get_version: returns the current version.
        return GetVersionResponse(version=app.state.version)

    # =========================================================================
    # Session management (admin key / session key required)
    # =========================================================================

    @app.post("/rl/start_session", status_code=201)
    async def start_session(
        body: StartSessionRequest, request: Request
    ) -> StartSessionResponse:
        # [FUNCTION DOC]
        # POST /rl/start_session (admin key): creates max(group_size, 1) sessions for
        # task_id, the first optionally using the caller-supplied api_key, with seed
        # identity "task:i" for groups. A ValueError from the store (e.g. duplicate key)
        # gives 409. Returns a new random group id and the session credentials.
        # KNOWN ISSUE: sessions created before a 409 failure within a group are not
        # rolled back.
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)

        group_id = f"grp-{uuid.uuid4()}"
        group_size = max(body.group_size, 1)
        credentials: list[SessionCredentials] = []
        for i in range(group_size):
            try:
                session_id, session_api_key = store.start_session(
                    body.task_id,
                    body.api_key if i == 0 else None,
                    sampling_seed_identity=(
                        f"{body.task_id}:{i}" if group_size > 1 else body.task_id
                    ),
                )
            except ValueError as e:
                raise HTTPException(status_code=409, detail=str(e))
            credentials.append(
                SessionCredentials(
                    session_id=session_id, session_api_key=session_api_key
                )
            )
        return StartSessionResponse(group_id=group_id, sessions=credentials)

    @app.post("/rl/set_reward", response_model=SetRewardResponse)
    async def set_reward(body: SetRewardRequest, request: Request):
        # [FUNCTION DOC]
        # POST /rl/set_reward: authenticates by session key (or admin key for the HITL
        # session; 401 otherwise), assigns the reward to the given or last interaction
        # (400 on ValueError), and returns the reward result fields.
        store: SessionStore = app.state.session_store
        token = _extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is None:
            raise HTTPException(
                status_code=401, detail="Invalid or expired session API key."
            )

        try:
            reward_result = session.set_reward(
                interaction_id=body.interaction_id,
                reward=body.reward,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SetRewardResponse(
            message="success",
            interaction_count=reward_result.interaction_count,
            session_id=reward_result.session_id,
            trajectory_id=reward_result.trajectory_id,
            trajectory_ready=reward_result.trajectory_id is not None,
            ready_transition=reward_result.ready_transition,
        )

    # =========================================================================
    # Chat completions — OpenAI-compatible
    #
    # If the API token is a known session key, use session cache.
    # Otherwise (no token, admin key, unknown key) → standalone mode.
    # Data proxy never rejects requests on /chat/completions.
    # =========================================================================

    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        # [FUNCTION DOC]
        # POST /chat/completions (never rejects auth). Parses JSON (400 if invalid),
        # resolves an optional session and touches its access time. If the model name is
        # a registered external model with a URL, forwards the request to
        # <url>/chat/completions (model renamed, auth replaced by the provider key, hop
        # headers dropped): streaming responses are relayed and, on success, the
        # concatenated text is recorded as a string interaction; non-streaming responses
        # are returned verbatim and recorded when 200 (502 on transport error).
        # Otherwise uses the internal backend, returning a ChatCompletion or an SSE
        # stream ending in "[DONE]".
        raw_body = await request.body()
        try:
            body_json = json.loads(raw_body)
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        model_name = body_json.get("model")
        store: SessionStore = app.state.session_store

        token = _try_extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is not None:
            session.update_last_access()

        # -----------------------------------------------------------------
        # External model path: model is a registered external model name
        # -----------------------------------------------------------------
        ext_info = _registered_models.get(model_name) if model_name else None
        if ext_info is not None and ext_info.get("url"):
            ext_url = (ext_info["url"] or "").rstrip("/")
            ext_model = ext_info["model"]
            provider_api_key = ext_info.get("api_key")

            forward_body = dict(body_json)
            if ext_model is not None:
                forward_body["model"] = ext_model
            else:
                forward_body.pop("model", None)

            _skip = {"host", "content-length", "transfer-encoding", "authorization"}
            forward_headers = {
                k: v for k, v in dict(request.headers).items() if k.lower() not in _skip
            }
            if provider_api_key:
                forward_headers["authorization"] = f"Bearer {provider_api_key}"

            is_streaming = forward_body.get("stream", False) or False
            messages = body_json.get("messages", [])

            if is_streaming:
                collected_chunks: list[str] = []

                async def _stream_and_cache():
                    # [FUNCTION DOC]
                    # HELPER: Streams the external provider response, emitting an SSE
                    # error event on non-200 or exceptions, and records the collected
                    # text as an interaction on success.
                    success = False
                    try:
                        async with create_httpx_client(
                            timeout=httpx.Timeout(config.request_timeout)
                        ) as stream_client:
                            async with stream_client.stream(
                                "POST",
                                f"{ext_url}/chat/completions",
                                json=forward_body,
                                headers=forward_headers,
                            ) as resp:
                                if resp.status_code != 200:
                                    error_body = await resp.aread()
                                    yield (
                                        f"data: {json.dumps({'error': error_body.decode()})}\n\n".encode()
                                    )
                                    return
                                async for chunk in resp.aiter_bytes():
                                    decoded = chunk.decode("utf-8", errors="replace")
                                    collected_chunks.append(decoded)
                                    yield chunk
                                success = True
                    except Exception as exc:
                        logger.error(
                            "External stream error for %s: %s", model_name, exc
                        )
                        yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
                    finally:
                        if success and collected_chunks and session is not None:
                            session.add_string_interaction(
                                messages,
                                "".join(collected_chunks),
                            )

                return StreamingResponse(
                    _stream_and_cache(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            full_url = f"{ext_url}/chat/completions"
            try:
                resp = await app.state.http_client.post(
                    full_url,
                    json=forward_body,
                    headers=forward_headers,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"External API error: {exc}"
                )

            if resp.status_code != 200:
                logger.error(
                    "External API returned %d for %s: %s",
                    resp.status_code,
                    full_url,
                    resp.text[:500],
                )

            response_str = resp.text

            if resp.status_code == 200 and session is not None:
                session.add_string_interaction(messages, response_str)

            return RawResponse(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )

        # -----------------------------------------------------------------
        # Internal model path: use AReaL inference server
        # -----------------------------------------------------------------
        result, is_streaming = await _create_internal_chat_result(body_json, session)

        if is_streaming:
            # result is an async generator of ChatCompletionChunk

            async def _sse_stream():
                # [FUNCTION DOC]
                # HELPER: Serializes internal completion chunks as SSE events followed
                # by "[DONE]".
                async for chunk in result:
                    yield f"data: {chunk.model_dump_json()}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return result

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(request: Request) -> Message | StreamingResponse:
        # [FUNCTION DOC]
        # POST /v1/messages: requires a session (or admin) key (401). Translates the
        # Anthropic request to OpenAI form with the message preprocessors (400 for
        # malformed input, 500 for unexpected errors), runs it internally, and
        # translates the result back to an Anthropic Message or Anthropic SSE stream
        # (500 on translation failures, closing the upstream).
        """Serve Anthropic Messages API requests for session-backed agent rollouts."""
        store: SessionStore = app.state.session_store
        token = _try_extract_bearer_token(request)
        session = _resolve_session_from_token(token, store)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid session key")
        session.update_last_access()

        try:
            anthropic_request = await request.json()
            openai_request = translate_anthropic_request(
                anthropic_request,
                message_preprocessors=message_preprocessors,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Anthropic request format: {exc}",
            ) from exc
        except Exception as exc:
            logger.error(
                "Unexpected Anthropic request conversion failure: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error during request conversion.",
            ) from exc

        result, is_streaming = await _create_internal_chat_result(
            openai_request,
            session,
        )
        model = str(anthropic_request.get("model", "default"))

        if is_streaming:
            try:
                anthropic_stream = translate_anthropic_stream(result, model=model)
                return StreamingResponse(
                    _safe_stream_wrapper(anthropic_stream),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            except Exception as exc:
                if hasattr(result, "aclose"):
                    await result.aclose()
                logger.error("Anthropic streaming setup failed: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail=f"Streaming setup failed: {exc}",
                ) from exc

        try:
            return translate_anthropic_response(result)
        except Exception as exc:
            logger.error("Anthropic response conversion failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to convert response: {exc}",
            ) from exc

    @app.post("/register_model", response_model=RegisterModelResponse)
    async def register_model(request: Request):
        # [FUNCTION DOC]
        # POST /register_model (admin key, to prevent SSRF): registers name (from "name"
        # or "model", 400 if missing) -> {url, upstream model, api_key}. Empty url means
        # internal.
        # /register_model configures an external upstream URL that the data
        # proxy will fetch on behalf of /chat/completions callers. Without
        # authentication, any network-adjacent attacker could point a model
        # name at an internal address (e.g. cloud metadata service) and then
        # trigger a server-side fetch via /chat/completions, exfiltrating the
        # response — a classic SSRF (CWE-918). Restrict to the admin key.
        _require_admin_key(request, app.state.session_store)
        body = await request.json()
        name = body.get("name") or body.get("model")
        url = body.get("url", "")
        model = body.get("model", name)
        api_key = body.get("api_key")
        if not name:
            raise HTTPException(status_code=400, detail="model name is required")
        _registered_models[name] = {"url": url, "model": model, "api_key": api_key}
        logger.info("Model registered: name=%s url=%s", name, url or "(internal)")
        return RegisterModelResponse(status="ok", name=name)

    # =========================================================================
    # Trajectory export (admin key required)
    # =========================================================================

    @app.post("/export_trajectories")
    async def export_trajectories(
        body: ExportTrajectoriesRequest, request: Request
    ) -> ExportTrajectoriesResponse:
        # [FUNCTION DOC]
        # POST /export_trajectories (admin key; 400 for empty session_ids). For each
        # session exports the requested (or latest) trajectory with
        # discount/style/retry-orphan options, using None for missing sessions or
        # KeyError. If discard_trajectory, drops everything; else with
        # reward_normalization and multiple sessions normalizes rewards across the
        # group, dropping it all if incomplete. Merges interactions; if all have tensor
        # data, concatenates padded tensors and remotizes them as RTensor on this node,
        # else concatenates string interactions. Optionally removes the sessions.
        # Returns the serialized trajectory.
        # KNOWN ISSUE: when nothing is merged (discard or all missing) the all() check
        # is vacuously true and an empty list is passed to concat_padded_tensors.
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)

        if not body.session_ids:
            raise HTTPException(
                status_code=400,
                detail="session_ids must be a non-empty list",
            )

        grouped_interactions: list[
            dict[str, InteractionWithTokenLogpReward] | None
        ] = []

        for sid in body.session_ids:
            session = store.get_session(sid)
            if session is None:
                grouped_interactions.append(None)
                continue

            try:
                _, interactions = session.export_trajectory(
                    discount=body.discount,
                    style=body.style,
                    trajectory_id=body.trajectory_id,
                    drop_retry_orphans=body.drop_retry_orphans,
                )
                grouped_interactions.append(interactions)
            except KeyError:
                grouped_interactions.append(None)
                continue

        if body.discard_trajectory:
            grouped_interactions = []
        elif body.reward_normalization and len(body.session_ids) > 1:
            if not normalize_group_rewards(grouped_interactions):
                logger.warning(
                    "Reward normalization dropped an incomplete group (%d sessions)",
                    len(body.session_ids),
                )
                grouped_interactions = []

        merged: dict[str, InteractionWithTokenLogpReward] = {}
        for interactions in grouped_interactions:
            if interactions is not None:
                merged.update(interactions)

        if all(v.has_tensor_data for v in merged.values()):
            traj = concat_padded_tensors([v.to_tensor_dict() for v in merged.values()])
            traj = _remotize_trajectory(traj, node_addr=config.serving_addr)
        else:
            traj = concat_string_interactions(merged)

        if body.remove_session:
            for sid in body.session_ids:
                store.remove_session(sid)

        serialized = serialize_value(traj)
        return ExportTrajectoriesResponse(traj=serialized)

    # =========================================================================
    # Runtime backend reconfiguration (for fork-based deployment)
    # =========================================================================

    @app.post("/configure_backend", response_model=ConfigureBackendResponse)
    async def configure_backend(request: Request):
        # [FUNCTION DOC]
        # POST /configure_backend (admin key; 400 without backend_addr): builds a new
        # InfBridge and ArealOpenAI client for the new address, swaps
        # config/bridge/client in app.state, then closes the old bridge.
        # KNOWN ISSUE: if the proxy started without a backend the tokenizer is None and
        # client creation fails; closing the old bridge right away can break in-flight
        # requests that still use it, contrary to the inline comment.
        """Reconfigure the inference backend address after process start."""
        store: SessionStore = app.state.session_store
        _require_admin_key(request, store)
        body = await request.json()
        new_addr = body.get("backend_addr")
        if not new_addr:
            raise HTTPException(status_code=400, detail="backend_addr is required")
        pause_state: PauseState = app.state.pause_state
        tok: TokenizerProxy = app.state.tokenizer

        old_inf_bridge: InfBridge | None = app.state.inf_bridge

        # Recreate InfBridge + ArealOpenAI with new backend address
        new_inf_bridge = _create_inf_bridge(new_addr, pause_state, app.state.config)
        try:
            new_areal_client = _create_areal_client(
                new_inf_bridge, tok, app.state.config
            )
        except Exception:
            await new_inf_bridge.aclose()
            raise

        # Build updated config copy, then swap all three state fields.
        # Swap references first so new requests immediately use the new bridge.
        from dataclasses import replace as _dc_replace

        new_config = _dc_replace(app.state.config, backend_addr=new_addr)
        app.state.config = new_config
        app.state.inf_bridge = new_inf_bridge
        app.state.areal_client = new_areal_client

        # Close old InfBridge after the swap so in-flight requests that already
        # hold a reference to the old bridge can still finish their HTTP calls.
        if old_inf_bridge is not None:
            await old_inf_bridge.aclose()

        logger.info("Backend reconfigured to %s", new_addr)
        return ConfigureBackendResponse(status="ok", backend_addr=new_addr)

    # =========================================================================
    # RTensor data storage endpoints
    #
    # These endpoints are now mounted from the legacy Flask data_blueprint
    # (areal.infra.rpc.guard.data_blueprint) to ensure a single source of
    # truth for RTensor storage logic.
    #
    # This mount provides:
    # - POST   /data/batch
    # - PUT    /data/<shard_id>
    # - GET    /data/<shard_id>
    # - DELETE /data/clear
    # =========================================================================
    flask_shim = Flask("data_proxy_shim")
    flask_shim.register_blueprint(data_bp)
    app.mount("/", WSGIMiddleware(flask_shim))

    return app
