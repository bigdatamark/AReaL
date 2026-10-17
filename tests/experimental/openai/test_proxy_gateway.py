# [MODULE DOC]
# Unit tests for the proxy gateway FastAPI app (create_proxy_gateway_app), which fronts
# a set of proxy workers: authenticates admin vs session bearer keys, assigns sessions
# to workers either from a "ready worker" queue (fed by internal/wait_for_session) or
# round robin, forwards session requests, and supports refreshing/reusing session API
# keys from a bounded key pool. The app runs in-process via httpx ASGITransport; backend
# forwarding uses a mock aiohttp session injected into app.state.http_session. Skipped
# if httpx is missing.
"""Unit tests for the proxy gateway (FastAPI gateway)."""

from __future__ import annotations

import asyncio
import json

import pytest

from areal.experimental.openai.proxy.proxy_gateway import (
    CompletedSessionInfo,
    _extract_bearer_token,
    _forwarding_headers,
    _ReadyWorkerEntry,
    _SessionRoute,
    create_proxy_gateway_app,
)
from areal.experimental.openai.proxy.server import DEFAULT_ADMIN_API_KEY

# ---------------------------------------------------------------------------
# We use httpx.AsyncClient with ASGITransport to test the FastAPI app
# in-process (no real HTTP server needed).
# ---------------------------------------------------------------------------

httpx = pytest.importorskip("httpx")


def _make_client(
    proxy_addrs: list[str] | None = None, admin_key: str = DEFAULT_ADMIN_API_KEY
):
    # [FUNCTION DOC]
    # HELPER: Builds the gateway app over proxy_addrs (default two fake workers) with
    # the given admin key, and returns an httpx.AsyncClient bound to it via
    # ASGITransport with base URL http://testserver. No backend mock is installed.
    """Create an httpx.AsyncClient backed by the proxy gateway ASGI app."""
    addrs = proxy_addrs or ["http://worker-0:8000", "http://worker-1:8000"]
    app = create_proxy_gateway_app(proxy_addrs=addrs, admin_api_key=admin_key)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    # [CLASS DOC]
    # Tests construction of the gateway's small dataclasses.
    def test_ready_worker_entry(self):
        # [FUNCTION DOC]
        # Asserts _ReadyWorkerEntry stores worker_addr and the exact future object
        # (created on a temporary event loop that is then closed).
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        entry = _ReadyWorkerEntry(worker_addr="http://w:8000", future=future)
        assert entry.worker_addr == "http://w:8000"
        assert entry.future is future
        loop.close()

    def test_completed_session_info(self):
        # [FUNCTION DOC]
        # Asserts CompletedSessionInfo stores session_api_key, session_id and
        # worker_addr.
        info = CompletedSessionInfo(
            session_api_key="sk-123",
            session_id="sess-abc",
            worker_addr="http://w:8000",
        )
        assert info.session_api_key == "sk-123"
        assert info.session_id == "sess-abc"
        assert info.worker_addr == "http://w:8000"

    def test_session_route_defaults(self):
        # [FUNCTION DOC]
        # Asserts _SessionRoute.pending_future defaults to None.
        route = _SessionRoute(worker_addr="http://w:8000", session_id="s1")
        assert route.pending_future is None

    def test_session_route_with_future(self):
        # [FUNCTION DOC]
        # Asserts _SessionRoute stores a provided pending_future object.
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        route = _SessionRoute(
            worker_addr="http://w:8000",
            session_id="s1",
            pending_future=future,
        )
        assert route.pending_future is future
        loop.close()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    # [CLASS DOC]
    # Tests for header helpers _extract_bearer_token and _forwarding_headers.
    def test_extract_bearer_token_valid(self):
        # [FUNCTION DOC]
        # Asserts a "Bearer <token>" authorization header yields the token.
        class FakeRequest:
            # [CLASS DOC]
            # HELPER: Stub request exposing only a headers mapping with a Bearer token.
            headers = {"authorization": "Bearer sk-test-token"}

        assert _extract_bearer_token(FakeRequest()) == "sk-test-token"

    def test_extract_bearer_token_missing(self):
        # [FUNCTION DOC]
        # Asserts a missing authorization header yields None.
        class FakeRequest:
            # [CLASS DOC]
            # HELPER: Stub request with empty headers.
            headers = {}

        assert _extract_bearer_token(FakeRequest()) is None

    def test_extract_bearer_token_not_bearer(self):
        # [FUNCTION DOC]
        # Asserts a non-Bearer (Basic) authorization header yields None.
        class FakeRequest:
            # [CLASS DOC]
            # HELPER: Stub request with a Basic authorization header.
            headers = {"authorization": "Basic abc123"}

        assert _extract_bearer_token(FakeRequest()) is None

    def test_forwarding_headers_filters(self):
        # [FUNCTION DOC]
        # Asserts _forwarding_headers keeps authorization and content-type but drops
        # custom and host headers.
        raw = {
            "authorization": "Bearer sk-123",
            "content-type": "application/json",
            "x-custom": "should-be-dropped",
            "host": "localhost",
        }
        result = _forwarding_headers(raw)
        assert "authorization" in result
        assert "content-type" in result
        assert "x-custom" not in result
        assert "host" not in result

    def test_forwarding_headers_empty(self):
        # [FUNCTION DOC]
        # Asserts _forwarding_headers of an empty mapping is empty.
        assert _forwarding_headers({}) == {}


# ---------------------------------------------------------------------------
# App endpoint tests (using httpx ASGI transport)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    # [CLASS DOC]
    # Tests for GET /health.
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        # [FUNCTION DOC]
        # Asserts /health returns 200 with status "ok" and workers equal to the number
        # of configured proxy addresses (2).
        async with _make_client(["http://w1:8000", "http://w2:8000"]) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["workers"] == 2


class TestStartSessionAuth:
    # [CLASS DOC]
    # Tests admin authentication on POST /rl/start_session.
    @pytest.mark.asyncio
    async def test_start_session_rejects_bad_token(self):
        # [FUNCTION DOC]
        # Asserts a wrong bearer key gets 401.
        async with _make_client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers={"Authorization": "Bearer wrong-key"},
                json={},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_start_session_rejects_missing_auth(self):
        # [FUNCTION DOC]
        # Asserts a missing authorization header gets 401.
        async with _make_client() as client:
            resp = await client.post("/rl/start_session", json={})
            assert resp.status_code == 401


class TestSessionForwardAuth:
    # [CLASS DOC]
    # Tests that session-scoped endpoints reject unknown session API keys.
    @pytest.mark.asyncio
    async def test_chat_completions_rejects_unknown_session(self):
        # [FUNCTION DOC]
        # Asserts /chat/completions with an unknown session key gets 401.
        async with _make_client() as client:
            resp = await client.post(
                "/chat/completions",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={"model": "test", "messages": []},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_set_reward_rejects_unknown_session(self):
        # [FUNCTION DOC]
        # Asserts /rl/set_reward with an unknown session key gets 401.
        async with _make_client() as client:
            resp = await client.post(
                "/rl/set_reward",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_end_session_rejects_unknown_session(self):
        # [FUNCTION DOC]
        # Asserts /rl/end_session with an unknown session key gets 401.
        async with _make_client() as client:
            resp = await client.post(
                "/rl/end_session",
                headers={"Authorization": "Bearer unknown-session-key"},
                json={},
            )
            assert resp.status_code == 401


class TestWaitForSessionAuth:
    # [CLASS DOC]
    # Tests admin authentication on POST /internal/wait_for_session.
    @pytest.mark.asyncio
    async def test_wait_for_session_rejects_bad_token(self):
        # [FUNCTION DOC]
        # Asserts a wrong bearer key gets 401.
        async with _make_client() as client:
            resp = await client.post(
                "/internal/wait_for_session",
                headers={"Authorization": "Bearer wrong-key"},
                json={"worker_addr": "http://w:8000"},
            )
            assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Mock aiohttp infrastructure for backend-forwarding tests
# ---------------------------------------------------------------------------


class _MockAiohttpResponse:
    # [CLASS DOC]
    # HELPER: Fake aiohttp response usable as an async context manager. Stores status,
    # JSON-encodes dict bodies to bytes (other bodies kept as-is), and fixes
    # content_type to application/json.
    """Simulates an aiohttp response for the gateway's ``_forward`` helper."""

    def __init__(self, status: int, body):
        # [FUNCTION DOC]
        # HELPER: Stores status, encoded body and content type.
        self.status = status
        self._body = json.dumps(body).encode() if isinstance(body, dict) else body
        self.content_type = "application/json"

    async def read(self):
        # [FUNCTION DOC]
        # HELPER: Returns the stored body bytes.
        return self._body

    async def __aenter__(self):
        # [FUNCTION DOC]
        # HELPER: Returns self on context entry.
        return self

    async def __aexit__(self, *a):
        # [FUNCTION DOC]
        # HELPER: No-op context exit (does not suppress exceptions).
        pass


class _MockAiohttpSession:
    # [CLASS DOC]
    # HELPER: Fake aiohttp ClientSession with a list of (url substring, response)
    # entries and a log of calls (url, kwargs).
    """Controllable aiohttp session returning pre-enqueued responses.

    Responses are matched by the first enqueued entry whose *url_contains*
    substring appears in the request URL.  Matched entries are consumed (FIFO
    per pattern).
    """

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes an empty response queue and call log.
        self._queue: list[tuple[str, _MockAiohttpResponse]] = []
        self.calls: list[tuple[str, dict]] = []

    def enqueue(self, url_contains: str, status: int, body) -> None:
        # [FUNCTION DOC]
        # HELPER: Appends a canned response with status/body matched when url_contains
        # is a substring of the request URL.
        self._queue.append((url_contains, _MockAiohttpResponse(status, body)))

    def post(self, url, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records the call, then returns and removes the first queued response
        # whose pattern occurs in the URL; if none matches, returns a 500 response with
        # an error body.
        self.calls.append((url, kwargs))
        for i, (pattern, resp) in enumerate(self._queue):
            if pattern in url:
                self._queue.pop(i)
                return resp
        return _MockAiohttpResponse(500, {"error": f"no mock for {url}"})

    async def __aenter__(self):
        # [FUNCTION DOC]
        # HELPER: Returns self on context entry.
        return self

    async def __aexit__(self, *a):
        # [FUNCTION DOC]
        # HELPER: No-op context exit.
        pass


def _make_mocked_client(
    proxy_addrs: list[str] | None = None,
    admin_key: str = DEFAULT_ADMIN_API_KEY,
    refresh_timeout: float = 120.0,
    key_pool_size: int = 4096,
):
    # [FUNCTION DOC]
    # HELPER: Builds the gateway app (default single worker, given admin key,
    # refresh_timeout and key_pool_size), injects a fresh _MockAiohttpSession into
    # app.state.http_session (since ASGI lifespan does not run), and returns a wrapper
    # object that is an async context manager around an httpx AsyncClient and exposes
    # get/post plus the mock as .mock.
    """Create gateway client + mock aiohttp session for forwarding tests.

    The httpx ASGI transport does not trigger the ASGI lifespan, so we
    inject the mock session directly into ``app.state.http_session``.
    """
    addrs = proxy_addrs or ["http://worker-0:8000"]
    mock_session = _MockAiohttpSession()

    app = create_proxy_gateway_app(
        proxy_addrs=addrs,
        admin_api_key=admin_key,
        refresh_timeout=refresh_timeout,
        key_pool_size=key_pool_size,
    )
    # Inject mock session directly (lifespan won't run under ASGI transport).
    app.state.http_session = mock_session

    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    class _ClientWrapper:
        # [CLASS DOC]
        # HELPER: Async context manager wrapper delegating get/post to the captured
        # httpx client and exposing the mock session.
        """Async context manager exposing both client and mock session."""

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Stores the mock session as self.mock.
            self.mock = mock_session

        async def __aenter__(self):
            # [FUNCTION DOC]
            # HELPER: Enters the underlying httpx client and returns the wrapper.
            await client.__aenter__()
            return self

        async def __aexit__(self, *a):
            # [FUNCTION DOC]
            # HELPER: Exits the underlying httpx client, passing exception info through.
            await client.__aexit__(*a)

        # Delegate HTTP methods to the underlying client.
        def get(self, *a, **kw):
            # [FUNCTION DOC]
            # HELPER: Delegates to the client's get, returning its coroutine.
            return client.get(*a, **kw)

        def post(self, *a, **kw):
            # [FUNCTION DOC]
            # HELPER: Delegates to the client's post, returning its coroutine.
            return client.post(*a, **kw)

    return _ClientWrapper()


def _admin_headers(key: str = DEFAULT_ADMIN_API_KEY) -> dict:
    # [FUNCTION DOC]
    # HELPER: Returns Authorization Bearer header for the admin key plus JSON content
    # type.
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _session_headers(api_key: str) -> dict:
    # [FUNCTION DOC]
    # HELPER: Returns Authorization Bearer header for a session API key plus JSON
    # content type.
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Refresh / key-reuse tests
# ---------------------------------------------------------------------------


class TestRefresh:
    # [CLASS DOC]
    # Tests for API key refresh/reuse on start_session (passing an api_key in the body).
    """Tests for the session refresh (API key reuse) feature."""

    @pytest.mark.asyncio
    async def test_refresh_known_active_key(self):
        # [FUNCTION DOC]
        # Registers a ready worker via a background wait_for_session call and starts
        # session s1 (key reuse-key) on it. Registers another ready worker, then calls
        # start_session with api_key reuse-key: the gateway must end the old session
        # (mocked end_session), resolve the first waiter's future, and start s2 with the
        # same key. Asserts both responses and that the first background waiter
        # completes with 200; cancels the second waiter.
        """Full refresh: end old session, wait for ready worker, start new."""
        async with _make_mocked_client() as ctx:
            # Enqueue backend responses:
            # 1) first start_session → success
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "reuse-key", "session_id": "s1"},
            )
            # 2) end_session during refresh → success
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 3},
            )
            # 3) second start_session (after refresh) → success
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "reuse-key", "session_id": "s2"},
            )

            # Step 1: populate ready queue and start initial session.
            bg1 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "reuse-key"
            assert resp1.json()["session_id"] == "s1"

            # Step 2: populate queue again for the post-refresh session.
            bg2 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            # Step 3: refresh — same api_key triggers auto-end + new session.
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "reuse-key"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "reuse-key"
            assert resp2.json()["session_id"] == "s2"

            # bg1 should have completed (future resolved during refresh).
            bg1_resp = await asyncio.wait_for(bg1, timeout=2.0)
            assert bg1_resp.status_code == 200

            # Clean up bg2 (still waiting on its future).
            bg2.cancel()
            try:
                await bg2
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_refresh_timeout_returns_429(self):
        # [FUNCTION DOC]
        # With refresh_timeout 0.1 and no second ready worker, asserts a refresh of an
        # active key returns 429 whose detail mentions "timed out"; cancels the
        # background waiter.
        """Refresh with no ready worker within timeout returns 429."""
        async with _make_mocked_client(refresh_timeout=0.1) as ctx:
            # Enqueue: first start_session + end_session, but NO second start.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 0},
            )

            # Start initial session via ready queue.
            bg1 = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200

            # Refresh — no ready worker, should timeout (0.1s).
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            assert resp2.status_code == 429
            assert "timed out" in resp2.json().get("detail", "").lower()

            bg1.cancel()
            try:
                await bg1
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_reuse_known_inactive_key_round_robin(self):
        # [FUNCTION DOC]
        # Starts a session by round robin, ends it (route removed, key retained in
        # pool), then starts again with that key and asserts the normal flow is used and
        # the key is reused with a new session id.
        """Known key with no active route goes through normal flow with key."""
        async with _make_mocked_client() as ctx:
            # First session — round-robin (no ready queue entries).
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "k1"

            # End session (removes route, key stays in pool).
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 1},
            )
            resp_end = await ctx.post(
                "/rl/end_session",
                headers=_session_headers("k1"),
                json={},
            )
            assert resp_end.status_code == 200

            # Reuse same key — known but no active route → normal flow.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s2"},
            )
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "k1"
            assert resp2.json()["session_id"] == "s2"

    @pytest.mark.asyncio
    async def test_unknown_key_stripped(self):
        # [FUNCTION DOC]
        # Asserts an api_key never issued by the gateway is stripped from the forwarded
        # start_session body so the worker generates its own key.
        """Unknown api_key is stripped — worker generates a fresh key."""
        async with _make_mocked_client() as ctx:
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "worker-generated", "session_id": "s1"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "user-invented-key"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "worker-generated"

            # Verify the forwarded body had api_key stripped.
            _, call_kwargs = ctx.mock.calls[0]
            forwarded_body = json.loads(call_kwargs["data"])
            assert "api_key" not in forwarded_body

    @pytest.mark.asyncio
    async def test_none_api_key_normal_flow(self):
        # [FUNCTION DOC]
        # Asserts start_session without api_key forwards normally and returns the worker
        # key.
        """api_key=None goes through normal flow."""
        async with _make_mocked_client() as ctx:
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "new-key", "session_id": "s1"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "new-key"

    @pytest.mark.asyncio
    async def test_key_pool_eviction(self):
        # [FUNCTION DOC]
        # With key_pool_size 2, creates and ends three sessions (k0, k1, k2), then
        # asserts k0 was evicted (treated as unknown so the worker issues a fresh key)
        # while k2 is still reused.
        """Key pool evicts oldest key when exceeding key_pool_size."""
        async with _make_mocked_client(key_pool_size=2) as ctx:
            # Create 3 sessions — pool can hold only 2.
            for i in range(3):
                ctx.mock.enqueue(
                    "start_session",
                    200,
                    {"api_key": f"k{i}", "session_id": f"s{i}"},
                )
                ctx.mock.enqueue(
                    "end_session",
                    200,
                    {"message": "success", "interaction_count": 0},
                )

                resp = await ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t"},
                )
                assert resp.status_code == 200

                # End session so the key has no active route.
                resp_end = await ctx.post(
                    "/rl/end_session",
                    headers=_session_headers(f"k{i}"),
                    json={},
                )
                assert resp_end.status_code == 200

            # k0 should have been evicted.  Sending it now should be treated
            # as an unknown key (stripped).
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "fresh", "session_id": "s-new"},
            )
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k0"},
            )
            assert resp.status_code == 200
            assert resp.json()["api_key"] == "fresh"  # worker generated, not k0

            # k2 (most recent) should still be known.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k2", "session_id": "s-reuse"},
            )
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k2"},
            )
            assert resp2.status_code == 200
            assert resp2.json()["api_key"] == "k2"  # reused


# ---------------------------------------------------------------------------
# Additional regression tests
# ---------------------------------------------------------------------------


class TestOrphanFutureCancellation:
    # [CLASS DOC]
    # Regression test: when the backend rejects start_session for a ready worker, the
    # worker's pending future is cancelled and its wait_for_session returns 503.
    """When the backend rejects a ready worker, the future is cancelled
    and wait_for_session returns 503."""

    @pytest.mark.asyncio
    async def test_ready_worker_reject_cancels_future(self):
        # [FUNCTION DOC]
        # Registers a ready worker, mocks start_session to fail with 500, asserts
        # start_session returns 500 and the background waiter returns 503 with
        # "cancelled" in the body.
        """Backend returns 500 for start_session → future.cancel() → 503."""
        async with _make_mocked_client() as ctx:
            # Backend rejects start_session with 500.
            ctx.mock.enqueue(
                "start_session",
                500,
                {"error": "internal error"},
            )

            # Register a worker via wait_for_session (runs in background).
            bg = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            # start_session picks the ready worker; backend rejects;
            # gateway cancels the future.
            resp = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 500

            # wait_for_session sees CancelledError → 503.
            bg_resp = await asyncio.wait_for(bg, timeout=2.0)
            assert bg_resp.status_code == 503
            assert b"cancelled" in bg_resp.content.lower()


class TestConcurrentRefreshRejected:
    # [CLASS DOC]
    # Regression test: concurrent refreshes of the same key are rejected.
    """Concurrent refresh for the same key is rejected with 429."""

    @pytest.mark.asyncio
    async def test_concurrent_refresh_same_key_returns_429(self):
        # [FUNCTION DOC]
        # Starts session k1 on a ready worker, launches a refresh for k1 that blocks
        # waiting for a ready worker, then asserts a second concurrent refresh for k1
        # returns 429. Cancels the pending tasks, swallowing any errors.
        """Second refresh while first is in progress returns 429."""
        async with _make_mocked_client() as ctx:
            # First session via ready queue.
            ctx.mock.enqueue(
                "start_session",
                200,
                {"api_key": "k1", "session_id": "s1"},
            )
            ctx.mock.enqueue(
                "end_session",
                200,
                {"message": "success", "interaction_count": 0},
            )

            bg_wait = asyncio.create_task(
                ctx.post(
                    "/internal/wait_for_session",
                    headers=_admin_headers(),
                    json={"worker_addr": "http://worker-0:8000"},
                )
            )
            await asyncio.sleep(0.01)

            resp1 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp1.status_code == 200
            assert resp1.json()["api_key"] == "k1"

            # First refresh blocks on ready_workers.get() (queue empty).
            refresh1 = asyncio.create_task(
                ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t", "api_key": "k1"},
                )
            )
            await asyncio.sleep(0.05)  # let refresh1 enter the refresh path

            # Second refresh with same key.
            resp2 = await ctx.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "k1"},
            )
            # The _refreshing guard now rejects concurrent refreshes for the
            # same key before checking `routes`, so this reliably returns 429.
            assert resp2.status_code == 429

            # Clean up.
            refresh1.cancel()
            bg_wait.cancel()
            for t in [refresh1, bg_wait]:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass


class TestWaitForSessionTimeoutStaleEntry:
    # [CLASS DOC]
    # Regression test: a ready-queue entry whose wait_for_session timed out is skipped
    # by later start_session calls.
    """After wait_for_session times out, the stale ready entry is skipped
    by the next start_session call."""

    @pytest.mark.asyncio
    async def test_timeout_stale_entry_skipped(self):
        # [FUNCTION DOC]
        # Temporarily patches the module global _DEFAULT_WAIT_TIMEOUT to 0.1s, registers
        # a stale worker and asserts its wait returns 408; restores a long timeout,
        # registers a fresh worker and asserts start_session uses it (fresh-key).
        # Restores the original timeout in finally. Side effect: mutates a module-level
        # global during the test.
        """Timed-out worker entry is skipped; fresh worker is used."""
        import areal.experimental.openai.proxy.proxy_gateway as gw_mod

        original_timeout = gw_mod._DEFAULT_WAIT_TIMEOUT
        gw_mod._DEFAULT_WAIT_TIMEOUT = 0.1  # very short timeout
        try:
            async with _make_mocked_client() as ctx:
                # Register worker — will timeout (nobody calls start_session).
                bg_stale = asyncio.create_task(
                    ctx.post(
                        "/internal/wait_for_session",
                        headers=_admin_headers(),
                        json={"worker_addr": "http://stale-worker:8000"},
                    )
                )
                # Let it timeout.
                stale_resp = await asyncio.wait_for(bg_stale, timeout=2.0)
                assert stale_resp.status_code == 408

                # Stale entry remains in queue. Register a fresh worker.
                ctx.mock.enqueue(
                    "start_session",
                    200,
                    {"api_key": "fresh-key", "session_id": "s-fresh"},
                )

                gw_mod._DEFAULT_WAIT_TIMEOUT = 3600.0  # restore for fresh worker
                bg_fresh = asyncio.create_task(
                    ctx.post(
                        "/internal/wait_for_session",
                        headers=_admin_headers(),
                        json={"worker_addr": "http://fresh-worker:8000"},
                    )
                )
                await asyncio.sleep(0.01)

                # Should skip the stale entry and use the fresh one.
                resp = await ctx.post(
                    "/rl/start_session",
                    headers=_admin_headers(),
                    json={"task_id": "t"},
                )
                assert resp.status_code == 200
                assert resp.json()["api_key"] == "fresh-key"

                # The fresh worker's wait_for_session is still pending.
                bg_fresh.cancel()
                try:
                    await bg_fresh
                except asyncio.CancelledError:
                    pass
        finally:
            gw_mod._DEFAULT_WAIT_TIMEOUT = original_timeout
