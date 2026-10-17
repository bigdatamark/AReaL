# [MODULE DOC]
# Unit tests for the v2 inference Router service: WorkerRegistry and SessionRegistry
# state classes, routing strategies (round robin, get_strategy), and the router FastAPI
# endpoints (/health, /register, /unregister, /route, /register_model,
# /register_session, /workers, /resolve_worker) exercised in-process via httpx ASGI.
"""Unit tests for the Router service (Plan 2, Task 3).

Tests worker registry, session registry, routing strategies,
and all router endpoints.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from areal.v2.inference_service.router.app import create_app
from areal.v2.inference_service.router.config import RouterConfig
from areal.v2.inference_service.router.state import (
    SessionRegistry,
    WorkerInfo,
    WorkerRegistry,
)
from areal.v2.inference_service.router.strategies import (
    RoundRobinStrategy,
    get_strategy,
)

# =============================================================================
# Constants
# =============================================================================

ADMIN_KEY = "test-admin-key"
WORKER_1 = "http://worker-1:18082"
WORKER_2 = "http://worker-2:18082"
WORKER_3 = "http://worker-3:18082"


# =============================================================================
# WorkerRegistry unit tests
# =============================================================================


class TestWorkerRegistry:
    # [CLASS DOC]
    # Tests for the async WorkerRegistry (register/deregister, health, lookups).
    @pytest.mark.asyncio
    async def test_register_worker(self):
        # [FUNCTION DOC]
        # register returns a non-empty string id and records a single healthy worker.
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        assert isinstance(worker_id, str)
        assert len(worker_id) > 0  # UUID string
        workers = await reg.get_all_workers()
        assert len(workers) == 1
        assert workers[0].worker_addr == WORKER_1
        assert workers[0].worker_id == worker_id
        assert workers[0].is_healthy is True

    @pytest.mark.asyncio
    async def test_register_duplicate_noop(self):
        # [FUNCTION DOC]
        # Registering the same address twice returns the same id and keeps one entry.
        reg = WorkerRegistry()
        worker_id_1 = await reg.register(WORKER_1)
        worker_id_2 = await reg.register(WORKER_1)
        assert worker_id_1 == worker_id_2
        workers = await reg.get_all_workers()
        assert len(workers) == 1

    @pytest.mark.asyncio
    async def test_deregister_worker(self):
        # [FUNCTION DOC]
        # deregister by address removes the worker.
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.deregister(WORKER_1)
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_deregister_unknown_noop(self):
        # [FUNCTION DOC]
        # deregister of an unknown address is a no-op.
        reg = WorkerRegistry()
        await reg.deregister("http://unknown:9999")
        assert len(await reg.get_all_workers()) == 0

    @pytest.mark.asyncio
    async def test_health_update(self):
        # [FUNCTION DOC]
        # update_health marks a worker unhealthy.
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.update_health(WORKER_1, False)
        workers = await reg.get_all_workers()
        assert workers[0].is_healthy is False

    @pytest.mark.asyncio
    async def test_get_healthy_workers(self):
        # [FUNCTION DOC]
        # get_healthy_workers excludes unhealthy workers.
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        healthy = await reg.get_healthy_workers()
        assert len(healthy) == 1
        assert healthy[0].worker_addr == WORKER_2

    @pytest.mark.asyncio
    async def test_get_all_workers(self):
        # [FUNCTION DOC]
        # get_all_workers includes unhealthy workers.
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        await reg.update_health(WORKER_1, False)
        all_w = await reg.get_all_workers()
        assert len(all_w) == 2

    @pytest.mark.asyncio
    async def test_list_worker_addrs(self):
        # [FUNCTION DOC]
        # list_worker_addrs returns all registered addresses.
        reg = WorkerRegistry()
        await reg.register(WORKER_1)
        await reg.register(WORKER_2)
        addrs = await reg.list_worker_addrs()
        assert set(addrs) == {WORKER_1, WORKER_2}

    @pytest.mark.asyncio
    async def test_deregister_by_id(self):
        # [FUNCTION DOC]
        # deregister_by_id returns the removed address and removes the worker.
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        addr = await reg.deregister_by_id(worker_id)
        assert addr == WORKER_1
        workers = await reg.get_all_workers()
        assert len(workers) == 0

    @pytest.mark.asyncio
    async def test_deregister_by_id_unknown_returns_none(self):
        # [FUNCTION DOC]
        # deregister_by_id of an unknown id returns None.
        reg = WorkerRegistry()
        result = await reg.deregister_by_id("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        # [FUNCTION DOC]
        # get_by_id returns WorkerInfo with matching address and id.
        reg = WorkerRegistry()
        worker_id = await reg.register(WORKER_1)
        info = await reg.get_by_id(worker_id)
        assert info is not None
        assert info.worker_addr == WORKER_1
        assert info.worker_id == worker_id

    @pytest.mark.asyncio
    async def test_get_by_id_unknown_returns_none(self):
        # [FUNCTION DOC]
        # get_by_id of an unknown id returns None.
        reg = WorkerRegistry()
        result = await reg.get_by_id("nonexistent-id")
        assert result is None


# =============================================================================
# SessionRegistry unit tests
# =============================================================================


class TestSessionRegistry:
    # [CLASS DOC]
    # Tests for SessionRegistry (session key/id to worker address mapping).
    @pytest.mark.asyncio
    async def test_register_session(self):
        # [FUNCTION DOC]
        # register_session makes the worker discoverable by both key and id.
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_key(self):
        # [FUNCTION DOC]
        # lookup_by_key returns the pinned worker.
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_key("key-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_by_id(self):
        # [FUNCTION DOC]
        # lookup_by_id returns the pinned worker.
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.lookup_by_id("id-1") == WORKER_1

    @pytest.mark.asyncio
    async def test_lookup_unknown_key(self):
        # [FUNCTION DOC]
        # Unknown key lookup returns None.
        reg = SessionRegistry()
        assert await reg.lookup_by_key("nonexistent") is None

    @pytest.mark.asyncio
    async def test_lookup_unknown_id(self):
        # [FUNCTION DOC]
        # Unknown id lookup returns None.
        reg = SessionRegistry()
        assert await reg.lookup_by_id("nonexistent") is None

    @pytest.mark.asyncio
    async def test_revoke_by_worker(self):
        # [FUNCTION DOC]
        # revoke_by_worker removes all sessions of that worker (returns count 2) and
        # leaves other
        # workers' sessions intact.
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-2", "id-2", WORKER_1)
        await reg.register_session("key-3", "id-3", WORKER_2)  # different worker
        count = await reg.revoke_by_worker(WORKER_1)
        assert count == 2
        assert await reg.lookup_by_key("key-1") is None
        assert await reg.lookup_by_key("key-2") is None
        assert await reg.lookup_by_id("id-1") is None
        assert await reg.lookup_by_id("id-2") is None
        # Worker 2 sessions untouched
        assert await reg.lookup_by_key("key-3") == WORKER_2

    @pytest.mark.asyncio
    async def test_count(self):
        # [FUNCTION DOC]
        # count reflects the number of registered sessions.
        reg = SessionRegistry()
        assert await reg.count() == 0
        await reg.register_session("key-1", "id-1", WORKER_1)
        assert await reg.count() == 1
        await reg.register_session("key-2", "id-2", WORKER_2)
        assert await reg.count() == 2

    @pytest.mark.asyncio
    async def test_upsert_semantics(self):
        # [FUNCTION DOC]
        # Re-registering a session key updates its worker (upsert).
        """Re-registering a session key updates the worker."""
        reg = SessionRegistry()
        await reg.register_session("key-1", "id-1", WORKER_1)
        await reg.register_session("key-1", "id-1", WORKER_2)
        assert await reg.lookup_by_key("key-1") == WORKER_2


# =============================================================================
# Routing strategies unit tests
# =============================================================================


class TestRoutingStrategies:
    # [CLASS DOC]
    # Tests for routing strategies.
    def test_round_robin_cycling(self):
        # [FUNCTION DOC]
        # RoundRobinStrategy cycles through workers in order.
        s = RoundRobinStrategy()
        w1 = WorkerInfo(worker_id="w1", worker_addr=WORKER_1)
        w2 = WorkerInfo(worker_id="w2", worker_addr=WORKER_2)
        w3 = WorkerInfo(worker_id="w3", worker_addr=WORKER_3)
        workers = [w1, w2, w3]
        picks = [s.pick(workers).worker_addr for _ in range(6)]  # type: ignore[union-attr]
        assert picks == [
            WORKER_1,
            WORKER_2,
            WORKER_3,
            WORKER_1,
            WORKER_2,
            WORKER_3,
        ]

    def test_round_robin_empty(self):
        # [FUNCTION DOC]
        # pick on an empty list returns None.
        s = RoundRobinStrategy()
        assert s.pick([]) is None

    def test_get_strategy_round_robin(self):
        # [FUNCTION DOC]
        # get_strategy("round_robin") returns a RoundRobinStrategy.
        s = get_strategy("round_robin")
        assert isinstance(s, RoundRobinStrategy)

    def test_get_strategy_least_busy_not_implemented(self):
        # [FUNCTION DOC]
        # get_strategy("least_busy") raises NotImplementedError.
        with pytest.raises(NotImplementedError, match="least_busy"):
            get_strategy("least_busy")

    def test_get_strategy_unknown_raises(self):
        # [FUNCTION DOC]
        # Unknown strategy names raise ValueError.
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            get_strategy("random")


# =============================================================================
# Router endpoint tests
# =============================================================================


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: RouterConfig with the test admin key, round robin, and poll_interval 999
    # so
    # health polling effectively never runs.
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,  # effectively disable polling in tests
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def client(config):
    # [FUNCTION DOC]
    # Async fixture: creates the router app and yields an httpx AsyncClient over an ASGI
    # transport (lifespan not run, so no background health poller).
    # KNOWN ISSUE: docstring says it sets state directly, but no state is set here; it
    # relies on create_app initializing app.state itself.
    """Create router app and yield an httpx async client.
    Bypasses lifespan (no background health poller) by setting state directly.
    """
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    # [FUNCTION DOC]
    # HELPER: Authorization bearer header with the admin key.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestRouterEndpoints:
    # [CLASS DOC]
    # Endpoint tests for the router app.
    # ----- Health -----

    @pytest.mark.asyncio
    async def test_health_200(self, client):
        # [FUNCTION DOC]
        # /health on a fresh router reports ok, 0 workers, 0 sessions, round_robin
        # strategy.
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] == 0
        assert data["sessions"] == 0
        assert data["strategy"] == "round_robin"

    @pytest.mark.asyncio
    async def test_hitl_route_binds_first_use(self, client):
        # [FUNCTION DOC]
        # Routing with the admin key (HITL) picks a worker on first use and stays pinned
        # to it.
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )

        resp = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

        pinned = await client.post(
            "/route",
            json={
                "api_key": ADMIN_KEY,
                "path": "/chat/completions",
            },
            headers=admin_headers(),
        )
        assert pinned.status_code == 200
        assert pinned.json()["worker_addr"] == WORKER_1

    # ----- Worker registration -----

    @pytest.mark.asyncio
    async def test_register_worker_admin_key(self, client):
        # [FUNCTION DOC]
        # /register with admin key returns ok and a worker_id; /health then shows 1
        # worker.
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert "worker_id" in resp.json()
        assert isinstance(resp.json()["worker_id"], str)
        assert len(resp.json()["worker_id"]) > 0

        # Verify via /health
        health = (await client.get("/health")).json()
        assert health["workers"] == 1

    @pytest.mark.asyncio
    async def test_register_worker_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /register without auth returns 401.
        resp = await client.post("/register", json={"worker_addr": WORKER_1})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_register_worker_wrong_key_403(self, client):
        # [FUNCTION DOC]
        # /register with a wrong key returns 403.
        resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    # ----- Worker deletion (with cascade) -----

    @pytest.mark.asyncio
    async def test_delete_worker_cascades(self, client):
        # [FUNCTION DOC]
        # /unregister of a worker revokes its pinned sessions (sessions_revoked 1); the
        # session
        # key then routes 404.
        # Register worker + session pinned to it
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-1",
            },
            headers=admin_headers(),
        )

        # Delete the worker
        resp = await client.post(
            "/unregister",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 1

        # Session key should no longer route
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    # ----- /route — admin key -----

    @pytest.mark.asyncio
    async def test_route_admin_key_sticky_hitl(self, client):
        # [FUNCTION DOC]
        # Repeated admin-key routes return the same (sticky) worker.
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        resp1 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp1.status_code == 200
        addr1 = resp1.json()["worker_addr"]

        resp2 = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp2.status_code == 200
        addr2 = resp2.json()["worker_addr"]

        assert addr1 == addr2

    @pytest.mark.asyncio
    async def test_route_model_preserves_model_address_order(self, client):
        # [FUNCTION DOC]
        # /route by model follows the address order given in /register_model rather than
        # registration order.
        """Model routing follows logical DP order, not registration arrival order."""
        for worker_addr in [WORKER_2, WORKER_1]:
            resp = await client.post(
                "/register",
                json={"worker_addr": worker_addr},
                headers=admin_headers(),
            )
            assert resp.status_code == 200

        register_model = await client.post(
            "/register_model",
            json={
                "model": "test-model",
                "data_proxy_addrs": [WORKER_1, WORKER_2],
            },
            headers=admin_headers(),
        )
        assert register_model.status_code == 200

        routed = []
        for _ in range(2):
            resp = await client.post(
                "/route",
                json={"model": "test-model"},
                headers=admin_headers(),
            )
            assert resp.status_code == 200
            routed.append(resp.json()["worker_addr"])

        assert routed == [WORKER_1, WORKER_2]

    @pytest.mark.asyncio
    async def test_route_admin_key_uses_model_order_before_sticky_pin(self, client):
        # [FUNCTION DOC]
        # Admin-key route pins to the first worker of the registered model order.
        """The admin session pins the first logical model worker it selects."""
        for worker_addr in [WORKER_2, WORKER_1]:
            resp = await client.post(
                "/register",
                json={"worker_addr": worker_addr},
                headers=admin_headers(),
            )
            assert resp.status_code == 200

        register_model = await client.post(
            "/register_model",
            json={
                "model": "test-model",
                "data_proxy_addrs": [WORKER_1, WORKER_2],
            },
            headers=admin_headers(),
        )
        assert register_model.status_code == 200

        routed = []
        for _ in range(2):
            resp = await client.post(
                "/route",
                json={"api_key": ADMIN_KEY, "path": "/rl/start_session"},
                headers=admin_headers(),
            )
            assert resp.status_code == 200
            routed.append(resp.json()["worker_addr"])

        assert routed == [WORKER_1, WORKER_1]

    @pytest.mark.asyncio
    async def test_route_model_deduplicates_model_addresses(self, client):
        # [FUNCTION DOC]
        # Duplicate addresses in a model registration are deduplicated for round robin.
        """Duplicate model addresses do not change round-robin weighting."""
        for worker_addr in [WORKER_1, WORKER_2]:
            resp = await client.post(
                "/register",
                json={"worker_addr": worker_addr},
                headers=admin_headers(),
            )
            assert resp.status_code == 200

        register_model = await client.post(
            "/register_model",
            json={
                "model": "test-model",
                "data_proxy_addrs": [WORKER_1, WORKER_1, WORKER_2],
            },
            headers=admin_headers(),
        )
        assert register_model.status_code == 200

        routed = []
        for _ in range(3):
            resp = await client.post(
                "/route",
                json={"model": "test-model"},
                headers=admin_headers(),
            )
            assert resp.status_code == 200
            routed.append(resp.json()["worker_addr"])

        assert routed == [WORKER_1, WORKER_2, WORKER_1]

    # ----- /route — session key -----

    @pytest.mark.asyncio
    async def test_route_session_key_pinned(self, client):
        # [FUNCTION DOC]
        # A registered session key routes to its pinned worker.
        # Register worker + session
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-2",
            },
            headers=admin_headers(),
        )

        # Route with session key → pinned to WORKER_1
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — unknown key -----

    @pytest.mark.asyncio
    async def test_route_unknown_key_404(self, client):
        # [FUNCTION DOC]
        # Unknown api_key returns 404 "Unknown API key".
        resp = await client.post(
            "/route",
            json={"api_key": "unknown-key", "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404
        assert "Unknown API key" in resp.json()["detail"]

    # ----- /route — no healthy workers -----

    @pytest.mark.asyncio
    async def test_route_no_healthy_workers_503(self, client):
        # [FUNCTION DOC]
        # Admin key with no registered workers returns 503 "No registered workers".
        """Admin key but no workers registered → 503."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
            headers=admin_headers(),
        )
        assert resp.status_code == 503
        assert "No registered workers" in resp.json()["detail"]

    # ----- /route — pinned worker unhealthy -----

    @pytest.mark.asyncio
    async def test_route_pinned_worker_unhealthy_still_routes(self, client):
        # [FUNCTION DOC]
        # A pinned session still routes to its worker even after it is marked unhealthy
        # (registry
        # accessed via the transport's app state).
        # Register worker + session
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-4",
            },
            headers=admin_headers(),
        )

        # Mark worker unhealthy
        wr: WorkerRegistry = client._transport.app.state.worker_registry  # type: ignore[attr-defined]
        await wr.update_health(WORKER_1, False)

        # Routing still returns the pinned worker even if unhealthy
        resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    # ----- /route — session_id lookup -----

    @pytest.mark.asyncio
    async def test_route_by_session_id(self, client):
        # [FUNCTION DOC]
        # /route by session_id returns the pinned worker.
        await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-3",
            },
            headers=admin_headers(),
        )
        resp = await client.post(
            "/route",
            json={"session_id": "task-0-0"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_route_by_session_id_unknown_404(self, client):
        # [FUNCTION DOC]
        # Unknown session_id returns 404.
        resp = await client.post(
            "/route",
            json={"session_id": "nonexistent"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    # ----- /route — missing both api_key and session_id -----

    @pytest.mark.asyncio
    async def test_route_missing_both_keys_422(self, client):
        # [FUNCTION DOC]
        # /route with neither api_key nor session_id returns 422.
        resp = await client.post(
            "/route",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /route — no auth -----

    @pytest.mark.asyncio
    async def test_route_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /route without auth returns 401.
        """Route without auth → 401."""
        resp = await client.post(
            "/route",
            json={"api_key": ADMIN_KEY, "path": "/generate"},
        )
        assert resp.status_code == 401

    # ----- /register_session — no auth -----

    @pytest.mark.asyncio
    async def test_register_session_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /register_session without auth returns 401.
        """Register session without auth → 401."""
        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-5",
            },
        )
        assert resp.status_code == 401

    # ----- /register_session -----

    @pytest.mark.asyncio
    async def test_register_session(self, client):
        # [FUNCTION DOC]
        # /register_session returns ok and enables pinned routing by session key.
        """Register a session, then verify pinned routing works."""
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )

        resp = await client.post(
            "/register_session",
            json={
                "sessions": [
                    {"session_api_key": "sess-key-1", "session_id": "task-0-0"}
                ],
                "worker_addr": WORKER_1,
                "group_id": "grp-test-6",
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify: /route with session key returns pinned worker
        route_resp = await client.post(
            "/route",
            json={"api_key": "sess-key-1", "path": "/chat/completions"},
            headers=admin_headers(),
        )
        assert route_resp.status_code == 200
        assert route_resp.json()["worker_addr"] == WORKER_1

    # ----- /workers -----

    @pytest.mark.asyncio
    async def test_workers_list_admin_key(self, client):
        # [FUNCTION DOC]
        # /workers lists all workers with addr, healthy True and string worker_id.
        await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        await client.post(
            "/register",
            json={"worker_addr": WORKER_2},
            headers=admin_headers(),
        )

        resp = await client.get("/workers", headers=admin_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["workers"]) == 2
        addrs = {w["addr"] for w in data["workers"]}
        assert addrs == {WORKER_1, WORKER_2}
        assert all(w["healthy"] is True for w in data["workers"])
        assert all("worker_id" in w for w in data["workers"])
        assert all(isinstance(w["worker_id"], str) for w in data["workers"])

    @pytest.mark.asyncio
    async def test_workers_list_no_auth_401(self, client):
        # [FUNCTION DOC]
        # /workers without auth returns 401.
        resp = await client.get("/workers")
        assert resp.status_code == 401

    # ----- /unregister by worker_id -----

    @pytest.mark.asyncio
    async def test_delete_worker_by_id(self, client):
        # [FUNCTION DOC]
        # /unregister by worker_id removes the worker.
        """Delete a worker by worker_id instead of worker_addr."""
        # Register
        reg_resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        # Delete by worker_id
        resp = await client.post(
            "/unregister",
            json={"worker_id": worker_id},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify worker is gone
        health = (await client.get("/health")).json()
        assert health["workers"] == 0

    @pytest.mark.asyncio
    async def test_delete_worker_by_id_not_found_404(self, client):
        # [FUNCTION DOC]
        # /unregister with unknown worker_id returns 404.
        """Delete by unknown worker_id → 404."""
        resp = await client.post(
            "/unregister",
            json={"worker_id": "nonexistent-id"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_worker_missing_both_422(self, client):
        # [FUNCTION DOC]
        # /unregister with neither field returns 422.
        """Delete without worker_id or worker_addr → 422."""
        resp = await client.post(
            "/unregister",
            json={},
            headers=admin_headers(),
        )
        assert resp.status_code == 422

    # ----- /resolve_worker/{worker_id} -----

    @pytest.mark.asyncio
    async def test_resolve_worker_200(self, client):
        # [FUNCTION DOC]
        # /resolve_worker/{id} returns id and address for a registered worker.
        """Resolve a registered worker by ID → 200 with worker_id and worker_addr."""
        reg_resp = await client.post(
            "/register",
            json={"worker_addr": WORKER_1},
            headers=admin_headers(),
        )
        worker_id = reg_resp.json()["worker_id"]

        resp = await client.get(
            f"/resolve_worker/{worker_id}",
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_id"] == worker_id
        assert data["worker_addr"] == WORKER_1

    @pytest.mark.asyncio
    async def test_resolve_worker_not_found_404(self, client):
        # [FUNCTION DOC]
        # /resolve_worker of an unknown id returns 404.
        """Resolve an unknown worker_id → 404."""
        resp = await client.get(
            "/resolve_worker/nonexistent-id",
            headers=admin_headers(),
        )
        assert resp.status_code == 404
