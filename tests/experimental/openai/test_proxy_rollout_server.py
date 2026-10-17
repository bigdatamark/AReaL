# [MODULE DOC]
# Unit tests for the proxy rollout server FastAPI app (proxy_rollout_server.app),
# focused on session API-key handling: honoring caller-provided keys, rejecting the
# admin key or keys bound to active sessions, cleaning up finished-session key bindings,
# reporting interaction_count on end_session, and admin-only export_trajectories
# addressed by explicit session_id. The app runs in-process via httpx ASGITransport;
# module-level server globals are reset per test. Skipped if httpx is missing.
"""Unit tests for the proxy rollout server's session key handling."""

from __future__ import annotations

import threading

import pytest

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture that monkeypatches the server module's globals to a clean state
    # before each test: empty session cache and key<->session maps, capacity 0, admin
    # key set to the test key, a fresh lock and last cleanup time 0. Restored after each
    # test.
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)


httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    # [FUNCTION DOC]
    # HELPER: Returns a new httpx.AsyncClient bound to the shared ASGI transport for the
    # server app with base URL http://testserver.
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    # [FUNCTION DOC]
    # HELPER: Returns an Authorization Bearer header for the test admin key.
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


# ---------------------------------------------------------------------------
# Tests: start_session with provided api_key
# ---------------------------------------------------------------------------


class TestStartSessionApiKey:
    # [CLASS DOC]
    # Tests start_session behavior when an api_key is (or is not) supplied. Each test
    # sets capacity 1 so one session can be started.
    @pytest.mark.asyncio
    async def test_uses_provided_api_key(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts a provided api_key is echoed back and mapped to the new session id in
        # _api_key_to_session.
        """Worker returns the caller-provided key instead of generating one."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "my-preferred-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "my-preferred-key"
        assert srv._api_key_to_session["my-preferred-key"] == data["session_id"]

    @pytest.mark.asyncio
    async def test_generates_key_when_none(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts that without api_key a random key (longer than 10 chars, not the admin
        # key) is generated.
        """No api_key → worker generates a random key (current behaviour)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
        assert resp.status_code == 200
        key = resp.json()["api_key"]
        assert key != _ADMIN_KEY
        assert len(key) > 10  # random token

    @pytest.mark.asyncio
    async def test_rejects_admin_key_as_session_key(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts using the admin key as the session key is rejected with 400.
        """Cannot use the admin key as a session key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": _ADMIN_KEY},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_cleans_up_finished_session_conflict(self, monkeypatch):
        # [FUNCTION DOC]
        # Pre-seeds a finished session bound to key reuse-me, then asserts start_session
        # with that key succeeds, returns the key and rebinds it to a new, different
        # session id.
        """Key reuse after a finished session cleans up old mappings."""
        # Pre-seed a finished session with the same key.
        sid_old = "old-session"
        old_session = SessionData(session_id=sid_old)
        old_session.finish()
        srv._session_cache[sid_old] = old_session
        srv._api_key_to_session["reuse-me"] = sid_old
        srv._session_to_api_key[sid_old] = "reuse-me"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "reuse-me"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "reuse-me"
        # Old session mapping should be gone; new one present.
        assert srv._api_key_to_session["reuse-me"] == data["session_id"]
        assert data["session_id"] != sid_old

    @pytest.mark.asyncio
    async def test_rejects_active_session_conflict(self, monkeypatch):
        # [FUNCTION DOC]
        # Pre-seeds an active (unfinished) session bound to busy-key and asserts
        # start_session with that key returns 409.
        """Key bound to an active (unfinished) session → 409."""
        sid_active = "active-session"
        active_session = SessionData(session_id=sid_active)
        srv._session_cache[sid_active] = active_session
        srv._api_key_to_session["busy-key"] = sid_active
        srv._session_to_api_key[sid_active] = "busy-key"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "busy-key"},
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Tests: end_session returns interaction_count
# ---------------------------------------------------------------------------


class TestEndSessionInteractionCount:
    # [CLASS DOC]
    # Tests the end_session response payload.
    @pytest.mark.asyncio
    async def test_end_session_returns_interaction_count(self, monkeypatch):
        # [FUNCTION DOC]
        # Starts a session, ends it immediately with its session key, and asserts 200
        # with interaction_count 0.
        """end_session response includes interaction_count field."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            # Start a session.
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            # End it immediately (0 interactions).
            resp_end = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )
            assert resp_end.status_code == 200
            data = resp_end.json()
            assert data["interaction_count"] == 0


# ---------------------------------------------------------------------------
# Tests: export_trajectories (requires session_id + admin auth)
# ---------------------------------------------------------------------------


class TestExportTrajectories:
    # [CLASS DOC]
    # Tests for /export_trajectories: requires admin auth and an explicit session_id.
    """Tests for the export_trajectories endpoint.

    The endpoint requires an explicit ``session_id`` in the request body
    and admin-key authentication.  This eliminates routing ambiguity when
    an API key has been reused across sessions.
    """

    @pytest.mark.asyncio
    async def test_export_with_session_id_and_admin_auth(self, monkeypatch):
        # [FUNCTION DOC]
        # Starts and ends a session, then asserts admin export by session_id returns 200
        # with an interactions field.
        """Export succeeds with required session_id + admin key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            # End the session so export doesn't block.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()

    @pytest.mark.asyncio
    async def test_export_rejects_non_admin_key(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts export authenticated with the session key (not admin) returns 403.
        """Export requires admin auth; a session key must be rejected."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 403

    @pytest.mark.asyncio
    async def test_export_rejects_missing_session_id(self, monkeypatch):
        # [FUNCTION DOC]
        # Asserts export without session_id in the body fails validation with 422.
        """Omitting session_id from body triggers a validation error (422)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={},
            )
            assert resp_export.status_code == 422

    @pytest.mark.asyncio
    async def test_export_survives_key_remap(self, monkeypatch):
        # [FUNCTION DOC]
        # With capacity 2, starts and ends a session with key shared-key, starts a
        # second one reusing it, asserts the key now maps to the new session, and
        # asserts exporting the old session by explicit id still returns 200 with
        # interactions.
        """Explicit session_id resolves correctly even after key remapping.

        After a session refresh the API key maps to the NEW session.
        Because export uses the explicit session_id, it still targets
        the OLD (completed) session without blocking.
        """
        monkeypatch.setattr(srv, "_capacity", 2)
        async with _client() as client:
            # Start first session with a specific key.
            resp1 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-0", "api_key": "shared-key"},
            )
            assert resp1.status_code == 200
            session_id_old = resp1.json()["session_id"]

            # End the first session.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": "Bearer shared-key"},
                json={},
            )

            # Start a second session reusing the same key.
            resp2 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-1", "api_key": "shared-key"},
            )
            assert resp2.status_code == 200
            session_id_new = resp2.json()["session_id"]
            assert session_id_new != session_id_old

            # API key now points to the NEW session.
            assert srv._api_key_to_session["shared-key"] == session_id_new

            # Export the OLD session by session_id — unaffected by the remap.
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id_old},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()
