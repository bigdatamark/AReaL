# [MODULE DOC]
# Tests for the v2 agent-service guard Flask app (a pass-through of the base RPC guard):
# /health reporting and /alloc_ports behaviour, with module-level guard state reset
# around each test and port discovery patched.
"""Unit tests for Agent Service Guard (pure pass-through).

Tests that the base guard routes are available on the agent guard app.
The agent_blueprint has been removed in v2 — all orchestration logic
now lives in AgentController.

Test structure mirrors ``tests/v2/inference_service/test_guard.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from areal.v2.agent_service.guard import app as guard_module
from areal.v2.agent_service.guard.app import app

GUARD_APP = "areal.infra.rpc.guard.app"


@pytest.fixture(autouse=True)
def _reset_guard_globals():
    # [FUNCTION DOC]
    # Autouse fixture: sets guard _state fields (ports, forked children, host 10.0.0.1,
    # experiment/trial names, fileroot None) before each test and clears ports/children
    # afterwards. Mutates module-global state.
    """Reset all guard state between tests."""
    guard_module._state.allocated_ports = set()
    guard_module._state.forked_children = []
    guard_module._state.forked_children_map = {}
    guard_module._state.server_host = "10.0.0.1"
    guard_module._state.experiment_name = "test-exp"
    guard_module._state.trial_name = "test-trial"
    guard_module._state.fileroot = None
    yield
    guard_module._state.allocated_ports = set()
    guard_module._state.forked_children = []
    guard_module._state.forked_children_map = {}


@pytest.fixture()
def client():
    # [FUNCTION DOC]
    # Fixture: enables Flask TESTING mode and yields a test client for the guard app.
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealth:
    # [CLASS DOC]
    # /health endpoint tests.
    def test_health_returns_200(self, client):
        # [FUNCTION DOC]
        # Asserts 200, status "healthy" and zero forked children.
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["forked_children"] == 0

    def test_health_counts_forked_children(self, client):
        # [FUNCTION DOC]
        # Sets two fake forked children and asserts health reports 2.
        guard_module._state.forked_children = [MagicMock(), MagicMock()]
        resp = client.get("/health")
        data = resp.get_json()
        assert data["forked_children"] == 2


class TestAllocPorts:
    # [CLASS DOC]
    # /alloc_ports endpoint tests.
    @patch(f"{GUARD_APP}.find_free_ports")
    def test_alloc_ports_success(self, mock_find, client):
        # [FUNCTION DOC]
        # With find_free_ports patched to [9001, 9002], asserts success response with those
        # ports and that they are recorded in guard state.
        mock_find.return_value = [9001, 9002]
        resp = client.post("/alloc_ports", json={"count": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["ports"] == [9001, 9002]
        assert guard_module._state.allocated_ports == {9001, 9002}

    def test_alloc_ports_missing_count(self, client):
        # [FUNCTION DOC]
        # Asserts a request without "count" returns 400.
        resp = client.post("/alloc_ports", json={})
        assert resp.status_code == 400
