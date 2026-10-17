# [MODULE DOC]
# Unit tests for the v2 AgentController, which uses a Scheduler to launch guard
# workers and then, via HTTP calls to the guards (alloc_ports, fork, kill), starts a
# router, a gateway and (worker, data-proxy) pairs, registers them, monitors health and
# supports scale up/down with session draining. The module-level requests object and
# scheduler are mocked; no processes or servers are started.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentController.

All Guard HTTP interactions are mocked — no real processes or servers.
Tests cover: initialize, destroy, scale_up, scale_down, and error handling.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from areal.api.cli_args import AgentConfig, SchedulingSpec
from areal.v2.agent_service.controller.controller import AgentController

CTRL = "areal.v2.agent_service.controller.controller"


@dataclass
class _FakeWorker:
    # [CLASS DOC]
    # HELPER: Dataclass mimicking a scheduler Worker (id, ip, worker/engine port lists).
    id: str
    ip: str
    worker_ports: list[str]
    engine_ports: list[str]


def _make_scheduler(*guard_specs: tuple[str, str]) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock scheduler whose get_workers returns one _FakeWorker per
    # (ip, port) spec, with ids agent-guard/<i> and the port as its single worker port.
    """Return a mock Scheduler whose get_workers returns _FakeWorkers."""
    workers = [
        _FakeWorker(id=f"agent-guard/{i}", ip=ip, worker_ports=[port], engine_ports=[])
        for i, (ip, port) in enumerate(guard_specs)
    ]
    scheduler = MagicMock()
    scheduler.get_workers.return_value = workers
    return scheduler


def _mock_alloc_ports_response(host: str, ports: list[int]) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Mock successful /alloc_ports response returning host and ports.
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "ports": ports}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_fork_response(host: str, pid: int) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Mock successful /fork response returning host and pid.
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "pid": pid}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_kill_response() -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Mock successful /kill_forked_worker response (json and text bodies).
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success"}
    resp.text = '{"status": "success"}'
    return resp


def _mock_register_response() -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Mock 200 response for /register and /unregister.
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _mock_health_response(active_sessions: int = 0) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Mock health response with status ok and the given active_sessions.
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "active_sessions": active_sessions}
    return resp


@pytest.fixture()
def config():
    # [FUNCTION DOC]
    # Fixture: AgentConfig with agent_cls_path "my.Agent" and admin key "test-key".
    return AgentConfig(
        agent_cls_path="my.Agent",
        admin_api_key="test-key",
    )


def _setup_mock_requests(mock_requests, port_start=9001):
    # [FUNCTION DOC]
    # HELPER: Installs fake post/get on the mocked requests module: alloc_ports hands out
    # sequential ports from port_start, fork/register/kill/unregister succeed, others 404;
    # get always returns a healthy zero-session response; RequestException = Exception.
    port_counter = iter(range(port_start, port_start + 100))

    def mock_post(url, **kwargs):
        # [FUNCTION DOC]
        # HELPER: URL-substring dispatcher returning the matching mock response.
        if "/alloc_ports" in url:
            return _mock_alloc_ports_response("10.0.0.1", [next(port_counter)])
        if "/fork" in url:
            return _mock_fork_response("10.0.0.1", 100)
        if "/register" in url:
            return _mock_register_response()
        if "/kill_forked_worker" in url:
            return _mock_kill_response()
        if "/unregister" in url:
            return _mock_register_response()
        return MagicMock(status_code=404)

    mock_requests.post = mock_post
    mock_requests.get = lambda url, **kw: _mock_health_response()
    mock_requests.RequestException = Exception


class TestConstruction:
    # [CLASS DOC]
    # Constructor tests.
    def test_construction(self, config):
        # [FUNCTION DOC]
        # Asserts a fresh controller has empty router/gateway addresses and no pairs.
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}


class TestInitialize:
    # [CLASS DOC]
    # initialize() tests.
    @patch(f"{CTRL}.requests")
    def test_initialize_forks_router_pairs_gateway(self, mock_requests, config):
        # [FUNCTION DOC]
        # Asserts initialize creates and fetches guard workers once, sets http router and
        # gateway addresses, creates one pair, and tracks 4 forked services; then destroys.
        """Initialize should create guards via scheduler, then fork services."""
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"), ("10.0.0.2", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        scheduler.create_workers.assert_called_once()
        scheduler.get_workers.assert_called_once()

        assert "http://" in ctrl.router_addr
        assert "http://" in ctrl.gateway_addr
        assert len(ctrl.pairs) == 1
        assert len(ctrl._forked_services) == 4

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_initialize_uses_scheduling_spec_env_vars(self, mock_requests):
        # [FUNCTION DOC]
        # With a SchedulingSpec carrying env var ANTHROPIC_API_KEY, asserts the guard job's
        # task has those env vars and runs "python -m areal.v2.agent_service.guard", and that
        # every fork payload passes the same env.
        fork_payloads = []

        def mock_post(url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake post recording fork JSON payloads; otherwise like the shared dispatcher.
            if "/alloc_ports" in url:
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                fork_payloads.append(kwargs["json"])
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        config = AgentConfig(
            agent_cls_path="my.Agent",
            admin_api_key="test-key",
            scheduling_spec=(
                SchedulingSpec(env_vars={"ANTHROPIC_API_KEY": "test-anthropic-key"}),
            ),
        )
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.tasks[0].env_vars == {"ANTHROPIC_API_KEY": "test-anthropic-key"}
        assert job.tasks[0].cmd == f"{sys.executable} -m areal.v2.agent_service.guard"
        assert all(
            payload.get("env") == {"ANTHROPIC_API_KEY": "test-anthropic-key"}
            for payload in fork_payloads
        )

        ctrl.destroy()


class TestScaleUp:
    # [CLASS DOC]
    # scale_up() tests.
    @patch(f"{CTRL}.requests")
    def test_scale_up_adds_pairs(self, mock_requests, config):
        # [FUNCTION DOC]
        # Asserts scale_up(3) returns new pair ids [1, 2, 3] and yields 4 pairs total.
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl.pairs) == 1

        created = ctrl.scale_up(3)
        assert created == [1, 2, 3]
        assert len(ctrl.pairs) == 4

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_scale_up_round_robins_guards(self, mock_requests, config):
        # [FUNCTION DOC]
        # With two guards, asserts scale_up(4) spreads alloc_ports calls evenly (4 per guard,
        # i.e. 2 services per pair, pairs round-robined).
        guards_called: list[str] = []

        def mock_post(url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake post recording the guard base URL of each alloc_ports call.
            if "/alloc_ports" in url:
                guards_called.append(url.split("/alloc_ports")[0])
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("g0", "8090"), ("g1", "8091"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        guards_called.clear()

        ctrl.scale_up(4)

        g0_calls = [g for g in guards_called if "g0" in g]
        g1_calls = [g for g in guards_called if "g1" in g]
        assert len(g0_calls) == 4
        assert len(g1_calls) == 4

        ctrl.destroy()


class TestScaleDown:
    # [CLASS DOC]
    # scale_down() tests.
    @patch(f"{CTRL}.requests")
    def test_scale_down_removes_newest_first(self, mock_requests, config):
        # [FUNCTION DOC]
        # After scaling to 3 pairs, asserts scale_down(2) removes ids 2 and 1 leaving pair 0.
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        ctrl.scale_up(2)
        assert len(ctrl.pairs) == 3

        removed = ctrl.scale_down(2)
        assert set(removed) == {2, 1}
        assert len(ctrl.pairs) == 1
        assert 0 in ctrl.pairs

        ctrl.destroy()


class TestDestroy:
    # [CLASS DOC]
    # destroy() tests.
    @patch(f"{CTRL}.requests")
    def test_destroy_clears_everything(self, mock_requests, config):
        # [FUNCTION DOC]
        # Asserts destroy resets addresses, pairs and forked services and deletes workers.
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl._forked_services) > 0

        ctrl.destroy()
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}
        assert ctrl._forked_services == []
        scheduler.delete_workers.assert_called()

    @patch(f"{CTRL}.requests")
    def test_destroy_tolerates_kill_errors(self, mock_requests, config):
        # [FUNCTION DOC]
        # With kill requests raising ConnectionError, asserts destroy still attempts all 4
        # kills and clears the forked service list.
        kill_count = 0

        def mock_post(url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake post counting kill calls and raising on them.
            nonlocal kill_count
            if "/alloc_ports" in url:
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/kill_forked_worker" in url:
                kill_count += 1
                raise ConnectionError("Guard down")
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        ctrl.destroy()
        assert kill_count == 4
        assert ctrl._forked_services == []


class TestDrain:
    # [CLASS DOC]
    # Draining behaviour during scale_down.
    @patch(f"{CTRL}.requests")
    def test_scale_down_waits_for_drain(self, mock_requests, config):
        # [FUNCTION DOC]
        # Health reports 2 active sessions for the first 5 calls then 0; with controller time
        # patched (real monotonic, mocked sleep), asserts scale_down(1) polls more than once
        # and removes the pair.
        """scale_down should poll DataProxy health until active_sessions reaches 0."""
        _setup_mock_requests(mock_requests)
        health_call_count = 0

        def mock_get(url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake get counting calls; busy for the first 5 /health calls, then idle.
            nonlocal health_call_count
            health_call_count += 1
            if "/health" in url and health_call_count <= 5:
                return _mock_health_response(active_sessions=2)
            return _mock_health_response(active_sessions=0)

        mock_requests.get = mock_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        health_call_count = 0
        with patch(f"{CTRL}.time") as mock_time:
            mock_time.monotonic = time.monotonic
            mock_time.sleep = MagicMock()
            ctrl.scale_down(1)

        assert len(ctrl.pairs) == 0
        assert health_call_count > 1

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_drain_uses_default_timeout(self, mock_requests, config):
        # [FUNCTION DOC]
        # Health always reports 5 active sessions; asserts scale_down polls at least once and
        # eventually proceeds (default drain timeout).
        # KNOWN ISSUE: time.sleep is not patched, so this test waits the full real timeout.
        _setup_mock_requests(mock_requests)
        get_count = 0

        def counting_get(url, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Fake get counting calls and always returning 5 active sessions.
            nonlocal get_count
            get_count += 1
            return _mock_health_response(active_sessions=5)

        mock_requests.get = counting_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        pre_get_count = get_count
        ctrl.scale_down(1)
        drain_gets = get_count - pre_get_count
        assert drain_gets > 0

        ctrl.destroy()


class TestHealthMonitor:
    # [CLASS DOC]
    # Background health monitor thread lifecycle tests.
    @patch(f"{CTRL}.requests")
    def test_health_monitor_starts_and_stops(self, mock_requests, config):
        # [FUNCTION DOC]
        # Asserts initialize starts a live health thread and destroy stops and clears it.
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert ctrl._health_thread is not None
        assert ctrl._health_thread.is_alive()

        ctrl.destroy()
        assert ctrl._health_thread is None
