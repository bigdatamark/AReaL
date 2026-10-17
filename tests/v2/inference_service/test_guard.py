# [MODULE DOC]
# Unit tests for the RPCGuard Flask app exposed via areal.v2.inference_service.guard
# (/health, /alloc_ports, /fork, /kill_forked_worker) and cleanup_forked_children. Uses
# the Flask test client; port finding, process spawning (run_with_streaming_logs) and
# kill_process_tree are patched in areal.infra.rpc.guard.app, which the v2 guard module
# is expected to share state/handlers with (module-level _state is mutated directly).
"""Unit tests for RPCGuard Flask app (areal.v2.inference_service.guard.app).

Tests all 4 endpoints (/health, /alloc_ports, /fork, /kill_forked_worker)
and the cleanup_forked_children() function using Flask test client with
mocked subprocess spawning.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from areal.v2.inference_service.guard import app as guard_module
from areal.v2.inference_service.guard.app import app, cleanup_forked_children

GUARD_APP = "areal.infra.rpc.guard.app"


@pytest.fixture(autouse=True)
def _reset_guard_globals():
    # [FUNCTION DOC]
    # Autouse fixture: resets guard _state (allocated ports, forked children list and
    # map,
    # server host 10.0.0.1, experiment/trial names, fileroot None) before each test and
    # clears ports/children afterward.
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
    # Fixture: sets Flask TESTING mode and yields a test client for the guard app.
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_mock_process(pid: int = 12345, running: bool = True) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Builds a MagicMock with Popen spec, the given pid, and poll() returning
    # None
    # when running else 0.
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.return_value = None if running else 0
    return proc


class TestHealth:
    # [CLASS DOC]
    # Tests for /health.
    def test_health_returns_200(self, client):
        # [FUNCTION DOC]
        # /health returns 200 with status healthy and zero forked children.
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["forked_children"] == 0

    def test_health_counts_forked_children(self, client):
        # [FUNCTION DOC]
        # /health reports the length of the forked children list.
        guard_module._state.forked_children = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        resp = client.get("/health")
        data = resp.get_json()
        assert data["forked_children"] == 3


class TestAllocPorts:
    # [CLASS DOC]
    # Tests for /alloc_ports.
    @patch(f"{GUARD_APP}.find_free_ports")
    def test_alloc_ports_success(self, mock_find, client):
        # [FUNCTION DOC]
        # Allocating 3 ports returns the found ports and host and records them in state.
        mock_find.return_value = [9001, 9002, 9003]
        resp = client.post("/alloc_ports", json={"count": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["ports"] == [9001, 9002, 9003]
        assert data["host"] == "10.0.0.1"
        assert guard_module._state.allocated_ports == {9001, 9002, 9003}

    @patch(f"{GUARD_APP}.find_free_ports")
    def test_alloc_ports_excludes_previous(self, mock_find, client):
        # [FUNCTION DOC]
        # A second allocation passes previously allocated ports as exclude_ports and
        # state ends
        # with the union of all allocated ports.
        mock_find.return_value = [9001, 9002, 9003]
        client.post("/alloc_ports", json={"count": 3})

        mock_find.return_value = [9004, 9005]
        resp = client.post("/alloc_ports", json={"count": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ports"] == [9004, 9005]

        _, kwargs = mock_find.call_args
        assert 9001 in kwargs.get("exclude_ports", set())
        assert 9002 in kwargs.get("exclude_ports", set())
        assert 9003 in kwargs.get("exclude_ports", set())

        assert guard_module._state.allocated_ports == {
            9001,
            9002,
            9003,
            9004,
            9005,
        }

    def test_alloc_ports_missing_count(self, client):
        # [FUNCTION DOC]
        # Missing count returns 400 mentioning count.
        resp = client.post("/alloc_ports", json={})
        assert resp.status_code == 400
        assert "count" in resp.get_json()["error"].lower()

    def test_alloc_ports_invalid_count_zero(self, client):
        # [FUNCTION DOC]
        # count 0 returns 400.
        resp = client.post("/alloc_ports", json={"count": 0})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_negative(self, client):
        # [FUNCTION DOC]
        # Negative count returns 400.
        resp = client.post("/alloc_ports", json={"count": -1})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_string(self, client):
        # [FUNCTION DOC]
        # Non-integer string count returns 400.
        resp = client.post("/alloc_ports", json={"count": "three"})
        assert resp.status_code == 400

    def test_alloc_ports_no_json_body(self, client):
        # [FUNCTION DOC]
        # Non-JSON body returns 400.
        resp = client.post("/alloc_ports", data="not json", content_type="text/plain")
        assert resp.status_code == 400


class TestFork:
    # [CLASS DOC]
    # Tests for /fork with raw commands.
    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_raw_cmd_passes_command_as_is(self, mock_run, client):
        # [FUNCTION DOC]
        # raw_cmd is passed verbatim to the spawner; response has status, host, pid and
        # no port.
        mock_proc = _make_mock_process(pid=55)
        mock_run.return_value = mock_proc

        raw = [
            "python",
            "-m",
            "areal.v2.inference_service.sglang.launch_server",
            "--model",
            "test-model",
        ]
        resp = client.post(
            "/fork",
            json={"role": "sglang", "worker_index": 0, "raw_cmd": raw},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["host"] == "10.0.0.1"
        assert data["pid"] == 55
        assert "port" not in data

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == raw

    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_tracks_child(self, mock_run, client):
        # [FUNCTION DOC]
        # A forked process is added to forked_children and mapped under (role,
        # worker_index).
        mock_proc = _make_mock_process(pid=42)
        mock_run.return_value = mock_proc

        client.post(
            "/fork",
            json={
                "role": "test",
                "worker_index": 0,
                "raw_cmd": ["echo", "hello"],
            },
        )

        assert mock_proc in guard_module._state.forked_children
        assert ("test", 0) in guard_module._state.forked_children_map
        assert guard_module._state.forked_children_map[("test", 0)] is mock_proc

    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_with_env_overrides(self, mock_run, client):
        # [FUNCTION DOC]
        # env overrides in the request appear in the env kwarg passed to the spawner.
        mock_run.return_value = _make_mock_process()

        resp = client.post(
            "/fork",
            json={
                "role": "test",
                "worker_index": 0,
                "raw_cmd": ["echo", "hello"],
                "env": {"MY_VAR": "my_value"},
            },
        )
        assert resp.status_code == 200

        call_kwargs = mock_run.call_args
        child_env = call_kwargs[1]["env"]
        assert child_env["MY_VAR"] == "my_value"


class TestForkErrorHandling:
    # [CLASS DOC]
    # Validation error tests for /fork (and one /alloc_ports case).
    def test_fork_missing_role(self, client):
        # [FUNCTION DOC]
        # Missing role returns 400 mentioning role.
        resp = client.post(
            "/fork",
            json={"worker_index": 0, "raw_cmd": ["echo"]},
        )
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_fork_missing_worker_index(self, client):
        # [FUNCTION DOC]
        # Missing worker_index returns 400 mentioning worker_index.
        resp = client.post(
            "/fork",
            json={"role": "test", "raw_cmd": ["echo"]},
        )
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    def test_fork_missing_raw_cmd(self, client):
        # [FUNCTION DOC]
        # Missing raw_cmd returns 400 mentioning raw_cmd.
        resp = client.post("/fork", json={"role": "test", "worker_index": 0})
        assert resp.status_code == 400
        assert "raw_cmd" in resp.get_json()["error"].lower()

    def test_fork_no_json_body(self, client):
        # [FUNCTION DOC]
        # Non-JSON body returns 400.
        resp = client.post("/fork", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count(self, client):
        # [FUNCTION DOC]
        # Float count on /alloc_ports returns 400 (misplaced in this class but valid).
        resp = client.post("/alloc_ports", json={"count": 1.5})
        assert resp.status_code == 400


class TestKillForkedWorker:
    # [CLASS DOC]
    # Tests for /kill_forked_worker.
    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_known_worker(self, mock_kill, client):
        # [FUNCTION DOC]
        # Killing a tracked running worker returns success with the pid in the message,
        # removes
        # it from state, and calls kill_process_tree(pid, timeout=3, graceful=True).
        mock_proc = _make_mock_process(pid=123)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("test", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "123" in data["message"]

        assert mock_proc not in guard_module._state.forked_children
        assert ("test", 0) not in guard_module._state.forked_children_map

        mock_kill.assert_called_once_with(123, timeout=3, graceful=True)

    def test_kill_unknown_worker_returns_404(self, client):
        # [FUNCTION DOC]
        # Unknown (role, worker_index) returns 404 with "not found".
        resp = client.post(
            "/kill_forked_worker",
            json={"role": "ghost", "worker_index": 99},
        )
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_already_exited_worker(self, mock_kill, client):
        # [FUNCTION DOC]
        # An already-exited tracked worker returns 200 without calling
        # kill_process_tree.
        mock_proc = _make_mock_process(pid=456, running=False)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("done", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker",
            json={"role": "done", "worker_index": 0},
        )
        assert resp.status_code == 200
        mock_kill.assert_not_called()

    def test_kill_missing_role(self, client):
        # [FUNCTION DOC]
        # Missing role returns 400.
        resp = client.post("/kill_forked_worker", json={"worker_index": 0})
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_kill_missing_worker_index(self, client):
        # [FUNCTION DOC]
        # Missing worker_index returns 400.
        resp = client.post("/kill_forked_worker", json={"role": "test"})
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_then_kill_again_returns_404(self, mock_kill, client):
        # [FUNCTION DOC]
        # Killing the same worker twice: first 200, second 404.
        mock_proc = _make_mock_process(pid=789)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("test", 0)] = mock_proc

        resp1 = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp2.status_code == 404


class TestCleanup:
    # [CLASS DOC]
    # Tests for cleanup_forked_children.
    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_kills_all_running_children(self, mock_kill):
        # [FUNCTION DOC]
        # Kills every running child by pid and empties the children list and map.
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._state.forked_children = [proc1, proc2]
        guard_module._state.forked_children_map = {
            ("a", 0): proc1,
            ("b", 0): proc2,
        }

        cleanup_forked_children()

        assert mock_kill.call_count == 2
        pids_killed = {call.args[0] for call in mock_kill.call_args_list}
        assert pids_killed == {100, 200}

        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_skips_already_exited(self, mock_kill):
        # [FUNCTION DOC]
        # Skips exited children (only the running pid is killed) and still clears state.
        running = _make_mock_process(pid=100, running=True)
        exited = _make_mock_process(pid=200, running=False)
        guard_module._state.forked_children = [running, exited]
        guard_module._state.forked_children_map = {
            ("a", 0): running,
            ("b", 0): exited,
        }

        cleanup_forked_children()

        mock_kill.assert_called_once_with(100, timeout=3, graceful=True)

        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_no_children_is_noop(self, mock_kill):
        # [FUNCTION DOC]
        # With no children, no kill is attempted.
        cleanup_forked_children()
        mock_kill.assert_not_called()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_tolerates_kill_exception(self, mock_kill):
        # [FUNCTION DOC]
        # An OSError from one kill does not stop killing the rest; state is cleared.
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._state.forked_children = [proc1, proc2]
        guard_module._state.forked_children_map = {
            ("a", 0): proc1,
            ("b", 0): proc2,
        }

        mock_kill.side_effect = [OSError("boom"), None]

        cleanup_forked_children()

        assert mock_kill.call_count == 2
        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}
