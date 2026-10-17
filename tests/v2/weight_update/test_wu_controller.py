# [MODULE DOC]
# Unit tests for WeightUpdateController (areal.v2.weight_update.controller). The
# controller's HTTP session is replaced with a MagicMock of requests.Session and its
# gateway URL is forced to GATEWAY_URL, so tests verify the exact HTTP calls (endpoints,
# JSON payloads, timeout) made for health_check, connect, update_weights and disconnect,
# plus local pair-name state. No network or GPU.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import requests

from areal.v2.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.v2.weight_update.controller.controller import (
    WeightUpdateController,
)
from areal.v2.weight_update.gateway.config import WeightUpdateResult

GATEWAY_URL = "http://localhost:7080"


@pytest.fixture()
def ctrl() -> WeightUpdateController:
    # [FUNCTION DOC]
    # Fixture: builds a WeightUpdateController with admin key 'test-admin-key' and
    # request_timeout 10.0, then directly sets private _gateway_url to GATEWAY_URL and
    # _session to a requests.Session mock (bypassing gateway launch). Returns the
    # controller.
    c = WeightUpdateController(
        config=WeightUpdateControllerConfig(
            admin_api_key="test-admin-key",
            request_timeout=10.0,
        )
    )
    c._gateway_url = GATEWAY_URL
    c._session = MagicMock(spec=requests.Session)
    return c


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Builds a MagicMock requests.Response with given status_code (default 200)
    # and json() returning json_data (or empty dict). raise_for_status is a mock that
    # raises requests.HTTPError when status_code >= 400, else no-op.
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp,
        )
    return resp


class TestHealthCheck:
    # [CLASS DOC]
    # Tests for WeightUpdateController.health_check.
    def test_health_check_success_returns_true(self, ctrl):
        # [FUNCTION DOC]
        # Mocks GET returning 200; asserts health_check is True and GET was called once
        # on <gateway>/health with timeout 10.0.
        ctrl._session.get.return_value = _mock_response(200, {"status": "healthy"})

        assert ctrl.health_check() is True
        ctrl._session.get.assert_called_once_with(f"{GATEWAY_URL}/health", timeout=10.0)

    def test_health_check_connect_error_returns_false(self, ctrl):
        # [FUNCTION DOC]
        # Mocks GET raising httpx.ConnectError; asserts health_check returns False.
        # KNOWN ISSUE: the controller uses a requests.Session but catches
        # httpx.ConnectError; a real requests.ConnectionError would not be caught. This
        # test only passes because it injects the httpx exception type.
        ctrl._session.get.side_effect = httpx.ConnectError("refused")

        assert ctrl.health_check() is False


class TestConnect:
    # [CLASS DOC]
    # Tests for WeightUpdateController.connect request payloads and state.
    def test_connect_stores_pair_name(self, ctrl):
        # [FUNCTION DOC]
        # Mocks POST success; asserts connect stores the pair name in _pair_name.
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])

        assert ctrl._pair_name == "pair0"

    def test_connect_sends_correct_request(self, ctrl):
        # [FUNCTION DOC]
        # Asserts connect with defaults POSTs to <gateway>/connect with the full
        # payload: given pair/URLs, mode 'awex', empty save_path, use_lora False, empty
        # lora_name, lora_keep_versions 0, colocate False, empty nccl_master_addr,
        # nccl_master_port 0, timeout 10.0.
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000", "http://train2:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect("pair0", train_urls, infer_urls)

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "awex",
                "save_path": "",
                "use_lora": False,
                "lora_name": "",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )

    def test_connect_disk_mode_sends_disk_fields(self, ctrl):
        # [FUNCTION DOC]
        # Asserts connect with mode='disk', save_path, use_lora=True and lora_name
        # forwards those values in the /connect JSON payload, other fields at defaults.
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect(
            "pair0",
            train_urls,
            infer_urls,
            mode="disk",
            save_path="/shared/weights",
            use_lora=True,
            lora_name="my-lora",
        )

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "disk",
                "save_path": "/shared/weights",
                "use_lora": True,
                "lora_name": "my-lora",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )


class TestUpdateWeights:
    # [CLASS DOC]
    # Tests for WeightUpdateController.update_weights.
    def test_update_weights_returns_result(self, ctrl):
        # [FUNCTION DOC]
        # With _pair_name set, mocks a successful response and asserts update_weights
        # returns a WeightUpdateResult with matching status/version/duration_ms/error
        # and POSTs {pair_name, version} to <gateway>/update_weights with timeout 10.0.
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200,
            {"status": "ok", "version": 5, "duration_ms": 123.4, "error": None},
        )

        result = ctrl.update_weights(version=5)

        assert isinstance(result, WeightUpdateResult)
        assert result.status == "ok"
        assert result.version == 5
        assert result.duration_ms == 123.4
        assert result.error is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/update_weights",
            json={"pair_name": "pair0", "version": 5},
            timeout=10.0,
        )

    def test_update_weights_raises_when_not_connected(self, ctrl):
        # [FUNCTION DOC]
        # Asserts update_weights raises RuntimeError matching 'Not connected' when no
        # pair is connected.
        with pytest.raises(RuntimeError, match="Not connected"):
            ctrl.update_weights(version=1)


class TestDisconnect:
    # [CLASS DOC]
    # Tests for WeightUpdateController.disconnect.
    def test_disconnect_clears_state(self, ctrl):
        # [FUNCTION DOC]
        # With _pair_name set, asserts disconnect POSTs {pair_name} to
        # <gateway>/disconnect (timeout 10.0) and clears _pair_name to None.
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200, {"status": "ok", "pair_name": "pair0"}
        )

        ctrl.disconnect()

        assert ctrl._pair_name is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/disconnect",
            json={"pair_name": "pair0"},
            timeout=10.0,
        )

    def test_disconnect_noop_when_not_connected(self, ctrl):
        # [FUNCTION DOC]
        # Asserts disconnect without a connected pair does not raise and leaves
        # _pair_name None.
        ctrl.disconnect()
        assert ctrl._pair_name is None


class TestLifecycle:
    # [CLASS DOC]
    # End-to-end lifecycle and error-propagation tests against the mocked session.
    def test_full_lifecycle(self, ctrl):
        # [FUNCTION DOC]
        # Queues three POST responses and runs connect -> update_weights(1) ->
        # disconnect, asserting pair name set, result ok/version 1, then pair name
        # cleared.
        connect_resp = _mock_response(200, {"pair_name": "pair0"})
        update_resp = _mock_response(
            200, {"status": "ok", "version": 1, "duration_ms": 50.0, "error": None}
        )
        disconnect_resp = _mock_response(200, {"status": "ok", "pair_name": "pair0"})
        ctrl._session.post.side_effect = [connect_resp, update_resp, disconnect_resp]

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])
        assert ctrl._pair_name == "pair0"

        result = ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        ctrl.disconnect()
        assert ctrl._pair_name is None

    def test_gateway_error_raises_http_error(self, ctrl):
        # [FUNCTION DOC]
        # Mocks a 500 response; asserts update_weights propagates requests.HTTPError
        # from raise_for_status.
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(500, {"error": "internal"})

        with pytest.raises(requests.HTTPError):
            ctrl.update_weights(version=1)
