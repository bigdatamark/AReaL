# [MODULE DOC]
# Unit tests for the v2 training-service worker Flask app: engine creation via
# /create_engine with the CPU FakeTrainEngine, topology reporting, and RPC-style engine
# method endpoints (train, forward, initialize, SFT, PPO) using serialized args/kwargs.
# Module-level worker globals (engine, node address, engine thread and work queue) are
# reset around every test.
"""Unit tests for training-service worker Flask app."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.v2.training_service.worker.config import TrainWorkerConfig

MODULE = "areal.v2.training_service.worker.app"


@pytest.fixture(autouse=True)
def reset_worker_state():
    # [FUNCTION DOC]
    # Autouse fixture: before and after each test, stops any engine worker thread
    # (sending
    # a None sentinel to its queue and joining up to 1s) and resets the worker module's
    # _engine, _node_addr, _engine_thread and _engine_work_queue globals.
    import areal.v2.training_service.worker.app as worker_app

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None

    yield

    if worker_app._engine_work_queue is not None:
        worker_app._engine_work_queue.put(None)
    if worker_app._engine_thread is not None:
        worker_app._engine_thread.join(timeout=1.0)

    worker_app._engine = None
    worker_app._node_addr = ""
    worker_app._engine_thread = None
    worker_app._engine_work_queue = None


@pytest.fixture
def client():
    # [FUNCTION DOC]
    # Fixture: creates the worker Flask app (127.0.0.1:19001, admin key) and returns its
    # test client.
    from areal.v2.training_service.worker.app import create_app

    app = create_app(
        TrainWorkerConfig(
            host="127.0.0.1",
            port=19001,
            admin_api_key="worker-admin",
        )
    )
    return app.test_client()


class TestWorkerEngineCreation:
    # [CLASS DOC]
    # Tests for /create_engine.
    def test_create_engine_requires_engine_class(self, client):
        # [FUNCTION DOC]
        # Missing engine_class returns 400 mentioning engine_class.
        resp = client.post(
            "/create_engine",
            json={"init_args": [], "init_kwargs": {}},
        )
        assert resp.status_code == 400
        assert "engine_class" in resp.get_json()["error"]

    def test_create_engine_success(self, client):
        # [FUNCTION DOC]
        # Creating FakeTrainEngine by import path with serialized args returns success.
        resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["status"] == "success"


class TestWorkerEndpoints:
    # [CLASS DOC]
    # Tests for worker endpoints after engine creation.
    def test_topology_before_create_engine_returns_400(self, client):
        # [FUNCTION DOC]
        # /topology before engine creation returns 400 "Engine not created".
        resp = client.get("/topology")
        assert resp.status_code == 400
        assert "Engine not created" in resp.get_json()["error"]

    def test_train_batch_after_create_engine(self, client):
        # [FUNCTION DOC]
        # /train_batch on the fake engine returns a serialized dict containing total.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        train_resp = client.post(
            "/train_batch",
            json={
                "args": serialize_value(
                    [{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]
                ),
                "kwargs": serialize_value({}),
            },
        )
        assert train_resp.status_code == 200
        result = deserialize_value(train_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_topology_after_create_engine(self, client):
        # [FUNCTION DOC]
        # /topology with RANK/WORLD_SIZE/LOCAL_RANK env set reports rank 0, world_size
        # 1,
        # dp_size 1.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        with patch.dict(
            "os.environ",
            {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"},
            clear=False,
        ):
            topo_resp = client.get("/topology")
        assert topo_resp.status_code == 200
        topo = topo_resp.get_json()
        assert topo["rank"] == 0
        assert topo["world_size"] == 1
        assert topo["dp_size"] == 1

    def test_ppo_endpoints_return_400_when_engine_method_missing(self, client):
        # [FUNCTION DOC]
        # PPO actor/critic endpoints return 400 "does not implement method" since the
        # fake
        # engine lacks those methods.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        payload = {
            "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
            "kwargs": serialize_value({}),
        }
        for path in [
            "/ppo/actor/compute_logp",
            "/ppo/actor/compute_advantages",
            "/ppo/actor/update",
            "/ppo/critic/compute_values",
            "/ppo/critic/update",
        ]:
            resp = client.post(path, json=payload)
            assert resp.status_code == 400
            assert "does not implement method" in resp.get_json()["error"]

    def test_forward_batch_after_initialize_succeeds_without_distributed_group(
        self, client
    ):
        # [FUNCTION DOC]
        # After /initialize, /forward_batch works without a distributed group and echoes
        # output_seqlens.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        init_resp = client.post(
            "/initialize",
            json={
                "args": serialize_value([]),
                "kwargs": serialize_value({"addr": None, "ft_spec": None}),
            },
        )
        assert init_resp.status_code == 200

        forward_resp = client.post(
            "/forward_batch",
            json={
                "args": serialize_value(
                    [[{"token_ids": [1, 2, 3], "metadata": {"weight": 2.0}}]]
                ),
                "kwargs": serialize_value({"output_seqlens": [3]}),
            },
        )
        assert forward_resp.status_code == 200
        result = deserialize_value(forward_resp.get_json()["result"])
        assert isinstance(result, dict)
        assert result["output_seqlens"] == [3]

    def test_sft_route_succeeds_without_distributed_group_for_single_worker(
        self, client
    ):
        # [FUNCTION DOC]
        # /sft/train succeeds for a single worker without a distributed group and
        # returns total.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result

    def test_sft_route_ignores_rpc_meta_override_for_single_worker(self, client):
        # [FUNCTION DOC]
        # /sft/train with an rpc_meta broadcast override still succeeds for a single
        # worker.
        create_resp = client.post(
            "/create_engine",
            json={
                "engine_class": "tests.v2.training_service.fake_train_engine.FakeTrainEngine",
                "init_args": serialize_value([]),
                "init_kwargs": serialize_value({"world_size": 1}),
            },
        )
        assert create_resp.status_code == 200

        resp = client.post(
            "/sft/train",
            json={
                "args": serialize_value([[{"token_ids": [1, 2, 3]}]]),
                "kwargs": serialize_value({}),
                "rpc_meta": {"broadcast": False},
            },
        )
        assert resp.status_code == 200
        result = deserialize_value(resp.get_json()["result"])
        assert isinstance(result, dict)
        assert "total" in result
