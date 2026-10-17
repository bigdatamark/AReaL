# [MODULE DOC]
# Unit tests for the v2 training-service data proxy: FastAPI route wiring (dispatch vs
# broadcast, eval-route padding opt-in, health/topology) using a fake dispatcher, and
# the
# real Dispatcher's input partitioning, fan-out to workers, result merging and scalar
# fan-out using a fake aiohttp-like session that captures serialized RPC payloads. Also
# checks partition parity with the legacy train controller's _dispatch_tensors.
"""Unit tests for training-service data proxy."""

from __future__ import annotations

from typing import Any

import httpx
import orjson
import pytest
import pytest_asyncio
import torch

from areal.infra.controller.train_controller import _dispatch_tensors
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.v2.training_service.data_proxy.app import create_app
from areal.v2.training_service.data_proxy.config import TrainDataProxyConfig
from areal.v2.training_service.data_proxy.dispatcher import Dispatcher
from areal.v2.training_service.data_proxy.topology import (
    WorkerInfo,
    WorkerTopology,
)

ADMIN_KEY = "dp-admin-key"


# ------------------------------------------------------------------
# aiohttp mock helpers
# ------------------------------------------------------------------


class _FakeAiohttpResponse:
    # [CLASS DOC]
    # HELPER: Minimal aiohttp response stand-in with a status and fixed body bytes.
    def __init__(self, content: bytes, status: int = 200):
        # [FUNCTION DOC]
        # Stores the body bytes and HTTP status (default 200).
        self._content = content
        self.status = status

    async def read(self) -> bytes:
        # [FUNCTION DOC]
        # Returns the stored body bytes.
        return self._content


class _AsyncCM:
    # [CLASS DOC]
    # HELPER: Async context manager that yields a fixed value (mimics aiohttp's request
    # context manager).
    def __init__(self, value):
        # [FUNCTION DOC]
        # Stores the value to yield.
        self._value = value

    async def __aenter__(self):
        # [FUNCTION DOC]
        # Returns the stored value on entry.
        return self._value

    async def __aexit__(self, *args):
        # [FUNCTION DOC]
        # No-op exit; does not suppress exceptions.
        pass


class _NoOpSession:
    # [CLASS DOC]
    # HELPER: Session stand-in whose only method, close, does nothing; used when no
    # worker
    # calls are expected.
    async def close(self):
        # [FUNCTION DOC]
        # No-op async close.
        pass


class _CapturingSession:
    # [CLASS DOC]
    # HELPER: Fake aiohttp ClientSession that records every POST (url, decoded JSON
    # payload,
    # deserialized args and kwargs) and answers either via a custom post_handler or with
    # a
    # success echo of the serialized args.
    def __init__(self, *, post_handler=None):
        # [FUNCTION DOC]
        # Initializes the captured payload list and optional post_handler(url, data,
        # headers)
        # returning response bytes.
        self.captured_payloads: list[dict[str, Any]] = []
        self._post_handler = post_handler

    def post(self, url, *, data=b"", headers=None):
        # [FUNCTION DOC]
        # Parses the JSON body, records it with deserialized args/kwargs, computes the
        # response
        # bytes (handler or default echo) and returns an async context manager yielding
        # a fake
        # 200 response. Raises if the body is not valid JSON or lacks args/kwargs keys.
        _ = headers
        payload = orjson.loads(data)
        self.captured_payloads.append(
            {
                "url": url,
                "payload": payload,
                "args": deserialize_value(payload["args"]),
                "kwargs": deserialize_value(payload["kwargs"]),
            }
        )
        if self._post_handler:
            content = self._post_handler(url, data, headers)
        else:
            content = orjson.dumps({"status": "success", "result": payload["args"]})
        return _AsyncCM(_FakeAiohttpResponse(content))

    async def close(self):
        # [FUNCTION DOC]
        # No-op async close.
        pass


# ------------------------------------------------------------------
# Fake Dispatcher for app-level route tests
# ------------------------------------------------------------------


class _FakeDispatchRequest:
    # [CLASS DOC]
    # HELPER: Fake per-path dispatch request returned by _FakeDispatcher.dispatch.
    def __init__(
        self,
        parent: _FakeDispatcher,
        path: str,
        *,
        pad_eval_batch: bool = False,
    ):
        # [FUNCTION DOC]
        # Stores parent dispatcher, path and the pad_eval_batch flag.
        self._parent = parent
        self._path = path
        self._pad_eval_batch = pad_eval_batch

    async def get(self) -> bytes:
        # [FUNCTION DOC]
        # Returns a fixed success JSON body for GET.
        return b'{"status":"success","result":{"path":"get"}}'

    async def post(self, body: bytes) -> bytes:
        # [FUNCTION DOC]
        # Records (path, pad_eval_batch) in the parent's dispatch_calls and returns a
        # fixed
        # success JSON body; body ignored.
        _ = body
        self._parent.dispatch_calls.append(
            {"path": self._path, "pad_eval_batch": self._pad_eval_batch}
        )
        return b'{"status":"success","result":{"path":"compute"}}'


class _FakeBroadcastRequest:
    # [CLASS DOC]
    # HELPER: Fake broadcast request returning the parent's configured response lists.
    def __init__(self, parent: _FakeDispatcher, path: str):
        # [FUNCTION DOC]
        # Stores parent dispatcher and path.
        self._parent = parent
        self._path = path

    async def get(self) -> list[bytes]:
        # [FUNCTION DOC]
        # Returns the parent's broadcast_get_return list of response bytes.
        return self._parent.broadcast_get_return

    async def post(self, body: bytes) -> list[bytes]:
        # [FUNCTION DOC]
        # Returns the parent's broadcast_return list of response bytes; body ignored.
        _ = body
        return self._parent.broadcast_return


class _FakeDispatcher:
    # [CLASS DOC]
    # HELPER: Fake Dispatcher used in app route tests; exposes configurable broadcast
    # return
    # values and a log of dispatch calls.
    def __init__(self):
        # [FUNCTION DOC]
        # Sets default single-success broadcast POST/GET responses and empty
        # dispatch_calls.
        self.broadcast_return: list[bytes] = [
            b'{"status":"success","result":{"ok":true}}'
        ]
        self.broadcast_get_return: list[bytes] = [
            b'{"status":"success","result":{"path":"broadcast_get"}}'
        ]
        self.dispatch_calls: list[dict[str, Any]] = []

    def dispatch(
        self, path: str, *, pad_eval_batch: bool = False
    ) -> _FakeDispatchRequest:
        # [FUNCTION DOC]
        # Returns a _FakeDispatchRequest for the path and padding flag.
        return _FakeDispatchRequest(self, path, pad_eval_batch=pad_eval_batch)

    def broadcast(self, path: str) -> _FakeBroadcastRequest:
        # [FUNCTION DOC]
        # Returns a _FakeBroadcastRequest for the path.
        return _FakeBroadcastRequest(self, path)


@pytest.fixture
def config() -> TrainDataProxyConfig:
    # [FUNCTION DOC]
    # Fixture: TrainDataProxyConfig with two worker addresses, admin key and 10s
    # timeout.
    return TrainDataProxyConfig(
        host="127.0.0.1",
        port=18082,
        worker_addrs=["http://worker-0:19001", "http://worker-1:19001"],
        admin_api_key=ADMIN_KEY,
        request_timeout=10.0,
    )


@pytest_asyncio.fixture
async def app_client(config):
    # [FUNCTION DOC]
    # Async fixture: creates the data proxy app, overrides app.state.topology with a
    # 2-worker DP=2 topology and app.state.dispatcher with a _FakeDispatcher, and yields
    # (app, httpx ASGI client).
    app = create_app(config)
    app.state.topology = WorkerTopology(
        workers=[
            WorkerInfo(addr="http://worker-0:19001", rank=0, dp_rank=0, dp_size=2),
            WorkerInfo(addr="http://worker-1:19001", rank=1, dp_rank=1, dp_size=2),
        ],
        dp_heads=[0, 1],
        dp_size=2,
        dp_groups=[[0], [1]],
    )
    app.state.dispatcher = _FakeDispatcher()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield app, c


def _admin_headers() -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Authorization bearer header with the admin key.
    # KNOWN ISSUE: unused in this module.
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


class TestDataProxyBasics:
    # [CLASS DOC]
    # Route-level tests of the data proxy app using the fake dispatcher.
    @pytest.mark.asyncio
    async def test_health_and_topology(self, app_client):
        # [FUNCTION DOC]
        # /health reports dp_size 2 and /topology lists two workers.
        _app, client = app_client
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["dp_size"] == 2

        topo = await client.get("/topology")
        assert topo.status_code == 200
        assert len(topo.json()["workers"]) == 2

    @pytest.mark.asyncio
    async def test_train_batch_uses_dispatcher(self, app_client):
        # [FUNCTION DOC]
        # /train_batch is answered via the dispatcher with success.
        _app, client = app_client
        resp = await client.post("/train_batch", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_ppo_actor_compute_logp_uses_dispatcher(self, app_client):
        # [FUNCTION DOC]
        # /ppo/actor/compute_logp is answered via the dispatcher.
        _app, client = app_client
        resp = await client.post("/ppo/actor/compute_logp", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_sft_train_uses_dispatcher(self, app_client):
        # [FUNCTION DOC]
        # /sft/train is answered via the dispatcher.
        _app, client = app_client
        resp = await client.post("/sft/train", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_rw_train_uses_dispatcher(self, app_client):
        # [FUNCTION DOC]
        # /rw/train is answered via the dispatcher.
        _app, client = app_client
        resp = await client.post("/rw/train", content=b"{}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @pytest.mark.asyncio
    async def test_eval_routes_opt_in_to_padding(self, app_client):
        # [FUNCTION DOC]
        # Eval routes (/eval_batch, /sft/evaluate, /rw/evaluate) dispatch with
        # pad_eval_batch
        # True.
        app, client = app_client

        await client.post("/eval_batch", content=b"{}")
        await client.post("/sft/evaluate", content=b"{}")
        await client.post("/rw/evaluate", content=b"{}")

        assert app.state.dispatcher.dispatch_calls == [
            {"path": "/eval_batch", "pad_eval_batch": True},
            {"path": "/sft/evaluate", "pad_eval_batch": True},
            {"path": "/rw/evaluate", "pad_eval_batch": True},
        ]

    @pytest.mark.asyncio
    async def test_training_routes_do_not_enable_padding(self, app_client):
        # [FUNCTION DOC]
        # Training/forward routes dispatch with pad_eval_batch False.
        app, client = app_client

        await client.post("/train_batch", content=b"{}")
        await client.post("/forward_batch", content=b"{}")
        await client.post("/sft/train", content=b"{}")
        await client.post("/ppo/actor/update", content=b"{}")
        await client.post("/ppo/critic/update", content=b"{}")
        await client.post("/rw/train", content=b"{}")

        assert app.state.dispatcher.dispatch_calls == [
            {"path": "/train_batch", "pad_eval_batch": False},
            {"path": "/forward_batch", "pad_eval_batch": False},
            {"path": "/sft/train", "pad_eval_batch": False},
            {"path": "/ppo/actor/update", "pad_eval_batch": False},
            {"path": "/ppo/critic/update", "pad_eval_batch": False},
            {"path": "/rw/train", "pad_eval_batch": False},
        ]

    @pytest.mark.asyncio
    async def test_optimizer_step_empty_broadcast_returns_502(self, app_client):
        # [FUNCTION DOC]
        # /optimizer_step with an empty broadcast result returns 502 "No worker
        # responses".
        app, client = app_client
        app.state.dispatcher.broadcast_return = []
        resp = await client.post("/optimizer_step", content=b"{}")
        assert resp.status_code == 502
        assert "No worker responses" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_export_stats_uses_broadcast_get(self, app_client):
        # [FUNCTION DOC]
        # /export_stats uses the broadcast GET and returns the first worker's result.
        app, client = app_client
        app.state.dispatcher.broadcast_get_return = [
            b'{"status":"success","result":{"stats":1}}'
        ]
        resp = await client.get("/export_stats")
        assert resp.status_code == 200
        assert resp.json()["result"]["stats"] == 1


def _make_dispatcher(
    *,
    dp_size: int,
    dp_heads: list[int],
    dp_ranks: list[int],
    session=None,
) -> Dispatcher:
    # [FUNCTION DOC]
    # HELPER: Builds a real Dispatcher over a WorkerTopology with one worker per entry
    # of
    # dp_ranks (addr http://worker-i:19001, is_dp_head when i is in dp_heads), DP groups
    # derived from dp_ranks, 10s timeout and the given session (default _NoOpSession).
    workers = [
        WorkerInfo(
            addr=f"http://worker-{i}:19001",
            rank=i,
            dp_rank=dp_ranks[i],
            dp_size=dp_size,
            is_dp_head=(i in dp_heads),
        )
        for i in range(len(dp_ranks))
    ]
    max_dp_rank = max(dp_ranks) if dp_ranks else 0
    dp_groups = [[] for _ in range(max_dp_rank + 1)]
    for i, dp_rank in enumerate(dp_ranks):
        dp_groups[dp_rank].append(i)
    topology = WorkerTopology(
        workers=workers,
        dp_heads=dp_heads,
        dp_size=dp_size,
        dp_groups=dp_groups,
    )
    return Dispatcher(
        topology=topology,
        request_timeout=10.0,
        _session=session or _NoOpSession(),
    )


def _make_tensor_item(seq_len: int) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns one sample dict with random input_ids and all-true attention_mask
    # of
    # shape 1 x seq_len.
    return {
        "input_ids": torch.randint(0, 100, (1, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
    }


class TestDispatcherParityWithTrainController:
    # [CLASS DOC]
    # Tests that Dispatcher partitioning/fan-out/merging matches the legacy train
    # controller semantics.
    def test_partition_inputs_matches_train_controller_dispatch(self):
        # [FUNCTION DOC]
        # _partition_inputs with DP=2 produces the same splits and group indices as
        # _dispatch_tensors, and replicates non-tensor kwargs per DP rank.
        dispatcher = _make_dispatcher(dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1])
        req = dispatcher.dispatch("/any")

        batch = [
            _make_tensor_item(16),
            _make_tensor_item(8),
            _make_tensor_item(12),
            _make_tensor_item(10),
        ]
        args = [batch]
        kwargs: dict[str, object] = {"tag": "x"}

        dp_args, dp_kwargs, group_indices = req._partition_inputs(
            args=args,
            kwargs=kwargs,
            group_size=1,
        )

        expected_splits, expected_indices = _dispatch_tensors(batch, 2, group_size=1)
        assert group_indices == expected_indices
        assert dp_args[0] == expected_splits
        assert dp_kwargs["tag"] == ["x", "x"]

    def test_partition_inputs_respects_group_size_atomicity(self):
        # [FUNCTION DOC]
        # With group_size 2, partitions keep consecutive pairs (even-aligned) together.
        dispatcher = _make_dispatcher(dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1])
        req = dispatcher.dispatch("/any")

        batch = [
            _make_tensor_item(16),
            _make_tensor_item(8),
            _make_tensor_item(12),
            _make_tensor_item(10),
        ]
        dp_args, _dp_kwargs, group_indices = req._partition_inputs(
            args=[batch],
            kwargs={},
            group_size=2,
        )

        assert len(dp_args[0]) == 2
        for idxs in group_indices:
            assert len(idxs) % 2 == 0
            for i in range(0, len(idxs), 2):
                left, right = idxs[i], idxs[i + 1]
                assert right == left + 1
                assert left % 2 == 0

    @pytest.mark.asyncio
    async def test_fan_out_dispatches_only_to_dp_heads_for_model_parallel(
        self,
    ):
        # [FUNCTION DOC]
        # With DP=1 and two model-parallel workers, _fan_out sends the shard and kwargs
        # to the
        # DP head and empty args/kwargs to the non-head; no rpc_meta in payloads.
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )
        req = dispatcher.dispatch("/forward_batch")

        dp_args = [[[_make_tensor_item(16), _make_tensor_item(8)]]]
        dp_kwargs = {"output_seqlens": [[2, 3]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert captured[0]["args"] != []
        assert captured[0]["kwargs"] == {"output_seqlens": [2, 3]}
        assert "rpc_meta" not in captured[0]["payload"]
        assert captured[1]["args"] == []
        assert captured[1]["kwargs"] == {}
        assert "rpc_meta" not in captured[1]["payload"]

    @pytest.mark.asyncio
    async def test_fan_out_omits_rpc_meta_for_algorithm_paths(
        self,
    ):
        # [FUNCTION DOC]
        # Algorithm paths (/sft/train) also omit rpc_meta in fan-out payloads.
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )
        req = dispatcher.dispatch("/sft/train")

        dp_args = [[[_make_tensor_item(16), _make_tensor_item(8)]]]
        dp_kwargs = {"output_seqlens": [[2, 3]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert "rpc_meta" not in captured[0]["payload"]
        assert "rpc_meta" not in captured[1]["payload"]

    @pytest.mark.asyncio
    async def test_fan_out_dispatches_per_dp_shard_when_dp_size_two(
        self,
    ):
        # [FUNCTION DOC]
        # With DP=2, each DP head receives its own shard and per-shard kwargs.
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )
        req = dispatcher.dispatch("/forward_batch")

        shard0 = [_make_tensor_item(4)]
        shard1 = [_make_tensor_item(12), _make_tensor_item(6)]
        dp_args = [[shard0, shard1]]
        dp_kwargs = {"output_seqlens": [[1], [2, 1]]}

        await req._fan_out(dp_args=dp_args, dp_kwargs=dp_kwargs)

        captured = session.captured_payloads
        assert len(captured) == 2
        assert len(captured[0]["args"][0]) == len(shard0)
        assert len(captured[1]["args"][0]) == len(shard1)
        assert (
            captured[0]["args"][0][0]["input_ids"].shape == shard0[0]["input_ids"].shape
        )
        assert (
            captured[1]["args"][0][0]["input_ids"].shape == shard1[0]["input_ids"].shape
        )
        assert captured[0]["kwargs"] == {"output_seqlens": [1]}
        assert captured[1]["kwargs"] == {"output_seqlens": [2, 1]}

    @pytest.mark.asyncio
    async def test_dispatch_post_merges_results_in_original_order(
        self,
    ):
        # [FUNCTION DOC]
        # dispatch().post splits a 4-item batch, collects per-shard results (sum of
        # attention
        # mask) and merges them back into original order.
        def _shard_handler(url, data, headers):
            # [FUNCTION DOC]
            # HELPER: Fake worker returning, for each item in its shard, the attention
            # mask sum.
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            shard_result = [int(item["attention_mask"].sum().item()) for item in shard]
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(shard_result),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [
            _make_tensor_item(5),
            _make_tensor_item(11),
            _make_tensor_item(7),
            _make_tensor_item(13),
        ]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/forward_batch").post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [5, 11, 7, 13]

    @pytest.mark.asyncio
    async def test_dispatch_post_does_not_pad_training_routes(self):
        # [FUNCTION DOC]
        # Training routes do not pad: a 3-item batch on DP=2 raises ValueError
        # "divisible by K".
        def _shard_handler(url, data, headers):
            # [FUNCTION DOC]
            # HELPER: Fake worker returning per-item attention mask sums.
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(
                        [int(item["attention_mask"].sum().item()) for item in shard]
                    ),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(5), _make_tensor_item(11), _make_tensor_item(7)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        with pytest.raises(ValueError, match="divisible by K"):
            await dispatcher.dispatch("/train_batch").post(body)

    @pytest.mark.asyncio
    async def test_dispatch_post_pads_eval_routes_only(self):
        # [FUNCTION DOC]
        # Eval routes with pad_eval_batch pad a 3-item batch to 4 and the padded item's
        # result
        # (0) appears last in the merged output.
        def _shard_handler(url, data, headers):
            # [FUNCTION DOC]
            # HELPER: Fake worker returning per-item attention mask sums.
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {
                    "status": "success",
                    "result": serialize_value(
                        [int(item["attention_mask"].sum().item()) for item in shard]
                    ),
                }
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(5), _make_tensor_item(11), _make_tensor_item(7)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch(
            "/eval_batch", pad_eval_batch=True
        ).post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [5, 11, 7, 0]


class TestScalarFanOut:
    # [CLASS DOC]
    # Tests for scalar fan-out of non-partitionable payloads.
    """Tests for _scalar_fan_out: non-partitionable payloads must reach all workers."""

    @pytest.mark.asyncio
    async def test_non_partitionable_tensor_fans_out_to_all_workers(self):
        # [FUNCTION DOC]
        # A non-list (dict) argument is sent whole to the DP head and as empty args to
        # the other
        # model-parallel worker.
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0, 0], session=session
        )

        packed = {"tensor_a": [1, 2, 3]}
        body = orjson.dumps(
            {
                "args": serialize_value([packed]),
                "kwargs": serialize_value({}),
            }
        )

        await dispatcher.dispatch("/forward_batch").post(body)

        captured = session.captured_payloads
        assert len(captured) == 2

        assert captured[0]["args"] == [packed]
        assert captured[0]["kwargs"] == {}

        assert captured[1]["args"] == []
        assert captured[1]["kwargs"] == {}

    @pytest.mark.asyncio
    async def test_returns_first_dp_head_with_non_contiguous_heads(self):
        # [FUNCTION DOC]
        # With non-contiguous DP heads (workers 1 and 3), a scalar request returns the
        # result of
        # the first DP head (worker-1).
        def _echo_handler(url, data, headers):
            # [FUNCTION DOC]
            # HELPER: Fake worker that returns its own base address as the result.
            _ = headers
            addr = url.rsplit("/", 1)[0]
            return orjson.dumps({"status": "success", "result": serialize_value(addr)})

        session = _CapturingSession(post_handler=_echo_handler)
        dispatcher = _make_dispatcher(
            dp_size=2,
            dp_heads=[1, 3],
            dp_ranks=[0, 0, 1, 1],
            session=session,
        )

        body = orjson.dumps(
            {
                "args": serialize_value(["scalar_value"]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/train_batch").post(body)
        result_payload = orjson.loads(result_bytes)

        assert result_payload == {
            "status": "success",
            "result": serialize_value("http://worker-1:19001"),
        }

    @pytest.mark.asyncio
    async def test_partitionable_list_still_uses_tensor_dispatch(self):
        # [FUNCTION DOC]
        # A list of tensor dicts is still partitioned across DP heads (each returns
        # shard length
        # 1, merged [1, 1]).
        def _shard_handler(url, data, headers):
            # [FUNCTION DOC]
            # HELPER: Fake worker returning the length of its shard.
            _ = url, headers
            payload = orjson.loads(data)
            args = deserialize_value(payload["args"])
            shard = args[0] if args else []
            return orjson.dumps(
                {"status": "success", "result": serialize_value(len(shard))}
            )

        session = _CapturingSession(post_handler=_shard_handler)
        dispatcher = _make_dispatcher(
            dp_size=2, dp_heads=[0, 1], dp_ranks=[0, 1], session=session
        )

        batch = [_make_tensor_item(4), _make_tensor_item(8)]
        body = orjson.dumps(
            {
                "args": serialize_value([batch]),
                "kwargs": serialize_value({}),
            }
        )

        result_bytes = await dispatcher.dispatch("/forward_batch").post(body)
        result_payload = orjson.loads(result_bytes)
        merged = deserialize_value(result_payload["result"])

        assert merged == [1, 1]

    @pytest.mark.asyncio
    async def test_single_worker_single_dp_head(self):
        # [FUNCTION DOC]
        # A single worker that is the only DP head receives exactly one request with the
        # args.
        session = _CapturingSession()
        dispatcher = _make_dispatcher(
            dp_size=1, dp_heads=[0], dp_ranks=[0], session=session
        )

        body = orjson.dumps(
            {
                "args": serialize_value(["value"]),
                "kwargs": serialize_value({}),
            }
        )

        await dispatcher.dispatch("/eval_batch").post(body)

        captured = session.captured_payloads
        assert len(captured) == 1
        assert captured[0]["args"] == ["value"]
