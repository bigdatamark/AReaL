# [MODULE DOC]
# Unit tests for the data proxy's RTensor shard storage endpoints (PUT/GET /data/{id},
# POST /data/batch, DELETE /data/clear), including validation and error paths. Tensors
# are encoded with areal.infra.rpc.serialization.serialize_value + orjson; the backing
# store is the module-global areal.infra.rpc.rtensor storage, reset around every test.
"""Unit tests for RTensor storage endpoints in the data proxy FastAPI app.

Covers all 4 RTensor endpoints:
  PUT  /data/{shard_id}   — store a tensor shard
  GET  /data/{shard_id}   — retrieve a tensor shard
  POST /data/batch        — batch retrieve tensor shards
  DELETE /data/clear      — clear specified tensor shards
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import orjson
import pytest
import pytest_asyncio
import torch

from areal.infra.rpc import rtensor as rtensor_storage
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.v2.inference_service.data_proxy.app import create_app
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionStore

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_rtensor_storage():
    # [FUNCTION DOC]
    # Autouse fixture: clears the global rtensor _storage and _storage_stats dicts
    # before and after each test (global state mutation).
    rtensor_storage._storage.clear()
    rtensor_storage._storage_stats.clear()
    yield
    rtensor_storage._storage.clear()
    rtensor_storage._storage_stats.clear()


@pytest.fixture
def config():
    # [FUNCTION DOC]
    # Fixture: DataProxyConfig on 127.0.0.1:18082 with fake backend, tokenizer
    # "mock-tokenizer" and 10s timeout.
    return DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
    )


@pytest_asyncio.fixture
async def client(config):
    # [FUNCTION DOC]
    # Async fixture: creates the app, injects MagicMock tokenizer/inf_bridge (async
    # pause/resume)/areal_client, a PauseState, config, SessionStore and version 0 into
    # app.state, and yields an httpx AsyncClient over ASGITransport at "http://test".
    from areal.v2.inference_service.data_proxy.pause import PauseState

    app = create_app(config)
    pause_state = PauseState()

    app.state.tokenizer = MagicMock()
    app.state.inf_bridge = MagicMock()
    app.state.inf_bridge.pause = AsyncMock()
    app.state.inf_bridge.resume = AsyncMock()
    app.state.areal_client = MagicMock()
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.version = 0

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# =============================================================================
# Helpers
# =============================================================================


def _serialize_tensor_body(tensor: torch.Tensor) -> bytes:
    # [FUNCTION DOC]
    # HELPER: Serializes a tensor via serialize_value and returns orjson-encoded bytes.
    return orjson.dumps(serialize_value(tensor))


def _deserialize_response_bytes(content: bytes) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: orjson-decodes response bytes and deserializes them into a tensor.
    return deserialize_value(orjson.loads(content))


def _deserialize_batch_response_bytes(content: bytes) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Same as _deserialize_response_bytes but typed as returning a list of
    # tensors.
    return deserialize_value(orjson.loads(content))


# =============================================================================
# Tests
# =============================================================================


class TestDataProxyRTensor:
    # [CLASS DOC]
    # Endpoint tests for RTensor shard storage.
    @pytest.mark.asyncio
    async def test_put_store_shard(self, client):
        # [FUNCTION DOC]
        # PUTs a serialized 1-D tensor under a random uuid; asserts 200 and status "ok".
        """PUT /data/{shard_id} stores tensor → 200 with status ok."""
        shard_id = str(uuid.uuid4())
        tensor = torch.tensor([1.0, 2.0, 3.0])
        body = _serialize_tensor_body(tensor)

        resp = await client.put(
            f"/data/{shard_id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_get_retrieve_shard(self, client):
        # [FUNCTION DOC]
        # PUTs a 2x2 tensor then GETs it; asserts 200 and the deserialized tensor equals
        # the original.
        """PUT a tensor then GET it back → deserialized tensor matches original."""
        shard_id = str(uuid.uuid4())
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        body = _serialize_tensor_body(tensor)

        put_resp = await client.put(
            f"/data/{shard_id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert put_resp.status_code == 200

        get_resp = await client.get(f"/data/{shard_id}")
        assert get_resp.status_code == 200

        retrieved = _deserialize_response_bytes(get_resp.content)
        torch.testing.assert_close(retrieved, tensor)

    @pytest.mark.asyncio
    async def test_get_unknown_shard_returns_404(self, client):
        # [FUNCTION DOC]
        # Asserts GET for an unknown random shard id returns 404.
        """GET /data/{random-uuid} for a non-existent shard → 404."""
        shard_id = str(uuid.uuid4())
        resp = await client.get(f"/data/{shard_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_post_batch_retrieve(self, client):
        # [FUNCTION DOC]
        # PUTs three tensors of different shapes, POSTs /data/batch with their ids, and
        # asserts the deserialized list has the same length and each tensor matches in
        # order.
        """PUT 3 tensors with different shapes, POST /data/batch → all 3 match originals."""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.zeros(2, 4),
            torch.ones(3, 3, 3),
        ]
        shard_ids = []

        for tensor in tensors:
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)
            body = _serialize_tensor_body(tensor)
            put_resp = await client.put(
                f"/data/{shard_id}",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert put_resp.status_code == 200

        batch_resp = await client.post(
            "/data/batch",
            json={"shard_ids": shard_ids},
        )
        assert batch_resp.status_code == 200

        retrieved_list = _deserialize_batch_response_bytes(batch_resp.content)
        assert len(retrieved_list) == len(tensors)
        for original, retrieved in zip(tensors, retrieved_list):
            torch.testing.assert_close(retrieved, original)

    @pytest.mark.asyncio
    async def test_post_batch_missing_shard_returns_400(self, client):
        # [FUNCTION DOC]
        # Stores one shard then batch-requests it plus "nonexistent-id"; asserts 400
        # with status "error", the exact not-found message and missing_shard_ids
        # ["nonexistent-id"].
        """POST /data/batch with a missing shard_id → 400 with error details."""
        shard_id = str(uuid.uuid4())
        tensor = torch.tensor([1.0, 2.0])
        body = _serialize_tensor_body(tensor)
        put_resp = await client.put(
            f"/data/{shard_id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert put_resp.status_code == 200

        nonexistent_id = "nonexistent-id"
        batch_resp = await client.post(
            "/data/batch",
            json={"shard_ids": [shard_id, nonexistent_id]},
        )
        assert batch_resp.status_code == 400
        data = batch_resp.json()
        assert data["status"] == "error"
        assert data["message"] == "One or more requested shards were not found"
        assert data["missing_shard_ids"] == [nonexistent_id]

    @pytest.mark.asyncio
    async def test_post_batch_invalid_body_returns_400(self, client):
        # [FUNCTION DOC]
        # Posts shard_ids as a string; asserts 400 or 422 and a "detail" or "message"
        # key.
        """POST /data/batch with shard_ids as a non-list → 400/422 validation error."""
        batch_resp = await client.post(
            "/data/batch",
            json={"shard_ids": "not-a-list"},
        )

        # If calling FastAPI, it might return 422. If calling Flask Blueprint, 400.
        assert batch_resp.status_code in (400, 422)

        # Pydantic errors are formatted differently in FastAPI vs Flask.
        # Check for the presence of an error rather than the exact old string.
        data = batch_resp.json()
        assert "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_delete_clear_shards(self, client):
        # [FUNCTION DOC]
        # PUTs two random tensors, sends DELETE /data/clear with their ids, asserts 200
        # and cleared_count 2, then asserts GET for each returns 404.
        """PUT 2 tensors, DELETE /data/clear → 200 with cleared_count. Then GET → 404."""
        shard_ids = []
        for _ in range(2):
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)
            tensor = torch.rand(4)
            body = _serialize_tensor_body(tensor)
            put_resp = await client.put(
                f"/data/{shard_id}",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert put_resp.status_code == 200

        delete_resp = await client.request(
            "DELETE",
            "/data/clear",
            json={"shard_ids": shard_ids},
        )
        assert delete_resp.status_code == 200
        data = delete_resp.json()
        assert "cleared_count" in data
        assert data["cleared_count"] == 2

        for shard_id in shard_ids:
            get_resp = await client.get(f"/data/{shard_id}")
            assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_post_batch_malformed_json_returns_error(self, client):
        # [FUNCTION DOC]
        # Posts a non-JSON body to /data/batch; asserts 400/422 with "detail" or
        # "message".
        """POST /data/batch with non-JSON body → Now returns an error.

        Previous behavior was a 200 with an empty list, but Pydantic
        requires a valid BatchShardRequest object.
        """
        resp = await client.post(
            "/data/batch",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)
        data = resp.json()
        assert "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_post_batch_serialization_error_returns_500(
        self, client, monkeypatch
    ):
        # [FUNCTION DOC]
        # Stores a shard, monkeypatches serialize_value inside
        # areal.infra.rpc.guard.data_blueprint to raise RuntimeError, then asserts
        # /data/batch returns 500 with status "error" and the exception text in message.
        """Serialization failure in /data/batch → 500 with Flask-compatible body."""
        shard_id = str(uuid.uuid4())
        tensor = torch.tensor([1.0, 2.0])
        body = _serialize_tensor_body(tensor)
        put_resp = await client.put(
            f"/data/{shard_id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert put_resp.status_code == 200

        from areal.infra.rpc.guard import data_blueprint

        def _boom(data):
            # [FUNCTION DOC]
            # HELPER: Replacement serializer that always raises
            # RuntimeError("serialization kaboom").
            raise RuntimeError("serialization kaboom")

        monkeypatch.setattr(data_blueprint, "serialize_value", _boom)

        batch_resp = await client.post(
            "/data/batch",
            json={"shard_ids": [shard_id]},
        )
        assert batch_resp.status_code == 500
        data = batch_resp.json()
        assert data["status"] == "error"
        assert "serialization kaboom" in data["message"]

    @pytest.mark.asyncio
    async def test_post_batch_null_json_returns_error(self, client):
        # [FUNCTION DOC]
        # Posts a JSON null body to /data/batch; asserts 400/422 with "detail" or
        # "message".
        """POST /data/batch with ``null`` JSON body → error."""
        resp = await client.post(
            "/data/batch",
            content=b"null",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)
        data = resp.json()
        assert "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_post_batch_non_dict_json_returns_error(self, client):
        # [FUNCTION DOC]
        # Posts a JSON array body to /data/batch; asserts 400/422 with "detail" or
        # "message".
        """POST /data/batch with a JSON array → error."""
        resp = await client.post(
            "/data/batch",
            content=b"[1, 2, 3]",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code in (400, 422)
        data = resp.json()
        assert "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_post_batch_fetch_runtime_error_returns_500(
        self, client, monkeypatch
    ):
        # [FUNCTION DOC]
        # Stores a shard, monkeypatches rtensor_storage.fetch to raise RuntimeError,
        # then asserts /data/batch returns 500 with status "error" and "storage
        # corrupted" in message (a non-KeyError fetch failure is not treated as a
        # missing shard).
        """Non-KeyError failure during fetch → outer try/except returns 500."""
        shard_id = str(uuid.uuid4())
        tensor = torch.tensor([1.0])
        body = _serialize_tensor_body(tensor)
        put_resp = await client.put(
            f"/data/{shard_id}",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert put_resp.status_code == 200

        def _broken_fetch(sid):
            # [FUNCTION DOC]
            # HELPER: Replacement fetch that always raises RuntimeError("storage
            # corrupted").
            raise RuntimeError("storage corrupted")

        monkeypatch.setattr(rtensor_storage, "fetch", _broken_fetch)

        batch_resp = await client.post(
            "/data/batch",
            json={"shard_ids": [shard_id]},
        )
        assert batch_resp.status_code == 500
        data = batch_resp.json()
        assert data["status"] == "error"
        assert "storage corrupted" in data["message"]
