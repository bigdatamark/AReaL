# [MODULE DOC]
# Async tests for the data service worker app (areal.infra.data_service.worker.app):
# health, dataset load/unload, sample fetching, concurrency between load/unload/fetch
# (per-dataset state locks and the global datasets lock), epoch reset, state save/load
# to disk, and tensor shard endpoints. The app runs in-process via httpx.ASGITransport;
# the dataset loader is replaced by small in-memory HuggingFace Datasets and the
# tokenizer loader is stubbed. Some tests reach into the app's closure variables to
# manipulate internal locks.
from __future__ import annotations

# pyright: reportMissingImports=false
import asyncio
import inspect
import threading
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from datasets import Dataset

from areal.infra.data_service.worker.app import create_worker_app
from areal.infra.data_service.worker.config import DataWorkerConfig

DATASET_ID = "test-train"


@pytest.fixture
def config() -> DataWorkerConfig:
    # [FUNCTION DOC]
    # Fixture returning a single-rank DataWorkerConfig on 127.0.0.1 with port 0 and no
    # dataloader workers.
    return DataWorkerConfig(
        host="127.0.0.1",
        port=0,
        rank=0,
        world_size=1,
        dataloader_num_workers=0,
    )


def _make_mock_dataset(n: int = 20) -> Dataset:
    # [FUNCTION DOC]
    # HELPER: Returns an in-memory datasets.Dataset with n (default 20) rows of 'text'
    # and 'label' columns.
    return Dataset.from_dict(
        {
            "text": [f"sample_{i}" for i in range(n)],
            "label": list(range(n)),
        }
    )


def _load_payload(**overrides: object) -> dict[str, object]:
    # [FUNCTION DOC]
    # HELPER: Returns the /datasets/load JSON body (DATASET_ID, dummy path, type 'rl',
    # shuffle False) updated with keyword overrides.
    payload: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_path": "test/dataset",
        "dataset_type": "rl",
        "shuffle": False,
    }
    payload.update(overrides)
    return payload


def _get_route_endpoint(app, path: str, method: str = "POST"):
    # [FUNCTION DOC]
    # HELPER: Scans app.routes for a route with the given path whose methods include
    # method (default POST) and returns its endpoint function. Raises AssertionError if
    # not found.
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


def _get_worker_nonlocals(app) -> Mapping[str, object]:
    # [FUNCTION DOC]
    # HELPER: Returns the nonlocal closure variables of the /datasets/unload endpoint
    # (via inspect.getclosurevars), exposing the app's internal 'datasets' map and
    # 'datasets_lock'.
    endpoint = _get_route_endpoint(app, "/datasets/unload")
    return inspect.getclosurevars(endpoint).nonlocals


@pytest_asyncio.fixture
async def client(config: DataWorkerConfig):
    # [FUNCTION DOC]
    # Async fixture yielding an AsyncClient bound to a fresh worker app with no
    # datasets.
    app = create_worker_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def loaded_client(config: DataWorkerConfig):
    # [FUNCTION DOC]
    # Async fixture: with the dataset loader patched to an 8-row mock and tokenizer
    # loading stubbed, creates the app, loads DATASET_ID (asserting 200) and yields the
    # client. Patches stay active for the test's duration.
    with (
        patch("areal.infra.data_service.worker.app._get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(8)
        mock_get.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            yield c


@pytest_asyncio.fixture
async def loaded_worker(config: DataWorkerConfig):
    # [FUNCTION DOC]
    # Async fixture like loaded_client but yields a dict with the app, the client, and
    # the app's internal datasets map and datasets_lock taken from endpoint closures.
    with (
        patch("areal.infra.data_service.worker.app._get_custom_dataset") as mock_get,
        patch(
            "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
        ) as mock_load,
    ):
        ds = _make_mock_dataset(8)
        mock_get.return_value = ds
        mock_load.return_value = (None, None)

        app = create_worker_app(config)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/datasets/load", json=_load_payload())
            assert resp.status_code == 200
            nonlocals = _get_worker_nonlocals(app)
            yield {
                "app": app,
                "client": c,
                "datasets": nonlocals["datasets"],
                "datasets_lock": nonlocals["datasets_lock"],
            }


@pytest.mark.asyncio
class TestWorkerHealth:
    # [CLASS DOC]
    # Tests GET /health.
    async def test_health_returns_200(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Fresh worker reports status 'ok' and 0 datasets.
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["datasets"] == 0

    async def test_health_shows_dataset_count(self, loaded_client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # After loading one dataset, /health reports 1 dataset.
        resp = await loaded_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["datasets"] == 1


@pytest.mark.asyncio
class TestDatasetLoading:
    # [CLASS DOC]
    # Tests dataset load and unload endpoints.
    async def test_load_dataset_returns_steps_per_epoch(self, config: DataWorkerConfig):
        # [FUNCTION DOC]
        # Loading a 20-row mock dataset returns steps_per_epoch > 0 and dataset_size 20.
        # KNOWN ISSUE: Unlike other tests, load_hf_processor_and_tokenizer is not
        # patched here, so the test depends on the real loader tolerating the dummy
        # path/config.
        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset"
            ) as mock_get,
        ):
            ds = _make_mock_dataset(20)
            mock_get.return_value = ds

            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                resp = await c.post("/datasets/load", json=_load_payload())

        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_per_epoch"] > 0
        assert data["dataset_size"] == 20

    async def test_load_dataset_duplicate_409(self, loaded_client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Loading an already loaded dataset id returns 409.
        resp = await loaded_client.post("/datasets/load", json=_load_payload())
        assert resp.status_code == 409

    async def test_unload_dataset_removes(self, loaded_client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Unloading a loaded dataset returns 200 and the dataset count drops to 0.
        resp = await loaded_client.post(
            "/datasets/unload", json={"dataset_id": DATASET_ID}
        )
        assert resp.status_code == 200

        health = await loaded_client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 0

    async def test_unload_unknown_dataset_404(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Unloading an unknown dataset returns 404.
        resp = await client.post("/datasets/unload", json={"dataset_id": "unknown"})
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestSampleFetch:
    # [CLASS DOC]
    # Tests POST /v1/samples/fetch.
    async def test_fetch_samples_returns_data(self, loaded_client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Fetching indices [0, 1] returns two samples.
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["samples"]) == 2

    async def test_fetch_samples_unknown_dataset_404(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Fetching from an unknown dataset returns 404.
        resp = await client.post(
            "/v1/samples/fetch",
            json={"dataset_id": "unknown", "indices": [0]},
        )
        assert resp.status_code == 404

    async def test_fetch_samples_returns_distinct_items(
        self, loaded_client: httpx.AsyncClient
    ):
        # [FUNCTION DOC]
        # Fetching three indices returns three samples, the first two distinct.
        resp = await loaded_client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0, 1, 2]},
        )
        assert resp.status_code == 200
        samples = resp.json()["samples"]
        assert len(samples) == 3
        assert samples[0] != samples[1]


@pytest.mark.asyncio
class TestWorkerConcurrency:
    # [CLASS DOC]
    # Tests locking and race handling between concurrent operations.
    async def test_load_dataset_returns_409_when_same_id_currently_loading(
        self, config: DataWorkerConfig
    ):
        # [FUNCTION DOC]
        # Makes the dataset loader block on a threading.Event; starts a first load as a
        # task, waits until loading has begun, then asserts a second load of the same id
        # returns 409 with 'currently loading' in detail. Releases the first load and
        # asserts it returns 200.
        load_started = threading.Event()
        release_load = threading.Event()

        def slow_get_dataset(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Signals load_started, waits up to 2s for release_load (asserting
            # it was set), then returns an 8-row mock dataset.
            del args, kwargs
            load_started.set()
            assert release_load.wait(timeout=2)
            return _make_mock_dataset(8)

        with (
            patch(
                "areal.infra.data_service.worker.app._get_custom_dataset",
                side_effect=slow_get_dataset,
            ),
            patch(
                "areal.infra.data_service.worker.app.load_hf_processor_and_tokenizer"
            ) as mock_load,
        ):
            mock_load.return_value = (None, None)
            app = create_worker_app(config)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                first_load = asyncio.create_task(
                    client.post("/datasets/load", json=_load_payload())
                )
                await asyncio.to_thread(load_started.wait, 1)

                second_load = await client.post("/datasets/load", json=_load_payload())
                assert second_load.status_code == 409
                assert "currently loading" in second_load.json()["detail"].lower()

                release_load.set()
                first_response = await first_load
                assert first_response.status_code == 200

    async def test_unload_waits_for_in_flight_state_lock(self, loaded_worker):
        # [FUNCTION DOC]
        # Holds the dataset's state lock, starts an unload task, asserts it does not
        # complete while the lock is held, then releases it and asserts the unload
        # returns 200 within 1s.
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)
        assert not unload_task.done()

        state.lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

    async def test_fetch_returns_409_after_unload_starts(self, loaded_worker):
        # [FUNCTION DOC]
        # Holds the state lock, starts an unload, then acquires the global datasets_lock
        # and releases the state lock so the unload is mid-flight; asserts a fetch now
        # returns 409 mentioning 'unloading'. Releases datasets_lock and asserts the
        # unload completes with 200.
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]
        datasets_lock = loaded_worker["datasets_lock"]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)

        await datasets_lock.acquire()
        state.lock.release()
        await asyncio.sleep(0)

        fetch_response = await client.post(
            "/v1/samples/fetch",
            json={"dataset_id": DATASET_ID, "indices": [0]},
        )
        assert fetch_response.status_code == 409
        assert "unloading" in fetch_response.json()["detail"].lower()

        datasets_lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

    async def test_unrelated_load_succeeds_while_unload_waits_on_state_lock(
        self, loaded_worker
    ):
        # [FUNCTION DOC]
        # Holds the state lock of DATASET_ID so its unload waits, then loads a different
        # dataset id and asserts it succeeds within 1s while the unload is still
        # pending; after releasing the lock the unload succeeds and exactly one dataset
        # remains.
        client = loaded_worker["client"]
        state = loaded_worker["datasets"][DATASET_ID]

        await state.lock.acquire()
        unload_task = asyncio.create_task(
            client.post("/datasets/unload", json={"dataset_id": DATASET_ID})
        )
        await asyncio.sleep(0)

        other_load = await asyncio.wait_for(
            client.post(
                "/datasets/load",
                json=_load_payload(
                    dataset_id="test-valid",
                    dataset_path="test/other-dataset",
                ),
            ),
            timeout=1,
        )
        assert other_load.status_code == 200
        assert not unload_task.done()

        state.lock.release()
        unload_response = await asyncio.wait_for(unload_task, timeout=1)
        assert unload_response.status_code == 200

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["datasets"] == 1


@pytest.mark.asyncio
class TestEpochReset:
    # [CLASS DOC]
    # Tests POST /epoch/reset.
    async def test_epoch_reset_updates_epoch(self, loaded_client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Resetting to epoch 1 returns 200 with epoch 1.
        reset = await loaded_client.post(
            "/epoch/reset", json={"dataset_id": DATASET_ID, "epoch": 1}
        )
        assert reset.status_code == 200
        assert reset.json()["epoch"] == 1

    async def test_epoch_reset_unknown_dataset_404(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # Resetting an unknown dataset returns 404.
        resp = await client.post(
            "/epoch/reset", json={"dataset_id": "unknown", "epoch": 1}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestStatePersistence:
    # [CLASS DOC]
    # Tests /state/save and /state/load.
    async def test_state_save_creates_file(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        # [FUNCTION DOC]
        # Saving state writes worker_<rank>.pkl (worker_0.pkl) under the given directory
        # and returns status 'ok'.
        resp = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert resp.status_code == 200
        out = resp.json()
        assert out["status"] == "ok"
        assert (tmp_path / "worker_0.pkl").exists()

    async def test_state_load_restores(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        # [FUNCTION DOC]
        # Saving and then loading state from the same directory both return 200, the
        # load with status 'ok'.
        save = await loaded_client.post(
            "/state/save", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert save.status_code == 200

        load = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(tmp_path)}
        )
        assert load.status_code == 200
        assert load.json()["status"] == "ok"

    async def test_state_load_missing_file_404(
        self, loaded_client: httpx.AsyncClient, tmp_path: Path
    ):
        # [FUNCTION DOC]
        # Loading state from a nonexistent directory returns 404.
        missing = tmp_path / "does-not-exist"
        resp = await loaded_client.post(
            "/state/load", json={"dataset_id": DATASET_ID, "path": str(missing)}
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
class TestTensorShardEndpoints:
    # [CLASS DOC]
    # Tests tensor shard storage endpoints.
    async def test_data_clear_returns_ok(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # DELETE /data/clear returns status 'ok' with tensor_shards 0.
        resp = await client.delete("/data/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["tensor_shards"] == 0

    async def test_data_shard_not_found_404(self, client: httpx.AsyncClient):
        # [FUNCTION DOC]
        # GET /data/<unknown id> returns 404.
        resp = await client.get("/data/nonexistent")
        assert resp.status_code == 404
