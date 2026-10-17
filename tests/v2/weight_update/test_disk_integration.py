# [MODULE DOC]
# Tests for disk-mode weight updates in the v2 weight update gateway. Unit tests drive
# the gateway app (/connect, /update_weights, /disconnect) with a Starlette TestClient
# and a mocked aiohttp session that records outbound worker calls; a GPU end-to-end test
# runs real FSDP training workers and SGLang inference workers, saves HF weights to a
# shared directory through the gateway, and checks the inference server loaded them.
# Shared helpers come from test_nccl_integration.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch

from tests.v2.weight_update.test_nccl_integration import (
    _get_test_model_path,
    _make_local_scheduler,
    _validate_weight_update_correctness,
)

from areal.infra.platforms import current_platform
from areal.v2.weight_update.gateway.app import create_app
from areal.v2.weight_update.gateway.config import (
    PairInfo,
    WeightUpdateConfig,
)


@pytest.fixture()
def config() -> WeightUpdateConfig:
    # [FUNCTION DOC]
    # Fixture: WeightUpdateConfig with admin key "test-key" and 5s init/update timeouts.
    return WeightUpdateConfig(
        admin_api_key="test-key",
        init_timeout_s=5,
        update_timeout_s=5,
    )


@pytest.fixture()
def app(config):
    # [FUNCTION DOC]
    # Fixture: creates the weight update gateway app from the config.
    return create_app(config)


@pytest.fixture()
def client(app):
    # [FUNCTION DOC]
    # Fixture: returns a Starlette TestClient for the app (lazy import).
    from starlette.testclient import TestClient

    return TestClient(app)


ADMIN_HEADERS = {"Authorization": "Bearer test-key"}


def _make_mock_aiohttp_session(called_urls: list[tuple[str, str]]):
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock session whose post/get are async context managers that
    # append ("POST"|"GET", url) to called_urls and yield a fake 200 response with
    # raise_for_status no-op and an async json() (success result for POST, empty dict
    # for
    # GET).
    mock_session = MagicMock()

    @asynccontextmanager
    async def _fake_post(url, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake aiohttp POST context manager; records the URL and yields a
        # success
        # response.
        called_urls.append(("POST", url))
        resp = MagicMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={"status": "success", "result": None})
        yield resp

    @asynccontextmanager
    async def _fake_get(url, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake aiohttp GET context manager; records the URL and yields an empty-
        # JSON
        # response.
        called_urls.append(("GET", url))
        resp = MagicMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={})
        yield resp

    mock_session.post = _fake_post
    mock_session.get = _fake_get
    return mock_session


class TestDiskConnect:
    # [CLASS DOC]
    # Validation and registration tests for /connect in disk mode.
    def test_disk_connect_requires_non_empty_save_path(self, client, app):
        # [FUNCTION DOC]
        # Empty save_path returns 400 with the exact error and registers nothing.
        resp = client.post(
            "/connect",
            json={
                "pair_name": "missing_path",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == "save_path is required when mode='disk'"
        assert app.state.registry.get_by_name("missing_path") is None

    def test_disk_connect_requires_absolute_save_path(self, client, app):
        # [FUNCTION DOC]
        # Relative save_path returns 400 requiring an absolute path and registers
        # nothing.
        resp = client.post(
            "/connect",
            json={
                "pair_name": "relative_path",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "shared/weights",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == (
            "save_path must be an absolute path when mode='disk'"
        )
        assert app.state.registry.get_by_name("relative_path") is None

    def test_disk_connect_requires_lora_name_when_lora_enabled(self, client, app):
        # [FUNCTION DOC]
        # use_lora True with empty lora_name returns 400 and registers nothing.
        resp = client.post(
            "/connect",
            json={
                "pair_name": "missing_lora_name",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/lora",
                "use_lora": True,
                "lora_name": "",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == "lora_name is required when use_lora=True"
        assert app.state.registry.get_by_name("missing_lora_name") is None

    def test_disk_connect_registers_pair(self, client, app):
        # [FUNCTION DOC]
        # Valid disk connect returns the pair name and registers a pair with mode,
        # save_path and
        # worker URLs.
        resp = client.post(
            "/connect",
            json={
                "pair_name": "disk_pair",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/weights",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["pair_name"] == "disk_pair"

        registry = app.state.registry
        pair = registry.get_by_name("disk_pair")
        assert pair is not None
        assert pair.mode == "disk"
        assert pair.save_path == "/shared/weights"
        assert pair.train_worker_urls == ["http://train:8000"]
        assert pair.inference_worker_urls == ["http://infer:9000"]

    def test_disk_connect_with_lora(self, client, app):
        # [FUNCTION DOC]
        # Disk connect with LoRA stores use_lora and lora_name on the pair.
        resp = client.post(
            "/connect",
            json={
                "pair_name": "lora_pair",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/lora",
                "use_lora": True,
                "lora_name": "my-adapter",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200

        pair = app.state.registry.get_by_name("lora_pair")
        assert pair.use_lora is True
        assert pair.lora_name == "my-adapter"

    def test_disk_connect_does_not_call_awex_endpoints(self, client, app):
        # [FUNCTION DOC]
        # Disk connect makes no outbound worker calls (no awex initialization).
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        resp = client.post(
            "/connect",
            json={
                "pair_name": "disk_only",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/tmp/w",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert len(called_urls) == 0


class TestDiskUpdateWeights:
    # [CLASS DOC]
    # /update_weights tests for a registered full-weight disk pair.
    @pytest.fixture(autouse=True)
    def _register_disk_pair(self, app):
        # [FUNCTION DOC]
        # Autouse fixture: registers disk pair "test_disk" (one train and one inference
        # URL,
        # save_path /shared/weights) in the app registry.
        pair_info = PairInfo(
            pair_name="test_disk",
            train_worker_urls=["http://train:8000"],
            inference_worker_urls=["http://infer:9000"],
            mode="disk",
            save_path="/shared/weights",
        )
        app.state.registry.register(pair_info)

    def test_disk_update_versioned_save_path(self, client, app):
        # [FUNCTION DOC]
        # /update_weights with version 42 triggers a POST to the train worker's /save
        # endpoint.
        # KNOWN ISSUE: despite its name it only checks the /save URL, not the versioned
        # save path
        # in the request body.
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        client.post(
            "/update_weights",
            json={"pair_name": "test_disk", "version": 42},
            headers=ADMIN_HEADERS,
        )

        save_url = next(url for _, url in called_urls if "/save" in url)
        assert save_url == "http://train:8000/save"

    def test_disk_update_not_found_pair(self, client, app):
        # [FUNCTION DOC]
        # /update_weights for an unknown pair returns 404.
        resp = client.post(
            "/update_weights",
            json={"pair_name": "nonexistent", "version": 1},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 404


class TestDiskUpdateWeightsLora:
    # [CLASS DOC]
    # /update_weights tests for a LoRA disk pair.
    @pytest.fixture(autouse=True)
    def _register_lora_pair(self, app):
        # [FUNCTION DOC]
        # Autouse fixture: registers LoRA disk pair "lora_disk" with adapter "my-
        # adapter".
        pair_info = PairInfo(
            pair_name="lora_disk",
            train_worker_urls=["http://train:8000"],
            inference_worker_urls=["http://infer:9000"],
            mode="disk",
            save_path="/shared/lora_weights",
            use_lora=True,
            lora_name="my-adapter",
        )
        app.state.registry.register(pair_info)

    def test_lora_update_uses_load_lora_adapter(self, client, app):
        # [FUNCTION DOC]
        # LoRA update returns ok and calls the inference worker's /load_lora_adapter
        # instead of
        # /update_weights_from_disk.
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        resp = client.post(
            "/update_weights",
            json={"pair_name": "lora_disk", "version": 3},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        urls = [url for _, url in called_urls]
        assert "http://infer:9000/load_lora_adapter" in urls
        assert "http://infer:9000/update_weights_from_disk" not in urls


class TestDiskDisconnect:
    # [CLASS DOC]
    # /disconnect tests.
    def test_disconnect_removes_disk_pair(self, client, app):
        # [FUNCTION DOC]
        # A connected disk pair is removed from the registry by /disconnect.
        client.post(
            "/connect",
            json={
                "pair_name": "to_remove",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/tmp/w",
            },
            headers=ADMIN_HEADERS,
        )
        assert app.state.registry.get_by_name("to_remove") is not None

        resp = client.post(
            "/disconnect",
            json={"pair_name": "to_remove"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert app.state.registry.get_by_name("to_remove") is None


# ---------------------------------------------------------------------------
# E2E disk weight update: real FSDPEngine + SGLang server + gateway
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_disk_e2e_weight_update(n_gpus, tmp_path_factory):
    # [FUNCTION DOC]
    # GPU E2E, parametrized over 2/4/8 GPUs (skipped if fewer): splits GPUs in half
    # between
    # an SGLang RolloutControllerV2 and an FSDP GatewayTrainController on one local
    # scheduler; randomizes inference parameters, starts a WeightUpdateController
    # gateway,
    # connects a disk pair, updates weights at version 1 (expects ok), disconnects,
    # checks
    # weight_update_v1/config.json exists under the shared dir, that /generate still
    # works,
    # and that training and inference parameters match. Always destroys all controllers
    # and
    # scheduler workers.
    """Full round trip: FSDPEngine save → shared disk → SGLang load.

    Same infrastructure as :func:`test_awex_e2e_weight_update` but uses
    the disk-based weight update path (``mode="disk"``).  The gateway
    orchestrates pause → FSDP save to HF → SGLang
    ``/update_weights_from_disk`` → resume.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_half = n_gpus // 2
    tmp = tmp_path_factory.mktemp("disk_e2e")
    model_path = _get_test_model_path()
    shared_weight_dir = str(tmp / "shared_weights")

    scheduler = _make_local_scheduler(tmp, "disk_e2e", gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_half}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"fsdp:d{n_half}",
        experiment_name="test-disk-e2e",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.fsdp_engine.FSDPEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None

    try:
        # -- 1. SGLang via inference controller ----------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. FSDP engine via training controller -------------------------
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_ctrl.initialize(role="actor", ft_spec=ft_spec, wait=True)
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway -------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(
                host="127.0.0.1",
                request_timeout=300.0,
            )
        )
        wu_ctrl.initialize()

        # -- 4. Disk weight update lifecycle -------------------------------
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name="test_disk_e2e",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            mode="disk",
            save_path=shared_weight_dir,
        )

        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok", f"Disk weight update failed: {result.error}"
        assert result.version == 1

        wu_ctrl.disconnect()

        # -- 5. Verify the HF checkpoint exists on disk --------------------
        versioned_path = os.path.join(shared_weight_dir, "weight_update_v1")
        assert os.path.isdir(versioned_path), (
            f"Expected HF checkpoint at {versioned_path}"
        )
        assert os.path.isfile(os.path.join(versioned_path, "config.json")), (
            "Missing config.json in saved checkpoint"
        )

        # -- 6. Verify inference server still works post-update -----------
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        # -- 7. Validate training ↔ inference parameter equality ----------
        _validate_weight_update_correctness(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
        )

    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)
