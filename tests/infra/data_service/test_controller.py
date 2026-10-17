# [MODULE DOC]
# Unit tests for DataServiceConfig and DataController
# (areal.infra.data_service.controller). Covers config defaults and derivation from
# dataset configs, controller initialization state, the synchronous _gateway_post helper
# (with aiohttp.ClientSession mocked), and dataset registration/unregistration (with
# run_async_task patched).
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from areal.api.cli_args import (
    PPOConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SFTConfig,
)
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.controller.controller import DataController


def _make_mock_aiohttp(status=200, json_data=None, text_data=""):
    # [FUNCTION DOC]
    # HELPER: Builds mocks for aiohttp: a response with the given status, async json()
    # returning json_data (or {}), async text() returning text_data, usable as an async
    # context manager; and a session whose post() returns that response, also an async
    # context manager. Returns (session_class_mock, session_mock).
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data if json_data is not None else {})
    mock_resp.text = AsyncMock(return_value=text_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_cls = MagicMock(return_value=mock_session)
    return mock_cls, mock_session


class TestDataServiceConfig:
    # [CLASS DOC]
    # Tests DataServiceConfig fields and from_dataset_config.
    def test_default_values(self):
        # [FUNCTION DOC]
        # Defaults: one worker, setup_timeout 120s, separation scheduling strategy.
        cfg = DataServiceConfig()

        assert cfg.num_workers == 1
        assert cfg.setup_timeout == 120.0
        assert isinstance(cfg.scheduling_strategy, SchedulingStrategy)
        assert cfg.scheduling_strategy.type == SchedulingStrategyType.separation

    def test_custom_values(self):
        # [FUNCTION DOC]
        # Custom num_workers and setup_timeout are stored.
        cfg = DataServiceConfig(
            num_workers=8,
            setup_timeout=300.0,
        )

        assert cfg.num_workers == 8
        assert cfg.setup_timeout == 300.0

    def test_scheduling_strategy_colocation(self):
        # [FUNCTION DOC]
        # A colocation strategy targeting 'rollout' is stored with fork defaulting to
        # True.
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            ),
        )

        assert cfg.scheduling_strategy.type == SchedulingStrategyType.colocation
        assert cfg.scheduling_strategy.target == "rollout"
        assert cfg.scheduling_strategy.fork is True

    def test_scheduling_strategy_separation(self):
        # [FUNCTION DOC]
        # A separation strategy given as a plain string keeps type 'separation' and
        # target None.
        cfg = DataServiceConfig(
            scheduling_strategy=SchedulingStrategy(type="separation"),
        )

        assert cfg.scheduling_strategy.type == "separation"
        assert cfg.scheduling_strategy.target is None

    def test_config_in_base_experiment_config(self):
        # [FUNCTION DOC]
        # from_dataset_config on a basic 'rl' TrainDatasetConfig returns a
        # DataServiceConfig.
        # KNOWN ISSUE: The name suggests a base experiment config is involved, but only
        # a TrainDatasetConfig is used (duplicate of test_config_in_rw_config).
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="rl")
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert isinstance(cfg, DataServiceConfig)

    def test_from_dataset_config_uses_num_dataset_workers(self):
        # [FUNCTION DOC]
        # from_dataset_config maps num_dataset_workers to num_workers (7) and
        # num_workers to dataloader_num_workers (5).
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="rl",
            num_workers=5,
            num_dataset_workers=7,
        )
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert cfg.num_workers == 7
        assert cfg.dataloader_num_workers == 5

    def test_config_in_ppo_config(self):
        # [FUNCTION DOC]
        # from_dataset_config works on a default PPOConfig's train_dataset.
        cfg = PPOConfig(experiment_name="exp", trial_name="trial")
        ds_cfg = DataServiceConfig.from_dataset_config(cfg.train_dataset)
        assert isinstance(ds_cfg, DataServiceConfig)

    def test_config_in_sft_config(self):
        # [FUNCTION DOC]
        # from_dataset_config works on a default SFTConfig's train_dataset.
        cfg = SFTConfig(experiment_name="exp", trial_name="trial")
        ds_cfg = DataServiceConfig.from_dataset_config(cfg.train_dataset)
        assert isinstance(ds_cfg, DataServiceConfig)

    def test_config_in_rw_config(self):
        # [FUNCTION DOC]
        # from_dataset_config on a plain 'rl' TrainDatasetConfig returns a
        # DataServiceConfig.
        # KNOWN ISSUE: Named for the reward-model config but never constructs one;
        # identical to test_config_in_base_experiment_config.
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="rl")
        cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert isinstance(cfg, DataServiceConfig)


class TestDataControllerInit:
    # [CLASS DOC]
    # Tests DataController construction.
    def test_init_stores_config(self):
        # [FUNCTION DOC]
        # The config object is stored as controller.config.
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.config is cfg

    def test_init_stores_scheduler(self):
        # [FUNCTION DOC]
        # The scheduler object is stored as controller.scheduler.
        cfg = DataServiceConfig()
        scheduler = MagicMock()

        controller = DataController(cfg, scheduler)

        assert controller.scheduler is scheduler

    def test_init_empty_state(self):
        # [FUNCTION DOC]
        # A fresh controller has no workers, an empty gateway address and no datasets.
        controller = DataController(DataServiceConfig(), MagicMock())

        assert controller.workers == []
        assert controller._gateway_addr == ""
        assert controller._datasets == {}


class TestDataControllerGatewayPost:
    # [CLASS DOC]
    # Tests DataController._gateway_post against a mocked aiohttp session.
    def test_gateway_post_sends_bearer_auth(self):
        # [FUNCTION DOC]
        # Returns the parsed JSON body and sends Authorization 'Bearer <api_key>'.
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, mock_session = _make_mock_aiohttp(status=200, json_data={"ok": True})

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            result = controller._gateway_post("/v1/test", "api-key", {"x": 1})

        assert result == {"ok": True}
        _, kwargs = mock_session.post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer api-key"}

    def test_gateway_post_sends_json_payload(self):
        # [FUNCTION DOC]
        # Sends the given payload as the json kwarg.
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, mock_session = _make_mock_aiohttp(status=200, json_data={"ok": True})

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            controller._gateway_post("/v1/test", "api-key", {"payload": "value"})

        _, kwargs = mock_session.post.call_args
        assert kwargs["json"] == {"payload": "value"}

    def test_gateway_post_raises_on_error(self):
        # [FUNCTION DOC]
        # A 500 response raises RuntimeError containing 'returned 500'; fails explicitly
        # if nothing is raised.
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._gateway_addr = "http://gateway"

        mock_cls, _ = _make_mock_aiohttp(status=500, text_data="boom")

        with patch(
            "areal.infra.data_service.controller.controller.aiohttp.ClientSession",
            mock_cls,
        ):
            try:
                controller._gateway_post("/v1/test", "api-key", {})
            except RuntimeError as exc:
                assert "returned 500" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError for gateway error response")


class TestDataControllerRegisterDataset:
    # [CLASS DOC]
    # Tests register_dataset / unregister_dataset with the async runner patched.
    def test_register_returns_dataset_metadata(self):
        # [FUNCTION DOC]
        # register_dataset returns api_key, dataset_id, total_samples (from
        # dataset_size) and num_workers, and caches the dataset under its api key.
        controller = DataController(DataServiceConfig(), MagicMock())

        payload = {
            "api_key": "ds-key",
            "dataset_id": "test-ds",
            "dataset_size": 32,
            "num_workers": 4,
        }

        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            return_value=payload,
        ) as mock_run:
            result = controller.register_dataset(
                dataset_id="test-ds",
                dataset_path="dummy",
                dataset_type="rl",
                drop_last=True,
            )

        assert mock_run.called
        assert result["api_key"] == "ds-key"
        assert result["dataset_id"] == "test-ds"
        assert result["total_samples"] == 32
        assert result["num_workers"] == 4
        assert controller._datasets["ds-key"]["dataset_id"] == "test-ds"

    def test_register_stores_drop_last_flag(self):
        # [FUNCTION DOC]
        # register_dataset stores drop_last=False in the cached dataset entry.
        controller = DataController(DataServiceConfig(), MagicMock())

        payload = {
            "api_key": "ds-key",
            "dataset_id": "test-ds",
            "dataset_size": 30,
            "num_workers": 4,
        }
        with patch(
            "areal.infra.utils.concurrent.run_async_task",
            return_value=payload,
        ):
            controller.register_dataset(
                dataset_id="test-ds",
                dataset_path="dummy",
                dataset_type="rl",
                drop_last=False,
            )
        assert controller._datasets["ds-key"]["drop_last"] is False

    def test_unregister_removes_local_dataset_cache(self):
        # [FUNCTION DOC]
        # unregister_dataset('a') removes only cache entries whose dataset_id is 'a'.
        controller = DataController(DataServiceConfig(), MagicMock())
        controller._datasets["key-1"] = {"dataset_id": "a"}
        controller._datasets["key-2"] = {"dataset_id": "b"}

        with patch("areal.infra.utils.concurrent.run_async_task"):
            controller.unregister_dataset("a")

        assert "key-1" not in controller._datasets
        assert "key-2" in controller._datasets
