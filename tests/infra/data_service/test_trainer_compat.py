# [MODULE DOC]
# Compatibility tests between trainers and the data service: cycle_dataloader epoch
# handling (areal.utils.data), DataServiceConfig.from_dataset_config defaults,
# importability/protocol of DataController and RDataset, get_custom_dataset bypassing
# the service when scheduling_spec is None, split fallback in _get_custom_dataset, the
# _EmptyDataLoader shim in rl_trainer, and trainer constructor signatures. Uses small
# fake dataloader/sampler classes; no network.
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from areal.infra.data_service.controller.config import DataServiceConfig
from areal.utils.data import cycle_dataloader


class _SamplerWithSetEpoch:
    # [CLASS DOC]
    # HELPER: Fake sampler that records every epoch passed to set_epoch.
    def __init__(self):
        # [FUNCTION DOC]
        # Initializes epochs as an empty list.
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        # [FUNCTION DOC]
        # Appends epoch to self.epochs.
        self.epochs.append(epoch)


class _FiniteDataloader:
    # [CLASS DOC]
    # HELPER: Fake finite dataloader yielding batches_per_epoch dicts {'batch': i} per
    # iteration, optionally with a _SamplerWithSetEpoch sampler.
    def __init__(self, batches_per_epoch: int = 2, with_sampler: bool = True):
        # [FUNCTION DOC]
        # Stores batches_per_epoch (default 2) and attaches a recording sampler when
        # with_sampler is True (default).
        self.batches_per_epoch = batches_per_epoch
        if with_sampler:
            self.sampler = _SamplerWithSetEpoch()

    def __iter__(self):
        # [FUNCTION DOC]
        # Generator yielding {'batch': i} for i in range(batches_per_epoch).
        for i in range(self.batches_per_epoch):
            yield {"batch": i}


class _DataloaderWithSamplerNoSetEpoch:
    # [CLASS DOC]
    # HELPER: Fake dataloader whose sampler lacks set_epoch; yields one batch.
    class _Sampler:
        # [CLASS DOC]
        # Empty sampler class with no set_epoch method.
        pass

    def __init__(self):
        # [FUNCTION DOC]
        # Attaches an instance of the nested _Sampler.
        self.sampler = self._Sampler()

    def __iter__(self):
        # [FUNCTION DOC]
        # Yields a single {'batch': 0}.
        yield {"batch": 0}


class TestCycleDataloader:
    # [CLASS DOC]
    # Tests cycle_dataloader.
    def test_cycle_yields_correct_batches(self):
        # [FUNCTION DOC]
        # Two cycles over 3 batches yield 6 batches.
        dl = _FiniteDataloader(batches_per_epoch=3)
        gen = cycle_dataloader(dl, num_cycles=2)
        all_batches = list(gen)
        assert len(all_batches) == 6

    def test_cycle_calls_set_epoch(self):
        # [FUNCTION DOC]
        # set_epoch is called with 0, 1, 2 across three cycles.
        dl = _FiniteDataloader(batches_per_epoch=2)
        gen = cycle_dataloader(dl, num_cycles=3)
        list(gen)
        assert dl.sampler.epochs == [0, 1, 2]

    def test_cycle_no_sampler_does_not_crash(self):
        # [FUNCTION DOC]
        # A dataloader without a sampler cycles without error.
        dl = _FiniteDataloader(batches_per_epoch=2, with_sampler=False)
        gen = cycle_dataloader(dl, num_cycles=1)
        all_batches = list(gen)
        assert len(all_batches) == 2

    def test_cycle_sampler_without_set_epoch(self):
        # [FUNCTION DOC]
        # A sampler lacking set_epoch is tolerated.
        dl = _DataloaderWithSamplerNoSetEpoch()
        gen = cycle_dataloader(dl, num_cycles=1)
        all_batches = list(gen)
        assert len(all_batches) == 1

    def test_infinite_cycle_generates_beyond_one_epoch(self):
        # [FUNCTION DOC]
        # Without num_cycles the generator is infinite; taking 6 batches from a 2-batch
        # loader advances set_epoch through 0, 1, 2.
        dl = _FiniteDataloader(batches_per_epoch=2)
        gen = cycle_dataloader(dl)
        collected = []
        for i, batch in enumerate(gen):
            collected.append(batch)
            if i >= 5:
                break
        assert len(collected) == 6
        assert dl.sampler.epochs == [0, 1, 2]


class TestDataServiceConfig:
    # [CLASS DOC]
    # Tests DataServiceConfig.from_dataset_config on 'sft' dataset configs.
    def test_from_dataset_config_defaults(self):
        # [FUNCTION DOC]
        # Defaults: one service worker, zero dataloader workers, separation strategy.
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft")
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)

        assert svc_cfg.num_workers >= 1
        assert svc_cfg.num_workers == 1
        assert svc_cfg.dataloader_num_workers == 0
        assert svc_cfg.scheduling_strategy.type == "separation"

    def test_from_dataset_config_custom_workers(self):
        # [FUNCTION DOC]
        # num_dataset_workers=3 becomes num_workers and num_workers=4 becomes
        # dataloader_num_workers.
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="sft",
            num_workers=4,
            num_dataset_workers=3,
        )
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.num_workers == 3
        assert svc_cfg.dataloader_num_workers == 4

    def test_from_dataset_config_zero_workers_defaults_to_one(self):
        # [FUNCTION DOC]
        # num_workers=0 still yields one service worker and zero dataloader workers.
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft", num_workers=0)
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.num_workers == 1
        assert svc_cfg.dataloader_num_workers == 0

    def test_from_dataset_config_allows_disabling_data_service(self):
        # [FUNCTION DOC]
        # scheduling_spec=None propagates to the service config (data service disabled).
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft", scheduling_spec=None)
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.scheduling_spec is None

    def test_dataset_config_has_scheduling_spec(self):
        # [FUNCTION DOC]
        # TrainDatasetConfig has a non-None scheduling_spec by default.
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft")
        assert ds_cfg.scheduling_spec is not None


class TestTrainerDataServicePath:
    # [CLASS DOC]
    # Tests that the data-service path is wired for trainers.
    def test_data_controller_importable(self):
        # [FUNCTION DOC]
        # DataController and RDataset are importable from areal.infra.data_service.
        from areal.infra.data_service import DataController, RDataset

        assert DataController is not None
        assert RDataset is not None

    def test_data_controller_config_importable(self):
        # [FUNCTION DOC]
        # DataServiceConfig is importable (trivially non-None).
        assert DataServiceConfig is not None

    def test_rdataset_has_required_protocol(self):
        # [FUNCTION DOC]
        # An unconnected RDataset exposes connect, close, __len__, __getitem__ and
        # connected=False.
        from areal.infra.data_service import RDataset

        dataset = RDataset(path="dummy", type="rl", split="train")

        assert hasattr(dataset, "connect")
        assert hasattr(dataset, "close")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        assert dataset.connected is False

    def test_get_custom_dataset_respects_scheduling_spec_none(self, monkeypatch):
        # [FUNCTION DOC]
        # With AREAL_SPMD_MODE=0 and scheduling_spec=None, get_custom_dataset returns
        # the local _get_custom_dataset result (patched to a sentinel) instead of an
        # RDataset.
        from areal.api.cli_args import TrainDatasetConfig
        from areal.dataset import get_custom_dataset

        sentinel = object()

        def _fake_custom_dataset(**_kwargs):
            # [FUNCTION DOC]
            # HELPER: Ignores kwargs and returns the sentinel.
            return sentinel

        monkeypatch.setenv("AREAL_SPMD_MODE", "0")
        monkeypatch.setattr("areal.dataset._get_custom_dataset", _fake_custom_dataset)

        cfg = TrainDatasetConfig(path="dummy", type="sft", scheduling_spec=None)
        dataset = get_custom_dataset(split="train", dataset_config=cfg)

        assert dataset is sentinel


class TestGenericDatasetFallback:
    # [CLASS DOC]
    # Tests split selection in _get_custom_dataset for on-disk DatasetDicts.
    def test_none_split_uses_first_available_split(self, tmp_path: Path):
        # [FUNCTION DOC]
        # With split=None, the first available split ('train', 2 rows) is loaded.
        from datasets import Dataset, DatasetDict

        from areal.dataset import _get_custom_dataset

        dataset_path = tmp_path / "dict-ds-none-split"
        dataset = DatasetDict({"train": Dataset.from_dict({"x": [1, 2]})})
        dataset.save_to_disk(str(dataset_path))

        loaded = _get_custom_dataset(
            path=str(dataset_path),
            type="sft",
            split=None,
        )
        assert len(loaded) == 2

    def test_explicit_missing_split_raises_error(self, tmp_path: Path):
        # [FUNCTION DOC]
        # Requesting a missing split raises ValueError mentioning "Requested split
        # 'test' not found".
        from datasets import Dataset, DatasetDict

        from areal.dataset import _get_custom_dataset

        dataset_path = tmp_path / "dict-ds"
        dataset = DatasetDict({"train": Dataset.from_dict({"x": [1, 2]})})
        dataset.save_to_disk(str(dataset_path))

        with pytest.raises(ValueError, match="Requested split 'test' not found"):
            _get_custom_dataset(
                path=str(dataset_path),
                type="sft",
                split="test",
            )


class TestEmptyDataLoaderCompat:
    # [CLASS DOC]
    # Tests rl_trainer._EmptyDataLoader.
    def test_empty_dataloader_still_works(self):
        # [FUNCTION DOC]
        # len equals steps_per_epoch, batch_size is kept, and iteration yields at least
        # 3 batches.
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader(batch_size=2, steps_per_epoch=3)

        assert len(dataloader) == 3
        assert dataloader.batch_size == 2

        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 3:
                break
        assert len(batches) == 3

    def test_empty_dataloader_state_dict(self):
        # [FUNCTION DOC]
        # state_dict returns {} and load_state_dict accepts arbitrary state without
        # error.
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader()
        assert dataloader.state_dict() == {}
        dataloader.load_state_dict({"some": "state"})


class TestTrainerSignature:
    # [CLASS DOC]
    # Checks trainer constructors accept dataset arguments.
    def test_sft_trainer_accepts_dataset_params(self):
        # [FUNCTION DOC]
        # SFTTrainer.__init__ has config, train_dataset and valid_dataset parameters.
        from areal.trainer.sft_trainer import SFTTrainer

        sig = inspect.signature(SFTTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params

    def test_rw_trainer_accepts_dataset_params(self):
        # [FUNCTION DOC]
        # RWTrainer.__init__ has config, train_dataset and valid_dataset parameters.
        from areal.trainer.rw_trainer import RWTrainer

        sig = inspect.signature(RWTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params

    def test_ppo_trainer_accepts_dataset_params(self):
        # [FUNCTION DOC]
        # PPOTrainer.__init__ has config, train_dataset and valid_dataset parameters.
        from areal.trainer.rl_trainer import PPOTrainer

        sig = inspect.signature(PPOTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params
