# [MODULE DOC]
# Configuration dataclass for the data service controller (internal, built by the
# trainer from the dataset config section).
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import SchedulingSpec, SchedulingStrategy


@dataclass
class DataServiceConfig:
    # [CLASS DOC]
    # Dataclass of data-service controller settings. Fields: num_workers (number of data
    # worker processes, default 1); scheduling_spec (resources/command for each worker;
    # default 1 CPU, 0 GPU, 10 memory units, command running the RPC guard module);
    # scheduling_strategy (always "separation" so data workers get their own allocation,
    # since the data controller starts before engines); setup_timeout (seconds to wait
    # for workers to come up, 120); dataloader_num_workers (torch DataLoader worker
    # processes per data worker, 4); seed (shuffling seed, 42).
    """Internal config for the data service controller.

    Constructed from ``_DatasetConfig`` fields by the trainer.
    Not exposed to end users directly.
    """

    num_workers: int = 1
    scheduling_spec: SchedulingSpec = field(
        default_factory=lambda: SchedulingSpec(
            cpu=1, gpu=0, mem=10, cmd="python3 -m areal.infra.rpc.guard"
        ),
    )
    # Always separation — data controller starts before engines.
    scheduling_strategy: SchedulingStrategy = field(
        default_factory=lambda: SchedulingStrategy(type="separation"),
    )
    setup_timeout: float = 120.0
    dataloader_num_workers: int = 4
    seed: int = 42

    @staticmethod
    def from_dataset_config(dataset_config, seed: int = 42) -> DataServiceConfig:
        # [FUNCTION DOC]
        # Static factory. Inputs: dataset_config (a _DatasetConfig-like object exposing
        # num_dataset_workers, scheduling_spec, num_workers, setup_timeout) and seed
        # (int, default 42). Returns a DataServiceConfig with num_workers clamped to at
        # least 1 and dataloader_num_workers clamped to at least 0; scheduling_strategy
        # keeps its default. No side effects.
        """Build from a ``_DatasetConfig`` instance."""
        return DataServiceConfig(
            num_workers=max(dataset_config.num_dataset_workers, 1),
            scheduling_spec=dataset_config.scheduling_spec,
            dataloader_num_workers=max(dataset_config.num_workers, 0),
            setup_timeout=dataset_config.setup_timeout,
            seed=seed,
        )


__all__ = ["DataServiceConfig"]
