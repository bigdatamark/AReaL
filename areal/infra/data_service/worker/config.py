# [MODULE DOC]
# Configuration dataclass for a data service worker.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataWorkerConfig:
    # [CLASS DOC]
    # Worker settings: host/port to bind (port 0 means unspecified/random), rank and
    # world_size of this worker among all data workers (used to shard data),
    # dataloader_num_workers (torch DataLoader processes, 4) and seed (42).
    host: str = "0.0.0.0"
    port: int = 0
    rank: int = 0
    world_size: int = 1
    dataloader_num_workers: int = 4
    seed: int = 42
