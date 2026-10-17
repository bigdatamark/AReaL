# [MODULE DOC]
# Pydantic request models exchanged between the data controller/gateway and data
# workers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportMissingImports=false
from typing import Any

from pydantic import BaseModel, Field


class WorkerLoadDatasetRequest(BaseModel):
    # [CLASS DOC]
    # Request to load a dataset on a worker: dataset_id, dataset_path, dataset_type
    # (registry name for the loader), split (default "train"),
    # tokenizer_or_processor_path (empty means none), max_length (optional filter
    # length), shuffle and drop_last flags for the dataloader (both default True), and
    # extra dataset_kwargs.
    dataset_id: str
    dataset_path: str
    dataset_type: str
    split: str = "train"
    tokenizer_or_processor_path: str = ""
    max_length: int | None = None
    shuffle: bool = True
    drop_last: bool = True
    dataset_kwargs: dict[str, Any] = Field(default_factory=dict)


class WorkerUnloadDatasetRequest(BaseModel):
    # [CLASS DOC]
    # Request to unload a dataset by dataset_id.
    dataset_id: str


class WorkerEpochResetRequest(BaseModel):
    # [CLASS DOC]
    # Request to reset a dataset's sampler/iterator to a given epoch number.
    dataset_id: str
    epoch: int


class WorkerStateSaveRequest(BaseModel):
    # [CLASS DOC]
    # Request to save a dataset's iteration state to a filesystem path.
    dataset_id: str
    path: str


class WorkerStateLoadRequest(BaseModel):
    # [CLASS DOC]
    # Request to restore a dataset's iteration state from a filesystem path.
    dataset_id: str
    path: str


class FetchSamplesRequest(BaseModel):
    # [CLASS DOC]
    # Request to fetch specific samples of a dataset by integer indices.
    dataset_id: str
    indices: list[int]
