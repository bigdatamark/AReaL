# [PACKAGE DOC]
# Package areal.infra.data_service: a distributed dataset-serving subsystem. A
# DataController launches data workers (HTTP services that load datasets and serve
# samples) behind a router and gateway; RDataset is the trainer-side remote dataset
# proxy. Re-exports DataController, DataServiceConfig and RDataset.
# SPDX-License-Identifier: Apache-2.0

from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.controller.controller import DataController
from areal.infra.data_service.rdataset import RDataset

__all__ = [
    "DataController",
    "DataServiceConfig",
    "RDataset",
]
