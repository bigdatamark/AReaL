# [PACKAGE DOC]
# v2 weight-update package: protocols and adapters that move trained weights from
# training workers to inference servers. Re-exports WeightUpdateController and its
# config (importing this package imports the controller module).
# SPDX-License-Identifier: Apache-2.0
"""Weight update protocol adapters for training and inference."""

from areal.v2.weight_update.controller import (
    WeightUpdateController,
    WeightUpdateControllerConfig,
)

__all__ = [
    "WeightUpdateController",
    "WeightUpdateControllerConfig",
]
