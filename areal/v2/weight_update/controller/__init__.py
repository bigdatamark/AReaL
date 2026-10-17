# [PACKAGE DOC]
# Client-side controller for the v2 weight-update gateway: spawns the gateway as a
# subprocess and drives connect/update/disconnect over HTTP. Re-exports
# WeightUpdateController and WeightUpdateControllerConfig.
# SPDX-License-Identifier: Apache-2.0

from areal.v2.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.v2.weight_update.controller.controller import (
    WeightUpdateController,
)

__all__ = [
    "WeightUpdateController",
    "WeightUpdateControllerConfig",
]
