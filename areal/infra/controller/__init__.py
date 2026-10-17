# [PACKAGE DOC]
# Package areal.infra.controller: single-controller components that drive distributed
# workers. Re-exports RolloutController (manages inference/rollout workers) and
# TrainController (manages training engine workers).
# SPDX-License-Identifier: Apache-2.0

"""Controller components for managing distributed training and inference."""

from .rollout_controller import RolloutController
from .train_controller import TrainController

__all__ = [
    "RolloutController",
    "TrainController",
]
