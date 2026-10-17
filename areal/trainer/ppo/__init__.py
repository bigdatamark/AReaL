# [PACKAGE DOC]
# PPO sub-package: re-exports the actor (PPOActor, PPOActorController) and critic
# (PPOCritic, PPOCriticController) wrappers that compute PPO losses on top of train
# engines.
# SPDX-License-Identifier: Apache-2.0

from .actor import PPOActor, PPOActorController
from .critic import PPOCritic, PPOCriticController

__all__ = ["PPOActor", "PPOActorController", "PPOCritic", "PPOCriticController"]
