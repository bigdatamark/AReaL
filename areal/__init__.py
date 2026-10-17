# [PACKAGE DOC]
# Top-level AReaL package. Exposes __version__ and eagerly re-exports the
# controller/infra primitives (RolloutController, TrainController, StalenessManager,
# WorkflowExecutor, current_platform, workflow_context) from areal.infra. Trainer
# classes (DPO/PPO/RW/SFT) are loaded lazily via a module-level __getattr__ to avoid
# heavy imports at package import time.
# SPDX-License-Identifier: Apache-2.0

"""AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning"""

from .version import __version__  # noqa

from .infra import (
    RolloutController,
    StalenessManager,
    TrainController,
    WorkflowExecutor,
    current_platform,
    workflow_context,
)


def __getattr__(name: str):
    # [FUNCTION DOC]
    # HELPER: Module-level lazy attribute hook (PEP 562). For DPOTrainer, PPOTrainer,
    # RWTrainer or SFTTrainer it imports all four from areal.trainer, caches them into
    # module globals (so later lookups bypass this hook) and returns the requested one.
    # Any other name raises AttributeError.
    if name in ("DPOTrainer", "PPOTrainer", "RWTrainer", "SFTTrainer"):
        from .trainer import DPOTrainer, PPOTrainer, RWTrainer, SFTTrainer

        _map = {
            "DPOTrainer": DPOTrainer,
            "PPOTrainer": PPOTrainer,
            "RWTrainer": RWTrainer,
            "SFTTrainer": SFTTrainer,
        }
        globals().update(_map)
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DPOTrainer",
    "PPOTrainer",
    "RolloutController",
    "RWTrainer",
    "SFTTrainer",
    "StalenessManager",
    "TrainController",
    "WorkflowExecutor",
    "current_platform",
    "workflow_context",
]
