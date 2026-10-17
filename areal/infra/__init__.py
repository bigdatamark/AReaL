# [PACKAGE DOC]
# Package areal.infra: infrastructure layer of AReaL. Re-exports the controllers
# (rollout/train), cluster launchers (local, Ray, Slurm) and inference server wrappers
# (SGLang, vLLM), platform helpers, schedulers (local/Ray/Slurm), remote inference
# engine, staleness manager and workflow executor, plus the workflow_context module.
# Importing it eagerly imports all those submodules (heavy: torch, ray, etc.).
# SPDX-License-Identifier: Apache-2.0

"""Core components for AREAL."""

from . import workflow_context
from .controller import RolloutController, TrainController
from .launcher import (
    LocalLauncher,
    RayLauncher,
    SGLangServerWrapper,
    SlurmLauncher,
    vLLMServerWrapper,
)
from .platforms import Platform, current_platform, is_npu_available
from .remote_inf_engine import (
    RemoteInfBackendProtocol,
    RemoteInfEngine,
)
from .scheduler import LocalScheduler, RayScheduler, SlurmScheduler
from .staleness_manager import StalenessManager
from .workflow_executor import (
    WorkflowExecutor,
    check_trajectory_format,
)

__all__ = [
    "RemoteInfBackendProtocol",
    "RemoteInfEngine",
    "StalenessManager",
    "WorkflowExecutor",
    "check_trajectory_format",
    "RolloutController",
    "TrainController",
    "workflow_context",
    "Platform",
    "current_platform",
    "is_npu_available",
    "LocalScheduler",
    "RayScheduler",
    "SlurmScheduler",
    "LocalLauncher",
    "RayLauncher",
    "SlurmLauncher",
    "SGLangServerWrapper",
    "vLLMServerWrapper",
]
