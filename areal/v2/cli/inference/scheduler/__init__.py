# [PACKAGE DOC]
# Task scheduler abstraction for "areal inf": re-exports the scheduler types (Scheduler,
# SchedulerError, TaskAllocation, TaskHandle, TaskResources, TaskSpec) and the
# build_scheduler factory from the base module.
# SPDX-License-Identifier: Apache-2.0

from areal.v2.cli.inference.scheduler.base import (
    Scheduler,
    SchedulerError,
    TaskAllocation,
    TaskHandle,
    TaskResources,
    TaskSpec,
    build_scheduler,
)

__all__ = [
    "Scheduler",
    "SchedulerError",
    "TaskAllocation",
    "TaskHandle",
    "TaskResources",
    "TaskSpec",
    "build_scheduler",
]
