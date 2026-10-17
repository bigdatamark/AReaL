# [MODULE DOC]
# Backend-agnostic scheduler data model: resource requests, allocations, task specs
# whose command is built after placement, handles identifying running tasks, the
# abstract Scheduler interface and a factory selecting a concrete backend.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class SchedulerError(Exception):
    # [CLASS DOC]
    # Raised for scheduling failures (unknown backend, insufficient GPUs, spawn errors).
    pass


@dataclass
class TaskResources:
    # [CLASS DOC]
    # Resource request: gpu count (default 0) and number of ports (default 1).
    gpu: int = 0
    ports: int = 1


@dataclass
class TaskAllocation:
    # [CLASS DOC]
    # Concrete placement given to a command builder: host, list of ports, list of GPU
    # ids.
    host: str
    ports: list[int]
    gpu_devices: list[int]


@dataclass
class TaskSpec:
    # [CLASS DOC]
    # Task description: name, cmd_builder (maps a TaskAllocation to an argv list),
    # log_file, resources (default 0 GPU / 1 port) and extra env vars.
    """cmd_builder receives the TaskAllocation so host/port/GPU can be filled
    in at placement time — required for K8s/Slurm where the host is only
    known after scheduling."""

    name: str
    cmd_builder: Callable[[TaskAllocation], list[str]]
    log_file: Path
    resources: TaskResources = field(default_factory=TaskResources)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class TaskHandle:
    # [CLASS DOC]
    # Identity of a running task: host, ports, gpu_devices and a backend-specific ref
    # dict (e.g. pid for local).
    """Backend-agnostic identity of a running task.

    ``ref`` carries backend-specific identity payload — ``{"pid": int}`` for
    Local, ``{"pod_name", "namespace"}`` for K8s, ``{"job_id"}`` for Slurm.
    """

    host: str
    ports: list[int]
    gpu_devices: list[int] = field(default_factory=list)
    ref: dict = field(default_factory=dict)

    @property
    def addr(self) -> str:
        # [FUNCTION DOC]
        # Property: "http://host:first_port", or "" when host or ports are missing.
        if not self.host or not self.ports:
            return ""
        return f"http://{self.host}:{self.ports[0]}"

    @property
    def pid(self) -> int:
        # [FUNCTION DOC]
        # Property: integer pid from ref["pid"], or 0 when absent/falsy.
        return int(self.ref.get("pid", 0) or 0)


class Scheduler(ABC):
    # [CLASS DOC]
    # Abstract scheduler interface.
    @abstractmethod
    def submit(self, spec: TaskSpec) -> TaskHandle: ...
        # [FUNCTION DOC]
        # Abstract: places and launches spec, returning its TaskHandle.


def build_scheduler(backend: str, **kwargs) -> Scheduler:
    # [FUNCTION DOC]
    # Returns a LocalScheduler(**kwargs) for backend "local" (lazy import); raises
    # SchedulerError for any other backend name.
    if backend == "local":
        from areal.v2.cli.inference.scheduler.local import LocalScheduler

        return LocalScheduler(**kwargs)
    raise SchedulerError(f"unknown scheduler backend: {backend!r}")
