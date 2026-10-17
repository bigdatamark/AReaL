# [MODULE DOC]
# Pluggable worker-selection strategies for the router plus a name-based factory.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable routing strategies for worker selection."""

from __future__ import annotations

from typing import Protocol

from areal.v2.inference_service.router.state import WorkerInfo


class RoutingStrategy(Protocol):
    # [CLASS DOC]
    # Structural protocol: any object with pick(workers) returning one WorkerInfo or
    # None.
    """Protocol for routing strategy implementations."""

    def pick(self, workers: list[WorkerInfo]) -> WorkerInfo | None:
        # [FUNCTION DOC]
        # Protocol method: choose a worker from the list, or None if the list is empty.
        """Select a worker from the list, or return None if empty."""
        ...


class RoundRobinStrategy:
    # [CLASS DOC]
    # Stateful round-robin selector keeping a monotonically increasing counter.
    """Cycle through workers in order."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes the counter to 0.
        self._idx = 0

    def pick(self, workers: list[WorkerInfo]) -> WorkerInfo | None:
        # [FUNCTION DOC]
        # Returns None for an empty list; otherwise returns the worker at counter modulo
        # list length and increments the counter. Not thread-safe; fairness assumes a
        # stable list.
        if not workers:
            return None
        w = workers[self._idx % len(workers)]
        self._idx += 1
        return w


def get_strategy(name: str) -> RoutingStrategy:
    # [FUNCTION DOC]
    # Factory: "round_robin" returns a new RoundRobinStrategy; "least_busy" raises
    # NotImplementedError; any other name raises ValueError.
    """Instantiate a routing strategy by name."""
    if name == "round_robin":
        return RoundRobinStrategy()
    if name == "least_busy":
        raise NotImplementedError(
            "least_busy strategy is not yet implemented: "
            "active_requests tracking is required but not wired up. "
            "Use 'round_robin' instead."
        )
    raise ValueError(f"Unknown routing strategy: {name}")
