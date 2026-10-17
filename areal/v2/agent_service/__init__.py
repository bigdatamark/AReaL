# [PACKAGE DOC]
# Agent Service package: an agent-level inference tier exposing whole agent sessions
# over HTTP microservices (Gateway -> Router -> DataProxy -> Worker), orchestrated by
# the controller subpackage. Re-exports the public protocol types AgentRequest,
# AgentResponse, AgentRunnable and EventEmitter from the types module.
# SPDX-License-Identifier: Apache-2.0

"""AReaL Agent Service — agent-level inference tier.

Exposes complete agent sessions (autonomous multi-step reasoning, tool use,
memory) via independent HTTP microservices: Gateway, Router, DataProxy,
and Worker.

Submodules
----------
- ``controller`` — :class:`AgentController` orchestrator
- ``gateway`` — public HTTP/WebSocket entry point
- ``router`` — session-affine routing
- ``data_proxy`` — stateful session proxy
- ``worker`` — stateless agent execution
- ``protocol`` — WebSocket frame types and helpers
"""

from .types import (
    AgentRequest,
    AgentResponse,
    AgentRunnable,
    EventEmitter,
)

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentRunnable",
    "EventEmitter",
]
