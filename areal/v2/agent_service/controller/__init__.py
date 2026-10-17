# [PACKAGE DOC]
# Controller subpackage: the AgentController orchestrator that launches and wires
# the agent micro-services. Re-exports AgentController and the AgentConfig dataclass
# (from areal.api.cli_args) for convenience.
# SPDX-License-Identifier: Apache-2.0

"""Agent Service Controller — orchestrator for agent micro-services."""

from areal.api.cli_args import AgentConfig

from .controller import AgentController

__all__ = [
    "AgentController",
    "AgentConfig",
]
