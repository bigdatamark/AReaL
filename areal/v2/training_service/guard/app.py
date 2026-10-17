# [MODULE DOC]
# Creates the module-level GuardState and shared guard app for the training service and
# registers the "TrainRPCGuard" logger.
# SPDX-License-Identifier: Apache-2.0

"""Training service guard backed by the shared RPC guard."""

from __future__ import annotations

from areal.infra.rpc.guard.app import GuardState, create_app
from areal.infra.rpc.guard.app import cleanup_forked_children as _cleanup_impl
from areal.utils import logging

logger = logging.getLogger("TrainRPCGuard")

_state = GuardState()

app = create_app(_state)


def cleanup_forked_children() -> None:
    # [FUNCTION DOC]
    # Terminates/cleans up processes forked by this guard via the shared implementation
    # using the module-level state.
    _cleanup_impl(_state)
