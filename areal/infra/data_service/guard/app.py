# [MODULE DOC]
# Re-export shim: exposes the generic RPC guard app pieces (GuardState, create_app,
# run_server, parser/state helpers, forked-child cleanup) under the data service
# namespace. No logic of its own.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.infra.rpc.guard.app import (  # noqa: F401
    GuardState,
    cleanup_forked_children,
    configure_state_from_args,
    create_app,
    make_base_parser,
    run_server,
)
