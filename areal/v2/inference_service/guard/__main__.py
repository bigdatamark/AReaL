# [MODULE DOC]
# CLI entrypoint for the inference RPC guard (python -m
# areal.v2.inference_service.guard).
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``python -m areal.v2.inference_service.guard``"""

from __future__ import annotations

from areal.infra.rpc.guard.app import (
    configure_state_from_args,
    make_base_parser,
    run_server,
)
from areal.v2.inference_service.guard.app import (
    _state,
    app,
)


def main():
    # [FUNCTION DOC]
    # Builds the shared guard base argument parser, parses known args, configures the
    # module-level guard state from them (returns the bind host) and runs the guard HTTP
    # server blocking on args.port.
    parser = make_base_parser(
        description=("AReaL RPCGuard — HTTP gateway for coordinating forked workers")
    )
    args, _ = parser.parse_known_args()

    bind_host = configure_state_from_args(_state, args)

    run_server(_state, app, bind_host, args.port)


if __name__ == "__main__":
    main()
