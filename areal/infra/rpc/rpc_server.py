# [MODULE DOC]
# Full synchronous RPC server for training/inference workers: composes the Guard app
# with the data blueprint (tensor shard storage) and the engine blueprint (engine
# creation and method calls). Run as a module with experiment/trial/role/worker-index
# arguments.
# SPDX-License-Identifier: Apache-2.0

"""AReaL Sync RPC Server — Guard + Data + Engine composition.

This module composes the shared Guard with data and engine blueprints
to create the full RPC server used by training workers.

Usage::

    python -m areal.infra.rpc.rpc_server \\
        --experiment-name exp1 --trial-name trial1 \\
        --role actor --worker-index 0
"""

from __future__ import annotations

import logging as stdlib_logging

from areal.infra.rpc.guard.app import (
    GuardState,
    configure_state_from_args,
    create_app,
    make_base_parser,
    run_server,
)
from areal.infra.rpc.guard.data_blueprint import data_bp
from areal.infra.rpc.guard.engine_blueprint import engine_bp, register_engine_hooks
from areal.utils import logging, perf_tracer

logger = logging.getLogger("SyncRPCServer")


def main():
    # [FUNCTION DOC]
    # Parses base guard args plus --werkzeug-log-level (default WARNING) and sets the
    # werkzeug logger level accordingly. Creates GuardState, configures it, builds the
    # app, registers data and engine blueprints and engine hooks, registers a cleanup
    # hook that force-saves perf_tracer output, logs the werkzeug level, then runs the
    # server (blocking).
    parser = make_base_parser(
        description="AReaL Sync RPC Server for TrainEngine/InferenceEngine"
    )
    parser.add_argument(
        "--werkzeug-log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for Werkzeug (Flask's WSGI server). Default: WARNING",
    )

    args, _ = parser.parse_known_args()

    werkzeug_logger = stdlib_logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(getattr(stdlib_logging, args.werkzeug_log_level))

    state = GuardState()
    bind_host = configure_state_from_args(state, args)

    app = create_app(state)
    app.register_blueprint(data_bp)
    app.register_blueprint(engine_bp)
    register_engine_hooks(state)

    state.register_cleanup_hook(lambda: perf_tracer.save(force=True))

    logger.info(f"Werkzeug log level: {args.werkzeug_log_level}")

    run_server(state, app, bind_host, args.port)


if __name__ == "__main__":
    main()
