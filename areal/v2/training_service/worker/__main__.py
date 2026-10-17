# [MODULE DOC]
# CLI entrypoint for a training worker (python -m areal.v2.training_service.worker).
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the train worker."""

from __future__ import annotations

import argparse


def main():
    # [FUNCTION DOC]
    # Parses args (host, port default 30000, admin key, log level; unknown args
    # ignored), validates the admin key for the bind host, builds TrainWorkerConfig,
    # silences HTTP loggers, creates the Flask app and runs Flask's built-in threaded
    # server (blocking).
    # KNOWN ISSUE: log_level is accepted but never applied to the server.
    parser = argparse.ArgumentParser(description="AReaL Train Worker")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=30000, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args, _ = parser.parse_known_args()

    from areal.infra.utils.http import validate_admin_api_key
    from areal.v2.training_service.worker.app import create_app
    from areal.v2.training_service.worker.config import TrainWorkerConfig

    validate_admin_api_key(args.host, args.admin_api_key)

    config = TrainWorkerConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        log_level=args.log_level,
    )

    from areal.utils.logging import suppress_http_loggers

    suppress_http_loggers()

    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
