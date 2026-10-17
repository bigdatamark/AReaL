# [MODULE DOC]
# Command-line launcher for a standalone Agent Worker (usually forked by the guard).
# SPDX-License-Identifier: Apache-2.0

"""``python -m areal.v2.agent_service.worker``

Start a standalone Agent Worker process. The Controller forks this
via Guard to create Worker+DataProxy pairs.

    python -m areal.v2.agent_service.worker \
        --agent examples.agent_service.agent.ClaudeAgent \
        --host 127.0.0.1 --port 9000
"""

import argparse

import uvicorn

from .app import create_worker_app


def main() -> None:
    # [FUNCTION DOC]
    # Parses --agent (required import path), --host 0.0.0.0, --port 9000, --log-level
    # warning; builds the worker app (which imports and instantiates the agent) and runs
    # it with uvicorn (access log off); blocks.
    parser = argparse.ArgumentParser(description="Agent Worker")
    parser.add_argument("--agent", required=True, help="Agent import path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="warning"
    )
    args = parser.parse_args()

    app = create_worker_app(args.agent)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
