# [MODULE DOC]
# Command-line launcher for a standalone Gateway server with both protocol bridges.
# SPDX-License-Identifier: Apache-2.0

"""``python -m areal.v2.agent_service.gateway``"""

import argparse

import uvicorn

from areal.infra.utils.http import validate_admin_api_key

from ..auth import DEFAULT_ADMIN_API_KEY
from .app import create_gateway_app
from .bridge import (
    ChatCompletionsBridge,
    OpenResponsesBridge,
    mount_bridge,
    mount_chat_bridge,
)
from .config import GatewayConfig


def main() -> None:
    # [FUNCTION DOC]
    # Parses CLI args (--router-addr required; --host 0.0.0.0, --port 8080,
    # --admin-api-key default key, --router-timeout 2s, --forward-timeout 120s,
    # --log-level warning). Validates the admin key against the bind host (may raise if
    # the default key is used on a non-local host), builds a GatewayConfig and the
    # gateway app, mounts the OpenResponses bridge (admin-key protected) and the chat
    # completions bridge, then runs uvicorn (access log disabled); blocks.
    parser = argparse.ArgumentParser(description="Agent Gateway")
    parser.add_argument("--router-addr", required=True, help="Router HTTP address")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--admin-api-key", default=DEFAULT_ADMIN_API_KEY)
    parser.add_argument("--router-timeout", type=float, default=2.0)
    parser.add_argument("--forward-timeout", type=float, default=120.0)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="warning"
    )
    args = parser.parse_args()

    validate_admin_api_key(
        args.host, args.admin_api_key, default_key=DEFAULT_ADMIN_API_KEY
    )

    config = GatewayConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        router_addr=args.router_addr,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
        log_level=args.log_level,
    )
    app = create_gateway_app(config)
    mount_bridge(
        app,
        OpenResponsesBridge(
            router_addr=config.router_addr, admin_api_key=config.admin_api_key
        ),
        admin_api_key=config.admin_api_key,
    )
    mount_chat_bridge(
        app,
        ChatCompletionsBridge(
            router_addr=config.router_addr, admin_api_key=config.admin_api_key
        ),
    )
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
