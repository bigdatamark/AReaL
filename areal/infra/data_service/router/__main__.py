# [MODULE DOC]
# CLI entrypoint for the data service router process.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib


def main():
    # [FUNCTION DOC]
    # Parses CLI args (host 0.0.0.0, port 8091, admin API key, worker poll interval 5s,
    # worker health timeout 3s; unknown args ignored), dynamically imports the router
    # app/config and logging modules, builds a RouterConfig, silences HTTP loggers and
    # serves the router app with uvicorn (blocking). No return value.
    parser = argparse.ArgumentParser(description="AReaL Data Service Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--admin-api-key", default="areal-data-admin")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--worker-health-timeout", type=float, default=3.0)
    args, _ = parser.parse_known_args()

    router_app_module = importlib.import_module("areal.infra.data_service.router.app")
    router_config_module = importlib.import_module(
        "areal.infra.data_service.router.config"
    )
    logging_module = importlib.import_module("areal.utils.logging")
    create_router_app = router_app_module.create_router_app
    RouterConfig = router_config_module.RouterConfig
    suppress_http_loggers = logging_module.suppress_http_loggers

    config = RouterConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        worker_health_timeout=args.worker_health_timeout,
    )

    uvicorn = importlib.import_module("uvicorn")

    suppress_http_loggers()
    app = create_router_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
