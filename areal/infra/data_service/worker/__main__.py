# [MODULE DOC]
# CLI entrypoint for a data service worker process.
# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: python -m areal.infra.data_service.worker"""

from __future__ import annotations

import argparse
import importlib


def main():
    # [FUNCTION DOC]
    # Parses CLI args (host, port 0 meaning any, rank, world size, dataloader worker
    # count 4, seed 42; unknown args ignored), dynamically imports the worker app/config
    # and logging modules, builds a DataWorkerConfig, silences HTTP loggers and serves
    # the worker app with uvicorn (blocking).
    # KNOWN ISSUE: with the default port 0 uvicorn binds a random port that is not
    # reported anywhere by this function.
    parser = argparse.ArgumentParser(description="AReaL Data Service Worker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    app_module = importlib.import_module("areal.infra.data_service.worker.app")
    config_module = importlib.import_module("areal.infra.data_service.worker.config")
    logging_module = importlib.import_module("areal.utils.logging")
    create_worker_app = getattr(app_module, "create_worker_app")
    DataWorkerConfig = getattr(config_module, "DataWorkerConfig")
    suppress_http_loggers = getattr(logging_module, "suppress_http_loggers")

    config = DataWorkerConfig(
        host=args.host,
        port=args.port,
        rank=args.rank,
        world_size=args.world_size,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
    )
    uvicorn = importlib.import_module("uvicorn")

    suppress_http_loggers()
    app = create_worker_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
