# [MODULE DOC]
# FastAPI app factory for the training data proxy. Topology discovery and dispatcher
# creation happen in the lifespan; engine routes come from engine.py.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response as RawResponse

from areal.utils import logging
from areal.v2.training_service.data_proxy.config import TrainDataProxyConfig
from areal.v2.training_service.data_proxy.dispatcher import Dispatcher
from areal.v2.training_service.data_proxy.engine import register_engine_routes
from areal.v2.training_service.data_proxy.topology import discover_topology

logger = logging.getLogger("TrainDataProxy")


def _raw_json_response(content: bytes) -> RawResponse:
    # [FUNCTION DOC]
    # HELPER: Wraps already-serialized JSON bytes in a raw Response with
    # application/json media type (no re-encoding).
    return RawResponse(content=content, media_type="application/json")


def create_app(config: TrainDataProxyConfig) -> FastAPI:
    # [FUNCTION DOC]
    # Creates the app with a lifespan that logs, discovers the worker topology (timeout
    # min(request_timeout, 30s); failure aborts startup), builds a Dispatcher, stores
    # config/topology/dispatcher on app.state, and on shutdown closes the dispatcher.
    # Registers engine routes and returns the app.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # [FUNCTION DOC]
        # HELPER: Startup/shutdown logic described above.
        # KNOWN ISSUE: dispatcher.close is not in a finally block, so it is skipped if
        # the server exits via an exception.
        logger.info(
            "Train data proxy starting with %d workers", len(config.worker_addrs)
        )
        topology = await discover_topology(
            config.worker_addrs,
            timeout=min(config.request_timeout, 30.0),
        )
        dispatcher = Dispatcher(
            topology=topology, request_timeout=config.request_timeout
        )

        app.state.config = config
        app.state.topology = topology
        app.state.dispatcher = dispatcher
        yield
        await dispatcher.close()
        logger.info("Train data proxy shutting down")

    app = FastAPI(title="AReaL Train Data Proxy", lifespan=lifespan)
    register_engine_routes(app, _raw_json_response=_raw_json_response)

    return app
