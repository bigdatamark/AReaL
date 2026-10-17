# [MODULE DOC]
# Registers AReaL "awex" weight-update HTTP endpoints on SGLang's FastAPI app. Each
# endpoint forwards a collective RPC through RpcProxy (ZMQ) to all scheduler
# subprocesses, where AwexSchedulerBridge-bound methods do the work.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from areal.utils import logging

if TYPE_CHECKING:
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy

logger = logging.getLogger("AwexInferenceEndpoints")


def register_awex_endpoints(app: FastAPI, rpc_proxy: RpcProxy) -> None:
    # [FUNCTION DOC]
    # Adds /awex/* routes to app. Every handler catches all exceptions, logs them and
    # returns HTTP 500 with {"error": message}; on success returns {"status": "ok"}
    # unless noted. Routes: POST report_weight_meta (returns {"status","meta"} with the
    # pushed serialized metadata); GET report_parallelism (returns the dict pushed by
    # rank 0, 500 if not a dict); POST init_weights_update_group, batch_isend_irecv,
    # debug/get_parameters, init_colocate_weight_update (JSON body passed as kwargs);
    # POST update_weights and execute_colocate_weight_update (body "version", default 0,
    # echoed back); POST debug/randomize_parameters (no body); POST release_memory and
    # resume_memory (optional "tags").
    # KNOWN ISSUE: RpcProxy calls are blocking ZMQ round-trips executed directly inside
    # async handlers, stalling the server event loop for the duration of the operation.
    """Register ``/awex/*`` weight-update endpoints on the SGLang FastAPI app.

    Each endpoint dispatches to all scheduler processes via the
    :class:`RpcProxy`, which sends :class:`RpcReqInput` over ZMQ.
    The :class:`AwexSchedulerBridge` handles the methods and packs return
    values into ``RpcReqOutput.message``.
    """

    @app.post("/awex/report_weight_meta")
    async def report_weight_meta() -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/report_weight_meta: collective RPC that returns aggregated weight
        # metadata from rank 0.
        try:
            result = rpc_proxy.collective_rpc_with_result("awex_report_weight_meta")
            return JSONResponse(content={"status": "ok", "meta": result})
        except Exception as e:
            logger.error("Failed to report weight meta: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/awex/report_parallelism")
    async def report_parallelism() -> JSONResponse:
        # [FUNCTION DOC]
        # GET /awex/report_parallelism: returns the adapter's parallelism strategy dict.
        try:
            result = rpc_proxy.collective_rpc_with_result("awex_report_parallelism")
            if not isinstance(result, dict):
                err_msg = f"Expected dict from awex_report_parallelism, but got {type(result).__name__}"
                logger.error(err_msg)
                return JSONResponse(status_code=500, content={"error": err_msg})
            return JSONResponse(content=result)
        except Exception as e:
            logger.error("Failed to report parallelism: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_weights_update_group")
    async def init_weights_update_group(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/init_weights_update_group: forwards JSON body as RPC kwargs.
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_init_weights_update_group", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init weights update group: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/update_weights")
    async def update_weights(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/update_weights: triggers a weight update to the given version.
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_proxy.collective_rpc("awex_execute_weight_update", version=version)
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to update weights: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/batch_isend_irecv")
    async def batch_isend_irecv(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/batch_isend_irecv: forwards JSON body as RPC kwargs (p2p
        # transfers).
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_batch_isend_irecv", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed batch_isend_irecv: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/get_parameters")
    async def get_parameters(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/debug/get_parameters: asks schedulers to save parameters
        # (save_path, optional names) to disk; nothing is returned in the response.
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_get_parameters", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to get parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/debug/randomize_parameters")
    async def randomize_parameters() -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/debug/randomize_parameters: randomizes model weights (debug).
        try:
            rpc_proxy.collective_rpc("awex_randomize_parameters")
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to randomize parameters: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/init_colocate_weight_update")
    async def init_colocate_weight_update(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/init_colocate_weight_update: forwards JSON body as RPC kwargs.
        try:
            data = await request.json()
            rpc_proxy.collective_rpc("awex_init_colocate_weight_update", **data)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to init colocate weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/execute_colocate_weight_update")
    async def execute_colocate_weight_update(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/execute_colocate_weight_update: runs colocated update to given
        # version.
        try:
            data = await request.json()
            version = data.get("version", 0)
            rpc_proxy.collective_rpc(
                "awex_execute_colocate_weight_update", version=version
            )
            return JSONResponse(content={"status": "ok", "version": version})
        except Exception as e:
            logger.error("Failed to execute colocate weight update: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/release_memory")
    async def release_memory(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/release_memory: releases GPU memory for optional tags.
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_proxy.collective_rpc("awex_release_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to release memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/awex/resume_memory")
    async def resume_memory(request: Request) -> JSONResponse:
        # [FUNCTION DOC]
        # POST /awex/resume_memory: re-acquires GPU memory for optional tags.
        try:
            data = await request.json()
            tags = data.get("tags")
            rpc_proxy.collective_rpc("awex_resume_memory", tags=tags)
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            logger.error("Failed to resume memory: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
