# [MODULE DOC]
# Flask Blueprint ("data") exposing the process-local RTensor shard store over HTTP.
# Payloads are serialize_value-encoded structures sent as orjson bytes; the store itself
# lives in the rtensor module (CPU tensors keyed by shard id).
# SPDX-License-Identifier: Apache-2.0

"""Data Blueprint: RTensor ``/data/*`` storage endpoints.

Provides a Flask Blueprint that handles tensor shard storage and
retrieval via HTTP.  Used by any service that needs local tensor
storage accessible over the network (e.g., the RPC server).

Routes:

- ``PUT    /data/<shard_id>``  — store a single shard
- ``GET    /data/<shard_id>``  — retrieve a single shard
- ``POST   /data/batch``       — retrieve multiple shards
- ``DELETE /data/clear``       — clear specified shards
"""

from __future__ import annotations

import traceback

import orjson
from flask import Blueprint, Response, jsonify, request
from pydantic import BaseModel, ValidationError

from areal.infra.rpc import rtensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging

logger = logging.getLogger("DataBP")

data_bp = Blueprint("data", __name__)


# ================================================================================
# Pydantic models for Data API
# ================================================================================


class ShardListRequest(BaseModel):
    # [CLASS DOC]
    # Pydantic request model with a single field shard_ids (list of strings).
    """Base model for requests containing a list of shard IDs."""

    shard_ids: list[str]


class BatchShardRequest(ShardListRequest):
    # [CLASS DOC]
    # Request body for POST /data/batch (inherits shard_ids).
    """Request to retrieve multiple tensor shards."""


class ClearShardRequest(ShardListRequest):
    # [CLASS DOC]
    # Request body for DELETE /data/clear (inherits shard_ids).
    """Request to clear specific tensor shards."""


# ================================================================================
# Flask Blueprint Definition
# ================================================================================


@data_bp.route("/data/<shard_id>", methods=["PUT"])
def store_batch_data(shard_id: str):
    # [FUNCTION DOC]
    # Route PUT /data/<shard_id>: parses the raw body as JSON, deserializes it into the
    # value (tensor or nested structure), stores it in the rtensor store under shard_id
    # (overwriting), and returns {"status": "ok", "shard_id"}. Any error -> 500 with
    # message.
    """Store batch data shard."""
    try:
        data_bytes = request.get_data()

        # Deserialize to get tensor (already on CPU)
        serialized_data = orjson.loads(data_bytes)
        data = deserialize_value(serialized_data)

        rtensor.store(shard_id, data)

        logger.debug(f"Stored batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return jsonify({"status": "ok", "shard_id": shard_id})

    except Exception as e:
        logger.error(f"Error storing batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/<shard_id>", methods=["GET"])
def retrieve_batch_data(shard_id: str):
    # [FUNCTION DOC]
    # Route GET /data/<shard_id>: fetches the shard (404 JSON if missing), serializes it
    # and returns orjson bytes with octet-stream mimetype. Other errors -> 500.
    """Retrieve batch data shard."""
    logger.debug(f"Received data get request for shard {shard_id}")
    try:
        try:
            data = rtensor.fetch(shard_id)
        except KeyError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Shard {shard_id} not found",
                    }
                ),
                404,
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)

        logger.debug(f"Retrieved batch shard {shard_id} (size={len(data_bytes)} bytes)")
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shard {shard_id}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/batch", methods=["POST"])
def retrieve_batch_data_many():
    # [FUNCTION DOC]
    # Route POST /data/batch with JSON {"shard_ids": [...]}: 400 if the body is not an
    # object or fails validation. Fetches every shard in order; if any are missing
    # returns 400 with missing_shard_ids. Otherwise returns the serialized list of
    # values as orjson bytes (octet-stream). Other errors -> 500.
    """Retrieve multiple batch data shards in one request."""
    try:
        raw_payload = request.get_json(silent=True) or {}

        # USE PYDANTIC MODEL FOR VALIDATION
        try:
            if not isinstance(raw_payload, dict):
                raise TypeError(
                    "Expected a JSON object, got " + type(raw_payload).__name__
                )
            payload_model = BatchShardRequest(**raw_payload)
        except (ValidationError, TypeError):
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Expected JSON body with string list field 'shard_ids'",
                    }
                ),
                400,
            )
        shard_ids = payload_model.shard_ids  # use the validated data

        data = []
        missing_shard_ids = []
        for shard_id in shard_ids:
            try:
                data.append(rtensor.fetch(shard_id))
            except KeyError:
                missing_shard_ids.append(shard_id)

        if missing_shard_ids:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": ("One or more requested shards were not found"),
                        "missing_shard_ids": missing_shard_ids,
                    }
                ),
                400,
            )

        serialized_data = serialize_value(data)
        data_bytes = orjson.dumps(serialized_data)
        logger.debug(
            "Retrieved %s batch shards (size=%s bytes)",
            len(shard_ids),
            len(data_bytes),
        )
        return Response(data_bytes, mimetype="application/octet-stream")

    except Exception as e:
        logger.error(f"Error retrieving batch shards: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500


@data_bp.route("/data/clear", methods=["DELETE"])
def clear_batch_data():
    # [FUNCTION DOC]
    # Route DELETE /data/clear with JSON {"shard_ids": [...]}: removes each shard from
    # the store, counting those that existed, and returns status "ok", cleared_count and
    # the store's storage stats merged in. Logs at INFO. Validation errors -> 400.
    # KNOWN ISSUE: A non-object JSON body (e.g. a list) raises TypeError on model
    # construction, which is not caught by the validation handler and yields 500 instead
    # of 400.
    """Clear specified batch data shards."""
    try:
        raw_data = request.get_json(silent=True) or {}

        # USE PYDANTIC MODEL FOR VALIDATION
        try:
            payload_model = ClearShardRequest(**raw_data)
        except ValidationError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "'shard_ids' must be a list",
                    }
                ),
                400,
            )
        shard_ids = payload_model.shard_ids  # use the validated data

        cleared_count = sum(rtensor.remove(sid) for sid in shard_ids)
        storage = rtensor.storage_stats()
        result = {
            "status": "ok",
            "cleared_count": cleared_count,
            **storage,
        }
        logger.info(f"Cleared {cleared_count} batch shards. Stats: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error clearing batch data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
