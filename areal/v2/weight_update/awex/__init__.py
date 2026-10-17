# [PACKAGE DOC]
# Shared helpers for the awex weight-update adapters: env flag for grouped p2p ops and
# fetching parameter metadata for a train/infer pair from the gateway KV store over
# HTTP.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp  # pyright: ignore[reportMissingImports]

from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging

logger = logging.getLogger("AwexHTTP")


def awex_wu_use_group() -> bool:
    # [FUNCTION DOC]
    # Returns whether batched send/recv should use grouped batch_isend_irecv, from env
    # AWEX_WU_USE_GROUP (integer, default 1). Setting 0 falls back to per-op send/recv
    # to avoid hangs on some hardware. Non-integer values raise ValueError.
    """Resolve whether ``batch_send_recv`` should use ``use_group=True``.

    Why: on some hardware/driver combinations, ``torch.distributed.batch_isend_irecv``
    can hang during weight update. ``AWEX_WU_USE_GROUP=0`` lets the caller fall back
    to per-op send/recv to bypass the hang. Defaults to ``1`` (True).
    """
    return bool(int(os.getenv("AWEX_WU_USE_GROUP", "1")))


async def _fetch_kv_metadata(
    kv_store_url: str,
    pair_name: str,
) -> tuple[Any, Any]:
    # [FUNCTION DOC]
    # HELPER: Concurrently GETs kv_store_url/weight_meta/{pair_name}/infer_params_meta
    # and /training_params_meta in one aiohttp session; each JSON response yields its
    # "value" field (or the whole body if absent). HTTP errors raise aiohttp
    # ClientResponseError. Returns (infer_meta, training_meta) after areal RPC
    # deserialization.
    """Fetch infer and training parameter metadata from the gateway KV store.

    Uses a shared ``aiohttp.ClientSession`` with ``asyncio.gather`` so both
    requests share a TCP connection pool and execute concurrently.

    Returns
    -------
    tuple[Any, Any]
        (infer_params_meta, training_params_meta) — deserialized Python objects.
    """
    infer_url = f"{kv_store_url}/weight_meta/{pair_name}/infer_params_meta"
    train_url = f"{kv_store_url}/weight_meta/{pair_name}/training_params_meta"

    async with aiohttp.ClientSession() as session:

        async def _get(url: str) -> Any:
            # [FUNCTION DOC]
            # HELPER: GETs one URL, raises on HTTP error, returns the "value" field or
            # the body.
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("value", data)

        infer_json, train_json = await asyncio.gather(_get(infer_url), _get(train_url))

    return deserialize_value(infer_json), deserialize_value(train_json)


def fetch_kv_metadata(kv_store_url: str, pair_name: str) -> tuple[Any, Any]:
    # [FUNCTION DOC]
    # Synchronous wrapper running _fetch_kv_metadata via run_async_task (works whether
    # or not an event loop is running in the caller thread). Returns the same tuple.
    """Sync wrapper around :func:`_fetch_kv_metadata`.

    Bridges async ``aiohttp`` into the synchronous adapter context using
    :func:`~areal.infra.utils.concurrent.run_async_task`.
    """
    return run_async_task(_fetch_kv_metadata, kv_store_url, pair_name)
