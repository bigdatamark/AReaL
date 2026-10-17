# [MODULE DOC]
# Pause/resume coordination for the data proxy: an async pause flag polled by the
# request bridge, and helpers that call the backend's pause_generation /
# continue_generation endpoints.
# SPDX-License-Identifier: Apache-2.0

"""Pause/resume state management for generation coordination.

The controller calls POST /pause on the data proxy to:
  1. Set the PauseState flag to True
  2. Call SGLang POST /pause_generation (aborting in-flight requests)

When ready to resume, the controller calls POST /resume:
  1. Call SGLang POST /continue_generation
  2. Set the PauseState flag to False

InfBridge (inference_service/inf_bridge.py) polls PauseState and transparently
resubmits aborted requests once resumed.
"""

from __future__ import annotations

import asyncio

import httpx

from areal.infra.utils.http import create_httpx_client
from areal.utils import logging

logger = logging.getLogger("InferenceDataProxy")


class PauseState:
    # [CLASS DOC]
    # Boolean paused flag guarded by an asyncio lock for writes.
    """Async-safe pause flag for weight-update coordination."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes unpaused with a new asyncio.Lock.
        self._paused: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    async def set_paused(self, paused: bool) -> None:
        # [FUNCTION DOC]
        # Sets the flag under the lock.
        async with self._lock:
            self._paused = paused

    async def is_paused(self) -> bool:
        # [FUNCTION DOC]
        # Returns the flag without locking.
        # Single bool read is atomic under CPython GIL — no lock needed.
        return self._paused


async def pause_backend(
    backend_addr: str, *, client: httpx.AsyncClient | None = None
) -> None:
    # [FUNCTION DOC]
    # POSTs {} to <backend_addr>/pause_generation (10s timeout) using the given httpx
    # client or a temporary one; raises on non-2xx; logs at INFO.
    # KNOWN ISSUE: the endpoint is SGLang-specific and does not consult the backend
    # type.
    """Call SGLang POST /pause_generation to abort in-flight requests."""
    if client is not None:
        resp = await client.post(
            f"{backend_addr}/pause_generation", json={}, timeout=10.0
        )
        resp.raise_for_status()
    else:
        async with create_httpx_client(timeout=10.0) as c:
            resp = await c.post(f"{backend_addr}/pause_generation", json={})
            resp.raise_for_status()
    logger.info("SGLang pause_generation called on %s", backend_addr)


async def resume_backend(
    backend_addr: str, *, client: httpx.AsyncClient | None = None
) -> None:
    # [FUNCTION DOC]
    # POSTs {} to <backend_addr>/continue_generation (10s timeout) using the given or a
    # temporary client; raises on non-2xx; logs at INFO.
    # KNOWN ISSUE: SGLang-specific endpoint, as in pause_backend.
    """Call SGLang POST /continue_generation to resume inference."""
    if client is not None:
        resp = await client.post(
            f"{backend_addr}/continue_generation", json={}, timeout=10.0
        )
        resp.raise_for_status()
    else:
        async with create_httpx_client(timeout=10.0) as c:
            resp = await c.post(f"{backend_addr}/continue_generation", json={})
            resp.raise_for_status()
    logger.info("SGLang continue_generation called on %s", backend_addr)
