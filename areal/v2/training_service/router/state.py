# [MODULE DOC]
# In-memory model registry for the training router with a one-to-one mapping between
# model addresses and API keys and debounced health tracking (unhealthy after 2
# consecutive failed probes). Guarded by an asyncio.Lock.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

_MAX_CONSECUTIVE_HEALTH_FAILURES = 2


@dataclass
class ModelInfo:
    # [CLASS DOC]
    # Dataclass for a registered data proxy: model_addr, api_key, name, is_healthy,
    # consecutive_health_failures and registered_at timestamp.
    model_addr: str
    api_key: str
    name: str = ""
    is_healthy: bool = True
    consecutive_health_failures: int = 0
    registered_at: float = field(default_factory=time.time)


class ModelRegistry:
    # [CLASS DOC]
    # Maps model_addr -> ModelInfo and api_key -> model_addr, keeping the mapping one-
    # to-one.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Creates both empty maps and the lock.
        self._models: dict[str, ModelInfo] = {}
        self._key_to_addr: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(self, model_addr: str, api_key: str, name: str = "") -> None:
        # [FUNCTION DOC]
        # Upserts the pinning: if the address already had a different key, that old key
        # is unmapped; if the key was mapped to a different address, that other model is
        # removed entirely. Existing entries keep their health state and update the key
        # (and name if non-empty); new ones start healthy.
        async with self._lock:
            existing_model = self._models.get(model_addr)
            if existing_model is not None and existing_model.api_key != api_key:
                self._key_to_addr.pop(existing_model.api_key, None)

            existing_addr_for_key = self._key_to_addr.get(api_key)
            if (
                existing_addr_for_key is not None
                and existing_addr_for_key != model_addr
            ):
                self._models.pop(existing_addr_for_key, None)

            if existing_model is None:
                info = ModelInfo(model_addr=model_addr, api_key=api_key, name=name)
            else:
                existing_model.api_key = api_key
                if name:
                    existing_model.name = name
                info = existing_model

            self._models[model_addr] = info
            self._key_to_addr[api_key] = model_addr

    async def deregister(self, model_addr: str) -> None:
        # [FUNCTION DOC]
        # Removes a model by address and its key mapping; no-op if absent.
        async with self._lock:
            model = self._models.pop(model_addr, None)
            if model is not None:
                self._key_to_addr.pop(model.api_key, None)

    async def lookup_by_key(self, api_key: str) -> ModelInfo | None:
        # [FUNCTION DOC]
        # Returns the ModelInfo pinned to api_key, or None.
        async with self._lock:
            model_addr = self._key_to_addr.get(api_key)
            if model_addr is None:
                return None
            return self._models.get(model_addr)

    async def update_health(self, model_addr: str, healthy: bool) -> None:
        # [FUNCTION DOC]
        # For a known address: a healthy probe resets the failure counter and marks
        # healthy; a failed probe increments the counter and marks unhealthy once it
        # reaches the threshold (2). Unknown addresses are ignored.
        async with self._lock:
            model = self._models.get(model_addr)
            if model is not None:
                if healthy:
                    model.is_healthy = True
                    model.consecutive_health_failures = 0
                    return

                model.consecutive_health_failures += 1
                if (
                    model.consecutive_health_failures
                    >= _MAX_CONSECUTIVE_HEALTH_FAILURES
                ):
                    model.is_healthy = False

    async def get_all(self) -> list[ModelInfo]:
        # [FUNCTION DOC]
        # Returns a list of all ModelInfo objects (live references).
        async with self._lock:
            return list(self._models.values())

    async def count(self) -> int:
        # [FUNCTION DOC]
        # Returns the number of registered models.
        async with self._lock:
            return len(self._models)
