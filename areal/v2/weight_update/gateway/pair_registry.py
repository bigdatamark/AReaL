# [MODULE DOC]
# Thread-safe registry of connected training/inference pairs keyed by pair name.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

from areal.utils import logging
from areal.v2.weight_update.gateway.config import PairInfo

logger = logging.getLogger("PairRegistry")


class PairRegistry:
    # [CLASS DOC]
    # Lock-guarded dict pair_name -> PairInfo.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes empty mapping and lock.
        self._by_name: dict[str, PairInfo] = {}
        self._lock = threading.Lock()

    def register(self, pair_info: PairInfo) -> None:
        # [FUNCTION DOC]
        # Adds pair_info; raises ValueError if the name is already registered. Logs.
        with self._lock:
            if pair_info.pair_name in self._by_name:
                raise ValueError(f"Pair '{pair_info.pair_name}' already registered")
            self._by_name[pair_info.pair_name] = pair_info
            logger.info("Registered pair '%s'", pair_info.pair_name)

    def get_by_name(self, pair_name: str) -> PairInfo | None:
        # [FUNCTION DOC]
        # Returns PairInfo for name or None.
        with self._lock:
            return self._by_name.get(pair_name)

    def unregister(self, pair_name: str) -> PairInfo | None:
        # [FUNCTION DOC]
        # Removes and returns the pair (None if absent); logs on removal.
        with self._lock:
            pair_info = self._by_name.pop(pair_name, None)
            if pair_info is not None:
                logger.info("Unregistered pair '%s'", pair_name)
            return pair_info

    def list_pairs(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns registered pair names.
        with self._lock:
            return list(self._by_name.keys())
