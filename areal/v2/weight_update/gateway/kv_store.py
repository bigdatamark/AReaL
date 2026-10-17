# [MODULE DOC]
# Thread-safe in-memory key/value and set store, namespaced by pair name, used by the
# gateway for metadata exchange and barrier counting.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from typing import Any

from areal.utils import logging

logger = logging.getLogger("WeightMetaStore")


class WeightMetaStore:
    # [CLASS DOC]
    # In-memory store with a per-pair dict of values and a per-pair dict of string sets,
    # guarded by one lock.
    """Thread-safe in-memory KV store for metadata exchange.

    Values are stored as-is (Python objects). Serialization/deserialization
    is handled at the HTTP boundary, not inside the store.
    Each key is scoped by pair_name to support multiple concurrent pairs.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes empty data and set maps and the lock.
        self._data: dict[str, dict[str, Any]] = {}
        self._sets: dict[str, dict[str, set]] = {}
        self._lock = threading.Lock()

    def get(self, pair_name: str, key: str) -> Any | None:
        # [FUNCTION DOC]
        # Returns the value for pair/key or None.
        """Get value for key under pair_name. Returns None if not found."""
        with self._lock:
            return self._data.get(pair_name, {}).get(key)

    def put(self, pair_name: str, key: str, value: Any) -> None:
        # [FUNCTION DOC]
        # Stores value under pair/key (overwrites).
        """Store value under pair_name/key."""
        with self._lock:
            self._data.setdefault(pair_name, {})[key] = value

    def delete(self, pair_name: str, key: str) -> bool:
        # [FUNCTION DOC]
        # Deletes pair/key; returns True if it existed.
        """Delete key from pair. Returns True if key existed."""
        with self._lock:
            pair_data = self._data.get(pair_name, {})
            if key in pair_data:
                del pair_data[key]
                return True
            return False

    def add_to_set(self, pair_name: str, key: str, value: str) -> None:
        # [FUNCTION DOC]
        # Adds value to the set at pair/key (creating it).
        """Add value to a set under pair_name/key. Used for barrier sync."""
        with self._lock:
            self._sets.setdefault(pair_name, {}).setdefault(key, set()).add(value)

    def set_size(self, pair_name: str, key: str) -> int:
        # [FUNCTION DOC]
        # Returns size of the set at pair/key (0 if absent).
        """Get size of set under pair_name/key. Returns 0 if not found."""
        with self._lock:
            return len(self._sets.get(pair_name, {}).get(key, set()))

    def clear_pair(self, pair_name: str) -> None:
        # [FUNCTION DOC]
        # Removes all values and sets for the pair and logs.
        """Remove all data and sets for a pair."""
        with self._lock:
            self._data.pop(pair_name, None)
            self._sets.pop(pair_name, None)
            logger.info("Cleared all data for pair '%s'", pair_name)

    def list_keys(self, pair_name: str) -> list[str]:
        # [FUNCTION DOC]
        # Returns the list of value keys for the pair (sets not included).
        """List all keys for a pair."""
        with self._lock:
            return list(self._data.get(pair_name, {}).keys())
