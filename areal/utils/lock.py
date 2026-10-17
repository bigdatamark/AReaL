# [MODULE DOC]
# Cross-process mutex built on the torch.distributed default c10d TCP store using an
# atomic counter plus an owner token key.
# SPDX-License-Identifier: Apache-2.0

import time
import uuid

import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_store


class DistributedLock:
    # [CLASS DOC]
    # Named distributed lock. Keys live under "lock/<namespace>/<name>" (namespace
    # defaults to "ws<world_size>"). Acquiring increments a counter; the process that
    # sees value 1 owns the lock. Usable as a context manager.
    def __init__(self, name, backoff=0.05, namespace=None):
        # [FUNCTION DOC]
        # Raises RuntimeError if torch.distributed is not initialized. Grabs the default
        # store, builds key names, stores backoff and ensures the counter key exists
        # (add 0; errors ignored).
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized")

        self.store = _get_default_store()
        ns = namespace or f"ws{dist.get_world_size()}"
        self.base = f"lock/{ns}/{name}"
        self.key_counter = f"{self.base}/counter"
        self.key_owner = f"{self.base}/owner"
        self.backoff = backoff
        self.token = None

        try:
            self.store.add(self.key_counter, 0)
        except RuntimeError:
            pass

    def acquire(self, timeout=None):
        # [FUNCTION DOC]
        # Spins trying to acquire: atomically increments the counter; on 1 records an
        # owner token "<rank>:<uuid>" and returns True; otherwise decrements back. Store
        # RuntimeErrors are ignored. Sleeps with exponential backoff (x1.5, capped
        # 0.5s). Returns False once timeout seconds elapse (None means wait forever).
        # KNOWN ISSUE: if an increment succeeds but raises afterwards, or a holder
        # crashes, the counter can stay elevated and the lock becomes permanently
        # unavailable.
        start = time.perf_counter()
        sleep = self.backoff
        my_token = f"{dist.get_rank()}:{uuid.uuid4()}".encode()

        while True:
            try:
                current = self.store.add(self.key_counter, 1)
                if current == 1:
                    self._set_owner(my_token)
                    self.token = my_token
                    return True

                self._rollback_counter()
            except RuntimeError:
                # A RuntimeError can occur from store.add under contention.
                # We'll let the loop back off and retry.
                pass

            if timeout is not None and (time.perf_counter() - start) > timeout:
                return False

            time.sleep(sleep)
            sleep = min(sleep * 1.5, 0.5)

    def release(self):
        # [FUNCTION DOC]
        # Releases the lock. Raises RuntimeError if not held by this instance or if the
        # stored owner differs from our token; otherwise clears the owner, decrements
        # the counter and forgets the token.
        if self.token is None:
            raise RuntimeError("Lock not held by this process")

        owner = self._get_owner()

        if owner is not None and owner != self.token:
            raise RuntimeError("Lock owner mismatch; refusing to release")

        self._clear_owner()
        self._rollback_counter()
        self.token = None

    def _rollback_counter(self):
        # [FUNCTION DOC]
        # HELPER: Decrements the counter, retrying forever with backoff on store errors.
        delay = self.backoff
        while True:
            try:
                self.store.add(self.key_counter, -1)
                return
            except RuntimeError:
                time.sleep(delay)
                delay = min(delay * 1.5, 0.5)

    def _set_owner(self, token: bytes):
        # [FUNCTION DOC]
        # HELPER: Writes the owner token; on store error rolls back the counter and
        # raises RuntimeError.
        try:
            self.store.set(self.key_owner, token)
        except RuntimeError as exc:
            self._rollback_counter()
            raise RuntimeError("Failed to record lock owner") from exc

    def _get_owner(self) -> bytes | None:
        # [FUNCTION DOC]
        # HELPER: Returns the stored owner token, or None if empty or on store error.
        try:
            value = self.store.get(self.key_owner)
            return value if value else None
        except RuntimeError:
            return None

    def _clear_owner(self):
        # [FUNCTION DOC]
        # HELPER: Sets the owner key to empty bytes, ignoring store errors.
        try:
            self.store.set(self.key_owner, b"")
        except RuntimeError:
            pass

    def __enter__(self):
        # [FUNCTION DOC]
        # Context entry: acquires (no timeout; return value ignored) and returns self.
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC]
        # Context exit: releases the lock and returns False (exceptions propagate).
        self.release()
        return False
