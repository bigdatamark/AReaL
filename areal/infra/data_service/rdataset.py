# [MODULE DOC]
# Trainer-side remote dataset proxy (RDataset): a map-style dataset whose samples live
# on data service workers and are fetched over HTTP through the gateway, with a
# background prefetch buffer driven by a sampler that knows the epoch's index order.
# SPDX-License-Identifier: Apache-2.0

"""RDataset — remote dataset proxy, analogous to RTensor for tensors.

Provides a map-style dataset interface backed by remote data workers.
Samples are fetched over HTTP and cached locally via a sampler-aware
prefetch buffer for near-zero latency ``__getitem__`` calls.

Lifecycle
---------
1. Created by ``get_custom_dataset()`` — stores metadata only (unconnected).
2. Trainer calls ``connect(controller, ...)`` — registers with the data
   service and enables fetching.
3. ``create_dataloader()`` wraps the dataset with ``_PrefetchAwareSampler``
   which triggers prefetch on each ``set_epoch`` call.
4. ``__getitem__(idx)`` pops from the prefetch buffer (cache hit) or falls
   back to a blocking HTTP fetch (cache miss).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from torch.utils.data import DistributedSampler

from areal.utils import logging

if TYPE_CHECKING:
    from areal.infra.data_service.controller.controller import DataController

logger = logging.getLogger("RDataset")


class _PrefetchBuffer:
    # [CLASS DOC]
    # HELPER: Thread-backed sample cache. Holds the fetch function, chunk size, max
    # cache size, a lock-protected dict from index to sample, the planned index order
    # and current position, the prefetch thread, a stop event and a "space available"
    # event used for backpressure.
    """Background thread that fetches samples from remote workers.

    Fetches proceed in the exact index order that the
    ``DistributedSampler`` will request, ensuring near-100 % cache hit
    rate.  The buffer is bounded by *max_cached*; when full the
    prefetch thread pauses until space is freed by ``get()`` calls.

    Parameters
    ----------
    fetch_fn : callable
        ``fn(indices: list[int]) -> list[Any]``  — batch-fetch samples
        by index from remote workers.
    chunk_size : int
        Number of indices to fetch in a single HTTP round-trip.
    max_cached : int
        Maximum number of samples to hold in the local cache before
        the prefetch thread pauses.
    """

    def __init__(
        self,
        fetch_fn: Any,
        chunk_size: int = 64,
        max_cached: int = 512,
    ) -> None:
        # [FUNCTION DOC]
        # Stores fetch_fn (takes a list of indices, returns samples in the same order),
        # chunk_size (indices per request, default 64) and max_cached (default 512);
        # initializes empty cache/order, no thread, and space-available set.
        self._fetch_fn = fetch_fn
        self._chunk_size = chunk_size
        self._max_cached = max_cached

        self._cache: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._indices: list[int] = []
        self._pos: int = 0

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._space_available = threading.Event()
        self._space_available.set()

    # -- Public API --------------------------------------------------------

    def set_index_order(self, indices: list[int]) -> None:
        # [FUNCTION DOC]
        # Stops and joins (up to 10s) any running prefetch thread, clears the cache,
        # installs a copy of indices as the new order from position 0, then starts a new
        # daemon prefetch thread.
        """Reset the cache and start prefetching in *indices* order.

        Called by ``_PrefetchAwareSampler.set_epoch`` at the beginning
        of each epoch.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

        with self._lock:
            self._cache.clear()
            self._indices = list(indices)
            self._pos = 0

        self._stop.clear()
        self._space_available.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get(self, idx: int) -> Any:
        # [FUNCTION DOC]
        # Returns the sample for idx: pops it from the cache on a hit (and signals space
        # available); on a miss logs DEBUG and does a blocking single-index fetch.
        # KNOWN ISSUE: after a miss, the later prefetched copy of that index is never
        # consumed and stays in the cache, so stale entries can accumulate until the
        # cache is full and prefetching stalls.
        """Return the sample for *idx*.

        Pops from the prefetch cache on hit.  On miss, performs a
        blocking single-index fetch.
        """
        with self._lock:
            if idx in self._cache:
                sample = self._cache.pop(idx)
                self._space_available.set()
                return sample

        logger.debug("Prefetch cache miss for index %d, fetching directly", idx)
        return self._fetch_fn([idx])[0]

    def stop(self) -> None:
        # [FUNCTION DOC]
        # Signals the prefetch thread to stop and joins it for up to 10s.
        """Signal the prefetch thread to stop."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    # -- Background thread -------------------------------------------------

    def _run(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Prefetch thread body. Until stopped: under the lock, clears space-
        # available if the cache is full, exits when all indices are consumed, otherwise
        # takes the next chunk and advances the position; waits (polling stop every
        # 0.1s) for space; fetches the chunk. On fetch error logs the exception, sleeps
        # 0.5s and rewinds the position to retry that chunk (retries forever). On
        # success stores samples in the cache by index.
        while not self._stop.is_set():
            with self._lock:
                if len(self._cache) >= self._max_cached:
                    self._space_available.clear()

                if self._pos >= len(self._indices):
                    break

                chunk = self._indices[self._pos : self._pos + self._chunk_size]
                self._pos += len(chunk)

            while not self._space_available.wait(timeout=0.1):
                if self._stop.is_set():
                    return

            if self._stop.is_set():
                return

            try:
                samples = self._fetch_fn(chunk)
            except Exception:
                logger.exception(
                    "Prefetch failed for chunk starting at pos %d",
                    self._pos - len(chunk),
                )
                time.sleep(0.5)
                with self._lock:
                    self._pos -= len(chunk)
                continue

            with self._lock:
                for idx, sample in zip(chunk, samples):
                    self._cache[idx] = sample


class RDataset:
    # [CLASS DOC]
    # Remote map-style dataset. Before connect it only stores metadata (path, loader
    # type, split, max_length, dataset kwargs); after connect it holds the
    # DataController, dataset API key, dataset ID, total sample count and a prefetch
    # buffer. Length and item access raise RuntimeError while unconnected.
    """Remote dataset proxy — map-style dataset backed by remote workers.

    Analogous to :class:`RTensor` for tensors.  Locally stores only
    dataset metadata; actual samples are fetched lazily from the
    distributed data loading service via the :class:`_PrefetchBuffer`.

    Parameters
    ----------
    path : str
        Dataset path (HuggingFace dataset name or local path).
    type : str
        Dataset type (``"sft"``, ``"rl"``, ``"rw"``).
    split : str | None
        Dataset split to load on workers.
    max_length : int | None
        Maximum sequence length for tokenisation on workers.
    dataset_kwargs : dict | None
        Extra keyword arguments forwarded to the dataset loader.
    """

    def __init__(
        self,
        path: str,
        type: str = "rl",
        split: str | None = None,
        max_length: int | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores path, type (default "rl"), split, max_length and dataset_kwargs
        # (default empty dict); marks the dataset unconnected. No I/O.
        self._path = path
        self._type = type
        self._split = split
        self._max_length = max_length
        self._dataset_kwargs = dataset_kwargs or {}

        self._controller: DataController | None = None
        self._api_key: str = ""
        self._dataset_id: str = ""
        self._total_samples: int = 0
        self._connected: bool = False
        self._prefetch_buffer: _PrefetchBuffer | None = None

    # -- Connection --------------------------------------------------------

    def connect(
        self,
        controller: DataController,
        dataset_id: str,
        tokenizer_or_processor_path: str = "",
        shuffle: bool = True,
        drop_last: bool = True,
        prefetch_chunk_size: int = 64,
        prefetch_max_cached: int = 512,
    ) -> None:
        # [FUNCTION DOC]
        # Registers the dataset with the controller (split defaults to "train"), passing
        # tokenizer path, shuffle and drop_last; stores the returned API key, dataset ID
        # and total samples, creates the prefetch buffer with the given chunk size and
        # cache limit, and logs INFO. Raises RuntimeError if already connected;
        # propagates registration errors.
        """Register with *controller* and enable data fetching.

        Called by the trainer after the ``DataController`` is
        initialised.  The controller broadcasts dataset loading to all
        workers and returns the total sample count.
        """
        if self._connected:
            raise RuntimeError("RDataset is already connected")

        handle = controller.register_dataset(
            dataset_id=dataset_id,
            dataset_path=self._path,
            dataset_type=self._type,
            split=self._split or "train",
            max_length=self._max_length,
            dataset_kwargs=self._dataset_kwargs,
            tokenizer_or_processor_path=tokenizer_or_processor_path,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        self._controller = controller
        self._api_key = handle["api_key"]
        self._dataset_id = handle["dataset_id"]
        self._total_samples = handle["total_samples"]
        self._connected = True
        self._prefetch_buffer = _PrefetchBuffer(
            fetch_fn=self._fetch_samples,
            chunk_size=prefetch_chunk_size,
            max_cached=prefetch_max_cached,
        )
        logger.info(
            "RDataset connected: id=%s, total_samples=%d",
            self._dataset_id,
            self._total_samples,
        )

    # -- Map-style dataset interface ---------------------------------------

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns the total sample count reported at registration (sum of worker shard
        # sizes). Raises RuntimeError if not connected.
        if not self._connected:
            raise RuntimeError(
                "RDataset is not connected to a DataController. "
                "Call connect() before using the dataset."
            )
        return self._total_samples

    def __getitem__(self, idx: int) -> Any:
        # [FUNCTION DOC]
        # Returns the sample at idx from the prefetch buffer (or a blocking fetch).
        # Raises RuntimeError if not connected.
        if not self._connected or self._prefetch_buffer is None:
            raise RuntimeError(
                "RDataset is not connected to a DataController. "
                "Call connect() before using the dataset."
            )
        return self._prefetch_buffer.get(idx)

    # -- Prefetch control (called by _PrefetchAwareSampler) ----------------

    def _start_prefetch(self, indices: list[int]) -> None:
        # [FUNCTION DOC]
        # HELPER: Restarts the prefetch buffer with the given index order (no-op if
        # unconnected).
        """Kick off the prefetch buffer in *indices* order."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.set_index_order(indices)

    # -- Remote fetch ------------------------------------------------------

    def _fetch_samples(self, indices: list[int]) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: Synchronously POSTs the indices to the gateway's /v1/samples/fetch
        # with the dataset API key and returns the deserialized samples. Asserts a
        # controller is set; raises RuntimeError on gateway errors.
        """Batch-fetch samples by index from remote workers via the gateway."""
        assert self._controller is not None
        from areal.infra.rpc.serialization import deserialize_value

        resp = self._controller._gateway_post(
            "/v1/samples/fetch",
            self._api_key,
            {"indices": indices},
        )
        return [deserialize_value(s) for s in resp["samples"]]

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        # [FUNCTION DOC]
        # Stops and drops the prefetch buffer and, if connected, unregisters the dataset
        # through the controller (failures logged at DEBUG), then marks it unconnected.
        """Stop prefetching and unregister from the controller."""
        if self._prefetch_buffer is not None:
            self._prefetch_buffer.stop()
            self._prefetch_buffer = None
        if self._connected and self._controller is not None:
            try:
                self._controller.unregister_dataset(self._dataset_id)
            except Exception:
                logger.debug(
                    "Failed to unregister dataset %s (expected during teardown)",
                    self._dataset_id,
                )
            self._connected = False

    @property
    def connected(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether connect has completed and close has not been called.
        return self._connected


class _PrefetchAwareSampler(DistributedSampler):
    # [CLASS DOC]
    # DistributedSampler subclass bound to an RDataset that, on construction and every
    # set_epoch, computes the full index order this sampler will yield and hands it to
    # the dataset's prefetch buffer so prefetching matches DataLoader access order.
    """``DistributedSampler`` that triggers ``RDataset`` prefetch on epoch change.

    When ``cycle_dataloader`` calls ``sampler.set_epoch(epoch)``, this
    sampler generates the deterministic index order for the new epoch
    and passes it to the ``RDataset``'s prefetch buffer so that
    samples are fetched in the order the ``DataLoader`` will request
    them.
    """

    def __init__(self, dataset: RDataset, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # Initializes the DistributedSampler with the dataset and any sampler args
        # (num_replicas, rank, shuffle, seed, drop_last), keeps a reference to the
        # RDataset and triggers the initial prefetch.
        super().__init__(dataset, *args, **kwargs)
        self._rdataset = dataset
        self._trigger_prefetch()

    def set_epoch(self, epoch: int) -> None:
        # [FUNCTION DOC]
        # Sets the epoch on the base sampler and re-triggers prefetch for the new order.
        super().set_epoch(epoch)
        self._trigger_prefetch()

    def _trigger_prefetch(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Materializes the base sampler's iteration order as a list and starts
        # the dataset's prefetch with it.
        indices = list(super().__iter__())
        self._rdataset._start_prefetch(indices)
