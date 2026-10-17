# [MODULE DOC]
# Distributed training-statistics tracker. Values are recorded under hierarchical
# "/"-joined keys (scope stack + key) as masked tensors, scalars or timings, and
# exported as aggregated floats, optionally all-reduced across a process group
# (with per-key group overrides and cross-rank key/metadata synchronization so all
# ranks issue identical collectives). Exposes a module-level DEFAULT_TRACKER with
# bound shortcuts (stat, denominator, export, scope, scalar, record_timing,
# scope_func_wrapper), a registry TRACKERS of named trackers, get() and export_all().
# MOE_AUX_LOSSES is a module-level dict placeholder for MoE auxiliary losses.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.seqpack import flat2d

logger = logging.getLogger("StatsTracker")


class ReduceType(Enum):
    # [CLASS DOC]
    # How a key is aggregated: AVG_MIN_MAX (emits key/avg, key/min, key/max), AVG, SUM,
    # MIN, MAX (all masked by a denominator mask) or SCALAR (mean of plain floats plus a
    # "__count" key).
    AVG_MIN_MAX = auto()
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


@dataclass(frozen=True)
class _StatMetadata:
    # [CLASS DOC]
    # HELPER: Frozen per-key metadata exchanged across ranks: reduce type, denominator
    # key (or None) and whether a per-key reduce group override exists.
    reduce_type: ReduceType
    denominator: str | None
    has_reduce_group: bool


MOE_AUX_LOSSES = {}


class DistributedStatsTracker:
    # [CLASS DOC]
    # Holds a lock, a scope stack (optionally seeded with the tracker's name), maps of
    # key -> denominator key, key -> ReduceType, key -> reduce-group override, and
    # key -> list of recorded values (tensors or floats). Recording is lock-protected;
    # export performs reductions.
    def __init__(self, name: str = ""):
        # [FUNCTION DOC]
        # Initializes the lock, scope stack (name with surrounding "/" stripped, if
        # given) and empty metadata/stat maps.
        self.lock = Lock()
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}  # key -> denominator key
        self.reduce_types = {}  # key -> ReduceType
        # Per-key override of the reduce_group. If set, its value takes
        # precedence over the `reduce_group` passed to `export`. This is used,
        # e.g., to make CP-local SFT stats (loss/entropy/vocab_*) reduce across
        # DP + CP so the reported numbers are CP-invariant (#1242 follow-up).
        self.reduce_groups = {}  # key -> dist.ProcessGroup

        self.stats = defaultdict(list)

    def scope(self, name):
        # [FUNCTION DOC]
        # Returns a Scope context manager that pushes name onto the scope stack while
        # active.
        # KNOWN ISSUE: the lock only guards creating the Scope object; the scope stack is
        # shared and mutated without locking, so concurrent threads interleave scopes.
        """Context manager for hierarchical scoping"""
        with self.lock:
            return self.Scope(self, name)

    def scope_func_wrapper(self, name):
        # [FUNCTION DOC]
        # Returns a decorator that runs the wrapped function inside scope(name).
        """Decorator for hierarchical scoping"""

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: Wraps func with the scoping wrapper (no functools.wraps, so metadata is
            # lost).
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Calls func with its arguments inside the scope and returns its result.
                with self.scope(name):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    class Scope:
        # [CLASS DOC]
        # HELPER: Context manager bound to a tracker and a scope name (slashes stripped).
        def __init__(self, tracker, name):
            # [FUNCTION DOC]
            # Stores the tracker and the stripped name.
            self.tracker = tracker
            self.name = name.strip("/")

        def __enter__(self):
            # [FUNCTION DOC]
            # Pushes the name on the tracker's scope stack; returns self.
            self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # [FUNCTION DOC]
            # Pops the top of the scope stack; exceptions propagate.
            self.tracker.scope_stack.pop()

    def _get_full_key(self, key):
        # [FUNCTION DOC]
        # HELPER: Joins the scope stack and key with "/" (just key when no scope).
        """Combine scope stack with current key"""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def disable_scope(self):
        # [FUNCTION DOC]
        # Context manager that temporarily empties the scope stack so keys are recorded
        # unscoped, restoring it afterwards.
        # KNOWN ISSUE: no try/finally, so the scope stack is not restored if the body
        # raises.
        tmp = self.scope_stack
        self.scope_stack = []
        yield
        self.scope_stack = tmp

    @contextmanager
    def record_timing(self, key):
        # [FUNCTION DOC]
        # Context manager timing its body with perf_counter; on exit (even on error)
        # appends the elapsed seconds as a SCALAR under "timeperf/<key>" (ignores the
        # current scope).
        start_time = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                # NOTE: timing records are fixed under the "timeperf" scope
                full_key = f"timeperf/{key}"
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(time.perf_counter() - start_time)

    def denominator(self, *, reduce_group=None, **kwargs):
        # [FUNCTION DOC]
        # Records boolean mask tensors (kwargs name -> tensor) as SUM-type keys under the
        # current scope (detached clones). Optional reduce_group becomes a per-key
        # override. Raises ValueError for non-bool or empty tensors.
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                    raise ValueError(
                        f"`{key}` must be a pytorch bool tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` must be non-empty")
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SUM)
                self.stats[full_key].append(value.detach().clone())
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def scalar(self, *, reduce_group=None, **kwargs):
        # [FUNCTION DOC]
        # Records each kwarg value converted to float as a SCALAR key under the current
        # scope; optional reduce_group is stored as a per-key override.
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(float(value))
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        *,
        reduce_group=None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Records float32 tensors masked by a previously recorded denominator key
        # (resolved under the current scope). Validates dtype float, non-empty, not SCALAR
        # type, denominator exists, and that each value's shape matches the corresponding
        # denominator entry (assert). Remembers the denominator, sets the reduce type
        # (default AVG_MIN_MAX), appends a detached clone and stores any per-key reduce
        # group.
        # KNOWN ISSUE: a non-tensor value triggers AttributeError when building the error
        # message (value.dtype).
        """Record multiple values from a dictionary.

        If `reduce_group` is provided, it overrides the `reduce_group` passed
        to `export` for these specific keys. This enables recording stats that
        must be reduced across a different topology than the default (e.g.,
        loss/vocab_* under CP-local loss need DP+CP reduce while n_seqs stays
        DP-only).
        """
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float:
                    raise ValueError(
                        f"`{key}` should be a pytorch float tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` should be non-empty")
                if reduce_type == ReduceType.SCALAR:
                    raise ValueError("Cannot use the scalar reduce type for a tensor")
                full_key = self._get_full_key(key)

                denorm = self._get_full_key(denominator)
                if denorm not in self.stats or not self.stats[denorm]:
                    raise ValueError(f"Denominator `{denorm}` does not exist")
                for x, y in zip(self.stats[denorm], self.stats[full_key] + [value]):
                    assert x.shape == y.shape, (x.shape, y.shape)
                self.denominators[full_key] = denorm

                if reduce_type is None:
                    reduce_type = ReduceType.AVG_MIN_MAX
                self._set_reduce_type(full_key, reduce_type)
                self.stats[full_key].append(value.detach().clone())
                if reduce_group is not None:
                    self.reduce_groups[full_key] = reduce_group

    def _set_reduce_type(self, key, reduce_type):
        # [FUNCTION DOC]
        # HELPER: Validates reduce_type is a ReduceType (ValueError otherwise) and records
        # it for key.
        if not isinstance(reduce_type, ReduceType):
            raise ValueError("reduce_type must be a ReduceType enum")
        self.reduce_types[key] = reduce_type

    def _effective_reduce_group(
        self,
        key,
        default_reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Chooses the process group to reduce key over. If synced metadata says
        # some rank has a per-key override: use key_sync_group, else the local override,
        # else raise RuntimeError. Otherwise the local override if present, else the
        # default group (may be None meaning no reduction).
        """Return the per-key reduce_group override if any, else the default."""
        if sync_metadata is not None:
            metadata = sync_metadata.get(key)
            if metadata is not None and metadata.has_reduce_group:
                if key_sync_group is not None:
                    return key_sync_group
                if key in self.reduce_groups:
                    return self.reduce_groups[key]
                raise RuntimeError(
                    f"Key `{key}` has a per-key reduce_group on another rank, "
                    "but no key_sync_group was provided for this rank."
                )
        if key in self.reduce_groups:
            return self.reduce_groups[key]
        return default_reduce_group

    def _device_for_placeholder_tensor(self, group=None):
        # [FUNCTION DOC]
        # HELPER: Picks a device for identity tensors: the accelerator if the group's
        # backend matches the platform's communication backend or the platform is
        # initialized; otherwise "cpu". Lookup errors are ignored.
        if group is not None:
            try:
                backend = str(dist.get_backend(group)).lower()
                platform_backend = current_platform.communication_backend
                if backend == platform_backend:
                    return current_platform.device_type
            except (AttributeError, RuntimeError, ValueError):
                pass
        try:
            if current_platform.is_initialized():
                return current_platform.device_type
        except AttributeError:
            pass
        return "cpu"

    def _placeholder_scalar(
        self,
        fill: float = 0.0,
        like: torch.Tensor | None = None,
        group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Returns a float32 scalar tensor with value fill on like's device, or on
        # the device chosen for group.
        """Scalar identity tensor for reductions (0.0 for SUM/AVG, +/-inf for
        MIN/MAX). Placed on ``like``'s device if given, else on the device
        matching ``group``."""
        device = (
            like.device
            if like is not None
            else self._device_for_placeholder_tensor(group)
        )
        return torch.tensor(
            fill,
            dtype=torch.float32,
            device=device,
        )

    def _local_metadata(self, keys: list[str]) -> dict[str, _StatMetadata]:
        # [FUNCTION DOC]
        # HELPER: Builds _StatMetadata for each key from local maps (missing reduce type
        # defaults to SCALAR).
        return {
            key: _StatMetadata(
                reduce_type=self.reduce_types.get(key, ReduceType.SCALAR),
                denominator=self.denominators.get(key),
                has_reduce_group=key in self.reduce_groups,
            )
            for key in keys
        }

    @staticmethod
    def _merge_metadata(
        all_metadata: list[dict[str, _StatMetadata]],
    ) -> dict[str, _StatMetadata]:
        # [FUNCTION DOC]
        # HELPER: Merges per-rank metadata dicts into one; raises ValueError if ranks
        # disagree on a key's metadata.
        metadata: dict[str, _StatMetadata] = {}
        for rank_metadata in all_metadata:
            for key, key_metadata in rank_metadata.items():
                if key not in metadata:
                    metadata[key] = key_metadata
                elif metadata[key] != key_metadata:
                    raise ValueError(
                        f"Inconsistent stats metadata for key `{key}`: "
                        f"{metadata[key]} vs {key_metadata}"
                    )
        return metadata

    def export(
        self,
        key=None,
        reduce_group=None,
        key_sync_group=None,
        reset=True,
    ) -> dict[str, float]:
        # [FUNCTION DOC]
        # Aggregates and returns stats as {name: float}. With key: aggregates that single
        # scoped key (over reduce_group) and optionally resets it. Otherwise: if
        # reduce_group is given, all-gathers metadata over it and merges (so ranks missing a
        # key still join its collectives); if key_sync_group is given, also syncs metadata
        # of per-key-override keys over that group (ValueError on conflict). Keys are then
        # sorted for consistent collective order, each is aggregated, state is cleared if
        # reset, and tensor results are moved to CPU floats. Collective when groups given.
        # KNOWN ISSUE: the single-key path returns unconverted tensors for avg values and
        # raises KeyError on reset if the key has no recorded stats.
        """Get aggregated statistics"""
        with self.lock:
            if key is not None:
                full_key = self._get_full_key(key)
                result = self._aggregate(full_key, reduce_group)
                if reset:
                    if full_key in self.denominators:
                        self.denominators.pop(full_key)
                    if full_key in self.reduce_types:
                        self.reduce_types.pop(full_key)
                    if full_key in self.reduce_groups:
                        self.reduce_groups.pop(full_key)
                    self.stats.pop(full_key)
                return result

            # synchronize keys across ranks
            keys = list(self.stats.keys())
            sync_metadata = self._local_metadata(keys)
            if reduce_group is not None:
                all_metadata = [None for _ in range(dist.get_world_size(reduce_group))]
                dist.all_gather_object(
                    all_metadata,
                    self._local_metadata(keys),
                    group=reduce_group,
                )
                sync_metadata = self._merge_metadata(all_metadata)

            # Per-key reduce groups may be wider than the default export group
            # (for example DP+CP vs DP). Only those override keys need the
            # wider key/metadata sync; default keys keep the original
            # reduce_group alignment.
            if key_sync_group is not None:
                override_keys = list(self.reduce_groups.keys())
                all_override_metadata = [
                    None for _ in range(dist.get_world_size(key_sync_group))
                ]
                dist.all_gather_object(
                    all_override_metadata,
                    self._local_metadata(override_keys),
                    group=key_sync_group,
                )
                override_metadata = self._merge_metadata(all_override_metadata)
                for override_key, override_value in override_metadata.items():
                    if (
                        override_key in sync_metadata
                        and sync_metadata[override_key] != override_value
                    ):
                        raise ValueError(
                            f"Inconsistent stats metadata for key `{override_key}`: "
                            f"{sync_metadata[override_key]} vs {override_value}"
                        )
                    sync_metadata[override_key] = override_value

            if reduce_group is not None or key_sync_group is not None:
                # Should ensure that the orders are the same
                keys = sorted(sync_metadata)
            results = {}
            for key in keys:
                results.update(
                    self._aggregate(
                        key,
                        reduce_group,
                        sync_metadata=sync_metadata,
                        key_sync_group=key_sync_group,
                    )
                )
            if reset:
                self.denominators = {}
                self.reduce_types = {}
                self.reduce_groups = {}
                self.stats = defaultdict(list)
            results = {
                k: v.cpu().item() if torch.is_tensor(v) else v
                for k, v in results.items()
            }
            return results

    def _aggregate(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Dispatches on the key's reduce type (local, else synced metadata, else
        # SCALAR) to the matching reducer and returns a dict of output names to values.
        # SCALAR: sums and counts local floats, all-reduces both over the effective group
        # and emits mean (0.0 if count is 0) plus "<key>__count". Drops None results.
        # Raises ValueError for unknown types.
        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        reduce_type = self.reduce_types.get(
            key,
            metadata.reduce_type if metadata is not None else ReduceType.SCALAR,
        )

        result = {}
        if reduce_type == ReduceType.AVG_MIN_MAX:
            result["/".join([key, "avg"])] = self._avg_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
            result["/".join([key, "min"])] = self._min_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
            result["/".join([key, "max"])] = self._max_of(
                key, reduce_group, sync_metadata, key_sync_group
            )
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key, reduce_group, sync_metadata, key_sync_group)
        elif reduce_type == ReduceType.SCALAR:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            # `.get`: a rank may learn about this key via key/metadata sync
            # without holding any local values for it.
            stats = self.stats.get(key, [])
            value = self._placeholder_scalar(
                fill=float(sum(stats)), group=effective_group
            )
            cnt = self._placeholder_scalar(
                fill=float(len(stats)), group=effective_group
            )
            if effective_group is not None:
                dist.all_reduce(value, group=effective_group)
                dist.all_reduce(cnt, group=effective_group)
            result[key] = float(value / cnt) if float(cnt) > 0 else 0.0
            result[key + "__count"] = int(cnt)
        else:
            raise ValueError(f"Unknown reduce type: {reduce_type}")

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Sums all recorded tensors (masked by the denominator if the key has one;
        # ValueError if the denominator is missing), all-reduces SUM over the effective
        # group and returns a float. With no local values contributes 0 to the collective.
        values = self.stats.get(key, [])
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if not values:
            x = self._placeholder_scalar(group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
            return float(x)

        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        denominator = self.denominators.get(
            key,
            metadata.denominator if metadata is not None else None,
        )
        if denominator is None:
            x = sum([x.sum() for x in values], self._placeholder_scalar(like=values[0]))
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        else:
            if denominator not in self.stats:
                raise ValueError(
                    f"Denominator `{denominator}` not set for key `{key}`."
                )
            xs = []
            for v, d in zip(values, self.stats[denominator]):
                xs.append(torch.where(d, v, 0.0).sum())
            x = sum(xs, self._placeholder_scalar(like=values[0]))
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
        return float(x)

    def _denominator_of(
        self,
        key,
        sync_metadata: dict[str, _StatMetadata] | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Returns the key's denominator from local map or synced metadata; raises
        # ValueError if none.
        if key in self.denominators:
            return self.denominators[key]
        metadata = sync_metadata.get(key) if sync_metadata is not None else None
        if metadata is not None and metadata.denominator is not None:
            return metadata.denominator
        raise ValueError(f"Denominator not set for key `{key}`.")

    def _avg_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Masked mean: sums masked values and mask counts, all-reduces both, returns
        # sum / count as a tensor, or None when the total count is 0. With no local values
        # contributes zeros to the collectives.
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(group=effective_group)
            d = self._placeholder_scalar(group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group)
                dist.all_reduce(d, group=effective_group)
            if d == 0:
                return None
            return x / d

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        ds = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs, self._placeholder_scalar(like=values[0]))
        d = sum(ds, self._placeholder_scalar(like=values[0]))
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group)
            dist.all_reduce(d, group=effective_group)
        if d == 0:
            return None
        return x / d

    def _min_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Masked minimum (unmasked positions treated as +inf), all-reduced with MIN;
        # returns a float or None if the result is infinite (no valid entries). With no
        # local values contributes +inf.
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(fill=float("inf"), group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MIN)
            if torch.isinf(x):
                return None
            return float(x)

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, float("inf")).min())
        x = torch.stack(xs).min()
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MIN)
        if torch.isinf(x):
            return None
        return float(x)

    def _max_of(
        self,
        key,
        reduce_group,
        sync_metadata: dict[str, _StatMetadata] | None = None,
        key_sync_group=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Masked maximum (unmasked positions -inf), all-reduced with MAX; returns a
        # float or None if infinite. With no local values contributes -inf.
        values = self.stats.get(key, [])
        if not values:
            effective_group = self._effective_reduce_group(
                key, reduce_group, sync_metadata, key_sync_group
            )
            x = self._placeholder_scalar(fill=-float("inf"), group=effective_group)
            if effective_group is not None:
                dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MAX)
            if torch.isinf(x):
                return None
            return float(x)

        denominator = self._denominator_of(key, sync_metadata)
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = torch.stack(xs).max()
        effective_group = self._effective_reduce_group(
            key, reduce_group, sync_metadata, key_sync_group
        )
        if effective_group is not None:
            dist.all_reduce(x, group=effective_group, op=dist.ReduceOp.MAX)
        if torch.isinf(x):
            return None
        return float(x)


DEFAULT_TRACKER = DistributedStatsTracker()
stat = DEFAULT_TRACKER.stat
denominator = DEFAULT_TRACKER.denominator
export = DEFAULT_TRACKER.export
scope = DEFAULT_TRACKER.scope
scalar = DEFAULT_TRACKER.scalar
record_timing = DEFAULT_TRACKER.record_timing
scope_func_wrapper = DEFAULT_TRACKER.scope_func_wrapper

TRACKERS = {"": DEFAULT_TRACKER}
LOCK = Lock()


def get(name: str = ""):
    # [FUNCTION DOC]
    # Returns the named tracker from the global registry, creating a
    # DistributedStatsTracker(name) under a global lock if absent. "" is the default.
    global TRACKERS, LOCK
    with LOCK:
        if name not in TRACKERS:
            TRACKERS[name] = DistributedStatsTracker(name)
        return TRACKERS[name]


def export_all(reduce_group=None, key_sync_group=None, reset=True) -> dict[str, float]:
    # [FUNCTION DOC]
    # Exports every registered tracker and merges results into one dict. If a sync group
    # (key_sync_group, else reduce_group) is given, all-gathers tracker names over it so
    # every rank exports the same sorted set (creating missing trackers). Passes
    # reduce_group, key_sync_group and reset to each export. Logs a warning listing
    # duplicate output keys (later trackers win).
    stat = {}
    duplicate_keys = set()
    tracker_keys = list(TRACKERS.keys())
    sync_group = key_sync_group if key_sync_group is not None else reduce_group
    if sync_group is not None:
        all_trackers = [None for _ in range(dist.get_world_size(sync_group))]
        dist.all_gather_object(all_trackers, list(TRACKERS.keys()), group=sync_group)
        tracker_keys = sorted(list(set(flat2d(all_trackers))))
    for tracker_key in tracker_keys:
        tracker = get(tracker_key)
        x = tracker.export(
            reduce_group=reduce_group,
            key_sync_group=key_sync_group,
            reset=reset,
        )
        for k in x.keys():
            if k in stat:
                duplicate_keys.add(k)
        stat.update(x)
    if duplicate_keys:
        logger.warning(f"Duplicate stat keys detected: {list(duplicate_keys)}")
    return stat
