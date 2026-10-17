# [MODULE DOC]
# Runtime configuration helpers for "separation" (separate-GPU) AdamW delta weight
# transfer implemented by the optional external dte package: world-size validation,
# wire-dtype agreement across ranks, and an env-derived config that lazily creates the
# sender-side DeltaTracker.
# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration for separation AdamW delta transfer.

The delta algorithm itself lives in the standalone ``dte`` package. This
module snapshots the environment propagated to a GPU worker and lazily creates
the sender-side tracker only when the opt-in separation path needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from areal.utils.environ import get_bool_env_var, get_env_var

_DTE_MISSING_MSG = (
    "DTE separation delta transfer requires the 'dte' package "
    "(delta-transfer-engine), which the AWEX adapters import "
    "lazily. Install it with `pip install -e <path>/delta-transfer-engine` "
    "(local dev) or add DTE_SRC to PYTHONPATH."
)

_DTE_TRUTHY_VALUES = ("1", "true", "yes", "on")
_DTE_FALSY_VALUES = ("0", "false", "no", "off")


def validate_dte_world_size(
    combined_world_size: int,
    infer_world_size: int,
    train_world_size: int,
) -> None:
    # [FUNCTION DOC]
    # Returns silently only if both inference and training world sizes are positive, sum
    # to the combined size, and the combined size is a power of two; otherwise raises
    # ValueError (requirement of awex 0.8.0's recursive sparse scheduler).
    """Reject group sizes unsupported by AWEX's recursive sparse scheduler."""
    if (
        infer_world_size > 0
        and train_world_size > 0
        and combined_world_size == infer_world_size + train_world_size
        and (combined_world_size & (combined_world_size - 1)) == 0
    ):
        return
    raise ValueError(
        "DTE separation weight transfer requires a positive power-of-two "
        "combined world size with awex==0.8.0; got "
        f"inference={infer_world_size}, training={train_world_size}, "
        f"combined={combined_world_size}"
    )


def _wire_dtype_from_name(name: str) -> torch.dtype:
    # [FUNCTION DOC]
    # HELPER: Converts a string such as "torch.bfloat16" to the torch dtype; raises
    # ValueError if it lacks the "torch." prefix or does not name a dtype.
    prefix = "torch."
    dtype = (
        getattr(torch, name.removeprefix(prefix), None)
        if name.startswith(prefix)
        else None
    )
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Unsupported DTE wire dtype name gathered from peer: {name!r}"
        )
    return dtype


def synchronize_wire_dtypes(
    transfer_plan: Any,
    process_group: Any,
) -> tuple[torch.dtype, ...]:
    # [FUNCTION DOC]
    # Collects the sorted set of receive-shard dtype names from all operations in
    # transfer_plan, all-gathers them across process_group (collective; every rank must
    # call), and returns the sorted union as a tuple of torch dtypes so all ranks agree
    # on order. Raises ValueError if the union is empty or a name is invalid.
    """Build one stable wire-dtype union shared by every transfer rank."""
    local_dtype_names = sorted(
        {
            str(op.recv_shard_meta.dtype)
            for operations in transfer_plan.operations.values()
            for op in operations
        }
    )
    gathered_dtype_names: list[list[str] | None] = [None] * dist.get_world_size(
        process_group
    )
    dist.all_gather_object(
        gathered_dtype_names,
        local_dtype_names,
        group=process_group,
    )
    canonical_names = sorted(
        {
            name
            for rank_dtype_names in gathered_dtype_names
            for name in (rank_dtype_names or ())
        }
    )
    if not canonical_names:
        raise ValueError(
            "DTE transfer plan has no wire dtypes across the combined group"
        )
    return tuple(_wire_dtype_from_name(name) for name in canonical_names)


@dataclass(frozen=True)
class DTERuntimeConfig:
    # [CLASS DOC]
    # Frozen dataclass of worker-local delta-transfer settings: delta_transfer flag,
    # separation_weight_update flag and anchor_interval (full-weight anchor period).
    """Worker-local runtime settings for separation delta transfer."""

    delta_transfer: bool
    separation_weight_update: bool
    anchor_interval: int

    @classmethod
    def from_env(cls) -> DTERuntimeConfig:
        # [FUNCTION DOC]
        # Classmethod snapshotting env: DTE_DELTA_ANCHOR_INTERVAL (fallback
        # AWEX_DELTA_ANCHOR_INTERVAL, default 0, parsed as int), DTE_DELTA_TRANSFER and
        # DTE_SEPARATION_WEIGHT_UPDATE (AWEX_* fallbacks) parsed as booleans from
        # 1/true/yes/on vs 0/false/no/off with whitespace stripped. Invalid values raise
        # from the env helpers or int parsing.
        """Snapshot DTE runtime settings from the worker environment."""
        anchor_value = get_env_var(
            "DTE_DELTA_ANCHOR_INTERVAL",
            "0",
            fallback_names=("AWEX_DELTA_ANCHOR_INTERVAL",),
        )
        assert anchor_value is not None
        anchor_interval = int(anchor_value)
        return cls(
            delta_transfer=get_bool_env_var(
                "DTE_DELTA_TRANSFER",
                fallback_names=("AWEX_DELTA_TRANSFER",),
                truthy_values=_DTE_TRUTHY_VALUES,
                falsy_values=_DTE_FALSY_VALUES,
                strip_value=True,
            ),
            separation_weight_update=get_bool_env_var(
                "DTE_SEPARATION_WEIGHT_UPDATE",
                fallback_names=("AWEX_SEPARATION_WEIGHT_UPDATE",),
                truthy_values=_DTE_TRUTHY_VALUES,
                falsy_values=_DTE_FALSY_VALUES,
                strip_value=True,
            ),
            anchor_interval=anchor_interval,
        )

    @property
    def enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: true only when both separation_weight_update and delta_transfer are
        # set.
        """Whether sparse separated-card transfer is explicitly enabled."""
        return self.separation_weight_update and self.delta_transfer

    def create_delta_tracker(self):
        # [FUNCTION DOC]
        # Lazily imports dte.core.DeltaTracker (ImportError with install hint if
        # missing) and returns a new tracker with anchor_interval.
        """Create the sender-side tracker without making DTE a base dependency."""
        try:
            from dte.core import DeltaTracker
        except ImportError as e:  # pragma: no cover - only without optional DTE
            raise ImportError(_DTE_MISSING_MSG) from e
        return DeltaTracker(anchor_interval=self.anchor_interval)
