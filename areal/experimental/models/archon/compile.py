# [MODULE DOC]
# torch.compile support for Archon models: compiles each transformer block separately
# (Inductor, fullgraph) rather than the whole model. Logs via a cached rank-aware logger.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn as nn

from areal.utils import logging


@functools.cache
def _get_logger() -> logging.Logger:
    # [FUNCTION DOC]
    # HELPER: Cached; returns a logger named "[Archon Compile Rank N]" (rank 0 if dist is not
    # initialized).
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon Compile Rank {rank}]")


class Compilable(Protocol):
    # [CLASS DOC]
    # Structural Protocol: any model exposing a "layers" nn.ModuleDict of blocks.
    """Protocol for models that can be compiled with apply_compile."""

    layers: nn.ModuleDict


def apply_compile(model: Compilable) -> None:
    # [FUNCTION DOC]
    # Replaces every entry of model.layers in place with torch.compile(block, backend
    # "inductor", fullgraph True) and logs the count. Should run after TP and AC and before
    # FSDP wrapping. Returns None; compilation itself is lazy (on first call).
    """Apply torch.compile to each TransformerBlock.

    Compiling per-block is more efficient than whole model due to:
    1. Repeated structure allows compilation reuse
    2. Avoids graph breaks from embedding/output layers

    Must be called AFTER TP and AC, BEFORE FSDP.

    Args:
        model: The model to compile. Must have a `layers` attribute (ModuleDict).
    """
    for name, block in model.layers.items():
        model.layers[name] = torch.compile(
            block,
            backend="inductor",
            fullgraph=True,
        )

    _get_logger().info(
        f"Compiled {len(model.layers)} TransformerBlocks with torch.compile"
    )
