# [MODULE DOC]
# Tree attention for the Archon engine: per-microbatch metadata (flex-attention
# BlockMask or Triton tree data) built from a prefix trie, and an nn.Module attention
# wrapper with the VarlenAttentionWrapper interface that dispatches to the Triton tree
# kernel or flex_attention.
# SPDX-License-Identifier: Apache-2.0

"""Tree attention module for Archon engine.

This module provides a TreeAttentionWrapper that has the same interface as
VarlenAttentionWrapper but uses tree attention (flex attention or Triton kernel)
instead of standard varlen attention.

The wrapper is designed to be a drop-in replacement for VarlenAttentionWrapper
when tree training is enabled. Set attn_type="tree" in model_args to use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from areal.models.tree_attn.constants import USE_TRITON_TREE_ATTN
from areal.models.tree_attn.tree import (
    build_block_mask_from_trie,
    build_triton_attn_data_from_trie,
)
from areal.models.tree_attn.triton_kernel import TRITON_AVAILABLE, TreeAttentionData
from areal.utils import logging

if TYPE_CHECKING:
    from areal.models.tree_attn.tree import TrieNode

logger = logging.getLogger("TreeAttentionWrapper")

__all__ = [
    "TreeAttentionMeta",
    "TreeAttentionWrapper",
]


@dataclass
class TreeAttentionMeta:
    # [CLASS DOC]
    # Dataclass holding tree attention metadata for one packed microbatch: block_mask
    # (flex attention BlockMask) or triton_data (TreeAttentionData); exactly one must be
    # set.
    """Tree attention metadata for one microbatch.

    Exactly one of block_mask or triton_data must be set,
    depending on the backend (flex attention vs Triton kernel).
    """

    block_mask: BlockMask | None = None
    triton_data: TreeAttentionData | None = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Validates that exactly one of block_mask / triton_data is non-None; raises
        # ValueError otherwise.
        if (self.block_mask is None) == (self.triton_data is None):
            raise ValueError("Exactly one of block_mask or triton_data must be set.")

    @classmethod
    def from_trie(
        cls, trie_node: TrieNode, padded_size: int, device: torch.device
    ) -> TreeAttentionMeta:
        # [FUNCTION DOC]
        # Builds metadata from a TrieNode for a sequence padded to padded_size. Uses the
        # Triton backend (build_triton_attn_data_from_trie) when USE_TRITON_TREE_ATTN is
        # set and Triton is available, else builds a flex attention BlockMask on device.
        # Returns a new instance.
        """Build tree attention metadata from a trie node.

        Automatically selects the Triton kernel or flex attention backend.
        """
        if USE_TRITON_TREE_ATTN and TRITON_AVAILABLE:
            return cls(
                triton_data=build_triton_attn_data_from_trie(trie_node, padded_size)
            )
        return cls(
            block_mask=build_block_mask_from_trie(trie_node, padded_size, device)
        )


class TreeAttentionWrapper(nn.Module):
    # [CLASS DOC]
    # Stateless nn.Module computing tree attention for packed sequences (batch 1).
    # Requires a TreeAttentionMeta via the tree_attn_meta keyword in forward.
    """Attention wrapper for tree training in Archon Engine.

    This wrapper has the same interface as VarlenAttentionWrapper but uses
    tree attention (flex attention with BlockMask or Triton tree attention)
    instead of standard flash attention.

    The wrapper expects a ``tree_attn_meta`` (:class:`TreeAttentionMeta`) to be
    passed as a keyword argument during forward.

    For packed sequences in Archon, batch is always 1.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Inputs q/k/v of shape [1, heads, seq, head_dim] (k/v may have fewer heads for
        # GQA), optional scale; cu_seqlens and max_seqlen are ignored (API
        # compatibility). Asserts batch == 1 and that tree_attn_meta is given. If
        # triton_data is set, calls the Triton tree_attention kernel with its packed
        # mask and kv/q index/offset arrays; otherwise calls (uncompiled) flex_attention
        # with the block mask, enabling GQA when head counts differ. Returns [1, heads,
        # seq, head_dim].
        """Compute tree attention.

        Args:
            q: Query tensor, shape [batch, heads, seq_len, head_dim]
            k: Key tensor, shape [batch, heads, seq_len, head_dim]
            v: Value tensor, shape [batch, heads, seq_len, head_dim]
            scale: Optional scale factor for attention scores.
            cu_seqlens: Cumulative sequence lengths (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            max_seqlen: Maximum sequence length (unused for tree attention,
                kept for API compatibility with VarlenAttentionWrapper).
            tree_attn_meta: Tree attention metadata containing either a
                BlockMask (flex attention) or TreeAttentionData (Triton).

        Returns:
            Attention output, shape [batch, heads, seq_len, head_dim]
        """
        # Input: [batch, heads, seq_len, head_dim]
        batch, n_heads, seq_len, head_dim = q.shape

        # For packed sequences, batch should be 1
        assert batch == 1, (
            f"TreeAttentionWrapper expects batch=1 for packed sequences, "
            f"got batch={batch}"
        )

        assert tree_attn_meta is not None, (
            "TreeAttentionWrapper requires tree_attn_meta. "
            "For standard attention, use VarlenAttentionWrapper instead."
        )

        if tree_attn_meta.triton_data is not None:
            from areal.models.tree_attn.triton_kernel import tree_attention

            triton_data = tree_attn_meta.triton_data
            return tree_attention(
                q,
                k,
                v,
                triton_data.packed_mask,
                triton_data.kv_indices,
                triton_data.kv_offsets,
                triton_data.q_indices,
                triton_data.q_offsets,
                sm_scale=scale,
            )

        assert tree_attn_meta.block_mask is not None
        enable_gqa = q.shape[1] != k.shape[1]
        return flex_attention(
            q,
            k,
            v,
            block_mask=tree_attn_meta.block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
        )
