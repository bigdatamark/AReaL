# [MODULE DOC]
# Partial rotary position embedding (RoPE) utilities for Qwen3.5: only the first
# rotary_dim = int(head_dim * partial_rotary_factor) channels of each head are rotated
# (HF "rotate_half" convention); remaining channels pass through unchanged. The cache
# layout is [max_seq_len, 2 * rotary_dim] with cos in the first half, sin in the second.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributed.tensor import DTensor


def _maybe_to_local(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns x.to_local() for DTensors, else x.
    """Convert DTensor to local tensor if needed."""
    if isinstance(x, DTensor):
        return x.to_local()
    return x


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    partial_rotary_factor: float = 1.0,
    base: float = 10_000.0,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Computes inverse frequencies base^(-2i/rotary_dim) for i < rotary_dim/2, outer
    # product with positions 0..max_seq_len-1, duplicates the angles to width
    # rotary_dim,
    # and returns cos and sin concatenated along the last dim: float32 tensor
    # [max_seq_len, 2 * rotary_dim]. CPU tensor unless a device context is active.
    """Precompute partial RoPE cos/sin cache.

    Args:
        head_dim: Full head dimension.
        max_seq_len: Maximum sequence length.
        partial_rotary_factor: Fraction of head_dim for RoPE (e.g. 0.25).
        base: RoPE base frequency.

    Returns:
        RoPE cache tensor of shape ``[max_seq_len, rotary_dim * 2]`` where
        ``rotary_dim = int(head_dim * partial_rotary_factor)``.
        First half is cos, second half is sin.
    """
    rotary_dim = int(head_dim * partial_rotary_factor)
    freqs = 1.0 / (
        base
        ** (torch.arange(0, rotary_dim, 2)[: (rotary_dim // 2)].float() / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=freqs.dtype)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Splits the last dim in halves (x1, x2) and returns concat(-x2, x1).
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
    rotary_dim: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Selects RoPE rows for x of shape [bz, seqlen, heads, head_dim] and returns
    # them shaped for broadcasting: positions None -> first seqlen rows as
    # [1, seqlen, 1, 2*rotary_dim]; positions with batch 1 -> rows indexed by positions
    # in
    # the same shape; otherwise gathers per-batch rows to [bz, seqlen, 1, 2*rotary_dim].
    # rotary_dim defaults to half the cache width.
    """Reshape RoPE cache for broadcasting with input tensor.

    Args:
        rope_cache: RoPE tensor of shape ``[max_seqlen, rotary_dim * 2]``.
        x: Target tensor, shape ``[bsz, seq_len, num_heads, head_dim]``.
        positions: Position indices, shape ``(1, seqlen)`` or ``(bz, seqlen)``.
        rotary_dim: The rotary dimension (may differ from head_dim for partial RoPE).
            If *None*, falls back to ``rope_cache.shape[-1] // 2``.

    Returns:
        Reshaped RoPE tensor suitable for broadcasting.
    """
    if rotary_dim is None:
        rotary_dim = rope_cache.shape[-1] // 2
    bz, seqlen, _, _ = x.shape
    rope_width = rotary_dim * 2  # cos + sin

    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        return rope_cache.view(-1, seqlen, 1, rope_width)
    elif positions.size(0) == 1:
        rope_cache = rope_cache[positions.squeeze(0)]
        return rope_cache.view(-1, seqlen, 1, rope_width)
    else:
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, rope_width),
        )
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Applies partial RoPE to xq [b, s, heads, head_dim] and xk [b, s, kv_heads,
    # head_dim]. Converts DTensor cache/positions to local, derives rotary_dim from the
    # cache width, gathers cos/sin for the positions, casts to xq's dtype/device,
    # rotates
    # the first rotary_dim channels (x*cos + rotate_half(x)*sin) and concatenates the
    # untouched rest. Returns (q, k) with input shapes and dtypes.
    """Apply partial rotary position embedding to query and key tensors.

    Only the first ``rotary_dim`` dimensions participate in rotation;
    the remaining dimensions pass through unchanged (bit-exact).

    Args:
        xq: Query tensor, shape ``[bsz, seq_len, num_heads, head_dim]``.
        xk: Key tensor, shape ``[bsz, seq_len, num_kv_heads, head_dim]``.
        rope_cache: RoPE cache, shape ``[max_seq_len, rotary_dim * 2]``.
        positions: Position indices.

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.
    """
    rope_cache = _maybe_to_local(rope_cache)
    if positions is not None:
        positions = _maybe_to_local(positions)

    rotary_dim = rope_cache.shape[-1] // 2
    rope_cache = reshape_for_broadcast(rope_cache, xq, positions, rotary_dim)

    cos = rope_cache[..., :rotary_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., rotary_dim:].to(dtype=xq.dtype, device=xq.device)

    # Split: only first rotary_dim dims participate in rotation
    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    xq_out = torch.cat([(xq_rot * cos) + (rotate_half(xq_rot) * sin), xq_pass], dim=-1)
    xk_out = torch.cat([(xk_rot * cos) + (rotate_half(xk_rot) * sin), xk_pass], dim=-1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # Repeats each KV head n_rep times consecutively (head-interleaved expand+reshape),
    # [bs, slen, n_kv, d] -> [bs, slen, n_kv*n_rep, d]; returns x unchanged if n_rep ==
    # 1.
    """Repeat KV heads to match number of query heads.

    Args:
        x: Input tensor of shape ``[bs, slen, n_kv_heads, head_dim]``.
        n_rep: Number of repetitions.

    Returns:
        Tensor of shape ``[bs, slen, n_kv_heads * n_rep, head_dim]``.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


__all__ = [
    "precompute_rope_cache",
    "rotate_half",
    "reshape_for_broadcast",
    "apply_rotary_emb",
    "repeat_kv",
]
