# [MODULE DOC]
# Archon implementation of the Qwen3.5 hybrid transformer for training on packed
# (varlen) batches. Each layer is either a gated full-attention layer (GatedAttention,
# flash varlen attention with partial RoPE and a sigmoid output gate) or a GatedDeltaNet
# linear-attention layer, followed by a dense SwiGLU FFN or MoE. Optional GPU kernels:
# causal_conv1d (depthwise causal conv) and flash-linear-attention (fla: chunked gated
# delta rule, fused gated RMSNorm); they are imported best-effort and set to None when
# missing (the delta-rule kernel is then mandatory at runtime). Designed to match the
# HuggingFace Qwen3.5 implementation numerically.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from areal.experimental.models.archon.qwen3_5.model.args import Qwen3_5ModelArgs
from areal.experimental.models.archon.attention import VarlenAttentionWrapper
from areal.experimental.models.archon.base import BaseArchonModel
from areal.experimental.models.archon.moe import MoE
from areal.experimental.models.archon.qwen3_5.model.rope import (
    apply_rotary_emb,
    precompute_rope_cache,
    repeat_kv,
)

# Optional GPU dependencies (fla, causal-conv1d).
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None
    FusedRMSNormGated = None


class Qwen3_5RMSNorm(nn.Module):
    # [CLASS DOC]
    # RMSNorm with "1-centered" weight: output = norm(x) * (1 + weight), weight starts
    # at zero; computed in fp32 and cast back. Used for block norms, final norm and Q/K
    # norm.
    """RMSNorm with ``(1 + weight) * norm(x)`` semantics.

    Used for: ``input_layernorm``, ``post_attention_layernorm``, final
    ``norm``, ``q_norm``, ``k_norm``.

    Weight is initialized to **zero** so the effective multiplier starts
    at 1.0 ("1-centered" design).  Computation is done in fp32 and cast
    back to the input dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # [FUNCTION DOC]
        # Creates a zero-initialized weight of size hidden_size and stores eps.
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Normalizes over the last dim in fp32, scales by (1 + weight) and casts to the
        # input
        # dtype. Same shape as x.
        input_dtype = x.dtype
        x_fp32 = x.float()
        output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).to(input_dtype)

    def reset_parameters(self):
        # [FUNCTION DOC]
        # Zeroes the weight.
        nn.init.zeros_(self.weight)


class Qwen3_5RMSNormGated(nn.Module):
    # [CLASS DOC]
    # Pure-PyTorch per-head gated RMSNorm used inside GatedDeltaNet when the fla fused
    # kernel is unavailable: weight * norm(x) * silu(gate). Inputs must be flattened to
    # [-1, head_v_dim]. Weight initialized to ones.
    """Gated RMSNorm: ``weight * norm(x) * silu(gate)``.

    Used **only** inside GatedDeltaNet layers.  The norm dimension is
    ``head_v_dim`` (per-head), NOT ``num_v_heads * head_v_dim``.

    Weight is initialized to **one** (standard ``nn.Parameter(torch.ones(…))``).
    Both ``hidden_states`` and ``gate`` must be reshaped to
    ``(-1, head_v_dim)`` before this norm, then reshaped back after.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # [FUNCTION DOC]
        # Creates a ones-initialized weight of size hidden_size and stores eps.
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Normalizes hidden_states in fp32, casts to input dtype and multiplies by
        # weight;
        # if gate is given multiplies by silu(gate) computed in fp32; returns input
        # dtype.
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def reset_parameters(self):
        # [FUNCTION DOC]
        # Fills weight with ones.
        nn.init.ones_(self.weight)


def cu_seqlens_to_seq_idx(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # Converts cumulative sequence lengths [N+1] to a per-token int32 sequence index of
    # length sum(lengths) via repeat_interleave (e.g. [0,3,5] -> [0,0,0,1,1]), on
    # cu_seqlens' device. total_len is accepted but not used for validation.
    """Convert ``cu_seqlens`` to ``seq_idx`` for ``causal_conv1d_fn``.

    Args:
        cu_seqlens: Cumulative sequence lengths of shape ``[N + 1]``
            (e.g. ``[0, 3, 5, 8]``).
        total_len: Total packed sequence length (should equal
            ``cu_seqlens[-1]``).

    Returns:
        Integer tensor of shape ``[total_len]`` mapping each position to
        its sequence index (e.g. ``[0, 0, 0, 1, 1, 2, 2, 2]``).
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lengths), device=cu_seqlens.device, dtype=torch.int32),
        lengths.to(torch.int64),
    )
    return seq_idx


def compute_decay_beta(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Returns (beta, g) for the gated delta rule: beta = sigmoid(b) and
    # g = -exp(A_log) * softplus(a + dt_bias), with A_log and a in fp32. Shapes [B, T,
    # H].
    """Compute decay (*g*) and beta for GatedDeltaNet.

    Args:
        A_log: Log-space decay base, shape ``[num_v_heads]``.
        dt_bias: Time-step bias, shape ``[num_v_heads]``.
        a: Decay input from ``in_proj_a``, shape ``[B, T, num_v_heads]``.
        b: Beta input from ``in_proj_b``, shape ``[B, T, num_v_heads]``.

    Returns:
        Tuple ``(beta, g)`` where:

        - ``beta = sigmoid(b)`` in ``(0, 1)``, shape ``[B, T, num_v_heads]``.
        - ``g = -exp(A_log) * softplus(a + dt_bias)`` < 0,
          shape ``[B, T, num_v_heads]``.
    """
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return beta, g


class GatedDeltaNet(nn.Module):
    # [CLASS DOC]
    # Training-only GatedDeltaNet linear-attention layer (no recurrent cache). Projects
    # hidden states to a combined QKV stream that passes through a depthwise causal conv
    # with SiLU, splits into multi-head Q/K/V, computes per-head decay g and write-
    # strength
    # beta, runs fla's chunked gated delta rule (with in-kernel Q/K L2 norm) respecting
    # packing boundaries, applies per-head gated RMSNorm with gate z, and projects out.
    """GatedDeltaNet for ``linear_attention`` layers in Qwen3.5.

    Training-only module (no KV-cache / decode paths).  Uses:

    - ``causal_conv1d_fn`` for depthwise causal convolution
      (packing isolation via ``seq_idx``).
    - ``chunk_gated_delta_rule`` from fla for the gated delta rule
      kernel (packing isolation via ``cu_seqlens``).
    - ``FusedRMSNormGated`` (fla) or ``Qwen3_5RMSNormGated``
      (pure-PyTorch fallback) for per-head gated normalisation.

    Produces numerically identical output to the HuggingFace
    ``Qwen3_5GatedDeltaNet`` when weights are shared.
    """

    def __init__(self, args: Qwen3_5ModelArgs, layer_idx: int) -> None:
        # [FUNCTION DOC]
        # Reads head counts/dims from args, computes key_dim, value_dim and conv_dim
        # (2*key_dim + value_dim). Creates a depthwise bias-free Conv1d with causal
        # padding
        # kernel-1, bare dt_bias (ones) and A_log (log of uniform(0,16)) parameters, the
        # gated
        # norm (fla FusedRMSNormGated with silu if available, else Qwen3_5RMSNormGated),
        # and
        # bias-free projections in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and
        # out_proj.
        super().__init__()
        self.hidden_size = args.dim
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        # Combined QKV conv dimension.
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Causal depthwise Conv1d on combined QKV.
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )

        # Bare parameters (no ``.weight`` suffix in state-dict keys).
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm — fused kernel when available, else pure-PyTorch.
        self.norm = (
            Qwen3_5RMSNormGated(self.head_v_dim, eps=args.norm_eps)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim, eps=args.norm_eps, activation="silu"
            )
        )

        # Linear projections.
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # x: [B, T, dim]. If cu_seqlens is None uses [0, T]; if seq_idx is None derives
        # it.
        # Projects qkv, z, a, b. Causal conv over time: with causal_conv1d_fn, passes
        # squeezed
        # conv weights, silu activation and seq_idx (unsqueezed to [1, T]) to isolate
        # packed
        # sequences; otherwise falls back to nn.Conv1d + SiLU per packed segment (using
        # .item(),
        # a GPU sync) or on the whole sequence, truncating padding. Splits into Q, K
        # (num_k_heads
        # x head_k_dim) and V (num_v_heads x head_v_dim), computes beta/g, repeat-
        # interleaves
        # Q/K heads when num_v_heads > num_k_heads, asserts the fla kernel is installed
        # (AssertionError otherwise) and runs chunk_gated_delta_rule with cu_seqlens.
        # Applies
        # gated norm per head with z, reshapes to [B, T, value_dim] and applies
        # out_proj.
        # Returns [B, T, dim].
        # KNOWN ISSUE: the fallback conv path slices along time per segment but ignores
        # batch,
        # so it assumes B == 1 for packed inputs (as do the varlen kernels).
        """Forward pass.

        Args:
            x: Hidden states ``[B, T, dim]``.
            cu_seqlens: Cumulative sequence lengths ``[N+1]`` for packing.
                Forwarded to ``chunk_gated_delta_rule``.  Synthesised as
                ``[0, T]`` when *None* so both kernels always take the
                varlen code path.
            seq_idx: Per-position sequence index ``[T]`` (or ``[B, T]``).
                Forwarded to ``causal_conv1d_fn``.  Derived from
                *cu_seqlens* when *None*.

        Returns:
            Output tensor ``[B, T, dim]``.
        """
        batch_size, seq_len, _ = x.shape

        # Normalise packing args: always use varlen mode so that
        # standalone and packed calls take identical kernel code paths.
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device=x.device)
        if seq_idx is None:
            seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, seq_len)
        # 1. Input projections.
        mixed_qkv = self.in_proj_qkv(x)  # [B, T, conv_dim]
        z = self.in_proj_z(x)  # [B, T, value_dim]
        a = self.in_proj_a(x)  # [B, T, num_v_heads]
        b = self.in_proj_b(x)  # [B, T, num_v_heads]

        # 2. Causal convolution on combined QKV.
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, T]
        if causal_conv1d_fn is not None:
            _seq_idx = seq_idx
            if _seq_idx is not None and _seq_idx.dim() == 1:
                _seq_idx = _seq_idx.unsqueeze(0)  # [T] -> [1, T]
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                bias=None,
                activation="silu",
                seq_idx=_seq_idx,
            )
        else:
            # Fallback: nn.Conv1d + SiLU.
            # When packing multiple sequences, process each separately to
            # avoid cross-sequence contamination through the conv kernel.
            if cu_seqlens.numel() > 2:
                segments = []
                for i in range(cu_seqlens.numel() - 1):
                    s = cu_seqlens[i].item()
                    e = cu_seqlens[i + 1].item()
                    seg = F.silu(self.conv1d(mixed_qkv[:, :, s:e])[:, :, : e - s])
                    segments.append(seg)
                mixed_qkv = torch.cat(segments, dim=2)
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, T, conv_dim]

        # 3. Split into Q, K, V and reshape to multi-head.
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # 4. Compute decay (g) and beta.
        beta, g = compute_decay_beta(self.A_log, self.dt_bias, a, b)

        # 5. Head grouping: expand Q/K when num_v_heads > num_k_heads.
        if self.num_v_heads > self.num_k_heads:
            repeats = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)

        # 6. Gated delta rule (chunk-based training kernel).
        assert chunk_gated_delta_rule is not None, (
            "fla.ops.gated_delta_rule.chunk_gated_delta_rule is required. "
            "Install: pip install flash-linear-attention"
        )
        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )

        # 7. Per-head gated norm.
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 8. Output projection.
        return self.out_proj(core_attn_out)

    def init_weights(self) -> None:
        # [FUNCTION DOC]
        # Resets dt_bias to ones and A_log to log(uniform(0,16)) in place. Projections,
        # conv
        # and norm are not re-initialized.
        """Re-initialise bare parameters (for from-scratch training)."""
        nn.init.ones_(self.dt_bias)
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(0, 16).log_())


# ---------------------------------------------------------------------------
# GatedAttention (full_attention layers)
# ---------------------------------------------------------------------------


class GatedAttention(nn.Module):
    # [CLASS DOC]
    # Gated GQA full attention: wq produces 2x width per head that is split into query
    # and an output gate; after flash varlen attention the output is multiplied by
    # sigmoid(gate) before wo. Uses Qwen3_5RMSNorm for Q/K norm and partial RoPE.
    """Gated multi-head attention for Qwen3.5 full_attention layers.

    Key difference from standard attention: ``q_proj`` outputs 2x width,
    split into ``query`` and ``gate``.  After attention computation:
    ``output = attn_output * sigmoid(gate)``.

    Uses :class:`VarlenAttentionWrapper` (flash attention) for packed
    sequences via ``cu_seqlens``.
    """

    def __init__(self, args: Qwen3_5ModelArgs):
        # [FUNCTION DOC]
        # Stores head counts, n_rep, head_dim and scaling head_dim^-0.5; builds wq (dim
        # ->
        # 2*n_heads*head_dim), wk, wv, wo (bias per attention_bias), q_norm, k_norm and
        # a
        # VarlenAttentionWrapper.
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scaling = self.head_dim**-0.5

        # Q outputs 2x: query + gate.
        self.wq = nn.Linear(
            args.dim, args.n_heads * self.head_dim * 2, bias=args.attention_bias
        )
        self.wk = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.wv = nn.Linear(
            args.dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim, args.dim, bias=args.attention_bias
        )

        # Q/K norm — Qwen3.5 RMSNorm with (1+weight) semantics.
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=args.norm_eps)

        self.packed_attn = VarlenAttentionWrapper()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # x [B, T, dim]. Splits wq output per head into query and gate halves (gate
        # flattened
        # to [B, T, n_heads*head_dim]), projects K/V, applies Q/K norm and partial RoPE
        # at
        # positions, repeats K/V heads for GQA, transposes to [B, H, T, D], runs varlen
        # flash
        # attention with cu_seqlens/max_seqlen, transposes back and flattens, multiplies
        # by
        # sigmoid(gate), applies wo. Returns [B, T, dim].
        """Forward pass.

        Args:
            x: Hidden states ``[B, T, dim]``.
            rope_cache: Precomputed partial RoPE cache.
            positions: Position indices for RoPE.
            cu_seqlens: Cumulative sequence lengths ``[N+1]``.
            max_seqlen: Maximum sequence length (for flash attention).

        Returns:
            Output tensor ``[B, T, dim]``.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Q projection (2x) → split into query + gate.
        qg = self.wq(x).view(batch_size, seq_len, -1, self.head_dim * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)  # [B, T, n_heads * head_dim]

        # 2. K, V projections.
        key = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        value = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 3. Q/K norm.
        query = self.q_norm(query)
        key = self.k_norm(key)

        # 4. Partial RoPE.
        query, key = apply_rotary_emb(query, key, rope_cache, positions)

        # 5. GQA expansion.
        if self.n_rep > 1:
            key = repeat_kv(key, self.n_rep)
            value = repeat_kv(value, self.n_rep)

        # 6. Transpose to [B, H, T, D] for flash attention.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 7. Flash attention via VarlenAttentionWrapper.
        attn_output = self.packed_attn(
            query,
            key,
            value,
            scale=self.scaling,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # 8. Transpose back and reshape.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # 9. Output gate: attn_output * sigmoid(gate).
        attn_output = attn_output * torch.sigmoid(gate)

        # 10. Output projection.
        return self.wo(attn_output)

    def init_weights(self, init_std: float) -> None:
        # [FUNCTION DOC]
        # Truncated-normal init: wq/wk/wv std 0.02, wo std init_std. Biases and Q/K
        # norms
        # are left untouched.
        """Initialise weights (for from-scratch training)."""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


# ---------------------------------------------------------------------------
# FeedForward, TransformerBlock, Qwen3_5Model
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    # [CLASS DOC]
    # SwiGLU MLP: w2(silu(w1 x) * w3 x), bias-free.
    """SwiGLU feedforward module."""

    def __init__(self, dim: int, hidden_dim: int):
        # [FUNCTION DOC]
        # Creates w1 and w3 (dim->hidden_dim) and w2 (hidden_dim->dim).
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns w2(silu(w1(x)) * w3(x)).
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        # [FUNCTION DOC]
        # Truncated-normal init: w1 std 0.02, w2/w3 std init_std.
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    # [CLASS DOC]
    # Pre-norm hybrid block: residual GatedAttention (for "full_attention") or
    # GatedDeltaNet (otherwise), then residual MoE or dense FFN. The unused mixer
    # attribute
    # is None.
    """Hybrid transformer block — dispatches to GatedAttention or GatedDeltaNet."""

    def __init__(self, layer_id: int, model_args: Qwen3_5ModelArgs):
        # [FUNCTION DOC]
        # Picks layer_type = model_args.layer_types[layer_id] and builds the matching
        # mixer,
        # the two Qwen3_5RMSNorms, MoE (if moe_enabled and moe_args present, every
        # layer) or
        # FeedForward, and depth-scaled weight_init_std as in Qwen3.
        # KNOWN ISSUE: layer_types is documented as optional (empty default), but an
        # empty
        # list raises IndexError here.
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.layer_type = model_args.layer_types[layer_id]

        # Hybrid: full_attention → GatedAttention, linear_attention → GatedDeltaNet.
        if self.layer_type == "full_attention":
            self.attention = GatedAttention(model_args)
            self.linear_attn = None
        else:
            self.attention = None
            self.linear_attn = GatedDeltaNet(model_args, layer_idx=layer_id)

        self.attention_norm = Qwen3_5RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = Qwen3_5RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # MoE vs dense FeedForward.
        self.moe_enabled = model_args.moe_enabled and model_args.moe_args is not None
        if self.moe_enabled:
            assert model_args.moe_args is not None
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
            self.feed_forward = None
        else:
            self.moe = None
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        seq_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Applies the residual mixer (full attention gets rope_cache, positions,
        # cu_seqlens,
        # max_seqlen; linear attention gets cu_seqlens and seq_idx), then residual
        # MoE/FFN.
        # Attention (hybrid dispatch).
        if self.layer_type == "full_attention":
            x = x + self.attention(
                self.attention_norm(x),
                rope_cache,
                positions,
                cu_seqlens,
                max_seqlen,
            )
        else:
            x = x + self.linear_attn(
                self.attention_norm(x),
                cu_seqlens=cu_seqlens,
                seq_idx=seq_idx,
            )

        # FeedForward (MoE or dense).
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self):
        # [FUNCTION DOC]
        # Resets both norms, initializes the active mixer and the MoE/FFN.
        """Initialize layer parameters."""
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        if self.layer_type == "full_attention":
            self.attention.init_weights(self.weight_init_std)
        else:
            self.linear_attn.init_weights()
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

    def init_buffers(self, buffer_device: torch.device | str):
        # [FUNCTION DOC]
        # Initializes MoE buffers on buffer_device if MoE is enabled.
        """Initialize layer buffers (MoE buffers if enabled)."""
        if self.moe_enabled:
            self.moe.init_buffers(buffer_device)


class Qwen3_5Model(BaseArchonModel):
    # [CLASS DOC]
    # Full Qwen3.5 model: embedding, non-persistent partial-RoPE cache, ModuleDict of
    # hybrid blocks, final norm, and LM head or critic score head. Pipeline splitting
    # may
    # set submodules to None.
    """Qwen3.5 hybrid transformer model."""

    def __init__(self, model_args: Qwen3_5ModelArgs):
        # [FUNCTION DOC]
        # Stores args, builds embedding, rope cache buffer, n_layers blocks, final norm
        # and
        # output/score head depending on is_critic.
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim
        self.is_critic = model_args.is_critic

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = Qwen3_5RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.is_critic:
            self.output = None
            self.score = nn.Linear(model_args.dim, model_args.num_labels, bias=False)
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
            self.score = None

    def _precompute_rope_cache(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Returns precompute_rope_cache(head_dim, max_seq_len,
        # partial_rotary_factor,
        # rope_theta).
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.partial_rotary_factor,
            self.model_args.rope_theta,
        )

    def init_weights(self):
        # [FUNCTION DOC]
        # Same scheme as Qwen3Model: embedding normal, per-layer init, norm reset, heads
        # truncated normal std dim^-0.5 clipped at +-3 std.
        """Initialize model parameters."""
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        if self.score is not None:
            nn.init.trunc_normal_(
                self.score.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def init_buffers(self, buffer_device: torch.device | str):
        # [FUNCTION DOC]
        # Recomputes rope_cache under the device context and initializes per-layer MoE
        # buffers.
        """Initialize model buffers (rope_cache and MoE buffers)."""
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        for layer in self.layers.values():
            if layer is not None:
                layer.init_buffers(buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        tree_attn_meta: None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Squeezes PP-shaped cu_seqlens [1, B+1], converts tensor max_seqlen via .item()
        # (GPU sync), embeds tokens (or passes hidden states through), computes seq_idx
        # once from
        # cu_seqlens for all linear-attention layers, runs all blocks, final norm, and
        # the score
        # or LM head. tree_attn_meta is accepted for interface compatibility and ignored
        # (tree
        # attention unsupported). Returns logits, values or hidden states.
        # When pipeline parallelism enabled, cu_seqlens is [1, B+1]
        if cu_seqlens.ndim == 2:
            cu_seqlens = cu_seqlens.squeeze(0)

        # When pipeline parallelism enabled, max_seqlen is [1]
        if isinstance(max_seqlen, torch.Tensor):
            max_seqlen = int(max_seqlen.item())

        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Compute seq_idx ONCE for all linear_attention layers.
        seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, h.shape[1])

        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                positions,
                cu_seqlens,
                max_seqlen,
                seq_idx=seq_idx,
            )

        h = self.norm(h) if self.norm else h

        if self.is_critic:
            output = self.score(h) if self.score else h
        else:
            output = self.output(h) if self.output else h
        return output


__all__ = [
    "Qwen3_5RMSNorm",
    "Qwen3_5RMSNormGated",
    "cu_seqlens_to_seq_idx",
    "compute_decay_beta",
    "GatedDeltaNet",
    "GatedAttention",
    "FeedForward",
    "TransformerBlock",
    "Qwen3_5Model",
]
