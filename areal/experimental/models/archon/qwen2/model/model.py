# [MODULE DOC]
# Archon Qwen2 decoder (adapted from torchtitan's Qwen3): RMSNorm, GQA attention with
# QKV bias, RoPE, pluggable packed attention backend (varlen / sdpa / tree) and optional
# Ulysses sequence parallelism, SwiGLU MLP, pre-norm blocks, and the top-level model
# with an LM head or a critic score head. Inputs are packed sequences with batch 1.
# Modules not owned by a pipeline stage may be None (forward tolerates this).
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/model.py

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup

from areal.experimental.models.archon.attention import (
    SDPAWrapper,
    TreeAttentionMeta,
    TreeAttentionWrapper,
    VarlenAttentionWrapper,
)
from areal.experimental.models.archon.base import BaseArchonModel
from areal.experimental.models.archon.qwen2.model.args import Qwen2ModelArgs
from areal.experimental.models.archon.qwen2.model.rope import (
    apply_rotary_emb,
    precompute_rope_cache,
    repeat_kv,
)
from areal.experimental.models.archon.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)


class RMSNorm(nn.Module):
    # [CLASS DOC]
    # RMS normalization with a learnable scale (init ones) computed in float32.
    """RMSNorm with float32 intermediate computation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # [FUNCTION DOC]
        # Creates the weight [hidden_size] of ones and stores eps.
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Upcasts to float32, divides by sqrt(mean of squares over last dim + eps), casts back to
        # the input dtype, multiplies by weight. Returns same shape.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

    def reset_parameters(self):
        # [FUNCTION DOC]
        # Resets weight to ones in place.
        nn.init.ones_(self.weight)


class Attention(nn.Module):
    # [CLASS DOC]
    # GQA self-attention: wq/wk/wv Linear (bias = attention_bias) to n_heads / n_kv_heads
    # times head_dim, wo without bias; scale head_dim^-0.5. Backend: TreeAttentionWrapper for
    # attn_type "tree", VarlenAttentionWrapper for "varlen", SDPAWrapper otherwise. Holds
    # Ulysses state (cp group, size, rank), set by set_cp_group.
    """Multi-head attention module with GQA and Ulysses SP support.

    Ulysses SP (Sequence Parallelism) uses All-to-All communication to split
    attention heads across GPUs while keeping the full sequence on each GPU.
    This is different from Ring Attention which splits the sequence.
    """

    def __init__(self, model_args: Qwen2ModelArgs):
        # [FUNCTION DOC]
        # Builds projections and the attention backend; Ulysses disabled initially. n_kv_heads
        # None falls back to n_heads.
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5

        # Linear projections
        attention_bias = model_args.attention_bias
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=attention_bias
        )
        self.wk = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=attention_bias
        )
        self.wv = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=attention_bias
        )
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        # Select attention backend
        if model_args.attn_type == "tree":
            self.packed_attn = TreeAttentionWrapper()
        elif model_args.attn_type == "varlen":
            self.packed_attn = VarlenAttentionWrapper()
        else:
            self.packed_attn = SDPAWrapper()

        # Ulysses SP state (set by parallelize_fn via set_cp_group)
        self._cp_group: ProcessGroup | None = None
        self._cp_size: int = 1
        self._cp_rank: int = 0

    def set_cp_group(self, cp_group: ProcessGroup | None):
        # [FUNCTION DOC]
        # Stores the CP group and caches its world size and this rank's index (1/0 when None).
        """Configure Ulysses sequence parallelism.

        Args:
            cp_group: Process group for Ulysses All-to-All communication.
                     If None or size 1, SP is disabled.
        """
        self._cp_group = cp_group
        if cp_group is not None:
            self._cp_size = dist.get_world_size(cp_group)
            self._cp_rank = dist.get_rank(cp_group)
        else:
            self._cp_size = 1
            self._cp_rank = 0

    @property
    def _sp_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: a CP group is set and its size > 1.
        """Check if Ulysses SP is enabled."""
        return self._cp_group is not None and self._cp_size > 1

    def init_weights(self, init_std: float):
        # [FUNCTION DOC]
        # Truncated-normal (std 0.02) for wq/wk/wv with zeroed biases; wo with init_std.
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # x [bs, seq, dim] -> projections viewed as [bs, seq, heads, head_dim], RoPE applied at
        # positions. With Ulysses: if kv heads < cp size, repeat KV heads by cp_size // kv_heads;
        # then all-to-all each of q/k/v from sequence-sharded to head-sharded (full sequence of
        # length seq * cp). Transposes to [bs, heads, seq, d], calls the backend with scale,
        # cu_seqlens, max_seqlen and tree meta, transposes back; with Ulysses, all-to-all back to
        # sequence-sharded. Flattens heads and applies wo. Returns [bs, seq, dim].
        # KNOWN ISSUE: If cp_size is not a multiple of kv_heads, the KV repeat factor is floored
        # and the head scatter would not divide evenly.
        bs, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # [Optional] Ulysses All-to-All
        if self._sp_enabled:
            # Repeat kv heads if needed for gather_seq_scatter_heads
            kv_heads = xk.size(2)
            if kv_heads < self._cp_size:
                repeats = self._cp_size // kv_heads
                xk = repeat_kv(xk, repeats)
                xv = repeat_kv(xv, repeats)

            xq = gather_seq_scatter_heads(
                xq,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )
            xk = gather_seq_scatter_heads(
                xk,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )
            xv = gather_seq_scatter_heads(
                xv,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )

            seqlen = xq.shape[1]

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = self.packed_attn(
            xq,
            xk,
            xv,
            scale=self.scaling,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            tree_attn_meta=tree_attn_meta,
        )

        output = output.transpose(1, 2).contiguous()

        # [Optional] Ulysses All-to-All
        if self._sp_enabled:
            output = gather_heads_scatter_seq(
                output, head_dim=2, seq_dim=1, group=self._cp_group
            )
            seqlen = output.shape[1]

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    # [CLASS DOC]
    # SwiGLU MLP: bias-free w1 (gate), w3 (up), w2 (down).
    """SwiGLU feedforward module."""

    def __init__(self, dim: int, hidden_dim: int):
        # [FUNCTION DOC]
        # Creates the three Linear layers.
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Returns w2(silu(w1 x) * w3 x).
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        # [FUNCTION DOC]
        # w1 truncated-normal std 0.02; w2 and w3 with init_std.
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    # [CLASS DOC]
    # Pre-norm block: x + attention(attention_norm(x)), then + feed_forward(ffn_norm(x)).
    # Per-layer init std is 0.02 / sqrt(2*(layer_id+1)) with depth_init, else
    # 0.02 / sqrt(2*n_layers).
    """Pre-norm transformer block with attention and feedforward."""

    def __init__(self, layer_id: int, model_args: Qwen2ModelArgs):
        # [FUNCTION DOC]
        # Builds attention, MLP and both RMSNorms and computes weight_init_std.
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim, hidden_dim=model_args.hidden_dim
        )
        self.attention_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

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
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Applies the two residual sub-layers, passing rope cache, positions, cu_seqlens,
        # max_seqlen and tree meta to attention. Returns [bs, seq, dim].
        x = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            positions,
            cu_seqlens,
            max_seqlen,
            tree_attn_meta=tree_attn_meta,
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self):
        # [FUNCTION DOC]
        # Resets norms and initializes attention and MLP with weight_init_std.
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Qwen2Model(BaseArchonModel):
    # [CLASS DOC]
    # Full model: tok_embeddings, non-persistent rope_cache buffer, layers ModuleDict keyed
    # "0".."n-1", final norm, and either output (LM head, vocab) or score (critic,
    # num_labels); the unused head is None.
    """Qwen2 transformer model."""

    def __init__(self, model_args: Qwen2ModelArgs):
        # [FUNCTION DOC]
        # Stores args and convenience fields, builds all submodules and registers the rope cache
        # computed on the current default device.
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

        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.is_critic:
            self.output = None
            self.score = nn.Linear(model_args.dim, model_args.num_labels, bias=False)
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
            self.score = None

    def _precompute_rope_cache(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: precompute_rope_cache(head_dim, max_seq_len, rope_theta).
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self):
        # [FUNCTION DOC]
        # Embedding normal(0, 1); each present layer's init_weights; norm reset; output/score
        # weights truncated normal with std dim^-0.5 cut at +-3 std. Skips None modules.
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
        # Recomputes rope_cache under the given device context (e.g. after meta init).
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Squeezes a 2-D cu_seqlens and converts a tensor max_seqlen to int (.item(), host sync)
        # for pipeline-parallel calls. Embeds tokens if this part owns tok_embeddings (otherwise
        # input is hidden states), runs all owned layers, applies norm if present, then score
        # (critic) or output if present. Returns logits [1, seq, vocab], values [1, seq, labels],
        # or hidden states for non-last pipeline stages.
        # When pipeline parallelism enabled, cu_seqlens is [1, B+1]
        if cu_seqlens.ndim == 2:
            cu_seqlens = cu_seqlens.squeeze(0)

        # When pipeline parallelism enabled, max_seqlen is [1]
        if isinstance(max_seqlen, torch.Tensor):
            max_seqlen = int(max_seqlen.item())

        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                positions,
                cu_seqlens,
                max_seqlen,
                tree_attn_meta=tree_attn_meta,
            )

        h = self.norm(h) if self.norm else h

        if self.is_critic:
            output = self.score(h) if self.score else h
        else:
            output = self.output(h) if self.output else h
        return output


__all__ = [
    "Qwen2ModelArgs",
    "Attention",
    "FeedForward",
    "TransformerBlock",
    "Qwen2Model",
]
