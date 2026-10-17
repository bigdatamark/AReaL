# [MODULE DOC]
# Rotary position embedding helpers for Qwen3. Pure re-export of the qwen2
# implementation (precompute_rope_cache, rotate_half, reshape_for_broadcast,
# apply_rotary_emb, repeat_kv), since Qwen3 RoPE is identical. No definitions here.
# SPDX-License-Identifier: Apache-2.0

# Re-export from qwen2 - implementation is identical
from areal.experimental.models.archon.qwen2.model.rope import (
    apply_rotary_emb,
    precompute_rope_cache,
    repeat_kv,
    reshape_for_broadcast,
    rotate_half,
)

__all__ = [
    "precompute_rope_cache",
    "rotate_half",
    "reshape_for_broadcast",
    "apply_rotary_emb",
    "repeat_kv",
]
