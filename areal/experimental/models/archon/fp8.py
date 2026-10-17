# [MODULE DOC]
# FP8 blockwise training support (via torchao's prototype fp8_blockwise_mm). Patches the
# forward of eligible nn.Linear and GroupedExperts modules on a meta-device model with
# per-instance bound methods that quantize on the fly (no weight conversion), and
# validates after parallelism that local weight dims stay multiples of the 128 block
# size. Logs via "ArchonFP8".
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

from areal.utils.logging import getLogger

logger = getLogger("ArchonFP8")

_FP8_BLOCK = 128


def enable_fp8_linear(
    model: nn.Module,
    *,
    exclude_fqns: set[str] | None = None,
    use_triton: bool = True,
) -> None:
    # [FUNCTION DOC]
    # Imports torchao lazily (ImportError if absent). Walks named_modules; for each nn.Linear
    # that _is_eligible (FQN not containing any exclude substring - default output, router,
    # score - no bias, both dims 128-aligned) patches forward with the FP8 path. Logs the
    # patched/total count. Mutates modules in place.
    """Enable FP8 blockwise matmul for eligible nn.Linear modules.

    Patches the forward of each eligible nn.Linear to use torchao's
    ``fp8_blockwise_mm`` (on-the-fly FP8 quantization, no weight conversion).
    Must be called on a meta-device model, before parallelism is applied.

    Args:
        model: Model residing on the meta device.
        exclude_fqns: FQN substrings to exclude (remain BF16).
            Defaults to ``{"output", "router", "score"}``.
        use_triton: Use Triton GEMM kernel instead of cuBLAS.
    """
    from torchao.prototype.blockwise_fp8_training.linear import fp8_blockwise_mm

    if exclude_fqns is None:
        exclude_fqns = {"output", "router", "score"}

    converted_count = 0
    total_linear = 0
    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        total_linear += 1
        if not _is_eligible(mod, fqn, exclude_fqns):
            continue
        _patch_fp8_forward(mod, fp8_blockwise_mm, use_triton)
        converted_count += 1

    logger.info(
        f"FP8 linear: patched {converted_count}/{total_linear} nn.Linear modules"
    )


def enable_fp8_experts(model: nn.Module, *, use_triton: bool = True) -> None:
    # [FUNCTION DOC]
    # For every GroupedExperts module with dim and hidden_dim multiples of 128, patches its
    # forward with the FP8 expert loop; others are skipped with a warning. Logs counts.
    """Enable FP8 blockwise matmul for MoE expert modules.

    Patches the forward of each eligible :class:`GroupedExperts` to use
    torchao's ``fp8_blockwise_mm``. Must be called on a meta-device model,
    before parallelism is applied.
    """
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    converted_count = 0
    skipped_count = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, GroupedExperts):
            continue
        if mod.dim % _FP8_BLOCK != 0 or mod.hidden_dim % _FP8_BLOCK != 0:
            logger.warning(
                f"Skipping FP8 for {name}: dimensions "
                f"({mod.dim}, {mod.hidden_dim}) not {_FP8_BLOCK}-aligned"
            )
            skipped_count += 1
            continue
        _patch_fp8_experts_forward(mod, use_triton)
        converted_count += 1

    logger.info(
        f"FP8 experts: enabled for {converted_count} GroupedExperts modules"
        + (
            f", skipped {skipped_count} (not {_FP8_BLOCK}-aligned)"
            if skipped_count
            else ""
        )
    )


def _patch_fp8_experts_forward(mod: nn.Module, use_triton: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Sets _fp8_use_triton and _fp8_block attributes on the module and binds a new
    # forward that fetches local weights via the module's _get_local_weights and runs
    # _run_experts_fp8_for_loop(w1, w2, w3, x, counts, use_triton). Bound with MethodType so
    # deepcopy keeps working and FSDP-swapped weights are read at call time.
    """Replace mod.forward with FP8 expert computation.

    Uses ``types.MethodType`` so that ``self`` refers to the module
    instance, making the patch safe under ``copy.deepcopy`` (used by
    pipeline-parallel stage splitting).  FSDP weight swaps are visible
    at call time because ``self.w1/w2/w3`` are read during forward.
    """
    from areal.experimental.models.archon.moe.grouped_experts import (
        _run_experts_fp8_for_loop,
    )

    mod._fp8_use_triton = use_triton  # type: ignore[attr-defined]
    mod._fp8_block = _FP8_BLOCK  # type: ignore[attr-defined]

    def _fp8_expert_fwd(
        self: nn.Module,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Patched GroupedExperts forward: x [tokens, dim] grouped by expert plus counts;
        # returns expert outputs computed in FP8.
        w1, w2, w3 = self._get_local_weights()
        return _run_experts_fp8_for_loop(
            w1, w2, w3, x, num_tokens_per_expert, use_triton=self._fp8_use_triton
        )

    mod.forward = types.MethodType(_fp8_expert_fwd, mod)  # type: ignore[assignment]


def _is_eligible(mod: nn.Linear, fqn: str, exclude_fqns: set[str]) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the Linear's FQN matches no exclude substring, it has no bias, and
    # in/out features are multiples of 128 (warns when skipping for alignment).
    for exclude in exclude_fqns:
        if exclude in fqn:
            return False
    if mod.bias is not None:
        return False
    if mod.in_features % _FP8_BLOCK != 0 or mod.out_features % _FP8_BLOCK != 0:
        logger.warning(
            f"Skipping FP8 for {fqn}: dimensions "
            f"({mod.in_features}, {mod.out_features}) not {_FP8_BLOCK}-aligned"
        )
        return False
    return True


def _patch_fp8_forward(mod: nn.Linear, fp8_blockwise_mm, use_triton: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Stores the fp8 mm autograd function, triton flag and block size on the module and
    # binds a replacement forward (see nested fn).
    """Replace mod.forward with FP8 blockwise matmul (padding + fp8_blockwise_mm).

    Uses ``types.MethodType`` so that ``self`` refers to the module
    instance, making the patch safe under ``copy.deepcopy`` (used by
    pipeline-parallel stage splitting).  FSDP weight swaps are visible
    at call time because ``self.weight`` is read during forward.
    """
    mod._fp8_mm = fp8_blockwise_mm  # type: ignore[attr-defined]
    mod._fp8_use_triton = use_triton  # type: ignore[attr-defined]
    mod._fp8_block = _FP8_BLOCK  # type: ignore[attr-defined]

    def _fp8_linear_fwd(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Flattens x to 2-D, zero-pads rows up to a multiple of the block size, unwraps a
        # DTensor weight to its local shard, applies fp8_blockwise_mm (output dtype = input
        # dtype), drops pad rows and reshapes to the original leading dims with the new last dim.
        leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        m = x.shape[0]
        pad = (self._fp8_block - m % self._fp8_block) % self._fp8_block
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        weight = self.weight
        if hasattr(weight, "to_local"):
            weight = weight.to_local()
        out = self._fp8_mm.apply(
            x, weight, self._fp8_block, x.dtype, self._fp8_use_triton
        )
        if pad > 0:
            out = out[:m]
        return out.view(*leading, -1)

    mod.forward = types.MethodType(_fp8_linear_fwd, mod)  # type: ignore[assignment]


def validate_fp8_shard_alignment(
    model_parts: list[nn.Module],
    block_size: int = _FP8_BLOCK,
) -> None:
    # [FUNCTION DOC]
    # After TP/PP, checks every module carrying _fp8_block: Linear local weight [out, in] and
    # each GroupedExperts w1/w2/w3 local per-expert dims (dims 1 and 2) must be multiples of
    # block_size; raises ValueError with a remediation hint otherwise. Returns None.
    """Check that FP8-patched modules have block-aligned local weight dims.

    Must be called **after** parallelism (TP/PP) is applied.  TP can shard
    weight columns or rows, producing local dimensions that are no longer
    multiples of ``block_size``.  The FP8 kernel pads the token (M)
    dimension automatically, but weight dimensions (N, K) must be
    pre-aligned — a mismatch causes a Triton/cuBLAS crash at runtime.

    Validates both ``nn.Linear`` modules (2D weights) and
    ``GroupedExperts`` modules (3D weights ``[num_experts, dim_a, dim_b]``
    where each per-expert slice must be block-aligned).
    """
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore[assignment, misc]

    def _local_shape(t: torch.Tensor) -> torch.Size:
        # [FUNCTION DOC]
        # HELPER: Local shard shape of a DTensor, or the plain tensor shape.
        if DTensor is not None and isinstance(t, DTensor):
            return t.to_local().shape
        return t.shape

    for part in model_parts:
        for fqn, mod in part.named_modules():
            if not hasattr(mod, "_fp8_block"):
                continue

            # --- nn.Linear: 2D weight (out_dim, in_dim) ---
            if isinstance(mod, nn.Linear):
                local_shape = _local_shape(mod.weight)
                out_dim, in_dim = local_shape[0], local_shape[1]
                if out_dim % block_size != 0 or in_dim % block_size != 0:
                    raise ValueError(
                        f"FP8 module {fqn!r} has non-{block_size}-aligned "
                        f"local weight shape {tuple(local_shape)} after "
                        f"parallelism. This will cause FP8 kernel failures "
                        f"at runtime. Adjust TP degree or add this module's "
                        f"name to fp8_config.exclude_modules."
                    )

            # --- GroupedExperts: 3D weights (num_experts, dim_a, dim_b) ---
            elif isinstance(mod, GroupedExperts):
                for wname in ("w1", "w2", "w3"):
                    w = getattr(mod, wname, None)
                    if w is None:
                        continue
                    local_shape = _local_shape(w)
                    # Per-expert slice is (dim_a, dim_b); both must be aligned.
                    dim_a, dim_b = local_shape[1], local_shape[2]
                    if dim_a % block_size != 0 or dim_b % block_size != 0:
                        raise ValueError(
                            f"FP8 expert {fqn!r}.{wname} has non-"
                            f"{block_size}-aligned local per-expert shape "
                            f"({dim_a}, {dim_b}) (full local shape "
                            f"{tuple(local_shape)}) after parallelism. "
                            f"This will cause FP8 kernel failures at "
                            f"runtime. Adjust TP/ETP degree or disable "
                            f"fp8_config.include_experts."
                        )
