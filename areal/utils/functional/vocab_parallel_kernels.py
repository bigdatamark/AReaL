# [MODULE DOC]
# Triton kernels and Python launchers for the destructive (in-place) vocab-parallel
# logprob/entropy path. All kernels operate on contiguous 2D CUDA FP32 [rows,
# vocab_shard] tensors with a 2D launch grid (row, column tile of _BLOCK_SIZE=1024). If
# Triton is not importable, TRITON_AVAILABLE is False, kernels are not defined and
# launchers raise.
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


_BLOCK_SIZE = 1024


if TRITON_AVAILABLE:

    @triton.jit
    def _exp_sum_inplace_kernel(
        logits,
        row_max,
        partial_sums,
        partial_weighted_sums,
        n_rows,
        n_cols,
        n_tiles,
        inv_temperature,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC]
        # Triton kernel: for one row and column tile, loads logits, computes shifted =
        # (logit - row_max) * inv_temperature and exp(shifted), stores the exponentials
        # in place, and writes the tile's sum of exp and sum of shifted*exp into
        # partial_sums / partial_weighted_sums at [row, tile] (deterministic, no
        # atomics).
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        values = tl.load(logits + offsets, mask=mask, other=-float("inf"))
        maximum = tl.load(row_max + row)
        shifted = (values - maximum) * inv_temperature
        exponentials = tl.exp(shifted)
        tl.store(logits + offsets, exponentials, mask=mask)
        tl.store(
            partial_sums + row * n_tiles + tile,
            tl.sum(tl.where(mask, exponentials, 0.0), axis=0),
        )
        tl.store(
            partial_weighted_sums + row * n_tiles + tile,
            tl.sum(tl.where(mask, shifted * exponentials, 0.0), axis=0),
        )

    @triton.jit
    def _normalize_inplace_kernel(
        logits,
        sum_exp,
        n_rows,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC]
        # Triton kernel: divides each element of a row tile in place by that row's
        # sum_exp, turning exponentials into softmax probabilities.
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        exponentials = tl.load(logits + offsets, mask=mask, other=0.0)
        denominator = tl.load(sum_exp + row)
        probabilities = exponentials / denominator
        tl.store(logits + offsets, probabilities, mask=mask)

    @triton.jit
    def _softmax_backward_inplace_kernel(
        softmax,
        local_targets,
        grad_logprobs,
        n_rows,
        n_cols,
        inv_temperature,
        HAS_LOGPROBS_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC]
        # Triton kernel: overwrites softmax in place with the logprob gradient w.r.t.
        # logits: (one_hot(target) - p) * grad_logprob * inv_temperature, where the one-
        # hot term applies only when the local target is >= 0 (label in this shard).
        # Without HAS_LOGPROBS_GRAD writes zeros.
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols
        probabilities = tl.load(softmax + offsets, mask=mask, other=0.0)
        gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        if HAS_LOGPROBS_GRAD:
            logprob_gradient = tl.load(grad_logprobs + row)
            target = tl.load(local_targets + row)
            gradient = -probabilities * logprob_gradient
            gradient += tl.where(
                mask & (target >= 0) & (cols == target), logprob_gradient, 0.0
            )

        tl.store(softmax + offsets, gradient * inv_temperature, mask=mask)

    @triton.jit
    def _logits_backward_inplace_kernel(
        logits,
        row_max,
        sum_exp,
        local_targets,
        grad_logprobs,
        n_rows,
        n_cols,
        inv_temperature,
        logit_scale,
        HAS_LOGPROBS_GRAD: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC]
        # Triton kernel: like the softmax backward but recomputes probabilities from raw
        # logits, row_max and sum_exp (exp((x - max)*inv_temperature)/sum_exp), and
        # additionally scales the stored gradient by logit_scale. Writes in place into
        # logits.
        row = tl.program_id(axis=0)
        tile = tl.program_id(axis=1)
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (row < n_rows) & (cols < n_cols)
        offsets = row * n_cols + cols

        values = tl.load(logits + offsets, mask=mask, other=-float("inf"))
        maximum = tl.load(row_max + row)
        denominator = tl.load(sum_exp + row)
        probabilities = tl.exp((values - maximum) * inv_temperature) / denominator
        gradient = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        if HAS_LOGPROBS_GRAD:
            logprob_gradient = tl.load(grad_logprobs + row)
            target = tl.load(local_targets + row)
            gradient = -probabilities * logprob_gradient
            gradient += tl.where(
                mask & (target >= 0) & (cols == target), logprob_gradient, 0.0
            )

        tl.store(
            logits + offsets,
            gradient * inv_temperature * logit_scale,
            mask=mask,
        )


def _check_input(tensor: torch.Tensor) -> None:
    # [FUNCTION DOC]
    # HELPER: Validates that Triton is available (RuntimeError) and that tensor is a
    # CUDA float32 contiguous 2D tensor (ValueError otherwise).
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused vocab-parallel kernels")
    if not tensor.is_cuda or tensor.dtype is not torch.float32:
        raise ValueError("Fused vocab-parallel kernels require CUDA FP32 tensors")
    if tensor.ndim != 2 or not tensor.is_contiguous():
        raise ValueError("Fused vocab-parallel kernels require contiguous 2D tensors")


def fused_exp_sum_inplace(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    partial_sums: torch.Tensor,
    partial_weighted_sums: torch.Tensor,
    inv_temperature: float,
) -> None:
    # [FUNCTION DOC]
    # Launches _exp_sum_inplace_kernel over (rows, tiles) after validation. Inputs:
    # logits [R, V] (overwritten with exponentials), row_max [R], partial_sums and
    # partial_weighted_sums [R, tiles] outputs, inv_temperature. Returns None.
    """Overwrite logits with shifted exponentials and emit deterministic tile sums."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    _exp_sum_inplace_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        partial_sums,
        partial_weighted_sums,
        n_rows,
        n_cols,
        n_tiles,
        inv_temperature,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_normalize_inplace(
    logits: torch.Tensor,
    sum_exp: torch.Tensor,
) -> None:
    # [FUNCTION DOC]
    # Launches _normalize_inplace_kernel to divide logits rows (holding exponentials) by
    # sum_exp [R] in place. Returns None.
    """Overwrite exponentials with normalized softmax probabilities."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    _normalize_inplace_kernel[(n_rows, n_tiles)](
        logits,
        sum_exp,
        n_rows,
        n_cols,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_softmax_backward_inplace(
    softmax: torch.Tensor,
    local_targets: torch.Tensor,
    grad_logprobs: torch.Tensor | None,
    inv_temperature: float,
) -> None:
    # [FUNCTION DOC]
    # Launches _softmax_backward_inplace_kernel turning softmax [R, V] into dlogits in
    # place given local_targets [R] (-1 if not local) and optional grad_logprobs [R] (a
    # dummy pointer is passed when None). Returns None.
    """Overwrite softmax with the selected-token logprob gradient."""
    _check_input(softmax)
    n_rows, n_cols = softmax.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    dummy = softmax
    _softmax_backward_inplace_kernel[(n_rows, n_tiles)](
        softmax,
        local_targets,
        grad_logprobs if grad_logprobs is not None else dummy,
        n_rows,
        n_cols,
        inv_temperature,
        HAS_LOGPROBS_GRAD=grad_logprobs is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def fused_logits_backward_inplace(
    logits: torch.Tensor,
    row_max: torch.Tensor,
    sum_exp: torch.Tensor,
    local_targets: torch.Tensor,
    grad_logprobs: torch.Tensor | None,
    inv_temperature: float,
    logit_scale: float = 1.0,
) -> None:
    # [FUNCTION DOC]
    # Launches _logits_backward_inplace_kernel turning recomputed logits [R, V] into
    # dlogits in place from row_max, sum_exp, local_targets, optional grad_logprobs,
    # inv_temperature and logit_scale. Returns None. (Not used by vocab_parallel.py in
    # this module set.)
    """Overwrite recomputed logits with selected-token logprob gradients."""
    _check_input(logits)
    n_rows, n_cols = logits.shape
    n_tiles = triton.cdiv(n_cols, _BLOCK_SIZE)
    dummy = logits
    _logits_backward_inplace_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        sum_exp,
        local_targets,
        grad_logprobs if grad_logprobs is not None else dummy,
        n_rows,
        n_cols,
        inv_temperature,
        logit_scale,
        HAS_LOGPROBS_GRAD=grad_logprobs is not None,
        BLOCK_SIZE=_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )


def reusable_vocab_parallel_logits(logits: torch.Tensor) -> torch.Tensor | None:
    # [FUNCTION DOC]
    # Returns a tensor safe to overwrite for the in-place path, or None. If logits is a
    # view, it must be contiguous, start at offset 0, cover all of its contiguous base,
    # and the base is used instead. The workspace must be CUDA float32 contiguous, not a
    # leaf (so it is an intermediate autograd result), and Triton must be available.
    """Return logits when they exclusively cover their underlying storage."""
    workspace = logits
    if logits._base is not None:
        base = logits._base
        if (
            not logits.is_contiguous()
            or logits.storage_offset() != 0
            or logits.numel() != base.numel()
            or not base.is_contiguous()
        ):
            return None
        workspace = base

    if (
        not TRITON_AVAILABLE
        or not workspace.is_cuda
        or workspace.dtype is not torch.float32
        or not workspace.is_contiguous()
        or workspace.is_leaf
    ):
        return None
    return workspace


def fused_vocab_parallel_available(logits: torch.Tensor) -> bool:
    # [FUNCTION DOC]
    # True if reusable_vocab_parallel_logits returns a workspace for logits.
    return reusable_vocab_parallel_logits(logits) is not None


def vocab_tile_count(vocab_size: int) -> int:
    # [FUNCTION DOC]
    # Returns the number of _BLOCK_SIZE column tiles needed for vocab_size (ceil
    # division).
    return (vocab_size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
