# [MODULE DOC]
# UE8M0 scale utilities (adapted from DeepGEMM / SGLang). UE8M0 keeps only the 8-bit
# biased exponent of a float32 scale, so scales are powers of two. Provides TMA
# alignment helpers, 128x128 blockwise FP8 quantization with power-of-two scales, and
# conversion of float32 scales into DeepGEMM's packed (4 exponents per int32), MN-major,
# TMA-aligned layout. Pure torch, runs on the tensors' device.
# SPDX-License-Identifier: Apache-2.0

# UE8M0 format utilities for FP8 quantization
# Adapted from DeepGEMM and SGLang
#
# UE8M0 = Unsigned Exponent, 8-bit, Mantissa=0
# This format stores only the exponent of a float32 scale as a uint8,
# effectively limiting scales to powers of 2.


import torch

from areal.utils.math import align, ceil_div

# TMA requires 16-byte alignment (from DeepGEMM csrc/utils/math.hpp)
_NUM_TMA_ALIGNMENT_BYTES = 16


def get_tma_aligned_size(x: int, element_size: int) -> int:
    # [FUNCTION DOC]
    # HELPER: Round size x up to a multiple of 16 bytes worth of elements (16 //
    # element_size). Asserts element_size divides 16. Returns int.
    """Get TMA aligned size (16 bytes alignment).

    Args:
        x: Size to align
        element_size: Size of each element in bytes

    Returns:
        Aligned size
    """
    assert _NUM_TMA_ALIGNMENT_BYTES % element_size == 0
    return align(x, _NUM_TMA_ALIGNMENT_BYTES // element_size)


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Round the absolute values of x up to the nearest power of two (2 **
    # ceil(log2|x|)). Returns a new float tensor. Zero inputs produce 0 (log2 gives
    # -inf).
    """Convert scale to power of 2 (UE8M0 format).

    UE8M0 format stores only the exponent, so scales are rounded up
    to the nearest power of 2.

    Args:
        x: Scale tensor (positive values)

    Returns:
        Tensor with values rounded up to powers of 2
    """
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def get_mn_major_tma_aligned_packed_ue8m0_tensor(x: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Convert a float32 scale tensor of shape (mn, k) or (batch, mn, k) to packed UE8M0:
    # takes bits 23..30 of each float as uint8 (assumes positive scales; the sign bit
    # would leak in otherwise), zero-pads mn to a TMA-aligned multiple of 4 and k to a
    # multiple of 4, reinterprets each 4 bytes as one int32, then copies into a buffer
    # whose memory is MN-major (stride 1 along mn) and slices back to mn rows. Returns
    # an int32 tensor of shape (..., mn, ceil(k/4)) with that non-contiguous layout;
    # batch dim removed for 2D input. AssertionError on wrong dtype/rank.
    """Convert FP32 scale tensor to TMA-aligned packed UE8M0 format.

    This is the pure PyTorch implementation from DeepGEMM tests/test_layout.py
    (get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl).

    The function:
    1. Extracts the exponent from FP32 (bits 23-30) as UE8M0
    2. Packs 4 uint8 values into 1 int32
    3. Transposes to MN-major layout with TMA alignment

    Args:
        x: FP32 scale tensor, shape (mn, k) or (batch, mn, k)

    Returns:
        Packed UE8M0 tensor, shape (mn, packed_k) or (batch, mn, packed_k),
        dtype=int32 with MN-major TMA-aligned layout
    """
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    # Extract exponent: FP32 bits [30:23] contain biased exponent
    ue8m0_tensor = (x.view(torch.int32) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
        ue8m0_tensor = ue8m0_tensor.unsqueeze(0)
    b = x.shape[0]

    # TMA alignment: 16 bytes, for 4-byte elements (int32) this is align to 4
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)

    # Pad and pack (4 uint8 -> 1 int32)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(torch.int32).view(b, aligned_mn, aligned_k // 4)

    # Transpose to MN-major layout
    # Create strided tensor with layout (batch, mn, packed_k) but stride (1, aligned_mn)
    transposed = torch.zeros(
        (b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int32
    ).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]

    return aligned_x.squeeze(0) if remove_dim else aligned_x


def per_block_cast_to_fp8_ue8m0(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Quantize a 2D tensor (m, n) to float8_e4m3fn with 128x128 blocks: zero-pads to
    # multiples of 128, computes per-block absmax (float, clamped >= 1e-4), scale =
    # absmax/448 rounded up to a power of two, divides and casts. Returns (fp8 tensor of
    # shape (m, n), contiguous; float32 scale of shape (ceil(m/128), ceil(n/128))).
    """Quantize 2D tensor to FP8 with 128x128 block-wise UE8M0 scaling.

    From SGLang fp8_utils.py, which is copied from DeepGEMM.

    Args:
        x: Input tensor, shape (m, n)

    Returns:
        (fp8_quantized, fp32_scale) where scale uses UE8M0 format (powers of 2)
    """
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    # 448.0 is the max representable value for FP8 E4M3 format
    # (torch.finfo(torch.float8_e4m3fn).max == 448.0)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # Quantize a bfloat16 weight of shape (..., n, k) to FP8 with UE8M0 128x128 block
    # scales by flattening leading dims into rows and calling
    # per_block_cast_to_fp8_ue8m0. Returns (fp8 weight of the same shape, float32 scale
    # of shape (..., ceil(n/128), ceil(k/128))). Asserts weight_block_size == [128, 128]
    # (list) and dtype bfloat16.
    # KNOWN ISSUE: for batched (>2D) input where n is not a multiple of 128, blocks span
    # batch boundaries in the flattened view and the scale reshape to (..., ceil(n/128),
    # ...) will not match the element count.
    """Quantize weight to FP8 + UE8M0 scale format.

    From SGLang sglang/srt/layers/quantization/fp8_utils.py.

    Args:
        weight_dequant: BF16 weight, shape (..., n, k)
        weight_block_size: Must be [128, 128]

    Returns:
        (fp8_weight, fp32_scale)
    """
    assert weight_block_size == [128, 128], (
        f"UE8M0 quantization requires [128, 128] block size, got {weight_block_size}"
    )
    assert weight_dequant.dtype == torch.bfloat16, (
        f"{weight_dequant.dtype=} {weight_dequant.shape=}"
    )

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8_ue8m0(weight_dequant_flat)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s


def transform_scale_ue8m0(sf: torch.Tensor, mn: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # Expand a block-scale tensor along the row dim to one row per original row (row i
    # takes block row i // 128, hard-coded 128 block) for mn rows, then pack it via
    # get_mn_major_tma_aligned_packed_ue8m0_tensor. Returns the packed int32 tensor for
    # DeepGEMM.
    """Transform scale to UE8M0 packed format for DeepGEMM.

    From SGLang sglang/srt/layers/quantization/fp8_utils.py.

    Args:
        sf: FP32 scale tensor, shape (scale_mn, scale_k) or (batch, scale_mn, scale_k)
        mn: Original mn dimension size (for broadcasting)

    Returns:
        Packed UE8M0 tensor with TMA-aligned MN-major layout
    """
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf
