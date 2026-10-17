# [PACKAGE DOC]
# FP8 blockwise quantization package for Megatron weight conversion (Megatron ->
# inference engine weight sync and FP8 checkpoint loading). Re-exports the public API of
# its submodules: high-level quantize/dequantize of named params, Triton kernels, UE8M0
# (power-of-two scale) helpers for DeepGEMM on Blackwell, DeepGEMM capability flags,
# block-size config parsing and the FP8BlockwiseTensorHelper wrapper. No logic of its
# own; importing it imports triton/torch via the submodules.
# KNOWN ISSUE: ENABLE_JIT_DEEPGEMM, DEEPGEMM_BLACKWELL and DEEPGEMM_SCALE_UE8M0 are
# pulled in with a from-import, which triggers deepgemm's lazy module __getattr__ at
# package import time, so the values are snapshotted eagerly (querying CUDA) rather than
# lazily as deepgemm.py intends.
# SPDX-License-Identifier: Apache-2.0

# FP8 quantization utilities
#
# This module provides FP8 quantization and dequantization utilities for
# model weight conversion and inference optimization.
#
# Structure:
#   - kernels.py: Triton kernels for blockwise quantization/dequantization
#   - ue8m0.py: UE8M0 format utilities (power-of-2 scales for DeepGEMM)
#   - deepgemm.py: DeepGEMM detection and configuration
#   - quantize.py: High-level quantization/dequantization API
#   - tensor_helper.py: FP8 blockwise tensor helper class
#   - config.py: Configuration utilities for extracting block size from quantization config

from areal.engine.megatron_utils.fp8.config import get_block_size_from_config
from areal.engine.megatron_utils.fp8.deepgemm import (
    DEEPGEMM_BLACKWELL,
    DEEPGEMM_SCALE_UE8M0,
    ENABLE_JIT_DEEPGEMM,
    should_deepgemm_weight_requant_ue8m0,
)
from areal.engine.megatron_utils.fp8.kernels import (
    blockwise_cast_to_fp8_triton,
    weight_dequant,
)
from areal.engine.megatron_utils.fp8.quantize import (
    dequantize_params,
    quantize_params,
)
from areal.engine.megatron_utils.fp8.tensor_helper import (
    FP8BlockwiseTensorHelper,
    convert_fp8_helper_to_pytorch_fp8,
)
from areal.engine.megatron_utils.fp8.ue8m0 import (
    ceil_to_ue8m0,
    get_tma_aligned_size,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)

__all__ = [
    # High-level API
    "quantize_params",
    "dequantize_params",
    "get_block_size_from_config",
    # Kernels
    "blockwise_cast_to_fp8_triton",
    "weight_dequant",
    # UE8M0 utilities
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "ceil_to_ue8m0",
    "get_tma_aligned_size",
    # DeepGEMM config
    "ENABLE_JIT_DEEPGEMM",
    "DEEPGEMM_BLACKWELL",
    "DEEPGEMM_SCALE_UE8M0",
    "should_deepgemm_weight_requant_ue8m0",
    # Tensor helper
    "FP8BlockwiseTensorHelper",
    "convert_fp8_helper_to_pytorch_fp8",
]
