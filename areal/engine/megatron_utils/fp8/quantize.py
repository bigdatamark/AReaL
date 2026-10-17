# [MODULE DOC]
# High-level FP8 API (adapted from slime) used when sending Megatron weights to an FP8
# inference engine or loading FP8 checkpoints: decides which converted HuggingFace
# parameters get quantized based on the originating Megatron parameter name, and
# dequantizes blockwise FP8 weights.
# SPDX-License-Identifier: Apache-2.0

# High-level FP8 quantization and dequantization utilities
# Adapted from slime

import re

import torch

from areal.engine.megatron_utils.fp8.config import get_block_size_from_config
from areal.engine.megatron_utils.fp8.deepgemm import (
    should_deepgemm_weight_requant_ue8m0,
)
from areal.engine.megatron_utils.fp8.kernels import (
    blockwise_cast_to_fp8_triton,
    weight_dequant,
)
from areal.engine.megatron_utils.fp8.ue8m0 import (
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)


def _quantize_param(
    name: str,
    weight: torch.Tensor,
    weight_block_size: tuple[int, int] | list[int] | None = None,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Quantize one ".weight" tensor to float8_e4m3fn. With weight_block_size
    # (list/tuple [m, n]) it uses blockwise quantization: the UE8M0 DeepGEMM path
    # (power-of-two scales, packed TMA-aligned int32 layout) when
    # should_deepgemm_weight_requant_ue8m0 says so, else the Triton kernel; the scale
    # param is named by replacing ".weight" with ".weight_scale_inv". Without a block
    # size it does per-tensor quantization (scale = absmax clamped to 1e-12 / FP8 max,
    # shape [1]) named ".weight_scale". Scale is cast to scale_dtype (default bfloat16).
    # Returns [(name, qweight), (scale_name, scale)]. Raises AssertionError if name does
    # not end with ".weight".
    # KNOWN ISSUE: the UE8M0 path calls quant_weight_ue8m0, which asserts block size
    # equals the list [128, 128]; a tuple (allowed by the type hint) or other size will
    # fail the assertion. Also the packed UE8M0 scale is int32, so casting it to
    # scale_dtype (bfloat16) corrupts the packed bits.
    """Quantize a single weight parameter to FP8 format.

    Args:
        name: Parameter name (must end with ".weight")
        weight: Weight tensor to quantize
        weight_block_size: Optional block size for blockwise quantization [block_m, block_n]
        scale_dtype: Dtype for the scale tensor

    Returns:
        List of (name, tensor) tuples: [(weight_name, quantized_weight), (scale_name, scale)]
    """
    assert name.endswith(".weight"), f"Expected weight parameter, got {name}"
    FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

    if weight_block_size is not None:
        # Blockwise quantization
        if should_deepgemm_weight_requant_ue8m0(weight_block_size=weight_block_size):
            # Use UE8M0 quantization for DeepGEMM on Blackwell
            qweight, scale = quant_weight_ue8m0(
                weight, weight_block_size=weight_block_size
            )
            scale = transform_scale_ue8m0(scale, mn=qweight.shape[-2])
        else:
            # Use triton-based blockwise quantization
            qweight, scale = blockwise_cast_to_fp8_triton(weight, weight_block_size)
        scale_name = name.replace(".weight", ".weight_scale_inv")
    else:
        # Per-tensor quantization
        scale = weight.abs().max().clamp(min=1e-12).to(torch.float32) / FP8_MAX
        qweight = (
            (weight / scale).clamp(min=FP8_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale = scale.view(1)
        scale_name = name.replace(".weight", ".weight_scale")

    scale = scale.to(scale_dtype)
    return [(name, qweight), (scale_name, scale)]


def quantize_params(
    megatron_name: str,
    converted_named_params: list[tuple[str, torch.Tensor]],
    quantization_config: dict[str, int | str | list[str]] | None,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC]
    # Quantize the HF parameters converted from one Megatron parameter. Inputs:
    # megatron_name (str, with or without "module.module." prefix; "decoder." and "mtp."
    # names get the prefix added), converted_named_params (list of (hf_name, tensor)),
    # quantization_config (HF dict or None), scale_dtype. Returns converted_named_params
    # unchanged when config is None or the param is not a quantizable linear weight.
    # Asserts config quant_method "fp8", fmt "e4m3", activation_scheme "dynamic"
    # (AssertionError otherwise). Parses layer index and remainder from decoder.layers.N
    # or mtp.layers.N (dropping "transformer_layer."). Quantizes (via _quantize_param,
    # block size from config) when the remainder is: routed expert linear_fc1/linear_fc2
    # weightK (skipping converted names ending in "_scale"), shared_experts
    # linear_fc1/fc2 weight, or attention/MLP linear weights (linear_proj, linear_qkv,
    # mlp fc1/fc2, and MLA q/q_down/q_up/kv_down/kv_up projections). Returns the
    # expanded list with interleaved weight and scale entries.
    """Apply FP8 quantization to converted HuggingFace parameters.

    Args:
        megatron_name: Megatron parameter name
        converted_named_params: List of (name, tensor) tuples
        quantization_config: Quantization configuration dict
        scale_dtype: Dtype for scale tensors

    Returns:
        List of (name, tensor) tuples with quantized parameters
    """
    if quantization_config is None:
        return converted_named_params

    assert quantization_config["quant_method"] == "fp8"
    assert quantization_config["fmt"] == "e4m3"
    assert quantization_config["activation_scheme"] == "dynamic"
    weight_block_size = quantization_config.get("weight_block_size", None)

    # handle both with and without "module.module." prefix
    if not megatron_name.startswith("module.module."):
        # Add prefix if missing for pattern matching
        if megatron_name.startswith("decoder."):
            megatron_name = "module.module." + megatron_name
        elif megatron_name.startswith("mtp."):
            megatron_name = "module.module." + megatron_name

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, megatron_name)

    if not match:
        # Check mtp layers
        mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
        match = re.match(mtp_layer_pattern, megatron_name)
        if not match:
            return converted_named_params
        layer_idx, rest = match.groups()
        rest = rest.replace("transformer_layer.", "")
    else:
        layer_idx, rest = match.groups()

    # Experts
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
    match = re.match(expert_pattern, rest)
    if match:
        rest, expert_idx = match.groups()
        if rest in ["linear_fc1", "linear_fc2"]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                # Skip bf16 weight_scale and input_scale
                # TODO: find a clearer way.
                if converted_name.endswith("_scale"):
                    continue
                quantize_named_params.extend(
                    _quantize_param(
                        converted_name, param, weight_block_size, scale_dtype
                    )
                )
            return quantize_named_params

    # Shared expert
    shared_expert_pattern = r"mlp.shared_experts\.(.+)"
    match = re.match(shared_expert_pattern, rest)
    if match:
        rest = match.groups()[0]
        if rest in ["linear_fc1.weight", "linear_fc2.weight"]:
            quantize_named_params = []
            for converted_name, param in converted_named_params:
                quantize_named_params.extend(
                    _quantize_param(
                        converted_name, param, weight_block_size, scale_dtype
                    )
                )
            return quantize_named_params

    # Regular attention and MLP layers
    if rest in [
        "self_attention.linear_proj.weight",
        "self_attention.linear_qkv.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        # mla
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_q_down_proj.weight",
        "self_attention.linear_q_up_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.linear_kv_up_proj.weight",
    ]:
        quantize_named_params = []
        for converted_name, param in converted_named_params:
            quantize_named_params.extend(
                _quantize_param(converted_name, param, weight_block_size, scale_dtype)
            )
        return quantize_named_params

    # For other parameters, return original converted_named_params
    return converted_named_params


def dequantize_params(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dst_dtype: torch.dtype = torch.bfloat16,
    quantization_config: dict[str, int | str | list[str]] | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Dequantize an FP8 weight with its per-block scale_inv into dst_dtype (default
    # bfloat16). Makes both inputs contiguous, reads the block size from
    # quantization_config in strict mode (default 128 if config is None; ValueError if
    # non-square), then calls weight_dequant. Returns the new tensor.
    """Dequantize FP8 weights to the given dtype.

    Args:
        weight: FP8 weight tensor
        scale_inv: Scale inverse tensor
        dst_dtype: Destination dtype
        quantization_config: Quantization configuration

    Returns:
        Dequantized weight tensor
    """
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not scale_inv.is_contiguous():
        scale_inv = scale_inv.contiguous()

    block_size = get_block_size_from_config(quantization_config, strict=True)

    return weight_dequant(weight, scale_inv, block_size, dst_dtype)
