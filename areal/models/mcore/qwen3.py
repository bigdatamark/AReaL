# [MODULE DOC]
# Megatron-Core (mcore) config conversion and layer-spec construction for dense Qwen3
# models (adapted from verl). Used by the mcore model registry to build GPTModel
# instances from HuggingFace Qwen3 configs.
# SPDX-License-Identifier: Apache-2.0

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import TransformerConfig
from transformers import PretrainedConfig

from areal.models.mcore.common import (
    check_and_construct_configs,
    hf_to_mcore_base_args,
)


# Modified from verl:
# https://github.com/volcengine/verl/blob/ea885f32f04d86c3a81de18083db7eef0d781421/verl/models/mcore/config_converter.py
def hf_to_mcore_config_qwen3_dense(
    hf_config: PretrainedConfig, dtype: torch.dtype
) -> TransformerConfig:
    # [FUNCTION DOC]
    # Converts a HF PretrainedConfig for a dense Qwen3 model into an mcore
    # TransformerConfig. Builds base args via hf_to_mcore_base_args with dtype, GPU
    # (non-CPU) initialization, no linear bias, QKV bias taken from
    # hf_config.attention_bias (default False) and QK layernorm enabled, then
    # validates/constructs the TransformerConfig through check_and_construct_configs.
    # Returns the TransformerConfig. No side effects.
    args: dict = hf_to_mcore_base_args(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=getattr(hf_config, "attention_bias", False),
        qk_layernorm=True,
    )
    return check_and_construct_configs(args, TransformerConfig)


def make_mcore_layer_specs_qwen3_dense(
    tfconfig: TransformerConfig, use_te: bool = True
):
    # [FUNCTION DOC]
    # Returns the mcore GPT decoder block spec for the given TransformerConfig, using
    # TransformerEngine layers when use_te is True (default). Raises AssertionError if
    # tfconfig.normalization is not "RMSNorm".
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
    return get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
