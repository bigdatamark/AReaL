# [MODULE DOC]
# Model-type registries and predicates used across engines: which HF model_type strings
# are supported vision / MoE / Qwen-family models, which sequence-packing mode to use,
# accessors for nested text configs, and a dropout-disabling utility. All functions are
# pure except disable_dropout_in_model.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch

VALID_VISION_MODELS = [
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "gemma3",
]
# This registry is used to check if a model is a vision model that we have checked it works with AReaL.
# As different vision models vary in their image processing, special tokens and keys, etc.
# We will add models to this registry as we test them.
# If you want to add a new vision model, please make sure it works with AReaL.


def is_valid_vision_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True if model_type is in the tested vision-model registry
    # (VALID_VISION_MODELS).
    return model_type in VALID_VISION_MODELS


def is_qwen2_vl_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for Qwen2-VL and Qwen2.5-VL model types.
    return model_type in ["qwen2_vl", "qwen2_5_vl"]


def is_qwen3_vl_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for Qwen3-VL dense or MoE model types.
    """True for the Qwen3-VL family (dense and MoE).

    Existing call sites in ``fsdp_engine``, ``fsdp_utils/parallel``, and
    ``awex/fsdp_adapter`` gate family-level behaviour (mRoPE index,
    attention-mask handling) that is identical for dense and MoE, so this
    helper covers both. Use ``is_qwen3_vl_moe_model`` when the MoE-vs-dense
    distinction matters.
    """
    return model_type in ("qwen3_vl", "qwen3_vl_moe")


def is_qwen3_vl_moe_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True only for the Qwen3-VL MoE model type.
    return model_type == "qwen3_vl_moe"


def is_qwen_vl_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for any Qwen2/2.5/3 VL model type (union of the two family
    # predicates).
    return is_qwen2_vl_model(model_type) or is_qwen3_vl_model(model_type)


def lang_config(hf_config):
    # [FUNCTION DOC]
    # Returns hf_config.text_config if the config has one (nested VLM configs),
    # otherwise the config itself. No side effects.
    """Return the language-model side of a (possibly nested) HF config.

    Qwen3-VL and similar VLMs nest text-model attributes (vocab_size,
    num_attention_heads, num_key_value_heads, hidden_size, head_dim) under
    ``hf_config.text_config``. Qwen2.5-VL and pure text models keep them
    flat. Use this anywhere the caller wants a language-side attribute and
    doesn't know the model family up front.
    """
    return getattr(hf_config, "text_config", hf_config)


def is_gemma3_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True if model_type is 'gemma3'.
    return model_type in ["gemma3"]


VALID_MOE_MODELS = [
    "qwen3_moe",
    "qwen3_vl_moe",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "bailing_moe_v2",
    "bailing_moe_linear",
    "bailing_hybrid",
]
# This registry is used to check if a model is a MoE model that we have checked it works with AReaL.


def is_moe_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True if model_type is in the tested MoE registry (VALID_MOE_MODELS).
    return model_type in VALID_MOE_MODELS


def is_qwen3_moe_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True only for text Qwen3 MoE ('qwen3_moe').
    return model_type in ["qwen3_moe"]


def is_qwen3_5_model(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for any Qwen3.5 variant (dense/MoE, with or without '_text' suffix).
    return model_type in ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"]


class SequencePackingMode(str, Enum):
    # [CLASS DOC]
    # String enum of how sequences are fed to the model: WRAPPER_THD (engine packs into
    # THD), MODEL_THD (model/bridge packs BSHD to THD itself), PADDED (padded BSHD
    # batches).
    WRAPPER_THD = "wrapper_thd"
    MODEL_THD = "model_thd"
    PADDED = "padded"


def supports_model_packed_seq(model_type: str, bridge_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True when the bridge type is 'megatron-bridge' and the model is Qwen3-VL
    # (dense or MoE), i.e. the bridge model handles THD packing internally.
    """Whether the bridge model owns BSHD-to-THD packing internally."""
    return bridge_type == "megatron-bridge" and is_qwen3_vl_model(model_type)


def resolve_sequence_packing_mode(
    model_type: str, bridge_type: str
) -> SequencePackingMode:
    # [FUNCTION DOC]
    # Chooses the SequencePackingMode: MODEL_THD if supports_model_packed_seq, PADDED
    # for other supported vision models or Qwen3.5, else WRAPPER_THD.
    """Select one packing path from the model and bridge contract."""
    if supports_model_packed_seq(model_type, bridge_type):
        return SequencePackingMode.MODEL_THD
    if is_valid_vision_model(model_type) or is_qwen3_5_model(model_type):
        return SequencePackingMode.PADDED
    return SequencePackingMode.WRAPPER_THD


def requires_padded_seq(model_type: str) -> bool:
    # [FUNCTION DOC]
    # Returns True for Qwen3.5 models (GDN/SSM kernels reject packed input), meaning the
    # padded forward path must be used.
    """Whether the model must run the padded (BSHD) forward instead of packed (THD).

    GDN/SSM models (currently the Qwen3.5 family) reject packed sequences in their
    attention/SSM kernels, so they must run on padded ``[B, S]`` input. THD stays
    the default for every other model.
    """
    return is_qwen3_5_model(model_type)


# Copied from trl
def disable_dropout_in_model(model: torch.nn.Module) -> None:
    # [FUNCTION DOC]
    # Sets p=0 on every torch.nn.Dropout submodule of model, in place. Returns None.
    # Does not affect functional dropout or non-Dropout attention dropout settings.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
