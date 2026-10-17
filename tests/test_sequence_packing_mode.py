# [MODULE DOC]
# Unit tests for resolve_sequence_packing_mode and supports_model_packed_seq, which
# choose how sequences are packed for Megatron depending on model type and bridge
# implementation (MODEL_THD, PADDED or WRAPPER_THD).
# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.core.model import (
    SequencePackingMode,
    resolve_sequence_packing_mode,
    supports_model_packed_seq,
)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
def test_qwen3_vl_family_uses_model_thd_with_megatron_bridge(model_type):
    # [FUNCTION DOC]
    # Parametrized on Qwen3-VL variants; asserts megatron-bridge supports model-level
    # packed sequences and resolves to MODEL_THD.
    assert supports_model_packed_seq(model_type, "megatron-bridge")
    assert (
        resolve_sequence_packing_mode(model_type, "megatron-bridge")
        == SequencePackingMode.MODEL_THD
    )


@pytest.mark.parametrize(
    ("model_type", "bridge_type"),
    [
        ("qwen3_vl", "mbridge"),
        ("qwen3_vl_moe", "mbridge"),
        ("qwen2_5_vl", "megatron-bridge"),
        ("qwen3_5", "megatron-bridge"),
        ("qwen3_5_moe", "megatron-bridge"),
    ],
)
def test_models_without_gpu_model_thd_contract_stay_padded(model_type, bridge_type):
    # [FUNCTION DOC]
    # Parametrized (model_type, bridge_type) combos lacking a model THD contract
    # (Qwen3-VL with mbridge, Qwen2.5-VL, Qwen3.5 variants); asserts no packed-seq
    # support and PADDED mode.
    assert not supports_model_packed_seq(model_type, bridge_type)
    assert (
        resolve_sequence_packing_mode(model_type, bridge_type)
        == SequencePackingMode.PADDED
    )


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "llama"])
def test_text_models_keep_wrapper_thd(model_type):
    # [FUNCTION DOC]
    # Parametrized on text-only models; asserts megatron-bridge resolves to WRAPPER_THD.
    assert (
        resolve_sequence_packing_mode(model_type, "megatron-bridge")
        == SequencePackingMode.WRAPPER_THD
    )
