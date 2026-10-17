# [MODULE DOC]
# Integration contract tests for native AWEX Qwen3-VL colocate support (skipped unless
# awex.models.qwen3_vl is importable): AWEX registry entries, weight converter
# construction with a composite VL config, vision-tower TP sharding, and the AReaL
# colocate reader helpers _get_awex_infer_hf_config and _get_router_dtype.
# SPDX-License-Identifier: Apache-2.0

"""Integration contracts for native AWEX Qwen3-VL colocate support."""

from types import SimpleNamespace

import pytest

qwen3_vl = pytest.importorskip(
    "awex.models.qwen3_vl",
    reason="requires an AWEX release with native Qwen3-VL support",
)

from awex.models.registry import (  # noqa: E402
    ModelRegistry,
    get_infer_weights_converter,
    get_sharding_strategy,
)
from awex.sharding.param_sharding import ShardingType  # noqa: E402

from areal.engine.awex.colocate_reader import (  # noqa: E402
    _get_awex_infer_hf_config,
    _get_router_dtype,
)


class _CompositeVLConfig:
    # [CLASS DOC]
    # HELPER: Fake composite Qwen3-VL HF config with architectures, a text_config
    # namespace (layers, sizes, heads, router_dtype) and a vision_config namespace.
    def __init__(self, architectures=None, router_dtype="fp32"):
        # [FUNCTION DOC]
        # HELPER: Stores architectures and builds text_config (28 layers, tiny dims,
        # router_dtype) and vision_config (2 heads, hidden 4).
        self.architectures = architectures
        self.text_config = SimpleNamespace(
            num_hidden_layers=28,
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            router_dtype=router_dtype,
        )
        self.vision_config = SimpleNamespace(num_heads=2, hidden_size=4)

    def to_dict(self):
        # [FUNCTION DOC]
        # HELPER: Returns a dict with architectures, model_type "qwen3_vl" and the
        # text/vision configs as plain dicts.
        return {
            "architectures": self.architectures,
            "model_type": "qwen3_vl",
            "text_config": vars(self.text_config),
            "vision_config": vars(self.vision_config),
        }


def _rank_info():
    # [FUNCTION DOC]
    # HELPER: Returns a SimpleNamespace rank info with all parallel ranks 0 and sizes 1.
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
    )


def _infer_engine_config():
    # [FUNCTION DOC]
    # HELPER: Returns a fake inference engine config (tp 1, ep 1, cpu backend).
    return SimpleNamespace(tp_size=1, ep_size=1, device_backend="cpu")


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"],
)
def test_awex_registry_provides_native_qwen3_vl_entries(architecture):
    # [FUNCTION DOC]
    # For dense and MoE Qwen3-VL architectures, asserts the AWEX registry maps them to
    # the Qwen3-VL SGLang converter and sharding strategy.
    entry = ModelRegistry.models[architecture]

    assert entry["sglang_converter"] is qwen3_vl.Qwen3VLSGlangToHFWeightConverter
    assert entry["sharding_strategy"] is qwen3_vl.Qwen3VLShardingStrategy
    assert get_sharding_strategy(architecture) is qwen3_vl.Qwen3VLShardingStrategy


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"],
)
def test_colocate_infer_converter_receives_composite_vl_config(architecture):
    # [FUNCTION DOC]
    # Asserts get_infer_weights_converter("sglang", ...) returns a Qwen3-VL converter
    # holding the composite config as vl_model_config and its text_config as
    # model_config.
    config = _CompositeVLConfig(architectures=[architecture])

    converter = get_infer_weights_converter(
        "sglang",
        architecture,
        config,
        _rank_info(),
        _infer_engine_config(),
    )

    assert isinstance(converter, qwen3_vl.Qwen3VLSGlangToHFWeightConverter)
    assert converter.vl_model_config is config
    assert converter.model_config is config.text_config


@pytest.mark.parametrize(
    ("name", "expected_dim"),
    [
        ("model.visual.blocks.0.attn.qkv.weight", 0),
        ("model.visual.blocks.0.attn.proj.weight", 1),
    ],
)
def test_colocate_vision_tower_uses_awex_tp_sharding(name, expected_dim):
    # [FUNCTION DOC]
    # Builds a Qwen3VLShardingStrategy with tp_size 2 and asserts vision attention qkv
    # shards on dim 0 and proj on dim 1 with TP_SHARDING over 2 ranks.
    rank_info = _rank_info()
    rank_info.tp_size = 2
    strategy = qwen3_vl.Qwen3VLShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    assert strategy.get_sharding_strategy(name) == (
        ShardingType.TP_SHARDING,
        expected_dim,
        2,
    )


def test_colocate_reader_serializes_complete_vl_config():
    # [FUNCTION DOC]
    # Model instance of a class named Qwen3VLForConditionalGeneration with the composite
    # config: asserts the serialized AWEX config keeps architectures, model_type, text
    # and vision configs.
    config = _CompositeVLConfig(architectures=["Qwen3VLForConditionalGeneration"])

    class Qwen3VLForConditionalGeneration:
        # [CLASS DOC]
        # HELPER: Empty stand-in class whose name provides the runtime architecture.
        pass

    model = Qwen3VLForConditionalGeneration()
    model.config = config

    awex_config = _get_awex_infer_hf_config(model)

    assert awex_config.architectures == ["Qwen3VLForConditionalGeneration"]
    assert awex_config.model_type == "qwen3_vl"
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config == {"num_heads": 2, "hidden_size": 4}


def test_colocate_reader_fills_missing_runtime_architecture():
    # [FUNCTION DOC]
    # With architectures=None, asserts the reader fills architectures from the model
    # class name and keeps text/vision configs.
    config = _CompositeVLConfig(architectures=None)

    class Qwen3VLForConditionalGeneration:
        # [CLASS DOC]
        # HELPER: Empty stand-in class whose name provides the runtime architecture.
        pass

    model = Qwen3VLForConditionalGeneration()
    model.config = config

    awex_config = _get_awex_infer_hf_config(model)

    assert awex_config.architectures == ["Qwen3VLForConditionalGeneration"]
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config["num_heads"] == 2


def test_colocate_reader_uses_composite_config_for_vl_moe():
    # [FUNCTION DOC]
    # MoE model whose config is only the text config: asserts the reader uses
    # model_runner.model_config.hf_config (composite) for architectures, text and vision
    # configs.
    """Use ModelRunner's composite config when the runtime model keeps text only."""
    config = _CompositeVLConfig(architectures=["Qwen3VLMoeForConditionalGeneration"])

    class Qwen3VLMoeForConditionalGeneration:
        # [CLASS DOC]
        # HELPER: Empty stand-in class for the MoE architecture name.
        pass

    model = Qwen3VLMoeForConditionalGeneration()
    model.config = config.text_config
    model_runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))

    awex_config = _get_awex_infer_hf_config(model, model_runner)

    assert awex_config.architectures == ["Qwen3VLMoeForConditionalGeneration"]
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config == {"num_heads": 2, "hidden_size": 4}


def test_colocate_reader_reads_router_dtype_from_text_config():
    # [FUNCTION DOC]
    # Asserts _get_router_dtype reads router_dtype from a composite config's text_config
    # and from a flat config.
    config = _CompositeVLConfig(router_dtype="fp32")

    assert _get_router_dtype(config) == "fp32"
    assert _get_router_dtype(SimpleNamespace(router_dtype="bf16")) == "bf16"
