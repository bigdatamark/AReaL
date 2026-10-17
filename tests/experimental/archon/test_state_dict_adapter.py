# [MODULE DOC]
# Unit tests for Archon state-dict adapters (HF <-> Archon key/tensor conversion).
# Covers Qwen3StateDictAdapter (dense and MoE roundtrips, plus a CUDA forward-equality
# check after an Archon->HF->Archon roundtrip) and, when importable (transformers >=
# 5.2), Qwen3_5StateDictAdapter (hybrid full/linear attention key mapping, MoE 2D<->3D
# expert stacking, full-coverage and full state-dict roundtrips). Module-level
# constants: _HAS_QWEN3_5_ADAPTER (import probe), _requires_qwen3_5 skip marker,
# _QWEN3_5_EXACT tolerances (1e-6), and _QWEN3_5_LAYER_TYPES (3 linear_attention layers
# then 1 full_attention layer).
"""Tests for Archon State Dict Adapter roundtrip conversion.

These tests verify key conversion between HuggingFace and Archon formats
by testing full roundtrip conversions (HF -> Archon -> HF).

Run tests:
    pytest tests/experimental/archon/test_state_dict_adapter.py -v

Note: For E2E weight sync tests with real models, see test_weight_sync.py
and test_checkpoint_e2e.py.
"""

import pytest
import torch
from torch.testing import assert_close

from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

try:
    from areal.experimental.models.archon.qwen3_5.model.state_dict_adapter import (
        Qwen3_5StateDictAdapter,
    )

    _HAS_QWEN3_5_ADAPTER = True
except ImportError:
    _HAS_QWEN3_5_ADAPTER = False

# =============================================================================
# Mock Configs
# =============================================================================


class MockQwen3Config:
    # [CLASS DOC]
    # HELPER: Minimal dense config stub for Qwen3StateDictAdapter (no experts, untied
    # embeddings).
    # KNOWN ISSUE: model_type is "qwen2" although it is described as a Qwen3 config;
    # harmless only if the adapter ignores model_type.
    """Mock Qwen3 model config for testing."""

    model_type = "qwen2"
    num_local_experts = 0  # Dense model
    tie_word_embeddings = False


class MockQwen3MoEConfig:
    # [CLASS DOC]
    # HELPER: MoE config stub with model_type "qwen3_moe", 8 local experts, untied
    # embeddings.
    """Mock Qwen3 MoE model config for testing."""

    model_type = "qwen3_moe"
    num_local_experts = 8
    tie_word_embeddings = False


# =============================================================================
# Dense Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterDense:
    # [CLASS DOC]
    # Roundtrip tests of Qwen3StateDictAdapter on a dense single-layer state dict.
    """Roundtrip tests for Qwen3StateDictAdapter with dense models."""

    @pytest.fixture
    def adapter(self):
        # [FUNCTION DOC]
        # HELPER: Pytest fixture returning a Qwen3StateDictAdapter built from
        # MockQwen3Config.
        return Qwen3StateDictAdapter(MockQwen3Config())

    def test_roundtrip_dense_state_dict(self, adapter):
        # [FUNCTION DOC]
        # Builds a random HF state dict for one layer (embeddings, q/k/v/o, q/k norms,
        # gate/up/ down MLP, two layer norms, final norm, lm_head), converts with
        # from_hf and asserts the exact Archon key set (tok_embeddings,
        # attention.wq/wk/wv/wo, q_norm/k_norm, feed_forward.w1/w3/w2, attention_norm,
        # ffn_norm, norm, output). Converts back with to_hf and asserts each original
        # key (except rotary_emb ones) exists with allclose values. Allocates several
        # large CPU tensors (32000x4096 etc.).
        """Test full roundtrip conversion for dense model."""
        # Create a minimal HF state dict
        hf_state = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(128),
            "model.layers.0.self_attn.k_norm.weight": torch.randn(128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
            "model.layers.0.input_layernorm.weight": torch.randn(4096),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(4096),
            "model.norm.weight": torch.randn(4096),
            "lm_head.weight": torch.randn(32000, 4096),
        }

        # HF -> Archon
        archon_state = adapter.from_hf(hf_state)

        # Verify Archon keys
        expected_archon_keys = {
            "tok_embeddings.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.attention.q_norm.weight",
            "layers.0.attention.k_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "norm.weight",
            "output.weight",
        }
        assert set(archon_state.keys()) == expected_archon_keys

        # Archon -> HF
        roundtrip_state = adapter.to_hf(archon_state)

        # Verify roundtrip
        for key in hf_state:
            if "rotary_emb" in key:
                continue
            assert key in roundtrip_state, f"Missing key: {key}"
            assert torch.allclose(hf_state[key], roundtrip_state[key]), (
                f"Mismatch at {key}"
            )


# =============================================================================
# MoE Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterMoE:
    # [CLASS DOC]
    # Roundtrip tests for MoE expert and router weights of Qwen3StateDictAdapter.
    """Roundtrip tests for Qwen3StateDictAdapter with MoE models."""

    @pytest.fixture
    def adapter(self):
        # [FUNCTION DOC]
        # HELPER: Pytest fixture returning a Qwen3StateDictAdapter built from
        # MockQwen3MoEConfig.
        return Qwen3StateDictAdapter(MockQwen3MoEConfig())

    def test_moe_roundtrip(self, adapter):
        # [FUNCTION DOC]
        # Starts from stacked 3D Archon expert tensors w1/w2/w3 for 8 experts, asserts
        # to_hf yields 24 per-expert 2D keys, then from_hf restores each original key
        # with allclose values.
        """Test roundtrip conversion for MoE experts."""
        # Create 3D Archon weight
        archon_state = {
            "layers.0.moe.experts.w1": torch.randn(8, 11008, 4096),
            "layers.0.moe.experts.w2": torch.randn(8, 4096, 11008),
            "layers.0.moe.experts.w3": torch.randn(8, 11008, 4096),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 keys (8 experts x 3 projections)
        assert len(hf_state) == 24

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip
        for key in archon_state:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

    def test_full_moe_state_dict_roundtrip(self, adapter):
        # [FUNCTION DOC]
        # Like test_moe_roundtrip but adds the router gate weight; asserts 25 HF keys
        # including "model.layers.0.mlp.gate.weight" and that experts and router
        # roundtrip unchanged.
        """Test roundtrip for complete MoE layer state dict including router."""
        # Create full MoE layer state dict
        archon_state = {
            # Expert weights
            "layers.0.moe.experts.w1": torch.randn(8, 1024, 512),
            "layers.0.moe.experts.w2": torch.randn(8, 512, 1024),
            "layers.0.moe.experts.w3": torch.randn(8, 1024, 512),
            # Router
            "layers.0.moe.router.gate.weight": torch.randn(8, 512),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 expert keys + 1 router key
        assert len(hf_state) == 25
        assert "model.layers.0.mlp.gate.weight" in hf_state

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip for experts
        for key in [
            "layers.0.moe.experts.w1",
            "layers.0.moe.experts.w2",
            "layers.0.moe.experts.w3",
        ]:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

        # Verify roundtrip for router
        router_key = "layers.0.moe.router.gate.weight"
        assert router_key in roundtrip_state
        assert torch.allclose(archon_state[router_key], roundtrip_state[router_key])


# =============================================================================
# MoE Model Forward Match Test with Adapter Roundtrip
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEWeightLoadingRoundtrip:
    # [CLASS DOC]
    # CUDA-only test that adapter roundtrip preserves MoE model numerics end to end.
    """Test MoE weight loading roundtrip with forward pass verification."""

    def test_moe_weight_roundtrip_forward_match(self):
        # [FUNCTION DOC]
        # Builds a tiny Qwen3 MoE model (dim 64, 2 layers, 4 experts top-2, no grouped
        # mm, SDPA) on CUDA, initializes weights/buffers, clones its state dict,
        # converts Archon->HF->Archon with an adapter configured for 4 experts, loads
        # the result non-strictly into a second freshly built model (expert_bias is
        # Archon-only), then runs both on the same seeded 16-token packed sequence
        # (cu_seqlens [0,16]) under no_grad and asserts outputs match within 1e-5. Side
        # effects: GPU allocation, global RNG seeding.
        """Test MoE forward output matches after Archon->HF->Archon roundtrip.

        This test verifies that the state dict adapter correctly converts
        MoE weights between Archon and HuggingFace formats by:
        1. Creating a model and running forward pass
        2. Converting state dict: Archon -> HF -> Archon via adapter
        3. Loading converted weights into a new model
        4. Verifying forward outputs match
        """
        from areal.experimental.models.archon.moe import MoEArgs
        from areal.experimental.models.archon.qwen3 import Qwen3Model, Qwen3ModelArgs

        # Create MoE model args
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=32,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,  # All layers are MoE
            attn_type="sdpa",
        )

        # Create and initialize first model
        model1 = Qwen3Model(model_args).cuda()
        model1.init_weights()
        model1.init_buffers(buffer_device=torch.device("cuda"))

        # Get original Archon state dict
        archon_state = {k: v.clone() for k, v in model1.state_dict().items()}

        # Create adapter config that matches the model's num_experts
        class AdapterConfig:
            # [CLASS DOC]
            # HELPER: Local adapter config stub (qwen3_moe, 4 experts, untied
            # embeddings) matching the test model.
            model_type = "qwen3_moe"
            num_local_experts = 4  # Must match MoEArgs(num_experts=4)
            tie_word_embeddings = False

        # Create adapter and perform roundtrip conversion
        adapter = Qwen3StateDictAdapter(AdapterConfig())

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Create second model and load roundtrip weights
        # Note: strict=False because expert_bias is an Archon-only buffer
        model2 = Qwen3Model(model_args).cuda()
        model2.init_buffers(buffer_device=torch.device("cuda"))
        model2.load_state_dict(roundtrip_state, strict=False)

        # Create test input (packed sequence format)
        torch.manual_seed(42)
        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        positions = torch.arange(16, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
        max_seqlen = 16

        # Forward pass on both models
        with torch.no_grad():
            output1 = model1(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            output2 = model2(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        # Verify outputs match
        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-5), (
            f"Forward output mismatch after adapter roundtrip. "
            f"Max diff: {(output1 - output2).abs().max()}"
        )


# =============================================================================
# Qwen3.5 State Dict Adapter Tests
# =============================================================================


_requires_qwen3_5 = pytest.mark.skipif(
    not _HAS_QWEN3_5_ADAPTER,
    reason="Qwen3.5 adapter not available (requires transformers >= 5.2)",
)

_QWEN3_5_EXACT = dict(rtol=1e-6, atol=1e-6)

_QWEN3_5_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


class _Qwen3_5DenseConfig:
    # [CLASS DOC]
    # HELPER: Minimal Qwen3.5 dense config: hidden 256, 4 layers using
    # _QWEN3_5_LAYER_TYPES, 4 heads / 2 KV heads, head_dim 128, GatedDeltaNet linear-
    # attention dims (conv kernel 4, key/value head dim 32, 4 key heads, 8 value heads),
    # FFN 512, untied embeddings.
    """Minimal dense config for Qwen3.5 adapter tests."""

    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    vocab_size = 1000
    head_dim = 128
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    partial_rotary_factor = 0.25
    tie_word_embeddings = False
    layer_types = list(_QWEN3_5_LAYER_TYPES)
    linear_conv_kernel_dim = 4
    linear_key_head_dim = 32
    linear_value_head_dim = 32
    linear_num_key_heads = 4
    linear_num_value_heads = 8
    attention_bias = False


class _Qwen3_5MoEConfig:
    # [CLASS DOC]
    # HELPER: Same as the dense Qwen3.5 config plus MoE fields: 4 experts, top-2, expert
    # FFN 128, shared expert FFN 512, moe_enabled True.
    """Minimal MoE config for Qwen3.5 adapter tests."""

    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    vocab_size = 1000
    head_dim = 128
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    partial_rotary_factor = 0.25
    tie_word_embeddings = False
    layer_types = list(_QWEN3_5_LAYER_TYPES)
    linear_conv_kernel_dim = 4
    linear_key_head_dim = 32
    linear_value_head_dim = 32
    linear_num_key_heads = 4
    linear_num_value_heads = 8
    attention_bias = False
    # MoE
    num_experts = 4
    num_experts_per_tok = 2
    moe_intermediate_size = 128
    shared_expert_intermediate_size = 512
    moe_enabled = True


# ---------------------------------------------------------------------------
# Qwen3.5 Helpers
# ---------------------------------------------------------------------------
def _all_full_attn_hf_keys(layer_idx: int) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns the six HF key names (q/k/v/o proj weights and q/k norm weights)
    # of a full_attention layer at layer_idx.
    """All HF keys for a Qwen3.5 full_attention layer."""
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ]
    return [f"model.layers.{layer_idx}.{s}" for s in suffixes]


def _all_linear_attn_hf_keys(layer_idx: int) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns HF key names of a linear_attention (GatedDeltaNet) layer at
    # layer_idx: seven ".weight" params (in_proj_qkv/z/a/b, conv1d, out_proj, norm)
    # followed by the bare parameters A_log and dt_bias.
    """All HF keys for a Qwen3.5 linear_attention (GatedDeltaNet) layer."""
    weight_suffixes = [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.out_proj.weight",
        "linear_attn.norm.weight",
    ]
    bare_suffixes = [
        "linear_attn.A_log",
        "linear_attn.dt_bias",
    ]
    keys = [f"model.layers.{layer_idx}.{s}" for s in weight_suffixes]
    keys += [f"model.layers.{layer_idx}.{s}" for s in bare_suffixes]
    return keys


def _common_per_layer_hf_keys(layer_idx: int, moe: bool = False) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns per-layer HF keys shared by all layer types: input and post-
    # attention layernorm weights, plus either the MoE router gate weight (moe=True) or
    # dense gate/up/down projection weights (moe=False, default). Expert weights are not
    # included.
    """Common per-layer HF keys (norms + FFN)."""
    keys = [
        f"model.layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    ]
    if moe:
        keys.append(f"model.layers.{layer_idx}.mlp.gate.weight")
    else:
        keys += [
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight",
        ]
    return keys


def _global_hf_keys() -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns the non-layer HF keys: embed_tokens, final norm and lm_head
    # weights.
    """Global (non-layer) HF keys for Qwen3.5."""
    return [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def _generate_qwen3_5_hf_state_dict(
    config: _Qwen3_5DenseConfig | _Qwen3_5MoEConfig,
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds a full random HF-format Qwen3.5 state dict for the given config:
    # global tensors, per-layer norms, then per layer_types either gated full attention
    # (q_proj has 2x output rows for the gate) or GatedDeltaNet tensors (qkv dim =
    # 2*nk*dk + nv*dv, conv1d shaped [qkv, 1, kernel], A_log/dt_bias of size nv), and
    # either per-expert 2D MoE weights plus router (if config has moe_enabled true) or
    # dense MLP weights. Returns dict of key to CPU tensor. Uses global torch RNG.
    """Generate a complete dummy HF state dict with random tensors."""
    sd: dict[str, torch.Tensor] = {}
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    is_moe = hasattr(config, "moe_enabled") and config.moe_enabled

    # Global
    sd["model.embed_tokens.weight"] = torch.randn(config.vocab_size, dim)
    sd["model.norm.weight"] = torch.randn(dim)
    sd["lm_head.weight"] = torch.randn(config.vocab_size, dim)

    for i in range(config.num_hidden_layers):
        lt = config.layer_types[i]

        # Norms (common to all layer types)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(dim)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(dim)

        if lt == "full_attention":
            # q_proj is 2x for gated attention
            sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(
                n_heads * head_dim * 2, dim
            )
            sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(
                n_kv_heads * head_dim, dim
            )
            sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(
                n_kv_heads * head_dim, dim
            )
            sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(
                dim, n_heads * head_dim
            )
            sd[f"model.layers.{i}.self_attn.q_norm.weight"] = torch.randn(head_dim)
            sd[f"model.layers.{i}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        else:
            # linear_attention (GatedDeltaNet)
            nk = config.linear_num_key_heads
            nv = config.linear_num_value_heads
            dk = config.linear_key_head_dim
            dv = config.linear_value_head_dim
            qkv_dim = 2 * nk * dk + nv * dv
            sd[f"model.layers.{i}.linear_attn.in_proj_qkv.weight"] = torch.randn(
                qkv_dim, dim
            )
            sd[f"model.layers.{i}.linear_attn.in_proj_z.weight"] = torch.randn(
                nv * dv, dim
            )
            sd[f"model.layers.{i}.linear_attn.in_proj_a.weight"] = torch.randn(nv, dim)
            sd[f"model.layers.{i}.linear_attn.in_proj_b.weight"] = torch.randn(nv, dim)
            sd[f"model.layers.{i}.linear_attn.conv1d.weight"] = torch.randn(
                qkv_dim, 1, config.linear_conv_kernel_dim
            )
            sd[f"model.layers.{i}.linear_attn.out_proj.weight"] = torch.randn(
                dim, nv * dv
            )
            sd[f"model.layers.{i}.linear_attn.norm.weight"] = torch.randn(dv)
            # Bare parameters
            sd[f"model.layers.{i}.linear_attn.A_log"] = torch.randn(nv)
            sd[f"model.layers.{i}.linear_attn.dt_bias"] = torch.randn(nv)

        # FFN / MoE
        if is_moe:
            num_exp = config.num_experts
            moe_dim = config.moe_intermediate_size
            sd[f"model.layers.{i}.mlp.gate.weight"] = torch.randn(num_exp, dim)
            for e in range(num_exp):
                sd[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                    moe_dim, dim
                )
                sd[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                    moe_dim, dim
                )
                sd[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                    dim, moe_dim
                )
        else:
            sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(hidden_dim, dim)
            sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(hidden_dim, dim)
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(dim, hidden_dim)

    return sd


# =============================================================================
# Qwen3.5 Key Mapping Tests
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5KeyMapping:
    # [CLASS DOC]
    # Key-level roundtrip tests of Qwen3_5StateDictAdapter (skipped without the
    # adapter).
    """Key mapping roundtrip for Qwen3.5 (full_attention + linear_attention)."""

    def test_key_mapping_full_attention_roundtrip(self):
        # [FUNCTION DOC]
        # For layer 3 (full_attention) asserts every HF key maps via the private
        # _convert_key_from_hf to non-None and back to itself via _convert_key_to_hf;
        # spot-checks q_proj -> attention.wq and q_norm -> attention.q_norm.
        """HF full_attention key → Archon → HF roundtrip."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())

        # Layer 3 is full_attention in our test config
        hf_keys = _all_full_attn_hf_keys(3)
        for hf_key in hf_keys:
            archon_key = adapter._convert_key_from_hf(hf_key)
            assert archon_key is not None, f"HF key unmapped: {hf_key}"

            roundtrip_key = adapter._convert_key_to_hf(archon_key)
            assert roundtrip_key == hf_key, (
                f"Roundtrip mismatch: {hf_key} → {archon_key} → {roundtrip_key}"
            )

        # Verify expected Archon key format
        wq = adapter._convert_key_from_hf("model.layers.3.self_attn.q_proj.weight")
        assert wq == "layers.3.attention.wq.weight"
        q_norm = adapter._convert_key_from_hf("model.layers.3.self_attn.q_norm.weight")
        assert q_norm == "layers.3.attention.q_norm.weight"

    def test_key_mapping_linear_attention_roundtrip(self):
        # [FUNCTION DOC]
        # For layer 0 (linear_attention) asserts all keys roundtrip and that A_log,
        # dt_bias and conv1d map to "layers.0.linear_attn.*" names unchanged apart from
        # the "model." prefix.
        """HF linear_attention key → Archon → HF roundtrip."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())

        # Layer 0 is linear_attention in our test config
        hf_keys = _all_linear_attn_hf_keys(0)
        for hf_key in hf_keys:
            archon_key = adapter._convert_key_from_hf(hf_key)
            assert archon_key is not None, f"HF key unmapped: {hf_key}"

            roundtrip_key = adapter._convert_key_to_hf(archon_key)
            assert roundtrip_key == hf_key, (
                f"Roundtrip mismatch: {hf_key} → {archon_key} → {roundtrip_key}"
            )

        # Verify bare params (no .weight suffix)
        a_log = adapter._convert_key_from_hf("model.layers.0.linear_attn.A_log")
        assert a_log == "layers.0.linear_attn.A_log"
        dt_bias = adapter._convert_key_from_hf("model.layers.0.linear_attn.dt_bias")
        assert dt_bias == "layers.0.linear_attn.dt_bias"

        # Verify conv1d
        conv = adapter._convert_key_from_hf("model.layers.0.linear_attn.conv1d.weight")
        assert conv == "layers.0.linear_attn.conv1d.weight"

    def test_key_mapping_skip_rotary(self):
        # [FUNCTION DOC]
        # Asserts a rotary_emb.inv_freq key maps to None (dropped).
        """Rotary emb inv_freq should map to None (skipped)."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())
        result = adapter._convert_key_from_hf(
            "model.layers.3.self_attn.rotary_emb.inv_freq"
        )
        assert result is None


# =============================================================================
# Qwen3.5 MoE Key Mapping
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5MoEKeyMapping:
    # [CLASS DOC]
    # MoE expert 2D<->3D conversion tests for Qwen3_5StateDictAdapter.
    """MoE expert 2D↔3D conversion for Qwen3.5."""

    def test_key_mapping_moe_roundtrip(self):
        # [FUNCTION DOC]
        # Creates per-expert 2D gate/up/down weights for layer 1, asserts from_hf
        # produces stacked w1 [E, moe, dim], w2 [E, dim, moe], w3 [E, moe, dim] with
        # w1[e] equal to expert e's gate_proj, then asserts to_hf restores every
        # original key exactly.
        """MoE expert weights 2D ↔ 3D roundtrip."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        dim = config.hidden_size
        moe_dim = config.moe_intermediate_size
        num_exp = config.num_experts

        # Create HF-format 2D expert weights for layer 1
        hf_sd = {}
        for e in range(num_exp):
            hf_sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                moe_dim, dim
            )
            hf_sd[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                moe_dim, dim
            )
            hf_sd[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                dim, moe_dim
            )

        # HF → Archon: should produce 3D tensors
        archon_sd = adapter.from_hf(hf_sd)
        assert "layers.1.moe.experts.w1" in archon_sd
        assert "layers.1.moe.experts.w2" in archon_sd
        assert "layers.1.moe.experts.w3" in archon_sd
        assert archon_sd["layers.1.moe.experts.w1"].shape == (num_exp, moe_dim, dim)
        assert archon_sd["layers.1.moe.experts.w2"].shape == (num_exp, dim, moe_dim)
        assert archon_sd["layers.1.moe.experts.w3"].shape == (num_exp, moe_dim, dim)

        # Verify per-expert data integrity (w1 = gate_proj)
        for e in range(num_exp):
            assert_close(
                archon_sd["layers.1.moe.experts.w1"][e],
                hf_sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"],
                **_QWEN3_5_EXACT,
            )

        # Archon → HF: should produce 2D tensors
        roundtrip_sd = adapter.to_hf(archon_sd)
        for key in hf_sd:
            assert key in roundtrip_sd, f"Missing roundtrip key: {key}"
            assert_close(roundtrip_sd[key], hf_sd[key], **_QWEN3_5_EXACT)


# =============================================================================
# Qwen3.5 Key Coverage
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5KeyCoverage:
    # [CLASS DOC]
    # Checks that no expected Qwen3.5 HF key is left unmapped by the adapter.
    """Verify complete HF key coverage for Qwen3.5."""

    def test_all_hf_keys_covered(self):
        # [FUNCTION DOC]
        # Collects all expected dense keys across the 4 layers plus global keys and
        # asserts every one maps to a non-None Archon key.
        """All HF keys map to Archon keys (zero unmapped)."""
        config = _Qwen3_5DenseConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        expected_keys: list[str] = []
        for i in range(config.num_hidden_layers):
            lt = config.layer_types[i]
            if lt == "full_attention":
                expected_keys += _all_full_attn_hf_keys(i)
            else:
                expected_keys += _all_linear_attn_hf_keys(i)
            expected_keys += _common_per_layer_hf_keys(i, moe=False)
        expected_keys += _global_hf_keys()

        unmapped = [k for k in expected_keys if adapter._convert_key_from_hf(k) is None]
        assert unmapped == [], f"Unmapped HF keys: {unmapped}"

    def test_all_hf_keys_covered_moe(self):
        # [FUNCTION DOC]
        # Asserts the MoE router key maps and that every per-expert projection key is
        # recognized by the private _parse_expert_key.
        """All MoE HF keys map correctly (router + experts via parse)."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        # Test router key
        router_key = "model.layers.0.mlp.gate.weight"
        assert adapter._convert_key_from_hf(router_key) is not None

        # Test expert keys parse correctly
        for e in range(config.num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                hf_key = f"model.layers.0.mlp.experts.{e}.{proj}.weight"
                parsed = adapter._parse_expert_key(hf_key)
                assert parsed is not None, f"Expert key not parseable: {hf_key}"


# =============================================================================
# Qwen3.5 Full State Dict Roundtrip
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5StateDictRoundtrip:
    # [CLASS DOC]
    # Full HF -> Archon -> HF state-dict roundtrips for Qwen3.5 (dense and MoE).
    """Full HF → Archon → HF roundtrip for Qwen3.5."""

    def test_state_dict_adapter_roundtrip(self):
        # [FUNCTION DOC]
        # Generates a dense dummy HF state dict, roundtrips it, asserts identical key
        # sets (reporting missing/extra) and per-key exact tensor equality.
        """Complete state dict roundtrip with per-key exact match."""
        config = _Qwen3_5DenseConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        hf_sd = _generate_qwen3_5_hf_state_dict(config)
        archon_sd = adapter.from_hf(hf_sd)
        roundtrip_sd = adapter.to_hf(archon_sd)

        # All original keys must be present
        assert set(roundtrip_sd.keys()) == set(hf_sd.keys()), (
            f"Key mismatch.\n"
            f"Missing: {set(hf_sd.keys()) - set(roundtrip_sd.keys())}\n"
            f"Extra: {set(roundtrip_sd.keys()) - set(hf_sd.keys())}"
        )

        # Per-key tensor equality
        for key in hf_sd:
            assert_close(
                roundtrip_sd[key],
                hf_sd[key],
                **_QWEN3_5_EXACT,
                msg=f"Tensor mismatch for key: {key}",
            )

    def test_state_dict_adapter_roundtrip_moe(self):
        # [FUNCTION DOC]
        # Same as the dense roundtrip test but with the MoE config, exercising expert
        # stacking and unstacking.
        """MoE state dict roundtrip with expert 2D↔3D."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        hf_sd = _generate_qwen3_5_hf_state_dict(config)
        archon_sd = adapter.from_hf(hf_sd)
        roundtrip_sd = adapter.to_hf(archon_sd)

        assert set(roundtrip_sd.keys()) == set(hf_sd.keys()), (
            f"Key mismatch.\n"
            f"Missing: {set(hf_sd.keys()) - set(roundtrip_sd.keys())}\n"
            f"Extra: {set(roundtrip_sd.keys()) - set(hf_sd.keys())}"
        )

        for key in hf_sd:
            assert_close(
                roundtrip_sd[key],
                hf_sd[key],
                **_QWEN3_5_EXACT,
                msg=f"Tensor mismatch for key: {key}",
            )
