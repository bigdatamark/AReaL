# [MODULE DOC]
# Tests that Archon <-> HF weight conversion used for weight sync is complete and
# correct: meta-device completeness/shape checks for dense and (slow) MoE models loaded
# from local model paths in tests.experimental.archon.utils, a CUDA value-match test
# against a real HF checkpoint, and CPU tests that per-parameter convert_single_to_hf
# (used by update_weights_from_dist) agrees with batch to_hf. Defines mock adapter
# configs and small CPU model fixtures.
"""Tests for weight synchronization and completeness verification.

These tests verify:
1. Weight completeness: All Archon parameters can be converted to HF format
2. Bidirectional completeness: HF keys <-> Archon keys mapping is complete
3. Shape matching: Converted weight shapes match HF model shapes
4. Value matching: Weight values are preserved after conversion (slow test)
5. Iterative conversion consistency: convert_single_to_hf matches batch to_hf

Run tests:
    pytest tests/experimental/archon/test_weight_sync.py -v

Note: Most tests use meta device for fast execution without GPU memory.
The slow test (test_archon_weights_match_hf) requires CUDA.
"""

import pytest
import torch
import torch.distributed as dist

from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

# =============================================================================
# Mock Configs for Iterative Conversion Tests
# =============================================================================


class MockDenseConfig:
    # [CLASS DOC]
    # HELPER: Adapter config stub for a dense Qwen3 (model_type "qwen3", no experts,
    # untied embeddings).
    """Mock Qwen3 dense model config."""

    model_type = "qwen3"
    num_local_experts = 0
    tie_word_embeddings = False


class MockMoEConfig:
    # [CLASS DOC]
    # HELPER: Adapter config stub for Qwen3 MoE (model_type "qwen3_moe", 4 local
    # experts, untied embeddings); expert count matches small_moe_model_args.
    """Mock Qwen3 MoE model config."""

    model_type = "qwen3_moe"
    num_local_experts = 4
    tie_word_embeddings = False


# =============================================================================
# Fixtures for Iterative Conversion Tests
# =============================================================================


@pytest.fixture
def small_dense_model_args():
    # [FUNCTION DOC]
    # HELPER: Fixture returning tiny dense Qwen3ModelArgs (dim 64, FFN 128, 4 heads/2
    # KV, head_dim 16, 2 layers, vocab 1000, SDPA, MoE off).
    """Create small dense model args for CPU testing."""
    from areal.experimental.models.archon.qwen3 import Qwen3ModelArgs

    return Qwen3ModelArgs(
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        n_layers=2,
        vocab_size=1000,
        max_seq_len=32,
        attn_type="sdpa",
        moe_enabled=False,
    )


@pytest.fixture
def small_moe_model_args():
    # [FUNCTION DOC]
    # HELPER: Fixture returning tiny MoE Qwen3ModelArgs (same dims as dense,
    # moe_inter_dim 128, 4 experts top-2, every layer MoE).
    """Create small MoE model args for CPU testing."""
    from areal.experimental.models.archon.moe import MoEArgs
    from areal.experimental.models.archon.qwen3 import Qwen3ModelArgs

    return Qwen3ModelArgs(
        dim=64,
        hidden_dim=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        n_layers=2,
        vocab_size=1000,
        max_seq_len=32,
        attn_type="sdpa",
        moe_enabled=True,
        moe_inter_dim=128,
        moe_args=MoEArgs(num_experts=4, top_k=2),
        decoder_sparse_step=1,  # All layers are MoE
    )


@pytest.fixture
def small_dense_model(small_dense_model_args):
    # [FUNCTION DOC]
    # HELPER: Fixture building a Qwen3Model from small_dense_model_args on CPU with
    # initialized weights and CPU buffers.
    """Create and initialize small dense model on CPU."""
    from areal.experimental.models.archon.qwen3 import Qwen3Model

    model = Qwen3Model(small_dense_model_args)
    model.init_weights()
    model.init_buffers(buffer_device=torch.device("cpu"))
    return model


@pytest.fixture
def small_moe_model(small_moe_model_args):
    # [FUNCTION DOC]
    # HELPER: Fixture building and initializing the tiny MoE Qwen3Model on CPU.
    """Create and initialize small MoE model on CPU."""
    from areal.experimental.models.archon.qwen3 import Qwen3Model

    model = Qwen3Model(small_moe_model_args)
    model.init_weights()
    model.init_buffers(buffer_device=torch.device("cpu"))
    return model


@pytest.fixture
def dense_adapter():
    # [FUNCTION DOC]
    # HELPER: Fixture returning Qwen3StateDictAdapter(MockDenseConfig()).
    """Create adapter for dense model."""
    return Qwen3StateDictAdapter(MockDenseConfig())


@pytest.fixture
def moe_adapter():
    # [FUNCTION DOC]
    # HELPER: Fixture returning Qwen3StateDictAdapter(MockMoEConfig()).
    """Create adapter for MoE model."""
    return Qwen3StateDictAdapter(MockMoEConfig())


# =============================================================================
# Lightweight Weight Completeness Tests (Meta Device, No GPU Memory)
# =============================================================================


class TestWeightCompletenessWithMetaDevice:
    # [CLASS DOC]
    # Dense-model completeness checks that instantiate HF and Archon models on the meta
    # device (no real memory) for every available entry in DENSE_MODEL_PATHS.
    """Tests that verify weight completeness using meta device (no GPU memory).

    These tests create models on meta device to check key names and shapes
    without actually allocating memory for weights. This makes them fast
    and runnable without GPU.

    Note: Only tests dense models (qwen2, qwen3). MoE tests are in a separate
    class marked as slow.
    """

    @pytest.fixture(scope="class")
    def model_configs_and_specs(self):
        # [FUNCTION DOC]
        # HELPER: Class-scoped fixture: for each non-None DENSE_MODEL_PATHS entry, loads
        # the HF AutoConfig (trust_remote_code) and the Archon model spec via
        # get_model_spec; returns {model_type: {config, spec, path}}. Reads config files
        # from disk.
        # KNOWN ISSUE: Any exception is silently swallowed, so if no model is available
        # the dependent tests iterate over nothing and pass vacuously instead of
        # skipping.
        """Load configs and specs for dense model types only."""
        from transformers import AutoConfig

        from tests.experimental.archon.utils import DENSE_MODEL_PATHS

        from areal.experimental.models.archon import get_model_spec

        results = {}
        for model_type, path in DENSE_MODEL_PATHS.items():
            if path is None:
                continue
            try:
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                spec = get_model_spec(model_type)
                results[model_type] = {
                    "config": config,
                    "spec": spec,
                    "path": path,
                }
            except Exception:
                pass
        return results

    def test_all_archon_params_have_hf_mapping(self, model_configs_and_specs):
        # [FUNCTION DOC]
        # For each model: builds Archon model args from the HF config (is_critic False),
        # builds the model on meta device, and for each named parameter feeds an empty
        # CPU tensor of the same shape to adapter.convert_single_to_hf; asserts no
        # parameter yields an empty list.
        """Verify that ALL Archon model parameters can be converted to HF format.

        This is the lightweight equivalent of test_archon_all_params_converted.
        Uses meta device so no GPU memory is needed.
        """
        for model_type, info in model_configs_and_specs.items():
            config = info["config"]
            spec = info["spec"]

            # Create model args from HF config
            model_args = spec.model_args_class.from_hf_config(config, is_critic=False)

            # Create model on meta device (no memory allocation)
            with torch.device("meta"):
                model = spec.model_class(model_args)

            # Create adapter
            adapter = spec.state_dict_adapter_class(config)

            # Check all parameters can be converted
            unconverted = []
            for name, param in model.named_parameters():
                # Create a fake tensor with correct shape on CPU
                # (meta tensors can't be used with adapter directly)
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                if not hf_pairs:
                    unconverted.append(name)

            assert not unconverted, (
                f"[{model_type}] Found {len(unconverted)} unconverted parameters: "
                f"{unconverted[:5]}..."
            )

    def test_hf_weight_keys_bidirectional_completeness(self, model_configs_and_specs):
        # [FUNCTION DOC]
        # For each model: gets HF state-dict key names from a meta-device
        # AutoModelForCausalLM (skips the model if that fails), collects all HF names
        # produced by converting every Archon parameter, and asserts no HF key is
        # missing (ignoring rotary_emb keys and lm_head.weight when embeddings are tied)
        # and no extra key is produced.
        """Verify bidirectional completeness: HF keys <-> Archon keys.

        This is the lightweight equivalent of test_archon_hf_weight_completeness.
        Compares expected HF keys with what the adapter produces.
        """
        from transformers import AutoModelForCausalLM

        for model_type, info in model_configs_and_specs.items():
            config = info["config"]
            spec = info["spec"]

            # Get expected HF keys by loading model on meta device
            with torch.device("meta"):
                try:
                    hf_model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
                    hf_weight_names = set(hf_model.state_dict().keys())
                except Exception:
                    # Some models may not support meta device loading
                    continue

            # Create Archon model on meta device
            model_args = spec.model_args_class.from_hf_config(config, is_critic=False)
            with torch.device("meta"):
                archon_model = spec.model_class(model_args)

            # Create adapter
            adapter = spec.state_dict_adapter_class(config)
            tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

            # Collect all converted names from Archon
            archon_converted_names = set()
            for name, param in archon_model.named_parameters():
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                for hf_name, _ in hf_pairs:
                    archon_converted_names.add(hf_name)

            # Check: HF weights not covered by Archon
            missing_in_archon = hf_weight_names - archon_converted_names

            # Expected missing: rotary_emb.inv_freq (computed at runtime)
            missing_in_archon = {k for k in missing_in_archon if "rotary_emb" not in k}

            # If tie_word_embeddings, lm_head.weight is shared
            if tie_word_embeddings and "lm_head.weight" in missing_in_archon:
                missing_in_archon.discard("lm_head.weight")

            # Check: Archon produces weights not in HF
            extra_in_archon = archon_converted_names - hf_weight_names

            assert not missing_in_archon, (
                f"[{model_type}] HF weights not produced by Archon: "
                f"{sorted(missing_in_archon)[:5]}..."
            )
            assert not extra_in_archon, (
                f"[{model_type}] Unexpected weights from Archon: "
                f"{sorted(extra_in_archon)[:5]}..."
            )

    def test_weight_shapes_match_hf(self, model_configs_and_specs):
        # [FUNCTION DOC]
        # For each model: compares shapes of HF tensors produced by convert_single_to_hf
        # against the meta HF model's state-dict shapes for names present in both;
        # asserts no mismatch.
        """Verify that converted weight shapes match HF model shapes.

        This is the lightweight equivalent of test_archon_hf_weight_shape_match.
        Uses meta device so no actual memory is allocated.
        """
        from transformers import AutoModelForCausalLM

        for model_type, info in model_configs_and_specs.items():
            config = info["config"]
            spec = info["spec"]

            # Get expected HF shapes by loading model on meta device
            with torch.device("meta"):
                try:
                    hf_model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
                    hf_shapes = {k: v.shape for k, v in hf_model.state_dict().items()}
                except Exception:
                    continue

            # Create Archon model on meta device
            model_args = spec.model_args_class.from_hf_config(config, is_critic=False)
            with torch.device("meta"):
                archon_model = spec.model_class(model_args)

            # Create adapter
            adapter = spec.state_dict_adapter_class(config)

            # Compare shapes
            shape_mismatches = []
            for name, param in archon_model.named_parameters():
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                for hf_name, hf_tensor in hf_pairs:
                    if hf_name in hf_shapes:
                        expected_shape = hf_shapes[hf_name]
                        actual_shape = hf_tensor.shape
                        if expected_shape != actual_shape:
                            shape_mismatches.append(
                                {
                                    "archon_name": name,
                                    "hf_name": hf_name,
                                    "expected": expected_shape,
                                    "actual": actual_shape,
                                }
                            )

            assert not shape_mismatches, (
                f"[{model_type}] Found {len(shape_mismatches)} shape mismatches: "
                f"{shape_mismatches[:3]}..."
            )


@pytest.mark.slow
class TestMoEWeightCompletenessWithMetaDevice:
    # [CLASS DOC]
    # Slow-marked MoE variant of the meta-device completeness checks over
    # MOE_MODEL_PATHS; models are built once in a class-scoped fixture.
    """MoE-specific weight completeness tests using meta device.

    These tests verify MoE models (qwen3_moe) separately from dense models
    because instantiating large MoE models is slow even on meta device.

    Marked as slow - skipped by default in CI.
    Run with: pytest -m slow
    """

    @pytest.fixture(scope="class")
    def moe_models_and_data(self):
        # [FUNCTION DOC]
        # HELPER: Class-scoped fixture: for each available MoE model path loads the
        # config and spec, builds the HF model and Archon model on meta device, records
        # HF key names and shapes, and builds the adapter; returns a dict per model_type
        # with those objects.
        # KNOWN ISSUE: Exceptions are swallowed silently, so tests can pass with zero
        # models.
        """Load configs, create models on meta device, and prepare test data.

        This fixture creates both HF and Archon models once for all tests,
        avoiding the expensive model instantiation in each test method.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        from tests.experimental.archon.utils import MOE_MODEL_PATHS

        from areal.experimental.models.archon import get_model_spec

        results = {}
        for model_type, path in MOE_MODEL_PATHS.items():
            if path is None:
                continue
            try:
                config = AutoConfig.from_pretrained(path, trust_remote_code=True)
                spec = get_model_spec(model_type)

                # Create HF model on meta device (once)
                with torch.device("meta"):
                    hf_model = AutoModelForCausalLM.from_config(
                        config, trust_remote_code=True
                    )
                hf_weight_names = set(hf_model.state_dict().keys())
                hf_shapes = {k: v.shape for k, v in hf_model.state_dict().items()}

                # Create Archon model on meta device (once)
                model_args = spec.model_args_class.from_hf_config(
                    config, is_critic=False
                )
                with torch.device("meta"):
                    archon_model = spec.model_class(model_args)

                # Create adapter
                adapter = spec.state_dict_adapter_class(config)

                results[model_type] = {
                    "config": config,
                    "spec": spec,
                    "path": path,
                    "hf_model": hf_model,
                    "hf_weight_names": hf_weight_names,
                    "hf_shapes": hf_shapes,
                    "archon_model": archon_model,
                    "adapter": adapter,
                }
            except Exception:
                pass
        return results

    def test_all_archon_params_have_hf_mapping(self, moe_models_and_data):
        # [FUNCTION DOC]
        # Asserts every parameter of each prebuilt MoE Archon model converts to at least
        # one HF tensor.
        """Verify that ALL Archon model parameters can be converted to HF format."""
        for model_type, info in moe_models_and_data.items():
            archon_model = info["archon_model"]
            adapter = info["adapter"]

            # Check all parameters can be converted
            unconverted = []
            for name, param in archon_model.named_parameters():
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                if not hf_pairs:
                    unconverted.append(name)

            assert not unconverted, (
                f"[{model_type}] Found {len(unconverted)} unconverted parameters: "
                f"{unconverted[:5]}..."
            )

    def test_hf_weight_keys_bidirectional_completeness(self, moe_models_and_data):
        # [FUNCTION DOC]
        # Same bidirectional key-set check as the dense version, using the precomputed
        # HF key names of each MoE model.
        """Verify bidirectional completeness: HF keys <-> Archon keys."""
        for model_type, info in moe_models_and_data.items():
            config = info["config"]
            archon_model = info["archon_model"]
            adapter = info["adapter"]
            hf_weight_names = info["hf_weight_names"]
            tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

            # Collect all converted names from Archon
            archon_converted_names = set()
            for name, param in archon_model.named_parameters():
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                for hf_name, _ in hf_pairs:
                    archon_converted_names.add(hf_name)

            # Check: HF weights not covered by Archon
            missing_in_archon = hf_weight_names - archon_converted_names

            # Expected missing: rotary_emb.inv_freq (computed at runtime)
            missing_in_archon = {k for k in missing_in_archon if "rotary_emb" not in k}

            # If tie_word_embeddings, lm_head.weight is shared
            if tie_word_embeddings and "lm_head.weight" in missing_in_archon:
                missing_in_archon.discard("lm_head.weight")

            # Check: Archon produces weights not in HF
            extra_in_archon = archon_converted_names - hf_weight_names

            assert not missing_in_archon, (
                f"[{model_type}] HF weights not produced by Archon: "
                f"{sorted(missing_in_archon)[:5]}..."
            )
            assert not extra_in_archon, (
                f"[{model_type}] Unexpected weights from Archon: "
                f"{sorted(extra_in_archon)[:5]}..."
            )

    def test_weight_shapes_match_hf(self, moe_models_and_data):
        # [FUNCTION DOC]
        # Same shape comparison as the dense version, using precomputed MoE HF shapes.
        """Verify that converted weight shapes match HF model shapes."""
        for model_type, info in moe_models_and_data.items():
            archon_model = info["archon_model"]
            adapter = info["adapter"]
            hf_shapes = info["hf_shapes"]

            # Compare shapes
            shape_mismatches = []
            for name, param in archon_model.named_parameters():
                fake_tensor = torch.empty(param.shape)
                hf_pairs = adapter.convert_single_to_hf(name, fake_tensor)
                for hf_name, hf_tensor in hf_pairs:
                    if hf_name in hf_shapes:
                        expected_shape = hf_shapes[hf_name]
                        actual_shape = hf_tensor.shape
                        if expected_shape != actual_shape:
                            shape_mismatches.append(
                                {
                                    "archon_name": name,
                                    "hf_name": hf_name,
                                    "expected": expected_shape,
                                    "actual": actual_shape,
                                }
                            )

            assert not shape_mismatches, (
                f"[{model_type}] Found {len(shape_mismatches)} shape mismatches: "
                f"{shape_mismatches[:3]}..."
            )


# =============================================================================
# Full Weight Value Tests (GPU Required, Slow)
# =============================================================================


# Skip if no CUDA available
cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@cuda_required
def test_archon_weights_match_hf():
    # [FUNCTION DOC]
    # CUDA-only. Calls setup_environment, loads the "qwen2" model path as both HF and
    # Archon models in bfloat16, and for a fixed list of Archon->HF key pairs
    # (embeddings, layer 0 attention/MLP, final norm, output) asserts equal shapes and
    # max abs difference < 1e-5; pairs missing from either model are skipped. Destroys
    # the default process group if one was initialized. Side effects: GPU memory, model
    # loading from disk, distributed teardown.
    """Verify Archon weight conversion from HuggingFace is correct.

    This test loads actual models and compares weights to ensure
    the state dict adapter correctly converts between formats.
    """
    from tests.experimental.archon.utils import (
        MODEL_PATHS,
        load_archon_model,
        load_hf_model,
        setup_environment,
    )

    setup_environment()

    model_path = MODEL_PATHS["qwen2"]
    dtype = torch.bfloat16

    hf_model = load_hf_model(model_path, dtype=dtype)
    archon_model, _ = load_archon_model(model_path, dtype=dtype)

    # Key mappings: archon_key -> hf_key
    key_mappings = [
        ("tok_embeddings.weight", "model.embed_tokens.weight"),
        ("layers.0.attention.wq.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("layers.0.attention.wk.weight", "model.layers.0.self_attn.k_proj.weight"),
        ("layers.0.attention.wv.weight", "model.layers.0.self_attn.v_proj.weight"),
        ("layers.0.attention.wo.weight", "model.layers.0.self_attn.o_proj.weight"),
        ("layers.0.feed_forward.w1.weight", "model.layers.0.mlp.gate_proj.weight"),
        ("layers.0.feed_forward.w2.weight", "model.layers.0.mlp.down_proj.weight"),
        ("layers.0.feed_forward.w3.weight", "model.layers.0.mlp.up_proj.weight"),
        ("norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
    ]

    archon_params = dict(archon_model.named_parameters())
    hf_params = dict(hf_model.named_parameters())

    for archon_key, hf_key in key_mappings:
        if archon_key not in archon_params or hf_key not in hf_params:
            continue

        archon_w = archon_params[archon_key].data
        hf_w = hf_params[hf_key].data

        assert archon_w.shape == hf_w.shape, (
            f"Shape mismatch for {archon_key}: {archon_w.shape} vs {hf_w.shape}"
        )

        max_diff = (archon_w.float() - hf_w.float()).abs().max().item()
        assert max_diff < 1e-5, f"Weight mismatch for {archon_key}: max_diff={max_diff}"

    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Iterative vs Batch Conversion Tests
# =============================================================================


class TestIterativeVsBatchConversion:
    # [CLASS DOC]
    # CPU tests that iterative per-parameter conversion matches batch to_hf and never
    # emits duplicate HF keys, for dense and MoE tiny models.
    """Verify iterative convert_single_to_hf matches batch to_hf.

    This is critical for weight synchronization (update_weights_from_dist)
    which uses iterative conversion to send weights to inference servers.
    """

    def test_dense_iterative_equals_batch(self, small_dense_model, dense_adapter):
        # [FUNCTION DOC]
        # Builds an HF dict by calling convert_single_to_hf on each named parameter of
        # the dense model, compares with to_hf(state_dict()): asserts identical key sets
        # and torch.equal values.
        # KNOWN ISSUE: Batch path uses state_dict() (parameters plus persistent buffers)
        # while the iterative path uses only parameters; the test relies on no
        # persistent buffer being converted to an HF key.
        """Dense: iterative convert_single_to_hf == batch to_hf."""
        # Method 1: Iterative conversion (simulates weight update flow)
        hf_iterative = {}
        for name, param in small_dense_model.named_parameters():
            hf_pairs = dense_adapter.convert_single_to_hf(name, param.data)
            for hf_name, hf_tensor in hf_pairs:
                hf_iterative[hf_name] = hf_tensor

        # Method 2: Batch conversion
        hf_batch = dense_adapter.to_hf(small_dense_model.state_dict())

        # Assert keys match
        assert set(hf_iterative.keys()) == set(hf_batch.keys()), (
            f"Key mismatch: iterative has {len(hf_iterative)} keys, "
            f"batch has {len(hf_batch)} keys"
        )

        # Assert values match
        for key in hf_iterative:
            assert torch.equal(hf_iterative[key], hf_batch[key]), (
                f"Value mismatch at {key}"
            )

    def test_moe_iterative_equals_batch(self, small_moe_model, moe_adapter):
        # [FUNCTION DOC]
        # Same iterative-vs-batch comparison for the MoE model, reporting extra keys on
        # either side on mismatch.
        """MoE: iterative convert_single_to_hf == batch to_hf."""
        # Method 1: Iterative conversion
        hf_iterative = {}
        for name, param in small_moe_model.named_parameters():
            hf_pairs = moe_adapter.convert_single_to_hf(name, param.data)
            for hf_name, hf_tensor in hf_pairs:
                hf_iterative[hf_name] = hf_tensor

        # Method 2: Batch conversion
        hf_batch = moe_adapter.to_hf(small_moe_model.state_dict())

        # Assert keys match
        assert set(hf_iterative.keys()) == set(hf_batch.keys()), (
            f"Key mismatch: iterative has {len(hf_iterative)} keys, "
            f"batch has {len(hf_batch)} keys. "
            f"Extra in iterative: {set(hf_iterative.keys()) - set(hf_batch.keys())}. "
            f"Extra in batch: {set(hf_batch.keys()) - set(hf_iterative.keys())}"
        )

        # Assert values match
        for key in hf_iterative:
            assert torch.equal(hf_iterative[key], hf_batch[key]), (
                f"Value mismatch at {key}"
            )

    def test_dense_no_duplicate_keys_in_iteration(
        self, small_dense_model, dense_adapter
    ):
        # [FUNCTION DOC]
        # Collects all HF names produced iteratively for the dense model and asserts
        # they are unique.
        """Dense: no duplicate HF keys produced during iteration."""
        hf_keys = []
        for name, param in small_dense_model.named_parameters():
            hf_pairs = dense_adapter.convert_single_to_hf(name, param.data)
            hf_keys.extend([hf_name for hf_name, _ in hf_pairs])

        assert len(hf_keys) == len(set(hf_keys)), (
            f"Duplicate keys found: {len(hf_keys)} total, {len(set(hf_keys))} unique"
        )

    def test_moe_no_duplicate_keys_in_iteration(self, small_moe_model, moe_adapter):
        # [FUNCTION DOC]
        # Collects all HF names produced iteratively for the MoE model and asserts they
        # are unique.
        """MoE: no duplicate HF keys produced during iteration."""
        hf_keys = []
        for name, param in small_moe_model.named_parameters():
            hf_pairs = moe_adapter.convert_single_to_hf(name, param.data)
            hf_keys.extend([hf_name for hf_name, _ in hf_pairs])

        assert len(hf_keys) == len(set(hf_keys)), (
            f"Duplicate keys found: {len(hf_keys)} total, {len(set(hf_keys))} unique"
        )
