# [MODULE DOC]
# CPU unit tests for generate_llm_fqn_per_model_part, which splits a decoder LLM into
# pipeline stages as lists of module FQNs ("tok_embeddings", "layers.i", "norm", and
# "output" or "score" for critics). The algorithm treats the embedding as costing
# first_stage_less_layers (default 1) and norm+head as costing last_stage_less_layers
# (default 1) "effective layers"; total effective layers are divided evenly with the
# remainder given one extra each to the earliest stages. Also checks validation errors
# and ZBV (V-shaped) stage-to-rank assignment.
"""Unit tests for pipeline_parallel.py (no GPU required)."""

import pytest

from areal.experimental.models.archon.pipeline_parallel import (
    generate_llm_fqn_per_model_part,
)


class TestGenerateLLMFqnPerModelPart:
    # [CLASS DOC]
    # Partitioning results and validation for various stage/layer counts.
    """Test generate_llm_fqn_per_model_part function."""

    def test_single_stage(self):
        # [FUNCTION DOC]
        # One stage holds embedding, all 4 layers, norm and output in order.
        """Single stage should contain all modules."""
        result = generate_llm_fqn_per_model_part(num_stages=1, num_layers=4)
        assert len(result) == 1
        assert result[0] == [
            "tok_embeddings",
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "norm",
            "output",
        ]

    def test_two_stages_four_layers(self):
        # [FUNCTION DOC]
        # 6 effective layers over 2 stages: [emb, l0, l1] and [l2, l3, norm, output].
        """Two stages with 4 layers should split evenly."""
        result = generate_llm_fqn_per_model_part(num_stages=2, num_layers=4)
        assert len(result) == 2
        # Stage 0: embedding + 2 layers (effective: 1 + 2 = 3)
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1"]
        # Stage 1: 2 layers + output (effective: 2 + 1 = 3)
        assert result[1] == ["layers.2", "layers.3", "norm", "output"]

    def test_four_stages_eight_layers(self):
        # [FUNCTION DOC]
        # 10 effective over 4 stages (3,3,2,2): [emb,l0,l1], [l2,l3,l4], [l5,l6],
        # [l7,norm,output].
        """Four stages with 8 layers."""
        result = generate_llm_fqn_per_model_part(num_stages=4, num_layers=8)
        assert len(result) == 4
        # 8 + 1 + 1 = 10 effective layers, 10 // 4 = 2 per stage, 2 extra
        # Stage 0: gets 3 effective (tok_embeddings counts as 1, so 2 layers)
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1"]
        # Stage 1: gets 3 effective (all transformer layers)
        assert result[1] == ["layers.2", "layers.3", "layers.4"]
        # Stage 2: gets 2 effective (all transformer layers)
        assert result[2] == ["layers.5", "layers.6"]
        # Stage 3: gets 2 effective (norm+output counts as 1, so 1 layer)
        assert result[3] == ["layers.7", "norm", "output"]

    def test_two_stages_eight_layers(self):
        # [FUNCTION DOC]
        # 10 effective over 2 stages: embedding plus layers 0-3, then layers 4-7 plus
        # norm and output.
        """Two stages with 8 layers."""
        result = generate_llm_fqn_per_model_part(num_stages=2, num_layers=8)
        assert len(result) == 2
        # 8 + 1 + 1 = 10 effective layers, 10 // 2 = 5 per stage
        # Stage 0: tok_embeddings + 4 layers
        assert result[0] == [
            "tok_embeddings",
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
        ]
        # Stage 1: 4 layers + norm + output
        assert result[1] == [
            "layers.4",
            "layers.5",
            "layers.6",
            "layers.7",
            "norm",
            "output",
        ]

    def test_three_stages_six_layers(self):
        # [FUNCTION DOC]
        # 8 effective over 3 stages (3,3,2): [emb,l0,l1], [l2,l3,l4], [l5,norm,output].
        """Three stages with 6 layers."""
        result = generate_llm_fqn_per_model_part(num_stages=3, num_layers=6)
        assert len(result) == 3
        # 6 + 1 + 1 = 8 effective layers, 8 // 3 = 2 per stage, 2 extra
        # Stage 0: gets 3 effective (tok_embeddings + 2 layers)
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1"]
        # Stage 1: gets 3 effective (3 layers)
        assert result[1] == ["layers.2", "layers.3", "layers.4"]
        # Stage 2: gets 2 effective (1 layer + norm + output)
        assert result[2] == ["layers.5", "norm", "output"]

    def test_custom_weights(self):
        # [FUNCTION DOC]
        # Embedding and head each cost 2: 8 effective over 2 stages gives [emb,l0,l1]
        # and [l2,l3,norm,output].
        """Test with custom first_stage_less_layers/last_stage_less_layers."""
        # first_stage_less_layers=2, last_stage_less_layers=2 means embedding "costs" 2 slots
        result = generate_llm_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            first_stage_less_layers=2,
            last_stage_less_layers=2,
        )
        assert len(result) == 2
        # 4 + 2 + 2 = 8 effective layers, 8 // 2 = 4 per stage
        # Stage 0: tok_embeddings (counts as 2) + 2 layers
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1"]
        # Stage 1: 2 layers + norm + output (counts as 2)
        assert result[1] == ["layers.2", "layers.3", "norm", "output"]

    def test_zero_first_stage_less_layers(self):
        # [FUNCTION DOC]
        # Embedding costs 0, head 1: 5 effective over 2 stages (3,2) gives
        # [emb,l0,l1,l2] and [l3,norm,output].
        """Test with zero first_stage_less_layers."""
        result = generate_llm_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            first_stage_less_layers=0,
            last_stage_less_layers=1,
        )
        assert len(result) == 2
        # 4 + 0 + 1 = 5 effective layers, 5 // 2 = 2 per stage, 1 extra
        # Stage 0: gets 3 effective (tok_embeddings counts as 0, so 3 layers)
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1", "layers.2"]
        # Stage 1: gets 2 effective (1 layer + norm + output)
        assert result[1] == ["layers.3", "norm", "output"]

    def test_validation_num_stages_zero(self):
        # [FUNCTION DOC]
        # Asserts num_stages 0 raises ValueError "num_stages must be >= 1".
        """Test validation error for num_stages < 1."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            generate_llm_fqn_per_model_part(num_stages=0, num_layers=4)

    def test_validation_num_stages_negative(self):
        # [FUNCTION DOC]
        # Asserts num_stages -1 raises the same error.
        """Test validation error for negative num_stages."""
        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            generate_llm_fqn_per_model_part(num_stages=-1, num_layers=4)

    def test_validation_too_many_stages(self):
        # [FUNCTION DOC]
        # Asserts 10 stages for 4 layers raises "cannot exceed effective layers".
        """Test validation error when num_stages exceeds effective layers."""
        with pytest.raises(ValueError, match="cannot exceed effective layers"):
            generate_llm_fqn_per_model_part(num_stages=10, num_layers=4)

    def test_validation_first_stage_less_layers_too_large(self):
        # [FUNCTION DOC]
        # Asserts an embedding cost of 5 exceeding layers_per_stage (2) raises a
        # matching ValueError.
        """Test validation error when first_stage_less_layers exceeds layers_per_stage."""
        # 4 + 5 + 1 = 10 effective, 10 // 4 = 2 per stage, but first_stage_less_layers=5 > 2
        with pytest.raises(
            ValueError, match="first_stage_less_layers.*exceeds layers_per_stage"
        ):
            generate_llm_fqn_per_model_part(
                num_stages=4,
                num_layers=4,
                first_stage_less_layers=5,
                last_stage_less_layers=1,
            )

    def test_validation_last_stage_less_layers_too_large(self):
        # [FUNCTION DOC]
        # Asserts a head cost of 5 exceeding layers_per_stage raises a matching
        # ValueError.
        """Test validation error when last_stage_less_layers exceeds layers_per_stage."""
        # 4 + 1 + 5 = 10 effective, 10 // 4 = 2 per stage, but last_stage_less_layers=5 > 2
        with pytest.raises(
            ValueError, match="last_stage_less_layers.*exceeds layers_per_stage"
        ):
            generate_llm_fqn_per_model_part(
                num_stages=4,
                num_layers=4,
                first_stage_less_layers=1,
                last_stage_less_layers=5,
            )

    def test_large_model(self):
        # [FUNCTION DOC]
        # 32 layers over 8 stages: asserts every layer appears exactly once in order,
        # the first stage starts with the embedding and the last stage ends with norm
        # and output.
        """Test with a large model (e.g., 32 layers, 8 stages)."""
        result = generate_llm_fqn_per_model_part(num_stages=8, num_layers=32)
        assert len(result) == 8

        # Verify all layers are covered exactly once
        all_layer_names = []
        for stage in result:
            all_layer_names.extend([m for m in stage if m.startswith("layers.")])
        expected_layers = [f"layers.{i}" for i in range(32)]
        assert all_layer_names == expected_layers

        # Verify first stage has tok_embeddings
        assert result[0][0] == "tok_embeddings"

        # Verify last stage has norm and output
        assert result[-1][-2:] == ["norm", "output"]

    def test_critic_model_single_stage(self):
        # [FUNCTION DOC]
        # is_critic: single stage ends with "score" instead of "output".
        """Critic model with single stage should use 'score' instead of 'output'."""
        result = generate_llm_fqn_per_model_part(
            num_stages=1, num_layers=4, is_critic=True
        )
        assert len(result) == 1
        assert result[0] == [
            "tok_embeddings",
            "layers.0",
            "layers.1",
            "layers.2",
            "layers.3",
            "norm",
            "score",
        ]

    def test_critic_model_two_stages(self):
        # [FUNCTION DOC]
        # is_critic with 2 stages: last stage is [l2, l3, norm, score].
        """Critic model with two stages should use 'score' on last stage."""
        result = generate_llm_fqn_per_model_part(
            num_stages=2, num_layers=4, is_critic=True
        )
        assert len(result) == 2
        assert result[0] == ["tok_embeddings", "layers.0", "layers.1"]
        assert result[1] == ["layers.2", "layers.3", "norm", "score"]

    def test_critic_model_four_stages(self):
        # [FUNCTION DOC]
        # is_critic with 4 stages and 8 layers: last stage is [l7, norm, score].
        """Critic model with four stages."""
        result = generate_llm_fqn_per_model_part(
            num_stages=4, num_layers=8, is_critic=True
        )
        assert len(result) == 4
        # Last stage should have 'score' instead of 'output'
        assert result[3] == ["layers.7", "norm", "score"]

    def test_exact_distribution(self):
        # [FUNCTION DOC]
        # 6 layers over 4 stages (8 effective, 2 each, no remainder): [emb,l0], [l1,l2],
        # [l3,l4], [l5,norm,output].
        """Test when layers divide evenly."""
        # 6 layers + 1 + 1 = 8, 8 // 4 = 2 per stage exactly
        result = generate_llm_fqn_per_model_part(num_stages=4, num_layers=6)
        assert len(result) == 4

        # All stages should have 2 effective layers
        # Stage 0: tok_embeddings (1) + 1 layer = 2
        assert result[0] == ["tok_embeddings", "layers.0"]
        # Stage 1: 2 layers = 2
        assert result[1] == ["layers.1", "layers.2"]
        # Stage 2: 2 layers = 2
        assert result[2] == ["layers.3", "layers.4"]
        # Stage 3: 1 layer + norm + output (1) = 2
        assert result[3] == ["layers.5", "norm", "output"]


class TestZBVFqnGeneration:
    # [CLASS DOC]
    # FQN generation as consumed by the ZBV (zero-bubble V) schedule.
    """Test FQN generation for ZBV pipeline configurations."""

    def test_zbv_fqn_generation(self):
        # [FUNCTION DOC]
        # 4 stages, 8 layers: with V-style assignment rank 0 owns stages 0 and 3 and
        # rank 1 owns stages 1 and 2. Asserts rank 0 holds embedding, norm and output,
        # rank 1 holds only transformer layers, and all layers are covered once in
        # order.
        """Verify FQN distribution for a typical ZBV config (pp_degree=2, 8 layers)."""
        result = generate_llm_fqn_per_model_part(num_stages=4, num_layers=8)
        assert len(result) == 4

        # Rank 0 gets stages (0, 3), rank 1 gets stages (1, 2)
        rank0_modules = result[0] + result[3]
        rank1_modules = result[1] + result[2]

        # Rank 0 has first and last stages
        assert "tok_embeddings" in rank0_modules
        assert "norm" in rank0_modules
        assert "output" in rank0_modules

        # Rank 1 has only middle layers (no embeddings or output head)
        assert all(m.startswith("layers.") for m in rank1_modules)

        # All layers covered exactly once
        all_layers = []
        for stage in result:
            all_layers.extend([m for m in stage if m.startswith("layers.")])
        assert all_layers == [f"layers.{i}" for i in range(8)]
