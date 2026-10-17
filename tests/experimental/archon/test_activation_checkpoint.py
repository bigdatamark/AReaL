# [MODULE DOC]
# CPU unit tests for Archon activation checkpointing: ActivationCheckpointConfig
# defaults and validation, and apply_ac wrapping of model.layers entries with
# CheckpointWrapper under the none / full / selective (every-N or per-op) modes, plus
# forward/backward behavior of wrapped models.
"""Unit tests for Archon activation checkpointing.

Tests cover:
1. ActivationCheckpointConfig
2. apply_ac with all modes

Run tests:
    pytest tests/experimental/archon/test_activation_checkpoint.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from areal.experimental.models.archon.activation_checkpoint import (
    ActivationCheckpointConfig,
    apply_ac,
)

# =============================================================================
# ActivationCheckpointConfig Tests
# =============================================================================


class TestActivationCheckpointConfig:
    # [CLASS DOC]
    # Validation and default-value tests for ActivationCheckpointConfig.
    """Test ActivationCheckpointConfig."""

    def test_default_config(self):
        # [FUNCTION DOC]
        # Asserts defaults: mode "selective", selective_ac_option "op",
        # preserve_rng_state False.
        """Default config should have mode='selective' and selective_ac_option='op'."""
        config = ActivationCheckpointConfig()
        assert config.mode == "selective"
        assert config.selective_ac_option == "op"
        assert config.preserve_rng_state is False

    def test_full_mode(self):
        # [FUNCTION DOC]
        # Asserts mode "full" is accepted.
        """Full mode should be valid."""
        config = ActivationCheckpointConfig(mode="full")
        assert config.mode == "full"

    def test_selective_mode_with_integer(self):
        # [FUNCTION DOC]
        # Asserts selective mode accepts the integer-string option "2".
        """Selective mode with integer option should be valid."""
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        assert config.mode == "selective"
        assert config.selective_ac_option == "2"

    def test_selective_mode_with_op(self):
        # [FUNCTION DOC]
        # Asserts selective mode accepts option "op".
        """Selective mode with 'op' option should be valid."""
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        assert config.selective_ac_option == "op"

    def test_preserve_rng_state(self):
        # [FUNCTION DOC]
        # Asserts preserve_rng_state True is stored.
        """preserve_rng_state should be configurable."""
        config = ActivationCheckpointConfig(mode="full", preserve_rng_state=True)
        assert config.preserve_rng_state is True

    def test_none_mode(self):
        # [FUNCTION DOC]
        # Asserts mode "none" is accepted.
        """None mode should be valid."""
        config = ActivationCheckpointConfig(mode="none")
        assert config.mode == "none"

    def test_memory_budget_mode(self):
        # [FUNCTION DOC]
        # Asserts mode "memory_budget" with memory_budget 0.7 is accepted and stored.
        """Memory budget mode should be valid."""
        config = ActivationCheckpointConfig(mode="memory_budget", memory_budget=0.7)
        assert config.mode == "memory_budget"
        assert config.memory_budget == 0.7

    def test_per_op_sac_force_recompute_default(self):
        # [FUNCTION DOC]
        # Asserts the default per_op_sac_force_recompute_mm_shapes_by_fqns contains
        # "moe.router.gate".
        """Default per_op_sac_force_recompute_mm_shapes_by_fqns should include moe.router.gate."""
        config = ActivationCheckpointConfig()
        assert "moe.router.gate" in config.per_op_sac_force_recompute_mm_shapes_by_fqns

    def test_invalid_mode_raises(self):
        # [FUNCTION DOC]
        # Asserts an unknown mode raises ValueError matching "Invalid AC mode" at
        # construction.
        """Invalid mode should raise ValueError at config creation."""
        with pytest.raises(ValueError, match="Invalid AC mode"):
            ActivationCheckpointConfig(mode="invalid")

    def test_invalid_selective_ac_option_raises(self):
        # [FUNCTION DOC]
        # Asserts selective mode with a non-integer, non-"op" option raises ValueError
        # matching "Invalid selective_ac_option".
        """Invalid selective_ac_option should raise ValueError at config creation."""
        with pytest.raises(ValueError, match="Invalid selective_ac_option"):
            ActivationCheckpointConfig(mode="selective", selective_ac_option="invalid")

    def test_selective_ac_option_validation_only_in_selective_mode(self):
        # [FUNCTION DOC]
        # Asserts an invalid selective_ac_option is tolerated when mode is "full" or
        # "none".
        """selective_ac_option validation should only apply when mode is 'selective'."""
        # These should NOT raise even with non-standard selective_ac_option
        # because they are not in selective mode
        config = ActivationCheckpointConfig(mode="full", selective_ac_option="invalid")
        assert config.mode == "full"

        config = ActivationCheckpointConfig(mode="none", selective_ac_option="invalid")
        assert config.mode == "none"


# =============================================================================
# apply_ac Tests
# =============================================================================


class DummyBlock(nn.Module):
    # [CLASS DOC]
    # HELPER: Minimal transformer-block stand-in: one dim x dim linear (with bias).
    """Dummy transformer block for testing."""

    def __init__(self, dim=8):
        # [FUNCTION DOC]
        # Creates self.linear with the given dim (default 8).
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        # [FUNCTION DOC]
        # Returns self.linear(x).
        return self.linear(x)


class DummyModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Minimal model exposing a "layers" ModuleDict keyed "0".."n-1" of
    # DummyBlock, as apply_ac requires.
    """Dummy model with layers attribute for testing."""

    def __init__(self, num_layers=4, dim=8):
        # [FUNCTION DOC]
        # Builds num_layers (default 4) DummyBlocks of size dim (default 8).
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(i): DummyBlock(dim) for i in range(num_layers)}
        )

    def forward(self, x):
        # [FUNCTION DOC]
        # Applies each layer sequentially in dict order and returns the result.
        for layer in self.layers.values():
            x = layer(x)
        return x


class TestApplyAc:
    # [CLASS DOC]
    # Tests for apply_ac layer wrapping.
    """Test apply_ac function."""

    def test_none_mode_no_wrapping(self):
        # [FUNCTION DOC]
        # Mode "none": asserts every layer remains a plain DummyBlock.
        """None mode should not wrap any layers."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="none")
        apply_ac(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, DummyBlock)
            assert not isinstance(layer, CheckpointWrapper)

    def test_full_mode_wraps_all_layers(self):
        # [FUNCTION DOC]
        # Mode "full": asserts every layer becomes a CheckpointWrapper.
        """Full mode should wrap all layers with CheckpointWrapper."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="full")
        apply_ac(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

    def test_selective_mode_wraps_every_n_layers(self):
        # [FUNCTION DOC]
        # Option "2" on 4 layers: asserts only layers "1" and "3" (every 2nd, 1-based
        # counting) are wrapped.
        """Selective mode should wrap every Nth layer."""
        model = DummyModel(num_layers=4)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        apply_ac(model, config)

        # With ac_freq=2, the 2nd and 4th layers are checkpointed.
        # This corresponds to layers with index 1 and 3.
        assert not isinstance(model.layers["0"], CheckpointWrapper)
        assert isinstance(model.layers["1"], CheckpointWrapper)
        assert not isinstance(model.layers["2"], CheckpointWrapper)
        assert isinstance(model.layers["3"], CheckpointWrapper)

    def test_selective_mode_every_layer(self):
        # [FUNCTION DOC]
        # Option "1": asserts all layers are wrapped.
        """Selective mode with option '1' should wrap every layer."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="1")
        apply_ac(model, config)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

    def test_selective_op_mode_wraps_all_layers(self):
        # [FUNCTION DOC]
        # Option "op" with op_sac_save_list containing aten.mm: asserts all layers are
        # wrapped.
        """Selective op mode should wrap all layers."""
        model = DummyModel(num_layers=3)
        config = ActivationCheckpointConfig(mode="selective", selective_ac_option="op")
        # op_sac_save_list is required for op mode
        op_sac_save_list = {torch.ops.aten.mm.default}
        apply_ac(model, config, op_sac_save_list=op_sac_save_list)

        for layer in model.layers.values():
            assert isinstance(layer, CheckpointWrapper)

    def test_model_without_layers_raises(self):
        # [FUNCTION DOC]
        # A bare nn.Linear has no layers attribute: asserts ValueError matching "must
        # have a 'layers' attribute".
        """Model without layers attribute should raise ValueError."""
        model = nn.Linear(4, 2)
        config = ActivationCheckpointConfig(mode="full")
        with pytest.raises(ValueError, match="must have a 'layers' attribute"):
            apply_ac(model, config)

    def test_wrapped_model_forward_works(self):
        # [FUNCTION DOC]
        # Compares no-grad output before and after full-mode wrapping; asserts allclose.
        """Wrapped model should still produce correct forward output."""
        model = DummyModel(num_layers=3, dim=4)

        # Get reference output before wrapping
        x = torch.randn(2, 4)
        with torch.no_grad():
            ref_output = model(x.clone())

        # Apply AC and verify output matches
        config = ActivationCheckpointConfig(mode="full")
        apply_ac(model, config)

        with torch.no_grad():
            wrapped_output = model(x.clone())

        assert torch.allclose(ref_output, wrapped_output)

    def test_wrapped_model_backward_works(self):
        # [FUNCTION DOC]
        # After full-mode wrapping, backprops output sum and asserts gradients exist for
        # input and all parameters.
        """Wrapped model should support backward pass."""
        model = DummyModel(num_layers=3, dim=4)
        config = ActivationCheckpointConfig(mode="full")
        apply_ac(model, config)

        x = torch.randn(2, 4, requires_grad=True)
        output = model(x)
        loss = output.sum()

        # Should not raise
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_multiple_models_independent(self):
        # [FUNCTION DOC]
        # Wraps one model selectively (every 2nd) and another fully; asserts each
        # follows its own config.
        """Multiple models should be wrapped independently."""
        model1 = DummyModel(num_layers=4)
        model2 = DummyModel(num_layers=4)

        config1 = ActivationCheckpointConfig(mode="selective", selective_ac_option="2")
        config2 = ActivationCheckpointConfig(mode="full")

        apply_ac(model1, config1)
        apply_ac(model2, config2)

        # model1: selective, every 2nd layer
        assert not isinstance(model1.layers["0"], CheckpointWrapper)
        assert isinstance(model1.layers["1"], CheckpointWrapper)

        # model2: all layers wrapped
        for layer in model2.layers.values():
            assert isinstance(layer, CheckpointWrapper)
