# [MODULE DOC]
# Unit tests for enable_fp8_linear and enable_fp8_experts from
# areal.experimental.models.archon.fp8. Builds tiny models on the meta device and checks
# which nn.Linear / GroupedExperts modules have their forward replaced, covering default
# exclusions (output, router), 128-alignment checks, bias exclusion and custom
# exclude_fqns. Skipped when the torchao blockwise FP8 prototype is not importable.
"""enable_fp8_linear() and enable_fp8_experts() unit tests.

Verifies FP8 patching, exclusion logic, alignment checks, and expert enabling.
"""

import pytest
import torch
from torch import nn

try:
    import torchao.prototype.blockwise_fp8_training.linear  # noqa: F401

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCHAO_AVAILABLE,
    reason="torchao blockwise FP8 prototype not available",
)


def _is_fp8_patched(mod: nn.Module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the module's forward is no longer the bound
    # nn.Linear.forward (either it is not a bound method or its underlying function
    # differs).
    """Check whether a module's forward has been replaced by enable_fp8_linear."""
    return (
        not hasattr(mod.forward, "__func__")
        or mod.forward.__func__ is not nn.Linear.forward
    )  # type: ignore[attr-defined]


def _is_fp8_experts_patched(mod: nn.Module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Same check as _is_fp8_patched but against GroupedExperts.forward (imported
    # lazily).
    """Check whether a module's forward has been replaced by enable_fp8_experts."""
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    return (
        not hasattr(mod.forward, "__func__")
        or mod.forward.__func__ is not GroupedExperts.forward
    )  # type: ignore[attr-defined]


def _make_model_on_meta(**linear_kwargs):
    # [FUNCTION DOC]
    # HELPER: Builds, on the meta device and cast to bfloat16, a SimpleModel wrapping a
    # single nn.Linear constructed from the given keyword arguments (in_features,
    # out_features, bias). Returns the model.
    class SimpleModel(nn.Module):
        # [CLASS DOC]
        # HELPER: Minimal module holding one linear layer as attribute "layer".
        def __init__(self):
            # [FUNCTION DOC]
            # Creates self.layer from the enclosing linear_kwargs.
            super().__init__()
            self.layer = nn.Linear(**linear_kwargs)

        def forward(self, x):
            # [FUNCTION DOC]
            # Applies self.layer to x and returns the result.
            return self.layer(x)

    with torch.device("meta"):
        return SimpleModel().to(torch.bfloat16)


def _make_multi_layer_model():
    # [FUNCTION DOC]
    # HELPER: Builds a FakeTransformer on meta in bfloat16 with an embedding (1000x256),
    # bias-free linears q_proj, k_proj (256->256), w1 (256->512), w2 (512->256), output
    # (256->1000) and router (256->8). Used to test name-based exclusions.
    class FakeTransformer(nn.Module):
        # [CLASS DOC]
        # HELPER: Container of named submodules mimicking transformer layer names;
        # forward is an identity.
        def __init__(self):
            # [FUNCTION DOC]
            # Creates the embedding and the six linear layers described above.
            super().__init__()
            self.tok_embeddings = nn.Embedding(1000, 256)
            self.q_proj = nn.Linear(256, 256, bias=False)
            self.k_proj = nn.Linear(256, 256, bias=False)
            self.w1 = nn.Linear(256, 512, bias=False)
            self.w2 = nn.Linear(512, 256, bias=False)
            self.output = nn.Linear(256, 1000, bias=False)
            self.router = nn.Linear(256, 8, bias=False)

        def forward(self, x):
            # [FUNCTION DOC]
            # Identity: returns x unchanged.
            return x

    with torch.device("meta"):
        return FakeTransformer().to(torch.bfloat16)


class TestEnableFP8Linear:
    # [CLASS DOC]
    # Tests for enable_fp8_linear patching rules.
    def test_basic_patching(self):
        # [FUNCTION DOC]
        # 256->512 bias-free linear: asserts unpatched before, patched after, weight
        # still bfloat16 and the module class still exactly nn.Linear.
        """FP8 patching: forward replaced, weight stays BF16, class stays nn.Linear."""
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=256, out_features=512, bias=False)
        assert not _is_fp8_patched(model.layer)

        enable_fp8_linear(model)
        assert _is_fp8_patched(model.layer)
        assert model.layer.weight.dtype == torch.bfloat16
        assert type(model.layer) is nn.Linear

    def test_default_exclusions_and_patches(self):
        # [FUNCTION DOC]
        # On FakeTransformer with default config, asserts output and router stay
        # unpatched while q_proj, k_proj, w1, w2 are patched.
        """Default config excludes output/router, patches eligible linears."""
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_multi_layer_model()
        enable_fp8_linear(model)

        # Excluded by default
        assert not _is_fp8_patched(model.output)
        assert not _is_fp8_patched(model.router)
        # Eligible modules patched
        assert _is_fp8_patched(model.q_proj)
        assert _is_fp8_patched(model.k_proj)
        assert _is_fp8_patched(model.w1)
        assert _is_fp8_patched(model.w2)

    def test_skips_non_aligned_dimensions(self):
        # [FUNCTION DOC]
        # 100->200 linear: patches the fp8 module logger, asserts the layer is not
        # patched and a warning mentioning "not 128-aligned" was emitted.
        from unittest.mock import patch

        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=100, out_features=200, bias=False)
        with patch("areal.experimental.models.archon.fp8.logger") as mock_logger:
            enable_fp8_linear(model)

        assert not _is_fp8_patched(model.layer)
        warning_msgs = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("not 128-aligned" in msg for msg in warning_msgs)

    def test_skips_linear_with_bias(self):
        # [FUNCTION DOC]
        # Asserts a 256->256 linear with bias is not patched.
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=256, out_features=256, bias=True)
        enable_fp8_linear(model)
        assert not _is_fp8_patched(model.layer)

    def test_custom_exclude_fqns(self):
        # [FUNCTION DOC]
        # With exclude_fqns {"proj","output","router"} (substring matching), asserts
        # q_proj stays unpatched while w1 and w2 are patched.
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_multi_layer_model()
        enable_fp8_linear(model, exclude_fqns={"proj", "output", "router"})

        assert not _is_fp8_patched(model.q_proj)
        assert _is_fp8_patched(model.w1)
        assert _is_fp8_patched(model.w2)


def _make_moe_model_on_meta():
    # [FUNCTION DOC]
    # HELPER: Builds on meta in bfloat16 a FakeMoEModel holding a 256x256 linear, an
    # aligned GroupedExperts (dim 256, hidden 512, 4 experts) and a non-aligned one (dim
    # 100, hidden 200, 2 experts), both with use_grouped_mm False.
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    class FakeMoEModel(nn.Module):
        # [CLASS DOC]
        # HELPER: Container for one linear and two GroupedExperts modules; forward is
        # identity.
        def __init__(self):
            # [FUNCTION DOC]
            # Creates linear, experts and small_experts as described by the factory.
            super().__init__()
            self.linear = nn.Linear(256, 256, bias=False)
            self.experts = GroupedExperts(
                dim=256, hidden_dim=512, num_experts=4, use_grouped_mm=False
            )
            self.small_experts = GroupedExperts(
                dim=100, hidden_dim=200, num_experts=2, use_grouped_mm=False
            )

        def forward(self, x):
            # [FUNCTION DOC]
            # Identity: returns x unchanged.
            return x

    with torch.device("meta"):
        return FakeMoEModel().to(torch.bfloat16)


class TestEnableFP8Experts:
    # [CLASS DOC]
    # Tests for enable_fp8_experts patching of GroupedExperts.
    def test_basic_expert_enabling(self):
        # [FUNCTION DOC]
        # Asserts the 128-aligned experts module has its forward replaced.
        from areal.experimental.models.archon.fp8 import enable_fp8_experts

        model = _make_moe_model_on_meta()
        enable_fp8_experts(model)

        assert _is_fp8_experts_patched(model.experts)

    def test_skips_non_aligned_experts(self):
        # [FUNCTION DOC]
        # Asserts the non-aligned small_experts module is left unpatched.
        from areal.experimental.models.archon.fp8 import enable_fp8_experts

        model = _make_moe_model_on_meta()
        enable_fp8_experts(model)

        assert not _is_fp8_experts_patched(model.small_experts)
