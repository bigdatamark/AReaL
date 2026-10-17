# [MODULE DOC]
# CPU unit tests verifying that make_mcore_model (areal.models.mcore.registry) with
# bridge_type "megatron-bridge" applies deterministic-mode settings to the Megatron-
# Bridge provider before provider.finalize() is called, and leaves defaults untouched
# when determinism is off. Uses fake bridge/provider objects; finalize raises a sentinel
# exception to stop model construction. Skips if mbridge or megatron.core is not
# installed.
"""Tests for Megatron-Bridge deterministic provider configuration."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch


class _FinalizeReached(RuntimeError):
    # [CLASS DOC]
    # HELPER: Sentinel exception raised by the fake provider's finalize to abort
    # make_mcore_model at the point of interest.
    pass


class _FakeProvider(SimpleNamespace):
    # [CLASS DOC]
    # HELPER: Attribute-bag provider whose finalize snapshots the determinism-related
    # fields.
    def finalize(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Records (deterministic_mode, attention_backend,
        # cross_entropy_loss_fusion, bias_dropout_fusion) into config_at_finalize, then
        # raises _FinalizeReached. Never returns normally.
        self.config_at_finalize = (
            self.deterministic_mode,
            self.attention_backend,
            self.cross_entropy_loss_fusion,
            self.bias_dropout_fusion,
        )
        raise _FinalizeReached


class _FakeBridge:
    # [CLASS DOC]
    # HELPER: Fake Megatron-Bridge bridge that hands out a preset provider.
    def __init__(self, provider: _FakeProvider) -> None:
        # [FUNCTION DOC]
        # HELPER: Stores the provider to return.
        self.provider = provider

    def to_megatron_provider(self, *, load_weights: bool) -> _FakeProvider:
        # [FUNCTION DOC]
        # HELPER: Asserts load_weights is False and returns the stored provider.
        assert load_weights is False
        return self.provider


@pytest.fixture(autouse=True)
def _restore_global_deterministic_state(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture: removes NVTE_ALLOW_NONDETERMINISTIC_ALGO, NCCL_ALGO and
    # CUBLAS_WORKSPACE_CONFIG from the environment (monkeypatch restores them), saves
    # torch's global deterministic-algorithms flags, yields, then restores those flags,
    # since make_mcore_model may mutate this global state.
    monkeypatch.delenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    yield

    torch.use_deterministic_algorithms(
        previous_enabled,
        warn_only=previous_warn_only,
    )


def _make_mcore_config(*, deterministic: bool) -> SimpleNamespace:
    # [FUNCTION DOC]
    # HELPER: Returns a SimpleNamespace mcore config with no VPP, no recompute, MTP
    # disabled, alltoall MoE dispatcher and use_deterministic_algorithms set to the
    # argument.
    return SimpleNamespace(
        virtual_pipeline_parallel_size=None,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        distribute_saved_activations=False,
        recompute_modules=None,
        enable_mtp=False,
        moe_token_dispatcher_type="alltoall",
        use_deterministic_algorithms=deterministic,
    )


def _make_provider(attention_backend) -> _FakeProvider:
    # [FUNCTION DOC]
    # HELPER: Returns a _FakeProvider with deterministic_mode False, the given
    # attention_backend, both fusions True and mtp_num_layers None.
    return _FakeProvider(
        deterministic_mode=False,
        attention_backend=attention_backend,
        cross_entropy_loss_fusion=True,
        bias_dropout_fusion=True,
        mtp_num_layers=None,
    )


def _run_until_provider_finalize(
    provider: _FakeProvider,
    *,
    deterministic: bool,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Skips unless mbridge and megatron.core import, then calls make_mcore_model
    # with the fake bridge and a mock mpu whose TP/PP/CP/EP/ETP world-size getters all
    # return 1, expecting _FinalizeReached to be raised. Mutates provider
    # (config_at_finalize). Returns None.
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore.registry import make_mcore_model

    tf_config = SimpleNamespace(params_dtype=None)
    mcore_config = _make_mcore_config(deterministic=deterministic)

    with (
        mock.patch.multiple(
            "areal.models.mcore.registry.mpu",
            get_tensor_model_parallel_world_size=mock.DEFAULT,
            get_pipeline_model_parallel_world_size=mock.DEFAULT,
            get_context_parallel_world_size=mock.DEFAULT,
            get_expert_model_parallel_world_size=mock.DEFAULT,
            get_expert_tensor_parallel_world_size=mock.DEFAULT,
        ) as mpu_mocks,
        pytest.raises(_FinalizeReached),
    ):
        for getter in mpu_mocks.values():
            getter.return_value = 1
        make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=tf_config,
            mcore_config=mcore_config,
            bridge=_FakeBridge(provider),
            bridge_type="megatron-bridge",
        )


def test_megatron_bridge_provider_applies_determinism_before_finalize():
    # [FUNCTION DOC]
    # deterministic=True with AttnBackend.auto: at finalize, deterministic_mode is True,
    # attention backend forced to flash, and both fusions disabled.
    """Deterministic settings reach the actual provider before finalization."""
    enums = pytest.importorskip("megatron.core.transformer.enums")
    provider = _make_provider(enums.AttnBackend.auto)

    _run_until_provider_finalize(provider, deterministic=True)

    assert provider.config_at_finalize == (
        True,
        enums.AttnBackend.flash,
        False,
        False,
    )


def test_megatron_bridge_provider_preserves_defaults_when_disabled():
    # [FUNCTION DOC]
    # deterministic=False: provider fields at finalize equal the originals (False, same
    # backend object, True, True).
    """The Megatron-Bridge provider remains unchanged without the opt-in."""
    attention_backend = object()
    provider = _make_provider(attention_backend)

    _run_until_provider_finalize(provider, deterministic=False)

    assert provider.config_at_finalize == (
        False,
        attention_backend,
        True,
        True,
    )
