# [MODULE DOC]
# Tests set_deterministic_algorithms (areal.engine.megatron_utils.deterministic) on a
# fake Megatron config (SimpleNamespace). Covers the prebuild flag (switching attention
# backend to flash), the fusion flags it disables, and the warning logged when
# transformer_engine is imported without NVTE_ALLOW_NONDETERMINISTIC_ALGO set.
"""Tests for Megatron deterministic-mode activation semantics."""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms


@pytest.fixture(autouse=True)
def _restore_global_state(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Autouse fixture. Clears NVTE_ALLOW_NONDETERMINISTIC_ALGO, NCCL_ALGO,
    # CUBLAS_WORKSPACE_CONFIG and NCCL_NVLS_ENABLE before each test and, after the test,
    # restores torch's global deterministic-algorithms setting and warn_only flag to
    # their previous values.
    monkeypatch.delenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", raising=False)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    yield

    torch.use_deterministic_algorithms(prev_deterministic, warn_only=prev_warn_only)


def _make_config(**extra):
    # [FUNCTION DOC]
    # HELPER: Builds a SimpleNamespace config with deterministic_mode=False,
    # cross_entropy_loss_fusion=True, bias_dropout_fusion=True plus any extra keyword
    # attributes.
    return SimpleNamespace(
        deterministic_mode=False,
        cross_entropy_loss_fusion=True,
        bias_dropout_fusion=True,
        **extra,
    )


def test_set_deterministic_algorithms_prebuild_selects_flash_backend():
    # [FUNCTION DOC]
    # Requires megatron (importorskip). With attention_backend=auto and prebuild=True,
    # asserts deterministic_mode is enabled, both fusions disabled and backend set to
    # flash.
    enums = pytest.importorskip("megatron.core.transformer.enums")
    config = _make_config(attention_backend=enums.AttnBackend.auto)

    set_deterministic_algorithms(config, prebuild=True)

    assert config.deterministic_mode is True
    assert config.cross_entropy_loss_fusion is False
    assert config.bias_dropout_fusion is False
    assert config.attention_backend == enums.AttnBackend.flash


def test_set_deterministic_algorithms_postbuild_keeps_attention_backend():
    # [FUNCTION DOC]
    # Without prebuild, asserts deterministic_mode is set but attention_backend is left
    # as the original sentinel object.
    sentinel = object()
    config = _make_config(attention_backend=sentinel)

    set_deterministic_algorithms(config)

    assert config.deterministic_mode is True
    assert config.attention_backend is sentinel


def test_set_deterministic_algorithms_te_imported_without_env_warns(monkeypatch):
    # [FUNCTION DOC]
    # Injects a dummy transformer_engine module into sys.modules with the NVTE env var
    # unset; asserts the module logger emits a warning.
    monkeypatch.setitem(
        sys.modules, "transformer_engine", types.ModuleType("transformer_engine")
    )
    config = _make_config()

    with mock.patch("areal.engine.megatron_utils.deterministic.logger") as mock_logger:
        set_deterministic_algorithms(config)

    assert mock_logger.warning.called


def test_set_deterministic_algorithms_env_preset_does_not_warn(monkeypatch):
    # [FUNCTION DOC]
    # Same as the previous test but with NVTE_ALLOW_NONDETERMINISTIC_ALGO="0" preset;
    # asserts no warning is logged.
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setitem(
        sys.modules, "transformer_engine", types.ModuleType("transformer_engine")
    )
    config = _make_config()

    with mock.patch("areal.engine.megatron_utils.deterministic.logger") as mock_logger:
        set_deterministic_algorithms(config)

    assert not mock_logger.warning.called
