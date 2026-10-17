# [MODULE DOC]
# Unit tests for MegatronEngine optimizer / LR-scheduler construction. Engines are built
# via __new__ (no real init, no GPU/distributed setup) with minimal attributes, and the
# Megatron factory functions (get_megatron_optimizer, OptimizerParamScheduler,
# MegatronCheckpointManager) are monkeypatched to capture arguments.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig
from areal.engine import megatron_engine as megatron_engine_module


def _make_test_engine(optimizer_config: OptimizerConfig):
    # [FUNCTION DOC]
    # HELPER: Builds a bare MegatronEngine (bypassing __init__) with the given optimizer
    # config, a SimpleNamespace config with use_lora=False, default MegatronEngineConfig, a
    # single dummy model chunk, bfloat16 dtype and FP8 disabled. Returns the engine.
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.optimizer_config = optimizer_config
    engine.config = SimpleNamespace(use_lora=False)
    engine.mcore_config = MegatronEngineConfig()
    engine.model = [object()]
    engine.dtype = torch.bfloat16
    engine.enable_fp8 = False
    engine.fp8_config = None
    return engine


def test_precision_aware_optimizer_fields_are_applied_before_validation(
    monkeypatch,
) -> None:
    # [FUNCTION DOC]
    # Verifies _create_optimizer forwards precision-aware optimizer settings from the mcore
    # config into the Megatron OptimizerConfig: use_precision_aware_optimizer and its derived
    # no-fp8 flag are True and the main grads / main params / exp_avg / exp_avg_sq dtypes are
    # converted from strings to torch dtypes. Captures the config via a monkeypatched
    # get_megatron_optimizer and stubs the scheduler and checkpoint manager.
    """MCore must derive its precision-aware mode from the final field values."""
    captured = {}
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.optimizer_config = OptimizerConfig(type="adam")
    engine.config = SimpleNamespace(use_lora=False)
    engine.mcore_config = MegatronEngineConfig(
        use_precision_aware_optimizer=True,
        main_grads_dtype="bfloat16",
        main_params_dtype="float32",
        exp_avg_dtype="float32",
        exp_avg_sq_dtype="float32",
    )
    engine.model = [object()]
    engine.dtype = torch.bfloat16
    engine.enable_fp8 = False
    engine.fp8_config = None

    def capture_optimizer(config, model):
        # [FUNCTION DOC]
        # HELPER: Fake get_megatron_optimizer; records config and model in the enclosing captured
        # dict and returns a dummy object.
        captured["config"] = config
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        megatron_engine_module, "get_megatron_optimizer", capture_optimizer
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1)
    )

    config = captured["config"]
    assert captured["model"] is engine.model
    assert config.use_precision_aware_optimizer is True
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True
    assert config.main_grads_dtype is torch.bfloat16
    assert config.main_params_dtype is torch.float32
    assert config.exp_avg_dtype is torch.float32
    assert config.exp_avg_sq_dtype is torch.float32


def test_scheduler_uses_fixed_warmup_and_resume_safe_initial_lr(
    monkeypatch,
) -> None:
    # [FUNCTION DOC]
    # Verifies the OptimizerParamScheduler receives init_lr=0.0, lr_warmup_steps equal to the
    # configured warmup_steps (3), and lr_decay_steps / wd_incr_steps equal to total train
    # steps (10) for a 10-sample, batch-size-1, one-epoch spec.
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="adam", warmup_steps=3))

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer",
        lambda config, model: object(),
    )

    def capture_scheduler(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake scheduler constructor; merges its kwargs into captured and returns a dummy.
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        capture_scheduler,
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1)
    )

    assert captured["init_lr"] == 0.0
    assert captured["lr_warmup_steps"] == 3
    assert captured["lr_decay_steps"] == 10
    assert captured["wd_incr_steps"] == 10


def test_dte_records_optimizer_step_lr_before_scheduler_advances() -> None:
    # [FUNCTION DOC]
    # Verifies that with DTE runtime enabled, optimizer_step stores each param group's current
    # LR under the "_areal_last_step_lr" key before lr_scheduler_step mutates the "lr" values,
    # using fake optimizer and scheduler objects sharing one param_groups list.
    """AdamW inversion must retain the LR consumed by the completed step."""
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    param_groups = [{"lr": 3e-6}, {"lr": 4e-6}]

    class _Optimizer:
        # [CLASS DOC]
        # HELPER: Fake Megatron optimizer exposing the shared param_groups and a step method.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Binds the enclosing param_groups list.
            self.param_groups = param_groups

        def step(self):
            # [FUNCTION DOC]
            # HELPER: Returns the Megatron-style tuple (update_successful=True, grad_norm tensor 1.0,
            # num_zeros=None).
            return True, torch.tensor(1.0), None

    class _Scheduler:
        # [CLASS DOC]
        # HELPER: Fake LR scheduler that lowers the shared param group LRs on step.
        def step(self, increment):
            # [FUNCTION DOC]
            # HELPER: Asserts increment == 1 and sets group LRs to 2e-6 and 1e-6 (side effect on the
            # shared param_groups).
            assert increment == 1
            param_groups[0]["lr"] = 2e-6
            param_groups[1]["lr"] = 1e-6

    engine.optimizer = _Optimizer()
    engine.lr_scheduler = _Scheduler()
    engine._dte_runtime_config = SimpleNamespace(enabled=True)

    engine.optimizer_step()
    engine.lr_scheduler_step()

    assert param_groups[0]["lr"] == 2e-6
    assert param_groups[1]["lr"] == 1e-6
    assert param_groups[0]["_areal_last_step_lr"] == 3e-6
    assert param_groups[1]["_areal_last_step_lr"] == 4e-6


@pytest.mark.parametrize(
    ("optimizer_config", "ft_spec", "match"),
    [
        (
            OptimizerConfig(type="adam", warmup_steps=0),
            FinetuneSpec(total_train_epochs=0, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires total_train_steps to be positive",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps=10),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps=11),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps_proportion=1.0),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
    ],
)
def test_scheduler_rejects_megatron_only_boundaries_before_optimizer_creation(
    monkeypatch,
    optimizer_config: OptimizerConfig,
    ft_spec: FinetuneSpec,
    match: str,
) -> None:
    # [FUNCTION DOC]
    # Parametrized: for zero total steps, or warmup steps (absolute or proportion=1.0) >= total
    # steps, _create_optimizer must raise ValueError with the given message before ever
    # calling get_megatron_optimizer (which is patched to fail the test).
    engine = _make_test_engine(optimizer_config)

    def fail_if_called(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fails the test immediately if the optimizer factory is invoked.
        pytest.fail("optimizer must not be created for an invalid scheduler config")

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer",
        fail_if_called,
    )

    with pytest.raises(ValueError, match=match):
        engine._create_optimizer(ft_spec)
