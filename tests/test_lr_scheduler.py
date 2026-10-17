# [MODULE DOC]
# Unit tests for get_num_warmup_steps (areal.utils.lr_scheduler) and the warmup
# validation in OptimizerConfig: fixed warmup_steps vs warmup_steps_proportion
# precedence, warnings, and negative-value rejection.
import warnings

import pytest

from areal.api.cli_args import OptimizerConfig
from areal.utils.lr_scheduler import get_num_warmup_steps


def test_get_num_warmup_steps_uses_proportion_when_fixed_steps_unset():
    # [FUNCTION DOC]
    # Only warmup_steps_proportion=0.1 set: 100 total steps yields 10 warmup steps.
    optimizer_config = OptimizerConfig(warmup_steps_proportion=0.1)

    assert get_num_warmup_steps(optimizer_config, total_train_steps=100) == 10


def test_get_num_warmup_steps_prefers_fixed_steps_over_proportion():
    # [FUNCTION DOC]
    # Setting both warmup_steps=7 and proportion 0.5 emits a UserWarning ("warmup_steps
    # takes precedence") at config construction, and the fixed 7 is returned.
    with pytest.warns(UserWarning, match="warmup_steps takes precedence"):
        optimizer_config = OptimizerConfig(
            warmup_steps=7,
            warmup_steps_proportion=0.5,
        )

    assert get_num_warmup_steps(optimizer_config, total_train_steps=100) == 7


def test_fixed_warmup_with_implicit_default_proportion_does_not_warn():
    # [FUNCTION DOC]
    # warmup_steps=7 with the default proportion emits no warnings at all.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OptimizerConfig(warmup_steps=7)

    assert not caught


def test_get_num_warmup_steps_rejects_negative_fixed_steps():
    # [FUNCTION DOC]
    # warmup_steps=-1 makes get_num_warmup_steps raise ValueError ("warmup_steps must be
    # non-negative").
    optimizer_config = OptimizerConfig(warmup_steps=-1)

    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        get_num_warmup_steps(optimizer_config, total_train_steps=100)


def test_get_num_warmup_steps_rejects_negative_proportion():
    # [FUNCTION DOC]
    # Negative proportion -0.1 makes get_num_warmup_steps raise ValueError.
    optimizer_config = OptimizerConfig(warmup_steps_proportion=-0.1)

    with pytest.raises(
        ValueError,
        match="warmup_steps_proportion must be non-negative",
    ):
        get_num_warmup_steps(optimizer_config, total_train_steps=100)


@pytest.mark.parametrize("warmup_steps", [0, 99, 100, 101])
def test_get_num_warmup_steps_accepts_non_negative_fixed_steps(
    warmup_steps: int,
) -> None:
    # [FUNCTION DOC]
    # Parametrized warmup_steps in {0, 99, 100, 101}: returned unchanged even when
    # exceeding total steps (100).
    optimizer_config = OptimizerConfig(warmup_steps=warmup_steps)

    assert get_num_warmup_steps(optimizer_config, total_train_steps=100) == warmup_steps


def test_get_num_warmup_steps_allows_proportion_covering_all_training():
    # [FUNCTION DOC]
    # Proportion 1.0 with 100 total steps yields 100 warmup steps.
    optimizer_config = OptimizerConfig(warmup_steps_proportion=1.0)

    assert get_num_warmup_steps(optimizer_config, total_train_steps=100) == 100


def test_get_num_warmup_steps_allows_zero_total_steps():
    # [FUNCTION DOC]
    # Default config with total_train_steps=0 yields 0 warmup steps.
    optimizer_config = OptimizerConfig()

    assert get_num_warmup_steps(optimizer_config, total_train_steps=0) == 0
