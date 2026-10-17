# [MODULE DOC]
# Unit tests for the "reuse_train_logp" proximal-logp method, which reuses the training
# forward's logprobs as the proximal policy (valid only with a single PPO minibatch).
# KNOWN ISSUE: the rejection of ppo_n_minibatches > 1 described in the docstring is not
# tested here; only the accepted single-minibatch case is covered.
"""Tests for the ``reuse_train_logp`` proximal-logp method.

``reuse_train_logp`` reuses the training forward-pass logprobs as the proximal
logp, skipping the extra decoupled-PPO forward. It requires
``ppo_n_minibatches == 1`` so the training forward still reflects the policy
that produced the rollout (with multiple minibatches the weights change between
steps, so the reused logprobs would no longer be the proximal policy).
"""

from areal.api.cli_args import PPOActorConfig
from areal.utils.constants import (
    PROX_LOGP_METHOD_REUSE_TRAIN_LOGP,
    PROX_LOGP_METHODS_ALL,
    ProxLogpMethod,
)


def test_enum_includes_reuse_train_logp():
    # [FUNCTION DOC]
    # ProxLogpMethod has REUSE_TRAIN_LOGP with value "reuse_train_logp", constructible from
    # the string, listed in PROX_LOGP_METHODS_ALL and matching the legacy constant.
    assert ProxLogpMethod.REUSE_TRAIN_LOGP.value == "reuse_train_logp"
    assert ProxLogpMethod("reuse_train_logp") is ProxLogpMethod.REUSE_TRAIN_LOGP
    assert "reuse_train_logp" in PROX_LOGP_METHODS_ALL
    assert PROX_LOGP_METHOD_REUSE_TRAIN_LOGP == "reuse_train_logp"


def test_reuse_train_logp_skips_forward_pass():
    # [FUNCTION DOC]
    # REUSE_TRAIN_LOGP reports that it skips the extra proximal forward pass.
    # It must skip the extra proximal forward, like loglinear does.
    assert ProxLogpMethod.REUSE_TRAIN_LOGP.skips_forward_pass()


def test_reuse_train_logp_allows_single_minibatch():
    # [FUNCTION DOC]
    # PPOActorConfig accepts reuse_train_logp with ppo_n_minibatches=1.
    config = PPOActorConfig(
        backend="fsdp:d1",
        prox_logp_method="reuse_train_logp",
        ppo_n_minibatches=1,
    )
    assert config.prox_logp_method == "reuse_train_logp"
    assert config.ppo_n_minibatches == 1
