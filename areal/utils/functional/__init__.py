# [PACKAGE DOC]
# Functional RL math package: re-exports policy/critic loss functions, DPO helpers,
# normalization, rejection sampling and overlong reward penalty from functional.py, and
# vocab-parallel-aware logprob/entropy gathering from vocab_parallel.py.
# SPDX-License-Identifier: Apache-2.0

from areal.utils.functional.functional import (
    RejectionSamplingResult,
    apply_rejection_sampling,
    cispo_loss_fn,
    dpo_pair_logratios,
    dpo_preference_loss,
    masked_normalization,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.functional.vocab_parallel import (
    gather_logprobs,
    gather_logprobs_entropy,
)

__all__ = [
    # functional.py
    "RejectionSamplingResult",
    "apply_rejection_sampling",
    "cispo_loss_fn",
    "dpo_pair_logratios",
    "dpo_preference_loss",
    "masked_normalization",
    "ppo_actor_loss_fn",
    "ppo_critic_loss_fn",
    "reward_overlong_penalty",
    "sapo_loss_fn",
    # vocab_parallel.py
    "gather_logprobs",
    "gather_logprobs_entropy",
]
