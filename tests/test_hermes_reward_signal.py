# [MODULE DOC]
# Config sanity test for examples/hermes/config.yaml: in single-sample online training,
# reward shaping must keep success and failure distinguishable with nonzero advantages.
from pathlib import Path

import torch
import yaml

from areal.api.cli_args import PPOActorConfig

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples/hermes/config.yaml"


def _load_config() -> dict:
    # [FUNCTION DOC]
    # HELPER: Reads and YAML-parses examples/hermes/config.yaml, returning a dict.
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def test_hermes_singleton_online_config_preserves_each_outcome_signal():
    # [FUNCTION DOC]
    # Asserts n_samples and batch_size are 1 and reward_norm/adv_norm are unset. Applies
    # the config's (or PPOActorConfig default) reward_bias and reward_scaling to
    # outcomes 0 and 1, then asserts each shaped reward is nonzero when broadcast to 3
    # tokens, that failure is negative and success positive, and that they differ.
    config = _load_config()
    actor = config["actor"]

    assert config["gconfig"]["n_samples"] == 1
    assert config["train_dataset"]["batch_size"] == 1
    assert actor.get("reward_norm") is None
    assert actor.get("adv_norm") is None

    counterfactual_outcomes = torch.tensor([0.0, 1.0])
    actor_defaults = PPOActorConfig()
    reward_bias = actor.get("reward_bias", actor_defaults.reward_bias)
    reward_scaling = actor.get("reward_scaling", actor_defaults.reward_scaling)
    signed_rewards = (counterfactual_outcomes + reward_bias) * reward_scaling

    for signed_reward in signed_rewards:
        single_trajectory_token_advantages = signed_reward.expand(3)
        assert torch.count_nonzero(single_trajectory_token_advantages).item() == 3

    assert signed_rewards[0] < 0 < signed_rewards[1]
    assert not torch.equal(signed_rewards[0], signed_rewards[1])
