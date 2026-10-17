# [MODULE DOC]
# Async unit tests for GroupedRolloutWorkflow (areal.infra.remote_inf_engine), which
# runs an inner workflow group_size times and optionally normalizes rewards within the
# group or drops incomplete groups, and for DistRolloutCoordinator forwarding those
# flags. Uses light fakes instead of real engines.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.infra import dist_rollout
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow


class _ListWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # HELPER: Fake RolloutWorkflow returning pre-set results in order, one per
    # arun_episode call.
    def __init__(self, results):
        # [FUNCTION DOC]
        # HELPER: Stores a list copy of results and resets the call index to 0.
        self.results = list(results)
        self.index = 0

    async def arun_episode(self, engine, data):
        # [FUNCTION DOC]
        # HELPER: Returns the next stored result and advances the index; ignores engine
        # and data. Raises IndexError when exhausted.
        result = self.results[self.index]
        self.index += 1
        return result


class _Logger:
    # [CLASS DOC]
    # HELPER: Fake logger that records warning messages.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes an empty message list.
        self.messages: list[str] = []

    def warning(self, message: str):
        # [FUNCTION DOC]
        # HELPER: Appends message to messages.
        self.messages.append(message)


class _TrainEngine:
    # [CLASS DOC]
    # HELPER: Fake train engine that always reports being data-parallel head.
    def is_data_parallel_head(self):
        # [FUNCTION DOC]
        # HELPER: Returns True.
        return True


class _RolloutEngine:
    # [CLASS DOC]
    # HELPER: Fake rollout engine recording kwargs passed to prepare_batch and
    # rollout_batch.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes both recorded kwargs to None.
        self.prepare_kwargs = None
        self.rollout_kwargs = None

    def prepare_batch(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs and returns a one-trajectory list.
        self.prepare_kwargs = kwargs
        return [{"trajectory": True}]

    def rollout_batch(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs and returns a one-trajectory list.
        self.rollout_kwargs = kwargs
        return [{"trajectory": True}]


def _interaction(reward: float) -> InteractionWithTokenLogpReward:
    # [FUNCTION DOC]
    # HELPER: Builds an InteractionWithTokenLogpReward with the given reward and a pre-
    # populated tensor cache (input_ids, loss_mask, logprobs, versions, attention_mask,
    # rewards) for a 2-token sequence.
    return InteractionWithTokenLogpReward(
        reward=reward,
        _cache={
            "input_ids": torch.tensor([[1, 2]]),
            "loss_mask": torch.tensor([[0, 1]]),
            "logprobs": torch.tensor([[0.0, -0.1]]),
            "versions": torch.tensor([[-1, 0]]),
            "attention_mask": torch.tensor([[True, True]]),
            "rewards": torch.tensor([reward]),
        },
    )


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_normalizes_rewards_and_updates_cache():
    # [FUNCTION DOC]
    # Group of rewards 1 and 3 with reward_normalization: asserts results are merged,
    # rewards become -1 and 1 (mean/std normalized), original_reward keeps raw values,
    # and cached "rewards"/"original_rewards" tensors are updated accordingly.
    first = _interaction(1.0)
    second = _interaction(3.0)
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": first}, {"b": second}]),
        group_size=2,
        logger=_Logger(),
        reward_normalization=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result == {"a": first, "b": second}
    assert first.reward == pytest.approx(-1.0)
    assert second.reward == pytest.approx(1.0)
    assert first.original_reward == pytest.approx(1.0)
    assert second.original_reward == pytest.approx(3.0)
    assert first._cache is not None
    assert second._cache is not None
    assert first._cache["rewards"].item() == pytest.approx(-1.0)
    assert second._cache["rewards"].item() == pytest.approx(1.0)
    assert first._cache["original_rewards"].item() == pytest.approx(1.0)
    assert second._cache["original_rewards"].item() == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_drops_incomplete_group():
    # [FUNCTION DOC]
    # With drop_incomplete_group and one None result: asserts None returned and a
    # "dropping entire group" warning logged.
    logger = _Logger()
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": _interaction(1.0)}, None]),
        group_size=2,
        logger=logger,
        drop_incomplete_group=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result is None
    assert "dropping entire group" in logger.messages[0]


@pytest.mark.asyncio
async def test_grouped_rollout_workflow_reward_normalization_requires_full_group():
    # [FUNCTION DOC]
    # With reward_normalization and an incomplete group: asserts None returned and a
    # "reward_normalization: dropping group" warning.
    logger = _Logger()
    workflow = GroupedRolloutWorkflow(
        _ListWorkflow([{"a": _interaction(1.0)}, None]),
        group_size=2,
        logger=logger,
        reward_normalization=True,
    )

    result = await workflow.arun_episode(engine=None, data={})

    assert result is None
    assert any("reward_normalization: dropping group" in m for m in logger.messages)


def test_dist_rollout_coordinator_forwards_reward_group_flags(monkeypatch):
    # [FUNCTION DOC]
    # Builds a DistRolloutCoordinator over fakes, monkeypatching redistribution, current
    # device ("cpu") and tensor transfer to identities; calls prepare_batch and
    # rollout_batch with both flags True and asserts they are forwarded to the rollout
    # engine.
    rollout_engine = _RolloutEngine()
    coordinator = DistRolloutCoordinator(rollout_engine, _TrainEngine())
    monkeypatch.setattr(
        coordinator,
        "_broadcast_and_redistribute_trajectories",
        lambda trajectories: trajectories,
    )
    monkeypatch.setattr(dist_rollout.current_platform, "current_device", lambda: "cpu")
    monkeypatch.setattr(dist_rollout, "tensor_container_to", lambda data, device: data)

    coordinator.prepare_batch(
        dataloader=object(),
        workflow=object(),
        reward_normalization=True,
        drop_incomplete_group=True,
    )
    coordinator.rollout_batch(
        data=[{}],
        workflow=object(),
        reward_normalization=True,
        drop_incomplete_group=True,
    )

    assert rollout_engine.prepare_kwargs["reward_normalization"] is True
    assert rollout_engine.prepare_kwargs["drop_incomplete_group"] is True
    assert rollout_engine.rollout_kwargs["reward_normalization"] is True
    assert rollout_engine.rollout_kwargs["drop_incomplete_group"] is True
