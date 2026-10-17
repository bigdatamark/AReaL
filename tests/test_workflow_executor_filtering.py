# [MODULE DOC]
# Async tests for WorkflowExecutor's should_accept_fn filtering of rollout
# trajectories containing remote RTensor shards: accepted trajectories keep their remote
# payload; rejected or filter-raising ones trigger RTensor.clear_node for each shard
# owner, with a per-node timeout so a hung cleanup does not block. Uses mocks only.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput


class _TrajectoryWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # HELPER: RolloutWorkflow that returns a fixed pre-built trajectory.
    def __init__(self, trajectory):
        # [FUNCTION DOC]
        # Stores the trajectory dict to return.
        self.trajectory = trajectory

    async def arun_episode(self, engine, data):
        # [FUNCTION DOC]
        # Returns the stored trajectory regardless of engine/data.
        return self.trajectory


def _remote_trajectory(rewards: list[float]):
    # [FUNCTION DOC]
    # HELPER: Builds a trajectory dict with input_ids and loss_mask as RTensors pointing at
    # shards "input"@node-a:1234 and "mask"@node-b:1234 (meta-device placeholders), plus
    # rewards and original_rewards tensors from the given list.
    return {
        "input_ids": RTensor(
            shard=TensorShardInfo(shard_id="input", node_addr="node-a:1234"),
            data=torch.empty((2, 2), device="meta"),
        ),
        "loss_mask": RTensor(
            shard=TensorShardInfo(shard_id="mask", node_addr="node-b:1234"),
            data=torch.empty((2, 2), device="meta"),
        ),
        "rewards": torch.tensor(rewards),
        "original_rewards": torch.tensor(rewards),
    }


def _executor():
    # [FUNCTION DOC]
    # HELPER: Creates a WorkflowExecutor (sglang:d1, consumer batch 1, no dumping) with
    # mocked inference engine, mocked staleness manager and mocked logger. Returns
    # (executor, staleness_manager_mock).
    manager = MagicMock()
    executor = WorkflowExecutor(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=1,
            dump_to_file=False,
        ),
        inference_engine=MagicMock(),
        staleness_manager=manager,
    )
    executor.logger = MagicMock()
    return executor, manager


@pytest.mark.asyncio
async def test_filter_accepts_local_rewards_without_clearing_remote_payload(
    monkeypatch,
):
    # [FUNCTION DOC]
    # Uses the SWE example filter_function with rewards [0,1]; asserts the task result
    # keeps the same trajectory with RTensors intact, on_rollout_accepted is called once,
    # and clear_node is never awaited.
    from examples.swe.filter_function import filter_function

    trajectory = _remote_trajectory([0.0, 1.0])
    executor, manager = _executor()
    clear_node = AsyncMock()
    monkeypatch.setattr(RTensor, "clear_node", clear_node)
    task = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={},
            workflow=_TrajectoryWorkflow(trajectory),
            should_accept_fn=filter_function,
        )
    )

    result = await task()

    assert result is not None
    assert result.trajectory is trajectory
    assert isinstance(result.trajectory["input_ids"], RTensor)
    manager.on_rollout_accepted.assert_called_once_with()
    clear_node.assert_not_awaited()


def _reject(_trajectory):
    # [FUNCTION DOC]
    # HELPER: Filter predicate that always rejects.
    return False


def _raise(_trajectory):
    # [FUNCTION DOC]
    # HELPER: Filter predicate that raises RuntimeError("filter failed").
    raise RuntimeError("filter failed")


@pytest.mark.parametrize("predicate", [_reject, _raise])
@pytest.mark.asyncio
async def test_filter_rejection_or_failure_clears_remote_payload(
    monkeypatch, predicate
):
    # [FUNCTION DOC]
    # Parametrized over rejecting and raising predicates: asserts result None, the
    # rejection callback fires once, and clear_node is awaited for both shard nodes.
    trajectory = _remote_trajectory([1.0, 1.0])
    executor, manager = _executor()
    clear_node = AsyncMock()
    monkeypatch.setattr(RTensor, "clear_node", clear_node)
    task = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={},
            workflow=_TrajectoryWorkflow(trajectory),
            should_accept_fn=predicate,
        )
    )

    result = await task()

    assert result is None
    manager.on_rollout_rejected.assert_called_once_with()
    clear_node.assert_has_awaits(
        [
            call("node-a:1234", ["input"]),
            call("node-b:1234", ["mask"]),
        ],
        any_order=True,
    )


@pytest.mark.asyncio
async def test_filter_rejection_times_out_blocked_remote_cleanup(monkeypatch):
    # [FUNCTION DOC]
    # Replaces clear_node with one that hangs forever for node-a, patches the clear timeout
    # to 0.01s, and asserts the task finishes within 1s, rejects, clears only node-b, and
    # logs a single timeout warning for node-a.
    trajectory = _remote_trajectory([1.0, 1.0])
    executor, manager = _executor()
    cleared = []

    async def partially_blocked_clear(node_addr, shard_ids):
        # [FUNCTION DOC]
        # HELPER: Fake clear_node: blocks indefinitely for node-a, else records the call.
        if node_addr == "node-a:1234":
            await asyncio.Event().wait()
        cleared.append((node_addr, shard_ids))

    monkeypatch.setattr(RTensor, "clear_node", partially_blocked_clear)
    monkeypatch.setattr(
        "areal.infra.workflow_executor._REJECTED_TRAJECTORY_CLEAR_TIMEOUT_SECONDS",
        0.01,
    )
    task = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={},
            workflow=_TrajectoryWorkflow(trajectory),
            should_accept_fn=_reject,
        )
    )

    result = await asyncio.wait_for(task(), timeout=1.0)

    assert result is None
    manager.on_rollout_rejected.assert_called_once_with()
    assert cleared == [("node-b:1234", ["mask"])]
    executor.logger.warning.assert_called_once_with(
        "Timed out after %.1fs clearing rejected trajectory shards on %s",
        0.01,
        "node-a:1234",
    )
