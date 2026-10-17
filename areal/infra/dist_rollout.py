# [MODULE DOC]
# SPMD (non-controller) rollout coordination for training processes: data-parallel head
# ranks generate rollouts through a local InferenceEngine client, trajectories are all-
# gathered and load-balanced across the data-parallel group by sequence length, and then
# broadcast to the context/model-parallel ranks of each DP group, with device syncs and
# CPU barriers around collectives.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import InferenceEngine, TrainEngine, WorkflowLike
from areal.infra.platforms import current_platform
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    split_and_unpad_tensor,
    tensor_container_to,
)
from areal.utils.seqpack import get_allocate_fn


@dataclass
class RedistributedData:
    # [CLASS DOC]
    # Result of redistribution: all_data (every gathered trajectory, unpadded), data
    # (the trajectories assigned to this rank), rank (this rank within the group) and
    # group_indices (per-rank lists of indices into all_data).
    all_data: list[dict[str, Any]]
    data: list[dict[str, Any]]
    rank: int
    group_indices: list[list[int]]


def redistribute_trajectories(
    trajectories: list[dict[str, Any]],
    group=None,
    packing_algorithm: str = "ffd",
) -> RedistributedData:
    # [FUNCTION DOC]
    # Collective over group (default process group); every rank must call it. All-
    # gathers the local trajectory dict lists, flattens them in rank order, computes
    # each trajectory's token count from the attention mask sum (one GPU-CPU sync per
    # trajectory), strips padding from each trajectory, and allocates indices to ranks
    # using the chosen packing algorithm ("ffd" or "kk") with effectively unlimited
    # capacity and at least world-size groups for balance. Returns a RedistributedData
    # with this rank's assigned trajectories (not concatenated).
    """Redistribute a list of trajectory dicts across a process group.

    Each trajectory dict should contain tensors with shape [batch_size, seqlen, *],
    where batch_size can vary per trajectory. This function gathers trajectories
    from all ranks and redistributes them for load balancing based on sequence lengths.

    Parameters
    ----------
    trajectories : list[dict[str, Any]]
        List of trajectory dictionaries from the local rank. Each trajectory
        contains tensors with shape [batch_size, seqlen, ...].
    group : dist.ProcessGroup, optional
        The process group for communication. If None, uses the default group.
    packing_algorithm : str, optional
        Packing algorithm to use ("ffd" or "kk"). Default is "ffd".

    Returns
    -------
    RedistributedData
        Contains:
        - all_data: All trajectories gathered from all ranks (with padding removed)
        - data: List of trajectories assigned to the local rank
        - rank: Local rank in the group
        - group_indices: Assignment of trajectory indices to each rank
    """
    # All-gather trajectories from all ranks
    all_gathered = all_gather_tensor_container(trajectories, group=group)

    # Flatten the list of lists into a single list of trajectories
    all_data = []
    for traj_list in all_gathered:
        all_data.extend(traj_list)

    # Compute sequence lengths for load balancing
    seqlens = [d["attention_mask"].sum().item() for d in all_data]

    # Remove pad positions from each trajectory (split_and_unpad_tensor
    # auto-derives trim lengths from attention_mask when traj_seqlens=None)
    all_data = [
        split_and_unpad_tensor(
            d, n_trajs=1, traj_group_sizes=[d["attention_mask"].shape[0]]
        )[0]
        for d in all_data
    ]

    allocate_fn = get_allocate_fn(packing_algorithm)
    # Allocate trajectories to ranks using the configured packing algorithm
    # No capacity limit leads to balanced partition across this group
    group_indices = allocate_fn(
        seqlens, capacity=int(1e12), min_groups=dist.get_world_size(group)
    )
    local_indices = group_indices[dist.get_rank(group=group)]

    # Select assigned trajectories for this rank (no concatenation — deferred to train side)
    data = [all_data[i] for i in local_indices]
    return RedistributedData(
        all_data=all_data,
        data=data,
        rank=dist.get_rank(group=group),
        group_indices=group_indices,
    )


class DistRolloutCoordinator:
    # [CLASS DOC]
    # Coordinates a rollout InferenceEngine and a TrainEngine inside SPMD training
    # processes so that only DP heads generate data and every rank ends up with its
    # share of the batch.
    def __init__(self, rollout_engine: InferenceEngine, train_engine: TrainEngine):
        # [FUNCTION DOC]
        # Stores the rollout (inference) engine and the train engine. No I/O.
        self.rollout_engine = rollout_engine
        self.train_engine = train_engine

    def _broadcast_and_redistribute_trajectories(
        self,
        trajectories: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Collective. On DP heads (trajectories not None), redistributes
        # trajectories across the train engine's data-parallel group using the packing
        # algorithm from train_engine.config.mb_spec (default "ffd"); other ranks start
        # with None. Then device-synchronizes and barriers on the CPU group, broadcasts
        # the batch from the current DP head to the context-and-model-parallel group,
        # and synchronizes/barriers again. Returns the local list of trajectories.
        """Broadcast and redistribute trajectories across distributed workers.

        This helper encapsulates:
        1. Redistribution within data parallel group (for load balancing)
        2. Broadcasting to context and model parallel group
        3. Synchronization barriers

        Parameters
        ----------
        trajectories : list[dict[str, Any]] | None
            List of trajectory dicts from data parallel head, None for other ranks.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory.

        Returns
        -------
        list[dict[str, Any]]
            Redistributed and broadcast batch available on all ranks (list of trajs)
        """
        if trajectories is not None:
            config = getattr(self.train_engine, "config", None)
            mb_spec = getattr(config, "mb_spec", None)
            packing_algorithm = getattr(mb_spec, "packing_algorithm", "ffd")
            redist = redistribute_trajectories(
                trajectories,
                group=self.train_engine.data_parallel_group,
                packing_algorithm=packing_algorithm,
            )
            batch = redist.data
        else:
            batch = None

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        batch = broadcast_tensor_container(
            batch,
            src_rank=self.train_engine.current_data_parallel_head(),
            group=self.train_engine.context_and_model_parallel_group,
        )

        current_platform.synchronize()
        dist.barrier(group=self.train_engine.cpu_group)

        return batch

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Synchronous rollout: on DP heads calls the rollout engine's rollout_batch with
        # the given workflow settings and moves results to the current device; then
        # redistributes/broadcasts so all ranks return their batch. Collective; all
        # ranks must call it.
        # KNOWN ISSUE: the docstring mentions a connect_engine requirement and a
        # RuntimeError, but this method performs no such check.
        """Generate rollout batch with distributed coordination (synchronous).

        This method orchestrates distributed rollout generation:
        - Only data parallel heads generate rollouts (avoid redundancy)
        - Results are transferred to device and redistributed
        - Batch is broadcast to all workers
        - Synchronization barriers ensure consistency

        Must call connect_engine() before using this method.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            Input data batch for rollout generation
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            Redistributed rollout trajectories on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.rollout_batch(
                data,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=group_size,
                reward_normalization=reward_normalization,
                drop_incomplete_group=drop_incomplete_group,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Asynchronous (staleness-controlled) rollout: on DP heads calls the rollout
        # engine's prepare_batch with the dataloader, workflow, should_accept_fn,
        # group_size, dynamic_bs and normalization flags, moves results to the device,
        # then redistributes/broadcasts. Returns the local batch. Collective.
        # KNOWN ISSUE: same as rollout_batch, the documented connect_engine check and
        # RuntimeError are not implemented here.
        """Prepare async rollout batch with distributed coordination.

        Similar to rollout_batch but uses prepare_batch for async training,
        where rollout generation happens concurrently with training.

        Must call connect_engine() before using this method.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            Dataloader to pull samples from
        workflow : WorkflowLike
            Workflow defining rollout logic
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            Filter function for accepting samples based on staleness
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            Prepared rollout trajectories on all ranks

        Raises
        ------
        RuntimeError
            If rollout engine not connected via connect_engine()
        """

        trajectories = None
        if self.train_engine.is_data_parallel_head():
            trajectories = self.rollout_engine.prepare_batch(
                dataloader,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                should_accept_fn=should_accept_fn,
                group_size=group_size,
                dynamic_bs=dynamic_bs,
                reward_normalization=reward_normalization,
                drop_incomplete_group=drop_incomplete_group,
            )
            trajectories = tensor_container_to(
                trajectories, current_platform.current_device()
            )

        return self._broadcast_and_redistribute_trajectories(trajectories)
