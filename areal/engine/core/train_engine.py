# [MODULE DOC]
# Stateless helpers shared by training engines (FSDP, Megatron, ...) for global loss
# normalization across data-parallel ranks, evaluation-loss aggregation with pipeline
# parallelism, and post-processing of micro-batched forward outputs back to the original
# batch order.
# SPDX-License-Identifier: Apache-2.0

"""Core operations for training engines.

This module provides stateless utility functions that are shared across
different training engine implementations (FSDP, Megatron, etc.).
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils.data import (
    MicroBatchList,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

__all__ = [
    "compute_total_loss_weight",
    "aggregate_eval_losses",
    "reorder_and_pad_outputs",
]


def compute_total_loss_weight(
    mb_list: MicroBatchList,
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    dp_group: dist.ProcessGroup,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Applies loss_weight_fn to every micro-batch in mb_list.mbs, sums the results into
    # a detached float32 scalar, all-reduces (sum) it over dp_group and returns it.
    # Raises AssertionError if the global total is not positive. Side effects: a
    # collective on dp_group; the positivity assert forces a GPU-CPU sync.
    """Compute total loss weight and all_reduce across data parallel group.

    This aggregates the loss weights from all micro-batches and reduces
    them across the data parallel group to get a global normalization factor.

    Parameters
    ----------
    mb_list : MicroBatchList
        The list of micro-batches.
    loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
        Function to compute loss weight for each micro-batch.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.

    Returns
    -------
    torch.Tensor
        The total loss weight (scalar tensor) after all_reduce.
    """
    total_weight = (
        torch.stack([loss_weight_fn(mb) for mb in mb_list.mbs])
        .sum()
        .detach()
        .clone()
        .to(dtype=torch.float32)
    )
    dist.all_reduce(total_weight, group=dp_group)
    assert total_weight > 0, (
        "Global total loss weight must be positive after all_reduce"
    )
    return total_weight


def aggregate_eval_losses(
    losses: list[torch.Tensor] | None,
    dp_group: dist.ProcessGroup,
    is_pp_last_stage: bool = True,
    pp_group: dist.ProcessGroup | None = None,
    pp_src_rank: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # On the last pipeline stage, sums the per-micro-batch losses in float32 and all-
    # reduces (sum) over dp_group; on other stages allocates an uninitialized 1-element
    # float32 buffer on the current device. If pp_group is given, broadcasts the loss
    # from pp_src_rank (global rank) over it so every stage holds the value. Returns the
    # loss tensor. Raises AssertionError if losses is None on the last stage or
    # pp_src_rank is missing with pp_group.
    # KNOWN ISSUE: shape differs between stages (0-d scalar on the last stage vs shape
    # [1] buffer elsewhere), which can break the broadcast or downstream code expecting
    # a consistent shape.
    """Aggregate evaluation losses from micro-batches.

    Parameters
    ----------
    losses : list[torch.Tensor] | None
        List of loss tensors from each micro-batch. None on non-last PP stages.
    dp_group : dist.ProcessGroup
        The data parallel process group for all_reduce.
    is_pp_last_stage : bool
        Whether this rank is the last PP stage. True by default.
    pp_group : dist.ProcessGroup | None
        Pipeline parallel group for broadcast. None if PP broadcast is not required.
    pp_src_rank : int | None
        Global rank of last PP stage (required if pp_group is set).

    Returns
    -------
    torch.Tensor
        The aggregated loss after summing and all_reduce.
    """
    if is_pp_last_stage:
        assert losses is not None, "losses required on last PP stage"
        loss = torch.stack(losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=dp_group)
    else:
        device = current_platform.current_device()
        loss = torch.empty(1, device=device, dtype=torch.float32)

    if pp_group is not None:
        assert pp_src_rank is not None, "pp_src_rank required when pp_group is set"
        dist.broadcast(loss, src=pp_src_rank, group=pp_group)

    return loss


def reorder_and_pad_outputs(
    outputs: list[torch.Tensor],
    output_seqlens: list[int],
    mb_list: MicroBatchList,
    aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Concatenates micro-batch outputs with aggregate_fn (default torch.cat), splits the
    # packed result along dim 0 using output_seqlens permuted by
    # mb_list.forward_indices, restores the original sample order with
    # mb_list.backward_indices, then pads and stacks the per-sequence tensors along a
    # new batch dimension. Returns the padded tensor [batch, max_len, ...]. No side
    # effects.
    """Aggregate, reorder, and pad forward outputs from micro-batches.

    This handles the output post-processing for forward_batch:
    1. Aggregate outputs from all micro-batches
    2. Unpack by sequence lengths
    3. Reorder to match original input order
    4. Pad and stack along batch dimension

    Parameters
    ----------
    outputs : list[torch.Tensor]
        List of output tensors from each micro-batch.
    output_seqlens : list[int]
        Sequence lengths for unpacking.
    mb_list : MicroBatchList
        The micro-batch list containing reordering indices.
    aggregate_fn : Callable[[list[Any]], Any], optional
        Function to aggregate outputs, by default torch.cat.

    Returns
    -------
    torch.Tensor
        The processed outputs, padded and stacked along batch dimension.
    """
    res = aggregate_fn(outputs)
    seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
    unpacked = unpack_sequence(res, lens=seqlens, dim=0)
    reordered = reorder_list(unpacked, mb_list.backward_indices)
    return pad_and_stack_tensors_along_first_dim(reordered)
