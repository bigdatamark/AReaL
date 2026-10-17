# [MODULE DOC]
# Ulysses sequence-parallel (context-parallel) helpers for Archon models. Re-exports
# the all-to-all primitives gather_seq_scatter_heads / gather_heads_scatter_seq from the
# FSDP Ulysses module and adds input slicing and output gathering across the CP group.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_F
from torch import Tensor
from torch.distributed import ProcessGroup

from areal.models.fsdp.ulysses import (
    _gather_heads_scatter_seq as gather_heads_scatter_seq,
)
from areal.models.fsdp.ulysses import (
    _gather_seq_scatter_heads as gather_seq_scatter_heads,
)

__all__ = [
    "ulysses_slice_inputs",
    "gather_seq_scatter_heads",
    "gather_heads_scatter_seq",
    "ulysses_gather_output",
]


def _ulysses_slice_tensor(
    tensor: Tensor,
    cp_rank: int,
    cp_size: int,
) -> Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns the contiguous cp_rank-th of cp_size equal chunks of tensor along
    # its last dim. Raises ValueError if the last-dim length is not divisible by cp_size
    # (inputs must be padded first).
    """Slice a tensor along the last dimension for Ulysses SP."""
    total_len = tensor.shape[-1]

    if total_len % cp_size != 0:
        raise ValueError(
            f"Tensor length {total_len} not aligned to cp_size {cp_size}. "
            "Ensure pad_mb_list is called with batch_align_to=seq_len_divisor first."
        )

    chunk_size = total_len // cp_size
    start = cp_rank * chunk_size
    end = start + chunk_size

    return tensor[..., start:end].contiguous()


def ulysses_slice_inputs(
    inputs: dict[str, Any],
    labels: Tensor,
    cp_rank: int,
    cp_size: int,
) -> tuple[dict[str, Any], Tensor]:
    # [FUNCTION DOC]
    # Shards packed inputs for Ulysses SP. If cp_size <= 1 returns inputs and labels
    # unchanged; otherwise returns a shallow copy of inputs with input_ids and
    # position_ids
    # sliced to this rank's chunk (last dim), plus the sliced labels. Other entries
    # (e.g.
    # cu_seqlens) are left as-is. Raises ValueError on misaligned lengths; KeyError if
    # input_ids/position_ids are missing.
    """Slice inputs and labels for Ulysses SP."""
    if cp_size <= 1:
        return inputs, labels

    inputs = dict(inputs)
    inputs["input_ids"] = _ulysses_slice_tensor(inputs["input_ids"], cp_rank, cp_size)
    labels = _ulysses_slice_tensor(labels, cp_rank, cp_size)

    inputs["position_ids"] = _ulysses_slice_tensor(
        inputs["position_ids"], cp_rank, cp_size
    )

    return inputs, labels


def ulysses_gather_output(
    output: Tensor,
    cp_group: ProcessGroup | None,
    seq_dim: int = 0,
) -> Tensor:
    # [FUNCTION DOC]
    # All-gathers output from every rank of cp_group (autograd-aware
    # torch.distributed.nn all_gather) and concatenates along seq_dim (default 0).
    # Returns
    # output unchanged when cp_group is None or has size <= 1. Collective communication.
    """Gather output tensor from all CP ranks after forward pass."""
    if cp_group is None:
        return output

    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return output

    gathered = dist_F.all_gather(output, group=cp_group)
    return torch.cat(gathered, dim=seq_dim)
