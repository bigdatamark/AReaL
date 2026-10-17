# [MODULE DOC]
# DTensor ParallelStyles for MoE layers, adapted from torchtitan. Each style shards a
# GroupedExperts module's weights (w1/w3 [num_experts, dim, hidden], w2 [num_experts,
# hidden, dim]) and installs input/output hooks through distribute_module:
# ExpertParallel (EP, all-to-all token dispatch/combine), TensorParallel (TP-only experts),
# ExpertTensorParallel (2-D EP x TP), and ReordererSequenceParallel (splits router output
# across TP ranks when TP ranks are borrowed by EP). All hooks issue collectives.
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/expert_parallel.py

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel.style import ParallelStyle

from areal.experimental.models.archon.moe.utils import _permute, _unpermute


class BaseExpertParallel(ParallelStyle, ABC):
    # [CLASS DOC]
    # Abstract ParallelStyle declaring the EP hooks: _partition_fn, _token_dispatch,
    # _token_combine (subclasses also implement _apply).
    """Abstract base class for Expert Parallelism styles.

    Subclasses implement specific communication patterns for
    dispatching tokens to experts and combining results.

    Subclasses must implement:
    - _partition_fn: Shard expert weights across devices
    - _token_dispatch: Dispatch tokens to devices holding their assigned experts
    - _token_combine: Combine expert outputs back to original token locations
    - _apply: Apply parallelism using distribute_module with hooks
    """

    @abstractmethod
    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # [FUNCTION DOC]
        # Abstract: shard the module's parameters across device_mesh in place.
        """Partition expert weights across devices."""

    @abstractmethod
    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Abstract: forward pre-hook; given (routed_input, num_tokens_per_expert) return the
        # tokens/counts as seen by the local experts.
        """Dispatch tokens to devices holding their assigned experts."""

    @abstractmethod
    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Abstract: forward hook; return expert outputs restored to the original token order/rank.
        """Combine expert outputs back to original token locations."""


class ExpertParallel(BaseExpertParallel):
    # [CLASS DOC]
    # EP with ETP=1: experts sharded on dim 0 across a 1-D EP mesh, each rank holding
    # num_experts / ep_size experts. Dispatch state (splits, pre-permutation shape, permutation
    # indices) is stored on the instance between the pre-hook and the hook.
    # KNOWN ISSUE: Because the dispatch state lives on the style instance, overlapping forwards
    # of the same module (e.g. multiple in-flight microbatches) would overwrite it.
    """Expert Parallelism with ETP=1 (no tensor parallelism within experts).

    This class implements EP by:
    1. Sharding expert weights on dim 0 (expert dimension) across EP ranks
    2. Using all-to-all to dispatch tokens to devices holding their experts
    3. Using all-to-all to combine results back

    Each EP rank holds num_experts // ep_size local experts.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes the dispatch state fields to None.
        super().__init__()
        # State saved during dispatch for use in combine
        self.input_splits: list[int] | None = None
        self.output_splits: list[int] | None = None
        self.input_shape: tuple[int, ...] | None = None
        self.permuted_indices: torch.Tensor | None = None

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # [FUNCTION DOC]
        # Replaces every direct parameter of module with an nn.Parameter wrapping a DTensor
        # sharded on dim 0 over device_mesh. name is unused.
        """Shard expert weights on expert dimension (dim 0).

        Args:
            name: Module name (unused).
            module: GroupedExperts module to partition.
            device_mesh: 1D mesh for EP.
        """
        for param_name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            module.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Inputs: routed_input [tokens sorted by global expert, dim] and per-expert counts
        # [num_experts]. Without grad: all-to-all the counts so each rank learns how many tokens
        # it will receive for each of its local experts from each source rank (waited explicitly
        # for compile), then computes per-rank send splits (sum of counts for each destination's
        # experts) and receive splits, copying both to CPU lists (device sync). Then an autograd
        # all-to-all moves the tokens. Finally _permute regroups received tokens by local expert
        # (with padding/alignment for grouped GEMM) and saves shape and permutation indices.
        # Returns (permuted tokens, aligned per-local-expert token counts).
        """Dispatch tokens to EP ranks via all-to-all.

        Args:
            module: GroupedExperts module.
            inputs: Tuple of (routed_input, num_tokens_per_expert).
                routed_input: Shape (total_tokens, dim), tokens sorted by expert.
                num_tokens_per_expert: Shape (num_experts,), token counts per expert.
            device_mesh: 1D mesh for EP.

        Returns:
            Tuple of (dispatched_tokens, local_num_tokens_per_expert).
        """
        routed_input, num_tokens_per_expert = inputs
        group = device_mesh.get_group()
        ep_degree = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # Step 1: Exchange token counts via all-to-all
        with torch.no_grad():
            num_tokens_per_expert_received = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=group,
            )
            # Need to wait explicitly because it is used by downstream operations
            # which don't realize that AsyncCollectiveTensor needs unwrapping.
            # This is required for torch.compile compatibility.
            num_tokens_per_expert_received = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_received
            )

            # Compute splits for variable-size all-to-all
            # input_splits: how many tokens to send to each rank
            # output_splits: how many tokens to receive from each rank
            counts_view = num_tokens_per_expert.view(ep_degree, num_local_experts)
            received_view = num_tokens_per_expert_received.view(
                ep_degree, num_local_experts
            )

            # NOTE: tolist() requires sync to CPU
            self.input_splits = counts_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=True
            )
            self.output_splits = received_view.sum(dim=1).to(
                torch.device("cpu"), non_blocking=False
            )
            self.input_splits = self.input_splits.tolist()
            self.output_splits = self.output_splits.tolist()

        # Step 2: All-to-all to dispatch tokens
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            group,
        )

        # Step 3: Permute for grouped_mm alignment
        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            aligned_num_tokens,
        ) = _permute(
            routed_input,
            num_tokens_per_expert_received,
            ep_degree,
            num_local_experts,
        )

        return routed_input, aligned_num_tokens

    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Undoes the permutation with _unpermute using the saved shape and indices, then runs the
        # reverse autograd all-to-all (send/receive splits swapped). Returns tokens in the
        # original pre-dispatch order on this rank.
        """Combine expert outputs via all-to-all back to original locations.

        Args:
            module: GroupedExperts module.
            output: Expert computation output.
            device_mesh: 1D mesh for EP.

        Returns:
            Combined output in original token order.
        """
        group = device_mesh.get_group()

        # Step 1: Unpermute to pre-permutation order
        output = _unpermute(output, self.input_shape, self.permuted_indices)

        # Step 2: All-to-all to combine (reverse direction)
        output = all_to_all_single_autograd(
            output,
            self.input_splits,  # Swap for reverse
            self.output_splits,
            group,
        )

        return output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC]
        # Calls distribute_module with the partition fn and the dispatch/combine hooks; returns
        # the module.
        """Apply expert parallelism to the module.

        Uses distribute_module to:
        1. Shard expert weights via partition_fn
        2. Register _token_dispatch as forward pre-hook (input_fn)
        3. Register _token_combine as forward hook (output_fn)

        This way EP communication happens automatically during forward pass.

        Args:
            module: GroupedExperts module to parallelize.
            device_mesh: 1D mesh for EP.

        Returns:
            The parallelized module.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def apply_expert_parallel(
    experts_module: nn.Module,
    ep_mesh: DeviceMesh,
) -> None:
    # [FUNCTION DOC]
    # Convenience: creates a new ExpertParallel and applies it to experts_module over the 1-D
    # ep_mesh. Returns None (module modified in place).
    """Apply Expert Parallelism to a GroupedExperts module.

    This is a convenience function that applies EP with ETP=1 to the
    given experts module. It uses distribute_module to register hooks
    for automatic token dispatch/combine during forward pass.

    Args:
        experts_module: GroupedExperts module to parallelize.
        ep_mesh: 1D device mesh for EP.

    Example:
        >>> from areal.experimental.models.archon import apply_expert_parallel
        >>> apply_expert_parallel(moe.experts, ep_mesh)
    """
    ep_style = ExpertParallel()
    ep_style._apply(experts_module, ep_mesh)


class TensorParallel(ParallelStyle):
    # [CLASS DOC]
    # TP-only experts (EP disabled): w1/w3 (actual layout [num_experts, hidden, dim]) are
    # sharded on dim 1 (hidden, column-parallel) and w2 ([num_experts, dim, hidden]) on dim 2
    # (hidden, row-parallel); inputs are replicated with Partial gradient placement so input
    # gradients are summed across TP ranks.
    # KNOWN ISSUE: The docstring and inline comments state w1/w3 shape (num_experts, dim,
    # hidden_dim) and w2 (num_experts, hidden_dim, dim); GroupedExperts actually stores them
    # transposed, so the comments are wrong (the sharded dims are still the hidden dim). Same
    # for ExpertTensorParallel's comments.
    """Tensor Parallelism for experts when EP is disabled.

    This class implements TP-only parallelism for MoE experts by:
    1. Sharding w1/w3 on dim 1 (column-wise)
    2. Sharding w2 on dim 2 (row-wise)
    3. All-reducing outputs during backward

    Weight sharding:
    - w1: [num_experts, dim, hidden_dim] -> Shard on dim 1
    - w2: [num_experts, hidden_dim, dim] -> Shard on dim 2
    - w3: [num_experts, dim, hidden_dim] -> Shard on dim 1

    Used when: EP is disabled but TP is enabled for the model.
    """

    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Wraps routed_input as a Replicate DTensor and converts back to local with
        # Partial grad placement (backward all-reduces grads); counts pass through unchanged.
        """Prepare input for TP by marking it for Partial gradient accumulation.

        The input is replicated across TP ranks, but we need Partial placement
        so that gradients are summed during backward.
        """
        routed_input, num_tokens_per_expert = inputs
        # Create DTensor with Replicate placement for forward,
        # Partial for backward gradient accumulation
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))
        return routed_input, num_tokens_per_expert

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # [FUNCTION DOC]
        # Re-registers w1 and w3 sharded on dim 1 and w2 sharded on dim 2 as DTensor parameters.
        """Shard expert weights for tensor parallelism.

        w1/w3: Column-wise (Shard on dim 1)
        w2: Row-wise (Shard on dim 2)
        """
        # w1 shape = (num_experts, dim, hidden_dim) -> Column-wise sharding
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )
        # w2 shape = (num_experts, hidden_dim, dim) -> Row-wise sharding
        module.register_parameter(
            "w2", nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)]))
        )
        # w3 shape = (num_experts, dim, hidden_dim) -> Column-wise sharding
        module.register_parameter(
            "w3", nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)]))
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC]
        # distribute_module with the TP partition fn and input fn (no output hook; the model's
        # outer TP output handling reduces partial outputs).
        """Apply tensor parallelism to the experts module."""
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._prepare_input_fn,
        )


class ExpertTensorParallel(ExpertParallel):
    # [CLASS DOC]
    # EP x TP on a 2-D mesh named ["ep", "tp"]: weights sharded [Shard(0), Shard(1)] for
    # w1/w3 and [Shard(0), Shard(2)] for w2; token dispatch/combine reuse ExpertParallel on the
    # "ep" submesh.
    """Expert Parallelism with Tensor Parallelism (EP + ETP).

    This class implements combined EP + TP for experts by:
    1. Sharding expert weights on both expert dimension (dim 0) and TP dimension
       - w1/w3: [Shard(0), Shard(1)]
       - w2: [Shard(0), Shard(2)]
    2. Using all-to-all for token dispatch/combine (inherited from ExpertParallel)
    3. All-reducing outputs within TP groups during backward

    Weight sharding (2D):
    - w1: [num_experts, dim, hidden_dim] -> [Shard(0), Shard(1)]
    - w2: [num_experts, hidden_dim, dim] -> [Shard(0), Shard(2)]
    - w3: [num_experts, dim, hidden_dim] -> [Shard(0), Shard(1)]

    device_mesh is 2D with dimensions: ["ep", "tp"]

    Used when: Both EP and ETP are enabled (etp=tp).
    """

    def _token_dispatch(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Marks routed_input as replicated with Partial gradient over the "tp" submesh, then runs
        # the parent EP dispatch over the "ep" submesh.
        """Dispatch tokens with TP input preparation.

        First prepares input for TP (Replicate with Partial gradient),
        then performs EP all-to-all dispatch.
        """
        routed_input, num_tokens_per_expert = inputs

        # Prepare input for TP (replicated input, partial gradient)
        routed_input = DTensor.from_local(
            routed_input, device_mesh["tp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        # Call parent's token dispatch with EP mesh
        return super()._token_dispatch(
            module, (routed_input, num_tokens_per_expert), device_mesh["ep"]
        )

    def _partition_fn(
        self,
        name: str,
        module: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        # [FUNCTION DOC]
        # Re-registers w1/w2/w3 as 2-D sharded DTensor parameters (see class doc).
        """Shard expert weights on both EP and TP dimensions.

        Uses 2D sharding:
        - First dimension (Shard(0)): Expert parallel
        - Second dimension (Shard(1/2)): Tensor parallel
        """
        # w1 shape = (num_experts, dim, hidden_dim) -> [Shard(0), Shard(1)]
        module.register_parameter(
            "w1",
            nn.Parameter(
                distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(1)])
            ),
        )
        # w2 shape = (num_experts, hidden_dim, dim) -> [Shard(0), Shard(2)]
        module.register_parameter(
            "w2",
            nn.Parameter(
                distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(2)])
            ),
        )
        # w3 shape = (num_experts, dim, hidden_dim) -> [Shard(0), Shard(1)]
        module.register_parameter(
            "w3",
            nn.Parameter(
                distribute_tensor(module.w3, device_mesh, [Shard(0), Shard(1)])
            ),
        )

    def _token_combine(
        self,
        module: nn.Module,
        output: torch.Tensor,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Delegates to the parent combine on the "ep" submesh.
        """Combine expert outputs via all-to-all back to original locations."""
        return super()._token_combine(module, output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC]
        # distribute_module over the 2-D mesh with this class's partition/dispatch/combine.
        """Apply expert + tensor parallelism to the module.

        Args:
            module: GroupedExperts module to parallelize.
            device_mesh: 2D mesh with dimensions [ep, tp].

        Returns:
            The parallelized module.
        """
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


class ReordererSequenceParallel(ParallelStyle):
    # [CLASS DOC]
    # For EP > 1 with etp = 1: wraps the TokenReorderer so each TP rank reorders only its
    # contiguous 1/tp slice of tokens, avoiding duplicate tokens in the EP all-to-all.
    """Sequence Parallel for TokenReorderer when etp=1.

    When EP>1 and etp=1 (TP borrowed by EP), this parallelism:
    1. Splits tokens across TP ranks (each rank gets 1/tp of tokens)
    2. Adjusts token indices from local to global after reordering

    This ensures each TP rank processes different tokens, so the
    subsequent EP all-to-all doesn't send duplicate data.
    """

    def _prepare_input_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Given (top_scores, selected_indices) with tokens on dim 0, raises ValueError
        # unless the token count divides by TP size, and returns this rank's contiguous slice of
        # both.
        """Split tokens across TP ranks."""
        top_scores, selected_indices = inputs
        num_tokens = top_scores.shape[0]
        tp_size = device_mesh.size()
        tp_rank = device_mesh.get_local_rank()

        if num_tokens % tp_size != 0:
            raise ValueError(
                f"Number of tokens ({num_tokens}) must be divisible by "
                f"TP size ({tp_size}) for ReordererSequenceParallel"
            )

        local_num_tokens = num_tokens // tp_size
        offset = tp_rank * local_num_tokens

        return (
            top_scores[offset : offset + local_num_tokens],
            selected_indices[offset : offset + local_num_tokens],
        )

    def _prepare_output_fn(
        self,
        module: nn.Module,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device_mesh: DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Shifts sorted token indices by (local token-slot count, i.e. local tokens times
        # top_k) times tp rank, so they index the global flattened token-slot space; scores and
        # per-expert counts pass through.
        """Adjust token indices from local to global.

        As we shard routed tokens along bs*slen dim across the TP ranks,
        the MoE gather and scatter still require global token indices.
        """
        top_scores_sorted, token_indices_sorted, num_tokens_per_expert = outputs
        tp_rank = device_mesh.get_local_rank()

        # token_indices_sorted are indices into local (num_tokens/tp * top_k) space
        # We need to adjust to global indices by adding the offset
        # top_scores_sorted.shape[0] == local_num_tokens * top_k
        token_indices_global = (
            token_indices_sorted + top_scores_sorted.shape[0] * tp_rank
        )

        return top_scores_sorted, token_indices_global, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        # [FUNCTION DOC]
        # distribute_module with only the input/output hooks (no parameter partitioning).
        """Apply sequence parallel to TokenReorderer."""
        return distribute_module(
            module,
            device_mesh,
            input_fn=self._prepare_input_fn,
            output_fn=self._prepare_output_fn,
        )


__all__ = [
    "ExpertParallel",
    "ExpertTensorParallel",
    "TensorParallel",
    "ReordererSequenceParallel",
    "apply_expert_parallel",
]
