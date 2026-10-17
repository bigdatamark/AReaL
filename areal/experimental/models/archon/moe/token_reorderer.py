# [MODULE DOC]
# TokenReorderer: sorts the flattened (token, top-k slot) routing assignments by expert.
# Kept as its own module so ReordererSequenceParallel can shard its inputs across TP ranks.
# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/moe/moe.py

from __future__ import annotations

import torch
from torch import nn


class TokenReorderer(nn.Module):
    # [CLASS DOC]
    # Parameter-free module storing num_experts and top_k.
    """Token reordering module for MoE routing.

    Separates the token reordering logic from the router to enable
    sequence parallel sharding when etp=1 (TP ranks borrowed by EP).

    When ReordererSequenceParallel is applied (etp=1 case):
    - Input is split across TP ranks (each rank gets 1/tp of tokens)
    - Output token_indices are adjusted to global indices

    Args:
        num_experts: Total number of experts.
        top_k: Number of experts each token is routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        # [FUNCTION DOC]
        # Stores num_experts and top_k.
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Counts slots per expert via a histogram of the flattened expert indices, computes a
        # stable argsort of the flattened indices (slot indices grouped by expert, original order
        # kept within an expert), and gathers flattened scores in that order. Returns (sorted
        # scores [T*k], sorted slot indices [T*k] - slot // top_k is the token, counts
        # [num_experts]).
        """Reorder tokens by expert assignment.

        Args:
            top_scores: Shape (num_tokens, top_k), routing scores.
            selected_experts_indices: Shape (num_tokens, top_k), expert indices.

        Returns:
            tuple of:
                - top_scores_experts_sorted: Shape (num_tokens * top_k,), scores sorted by expert
                - token_indices_experts_sorted: Shape (num_tokens * top_k,), sorting indices
                  (divide by top_k to get original token index)
                - num_tokens_per_expert: Shape (num_experts,)
        """
        # Count tokens per expert
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (num_tokens * top_k,)
        # These are indices into the flattened (num_tokens * top_k,) dimension
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


__all__ = ["TokenReorderer"]
