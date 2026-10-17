# [MODULE DOC]
# Shared pytest configuration and helpers for the Archon FP8 test suite. Marks every
# collected FP8 test as slow and provides a factory for random MoE expert weights.
import pytest
import torch


def pytest_collection_modifyitems(items):
    # [FUNCTION DOC]
    # Pytest hook. Iterates over all collected items and adds the "slow" marker to any
    # item whose file path contains "/fp8/". Mutates the items in place; returns
    # nothing.
    for item in items:
        if "/fp8/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


def make_expert_weights(
    num_experts: int,
    dim: int,
    hidden_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds random grouped-MoE expert weights. Inputs: num_experts, dim (model
    # dim), hidden_dim (FFN dim), device (default "cuda"), dtype (default bfloat16).
    # Returns (w1, w2, w3): w1 and w3 have shape [num_experts, hidden_dim, dim] scaled
    # by 1/sqrt(dim); w2 has shape [num_experts, dim, hidden_dim] scaled by
    # 1/sqrt(hidden_dim). Allocates on the given device (GPU by default).
    """Create MoE expert weights with Xavier-like 1/sqrt(fan_in) scale."""
    scale_in = dim**0.5
    scale_hidden = hidden_dim**0.5
    w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) / scale_in
    )
    w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype)
        / scale_hidden
    )
    w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) / scale_in
    )
    return w1, w2, w3
