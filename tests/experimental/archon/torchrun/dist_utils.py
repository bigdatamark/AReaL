# [MODULE DOC]
# HELPER: Shared helpers for torchrun-launched Archon distributed tests (EP/TP/CP/PP):
# result-file writing, tiny Qwen3 dense/MoE model-arg factories, a seeded golden (non-
# parallel) model builder, packed test-input creation, DTensor-aware state gathering,
# and output/gradient validation.
"""Shared utilities for distributed EP tests."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.experimental.models.archon.moe import MoEArgs
from areal.experimental.models.archon.qwen3 import Qwen3Model, Qwen3ModelArgs


def write_result(out: str, succ: bool, error: str = "") -> None:
    # [FUNCTION DOC]
    # HELPER: Writes the test outcome to file path out (overwriting): "Passed" if succ,
    # else "Failed" followed by a newline and error when error is non-empty. Returns
    # None.
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed")
            if error:
                f.write(f"\n{error}")


def create_moe_model_args(
    num_experts: int = 4,
    dim: int = 64,
    hidden_dim: int = 128,
    moe_inter_dim: int = 128,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
    n_layers: int = 2,
    vocab_size: int = 1000,
    max_seq_len: int = 8192,
    top_k: int = 2,
) -> Qwen3ModelArgs:
    # [FUNCTION DOC]
    # HELPER: Returns Qwen3ModelArgs for a tiny MoE model (SDPA attention, every layer
    # MoE, grouped mm disabled) from the given sizes: num_experts, dim, hidden_dim,
    # moe_inter_dim, heads, KV heads, head_dim, n_layers, vocab_size, max_seq_len and
    # router top_k.
    return Qwen3ModelArgs(
        dim=dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        attn_type="sdpa",
        moe_enabled=True,
        moe_inter_dim=moe_inter_dim,
        moe_args=MoEArgs(
            num_experts=num_experts,
            top_k=top_k,
            use_grouped_mm=False,
        ),
        decoder_sparse_step=1,
    )


def gather_full_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds a dict of every named parameter and buffer of model as detached
    # clones; DTensors are materialized with full_tensor() (a collective, so all ranks
    # must call it). Buffers with the same name as a parameter would overwrite it.
    """Gather all parameters (including DTensors) to full tensors."""
    full_state = {}
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            full_state[name] = param.full_tensor().detach().clone()
        else:
            full_state[name] = param.detach().clone()

    for name, buffer in model.named_buffers():
        if isinstance(buffer, DTensor):
            full_state[name] = buffer.full_tensor().detach().clone()
        else:
            full_state[name] = buffer.detach().clone()

    return full_state


def create_test_input(
    num_seqs: int = 4,
    seq_len_per_seq: int = 8,
    vocab_size: int = 1000,
    device: torch.device | str = "cuda",
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # [FUNCTION DOC]
    # HELPER: Seeds the global torch RNG with seed, then returns a packed batch: tokens
    # [1, num_seqs*seq_len_per_seq] random ids below vocab_size, positions 0..total-1 as
    # [1, total], int32 cu_seqlens of equal-length boundaries, and seq_len_per_seq (used
    # as max_seqlen). Tensors are on device.
    # KNOWN ISSUE: positions run continuously across sequences rather than restarting at
    # 0 per packed sequence.
    """Return (tokens, positions, cu_seqlens, seq_len_per_seq)."""
    torch.manual_seed(seed)
    total_len = num_seqs * seq_len_per_seq

    tokens = torch.randint(0, vocab_size, (1, total_len), device=device)
    positions = torch.arange(total_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [i * seq_len_per_seq for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=device,
    )

    return tokens, positions, cu_seqlens, seq_len_per_seq


def create_golden_model(
    model_args: Qwen3ModelArgs,
    device: torch.device | str,
    seed: int = 42,
) -> Qwen3Model:
    # [FUNCTION DOC]
    # HELPER: Seeds the RNG, builds a Qwen3Model from model_args, initializes weights
    # and buffers on device, moves it to device and returns it (reference for parallel
    # runs).
    """Create non-parallelized model for comparison."""
    torch.manual_seed(seed)
    model = Qwen3Model(model_args)
    model.init_weights()
    model.init_buffers(buffer_device=device)
    model = model.to(device)
    return model


def verify_outputs_match(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    max_diff_threshold: float = 1e-2,
) -> tuple[bool, float, float]:
    # [FUNCTION DOC]
    # HELPER: Computes max and mean absolute difference of two tensors (GPU sync via
    # item()) and returns (passed, max_diff, mean_diff) where passed is allclose(rtol,
    # atol) OR max_diff <= max_diff_threshold (default 1e-2, a looser fallback).
    """Return (success, max_diff, mean_diff)."""
    max_diff = (output1 - output2).abs().max().item()
    mean_diff = (output1 - output2).abs().mean().item()

    allclose = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    within_threshold = max_diff <= max_diff_threshold

    return (allclose or within_threshold), max_diff, mean_diff


def print_rank0(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints msg only on global rank 0; requires an initialized process group.
    if dist.get_rank() == 0:
        print(msg)


def create_dense_model_args(
    dim: int = 64,
    hidden_dim: int = 128,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 16,
    n_layers: int = 2,
    vocab_size: int = 1000,
    max_seq_len: int = 8192,
) -> Qwen3ModelArgs:
    # [FUNCTION DOC]
    # HELPER: Returns Qwen3ModelArgs for a tiny dense (MoE off, SDPA) model with the
    # given sizes.
    return Qwen3ModelArgs(
        dim=dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        attn_type="sdpa",
        moe_enabled=False,
    )


def validate_gradients(model: Qwen3Model) -> tuple[bool, list[str]]:
    # [FUNCTION DOC]
    # HELPER: Checks after backward that key parameters have .grad set: token
    # embeddings, each layer's attention.wq, MoE router gate and experts.w1 (or dense
    # feed_forward.w1), and the output projection if present. Returns (no_errors, list
    # of messages).
    """Return (success, error_messages)."""
    errors = []

    if model.tok_embeddings.weight.grad is None:
        errors.append("tok_embeddings has no gradient")

    for layer_id, layer in model.layers.items():
        if layer.attention.wq.weight.grad is None:
            errors.append(f"Layer {layer_id} attention.wq has no gradient")

        if layer.moe is not None:
            if layer.moe.router.gate.weight.grad is None:
                errors.append(f"Layer {layer_id} MoE router has no gradient")
            if layer.moe.experts.w1.grad is None:
                errors.append(f"Layer {layer_id} MoE experts.w1 has no gradient")
        elif layer.feed_forward is not None:
            if layer.feed_forward.w1.weight.grad is None:
                errors.append(f"Layer {layer_id} FFN has no gradient")

    if model.output is not None and model.output.weight.grad is None:
        errors.append("output projection has no gradient")

    return len(errors) == 0, errors


def validate_no_nan(output: torch.Tensor) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns False if output contains any NaN or Inf, else True (causes GPU
    # sync).
    """Return True if output has no NaN/Inf."""
    if torch.isnan(output).any():
        return False
    if torch.isinf(output).any():
        return False
    return True
