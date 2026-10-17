# [MODULE DOC]
# Single-GPU HF parity test for the Archon Qwen2 dense model with per-op diagnostics.
# Loads a truncated (first PARTIAL_LAYERS=4 layers) HF model with flash_attention_2 and
# an Archon model built from the same config and weights, captures intermediate
# activations with forward hooks (embedding, attention norm, attention, FFN norm, FFN,
# layer output, final norm), prints per-op diff and cosine statistics, and asserts final
# logits parity. Module skipped without CUDA.
from __future__ import annotations

import gc
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from tests.experimental.archon.utils import (
    DENSE_MODEL_PATHS,
    compare_outputs,
    setup_environment,
)

from areal.experimental.models.archon import get_model_spec, is_supported_model
from areal.infra.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

PARTIAL_LAYERS = 4
SEQ_LEN = 32


def _capture_hf(
    hf_model: Any, n_layers: int
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Registers forward hooks on an HF causal LM (embed_tokens, and for each of
    # the first n_layers: input_layernorm, self_attn, post_attention_layernorm, mlp, the
    # layer itself; then the final norm). Each hook stores a detached clone of the
    # output (or of the first element if the output is a tuple whose first item is a
    # tensor) into a dict under keys like "emb", "layer{i}_attn", "final_norm". Returns
    # (outputs dict filled during forward, list of hook handles to remove later).
    outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _mk(name: str):
        # [FUNCTION DOC]
        # HELPER: Returns a forward hook closure that records outputs under the given
        # name.
        def _hook(module: torch.nn.Module, args: tuple[Any, ...], out: Any) -> None:
            # [FUNCTION DOC]
            # HELPER: Forward hook body: stores a detached clone of a tensor output, or
            # of out[0] for non-empty tuple outputs; ignores other output types.
            if isinstance(out, torch.Tensor):
                outputs[name] = out.detach().clone()
            elif (
                isinstance(out, tuple)
                and len(out) > 0
                and isinstance(out[0], torch.Tensor)
            ):
                outputs[name] = out[0].detach().clone()

        return _hook

    handles.append(hf_model.model.embed_tokens.register_forward_hook(_mk("emb")))
    for i in range(n_layers):
        layer = hf_model.model.layers[i]
        handles.append(
            layer.input_layernorm.register_forward_hook(_mk(f"layer{i}_attn_norm"))
        )
        handles.append(layer.self_attn.register_forward_hook(_mk(f"layer{i}_attn")))
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _mk(f"layer{i}_ffn_norm")
            )
        )
        handles.append(layer.mlp.register_forward_hook(_mk(f"layer{i}_ffn")))
        handles.append(layer.register_forward_hook(_mk(f"layer{i}_out")))
    handles.append(hf_model.model.norm.register_forward_hook(_mk("final_norm")))
    return outputs, handles


def _capture_archon(
    archon_model: Any, n_layers: int
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    # [FUNCTION DOC]
    # HELPER: Same as _capture_hf but for the Archon model layout: tok_embeddings,
    # layers keyed by string index with attention_norm, attention, ffn_norm,
    # feed_forward (only if not None) and the layer, plus the final norm. Uses identical
    # key names so outputs can be paired. Returns (outputs dict, handles).
    outputs: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _mk(name: str):
        # [FUNCTION DOC]
        # HELPER: Returns a forward hook closure that records outputs under the given
        # name.
        def _hook(module: torch.nn.Module, args: tuple[Any, ...], out: Any) -> None:
            # [FUNCTION DOC]
            # HELPER: Forward hook body identical to the HF version.
            if isinstance(out, torch.Tensor):
                outputs[name] = out.detach().clone()
            elif (
                isinstance(out, tuple)
                and len(out) > 0
                and isinstance(out[0], torch.Tensor)
            ):
                outputs[name] = out[0].detach().clone()

        return _hook

    handles.append(archon_model.tok_embeddings.register_forward_hook(_mk("emb")))
    for i in range(n_layers):
        layer = archon_model.layers[str(i)]
        handles.append(
            layer.attention_norm.register_forward_hook(_mk(f"layer{i}_attn_norm"))
        )
        handles.append(layer.attention.register_forward_hook(_mk(f"layer{i}_attn")))
        handles.append(layer.ffn_norm.register_forward_hook(_mk(f"layer{i}_ffn_norm")))
        if layer.feed_forward is not None:
            handles.append(
                layer.feed_forward.register_forward_hook(_mk(f"layer{i}_ffn"))
            )
        handles.append(layer.register_forward_hook(_mk(f"layer{i}_out")))
    handles.append(archon_model.norm.register_forward_hook(_mk("final_norm")))
    return outputs, handles


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # [FUNCTION DOC]
    # HELPER: Flattens both tensors to float, truncates to the shorter length, and
    # returns their cosine similarity as a Python float (1.0 if either is empty). GPU
    # sync via item.
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    n = min(a_f.numel(), b_f.numel())
    if n == 0:
        return 1.0
    a_f = a_f[:n]
    b_f = b_f[:n]
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


@pytest.mark.slow
def test_hf_parity_qwen2_with_op_diagnostics() -> None:
    # [FUNCTION DOC]
    # Slow. Calls setup_environment, loads the config from DENSE_MODEL_PATHS["qwen2"]
    # and a copy truncated to min(4, num_hidden_layers) layers, builds the HF model in
    # bfloat16 on device, skips if the model type is unsupported by Archon, constructs
    # the Archon model on device from the model spec's args, converts the HF state dict
    # (moved to CPU) through the spec's state-dict adapter and loads it non-strictly,
    # casts to bfloat16. Installs hooks, runs both models on a seeded random sequence of
    # length 32 (single cu_seqlens segment), removes hooks, prints per-key max/mean diff
    # and cosine (flattening and truncating when shapes differ) and the layer-output
    # cosine trend. Asserts logits max diff below 6.0, mean diff below 0.5 and cosine
    # above 0.95. Frees models, empties the CUDA cache and destroys the process group if
    # initialized.
    # KNOWN ISSUE: The Archon state dict is loaded with strict=False, so missing or
    # unexpected keys from the adapter would go unnoticed (other than via numeric
    # mismatch). The support check happens only after the HF model is already loaded.
    setup_environment()
    model_path = DENSE_MODEL_PATHS["qwen2"]
    device = torch.device(current_platform.device_type)
    dtype = torch.bfloat16

    # Use partial config to avoid bf16 error accumulation across all layers
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    partial_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    partial_config.num_hidden_layers = min(
        PARTIAL_LAYERS, int(config.num_hidden_layers)
    )
    n_layers = int(partial_config.num_hidden_layers)

    hf_model_any: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=partial_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    hf_model = hf_model_any.to(device).eval()

    model_type = partial_config.model_type
    if not is_supported_model(model_type):
        pytest.skip(f"Model type {model_type} not supported by Archon")
    spec = get_model_spec(model_type)
    model_args = spec.model_args_class.from_hf_config(partial_config, is_critic=False)

    with torch.device(device):
        archon_model: Any = spec.model_class(model_args)

    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    adapter = spec.state_dict_adapter_class(partial_config)
    archon_state_dict = adapter.from_hf(hf_state_dict)
    del hf_state_dict
    archon_model.load_state_dict(archon_state_dict, strict=False)
    del archon_state_dict
    archon_model = archon_model.to(dtype).eval()
    hf_out, hf_handles = _capture_hf(hf_model, n_layers)
    ar_out, ar_handles = _capture_archon(archon_model, n_layers)

    torch.manual_seed(42)
    seq_len = SEQ_LEN
    input_ids = torch.randint(
        100, int(config.vocab_size) - 100, (1, seq_len), device=device
    )
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        archon_logits = archon_model(
            input_ids,
            positions=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=seq_len,
        )

    for h in hf_handles + ar_handles:
        h.remove()

    keys = [k for k in sorted(hf_out.keys()) if k in ar_out]
    print("\n[QWEN2 PER-OP]")
    out_cos_trend: list[tuple[str, float]] = []
    for k in keys:
        hf_t = hf_out[k].float()
        ar_t = ar_out[k].float()
        if hf_t.shape != ar_t.shape:
            hf_t = hf_t.reshape(-1)
            ar_t = ar_t.reshape(-1)
            n = min(hf_t.numel(), ar_t.numel())
            hf_t = hf_t[:n]
            ar_t = ar_t[:n]
        diff = (hf_t - ar_t).abs()
        cos = _cos_sim(hf_t, ar_t)
        print(
            f"{k:24s} max={diff.max().item():.6f} mean={diff.mean().item():.6f} cos={cos:.6f}"
        )
        if k.startswith("layer") and k.endswith("_out"):
            out_cos_trend.append((k, cos))

    if out_cos_trend:
        print("\n[QWEN2 COS SIM TREND: layer*_out]")
        prev: float | None = None
        for name, cos in out_cos_trend:
            if prev is None:
                print(f"{name:24s} cos={cos:.6f} delta=---")
            else:
                print(f"{name:24s} cos={cos:.6f} delta={cos - prev:+.6f}")
            prev = cos

    logits_cmp = compare_outputs(archon_logits, hf_logits, "qwen2_logits")
    assert logits_cmp["max_diff"] < 6.0
    assert logits_cmp["mean_diff"] < 0.5

    logits_cos = _cos_sim(hf_logits, archon_logits)
    print(f"\n[QWEN2 LOGITS COS SIM] {logits_cos:.6f}")
    assert logits_cos > 0.95

    del hf_model, archon_model
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
