# [MODULE DOC]
# Parameter-balanced pipeline-parallel layer partitioning for Megatron. Estimates per-
# decoder-layer parameter counts (dense and MoE) plus embedding/output head sizes from
# HF and Megatron configs, finds the split of contiguous layers over pp*vpp stages that
# minimizes the maximum per-stage parameter load (dynamic programming), and writes it
# into the TransformerConfig as an explicit PipelineParallelLayerLayout. Logs via
# "MCoreParallel".
# SPDX-License-Identifier: Apache-2.0

import math

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import (
    PipelineParallelLayerLayout,
)
from transformers import PretrainedConfig

from areal.api import MegatronParallelStrategy
from areal.utils import logging

logger = logging.getLogger("MCoreParallel")


def configure_pipeline_layer_splits(
    parallel_strategy: MegatronParallelStrategy,
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
) -> TransformerConfig:
    # [FUNCTION DOC]
    # Configure a custom pipeline layer layout on tf_config (mutated in place and
    # returned). Inputs: parallel_strategy (pipeline_parallel_size,
    # virtual_pipeline_parallel_size), hf_config (HF PretrainedConfig) and tf_config
    # (Megatron TransformerConfig). Returns tf_config untouched when pp <= 1, num_layers
    # is invalid, parameter estimates are empty/non-positive, or no partition is found
    # (the last case logs a warning). Otherwise computes stage lengths for pp*vpp
    # stages, builds a layout where stage 0 starts with "embedding", each stage has its
    # "decoder" entries and the last stage ends with "loss", sets
    # pipeline_model_parallel_layout, clears num_layers_in_first/last_pipeline_stage and
    # disables account_for_embedding/loss_in_pipeline_split when those attributes exist,
    # and logs per-stage decoder counts and estimated parameter loads (in millions).
    # KNOWN ISSUE: if the estimated layer list is shorter than num_layers, the layout is
    # built for the truncated count and will not cover all model layers.
    pp_size = parallel_strategy.pipeline_parallel_size
    vpp_size = parallel_strategy.virtual_pipeline_parallel_size

    if pp_size <= 1:
        return tf_config

    total_stages = pp_size * vpp_size
    total_layers = getattr(tf_config, "num_layers", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        return tf_config

    layer_param_weights, embedding_params, output_params = (
        estimate_stage_parameter_buckets(hf_config, tf_config)
    )
    if not layer_param_weights or all(weight <= 0 for weight in layer_param_weights):
        return tf_config

    if len(layer_param_weights) != total_layers:
        total_layers = min(total_layers, len(layer_param_weights))
        layer_param_weights = layer_param_weights[:total_layers]

    stage_lengths = _compute_stage_layer_lengths(
        layer_param_weights,
        embedding_params,
        output_params,
        total_stages,
    )
    if not stage_lengths:
        logger.warning(
            "Falling back to default pipeline layout; unable to find valid stage partition."
        )
        return tf_config

    layout: list[list[str]] = []
    for idx, length in enumerate(stage_lengths):
        stage_layers: list[str] = []
        if idx == 0:
            stage_layers.append("embedding")
        stage_layers.extend(["decoder"] * length)
        if idx == total_stages - 1:
            stage_layers.append("loss")
        layout.append(stage_layers)
    layout = PipelineParallelLayerLayout(
        layout=layout,
        pipeline_model_parallel_size=pp_size,
    )

    setattr(tf_config, "pipeline_model_parallel_layout", layout)
    if hasattr(tf_config, "num_layers_in_first_pipeline_stage"):
        setattr(tf_config, "num_layers_in_first_pipeline_stage", None)
    if hasattr(tf_config, "num_layers_in_last_pipeline_stage"):
        setattr(tf_config, "num_layers_in_last_pipeline_stage", None)
    if hasattr(tf_config, "account_for_embedding_in_pipeline_split"):
        setattr(tf_config, "account_for_embedding_in_pipeline_split", False)
    if hasattr(tf_config, "account_for_loss_in_pipeline_split"):
        setattr(tf_config, "account_for_loss_in_pipeline_split", False)

    stage_loads: list[float] = []
    cursor = 0
    for idx, length in enumerate(stage_lengths):
        load = 0.0
        if idx == 0:
            load += embedding_params
        stage_end = min(cursor + length, len(layer_param_weights))
        if stage_end > cursor:
            load += sum(layer_param_weights[cursor:stage_end])
        cursor = stage_end
        if idx == total_stages - 1:
            load += output_params
        stage_loads.append(load)

    logger.info(
        "Configured pipeline layout (per-stage decoder counts / params): %s / %s (pp=%s, vpp=%s)",
        stage_lengths,
        [f"{value / 1e6:.2f}M" for value in stage_loads],
        pp_size,
        vpp_size,
    )
    return tf_config


def _compute_stage_layer_lengths(
    layer_param_weights: list[float],
    embedding_params: float,
    output_params: float,
    pp_size: int,
) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Min-max partition of a sequence of layer weights into pp_size contiguous
    # stages. Stage 1 additionally carries embedding_params and the last stage
    # output_params; middle stages must be non-empty while first/last may hold zero
    # decoder layers. Uses an O(stages * layers^2) DP over prefix sums tracking the best
    # previous split, then backtracks. Returns a list of per-stage layer counts summing
    # to the number of layers, or [] if no layers, pp_size <= 0 or no feasible
    # partition. With pp_size 1 the single stage gets embedding_params only
    # (output_params ignored).
    total_layers = len(layer_param_weights)
    if total_layers == 0 or pp_size <= 0:
        return []

    prefix_sums = [0.0]
    for value in layer_param_weights:
        prefix_sums.append(prefix_sums[-1] + float(value))

    def segment_sum(start: int, end: int) -> float:
        # [FUNCTION DOC]
        # HELPER: Sum of layer weights in [start, end) via prefix sums.
        return prefix_sums[end] - prefix_sums[start]

    stages = pp_size
    dp = [[math.inf] * (total_layers + 1) for _ in range(stages + 1)]
    choice = [[-1] * (total_layers + 1) for _ in range(stages + 1)]
    dp[0][0] = 0.0

    for stage in range(1, stages + 1):
        base = 0.0
        if stage == 1:
            base = embedding_params
        elif stage == stages:
            base = output_params

        for end in range(0, total_layers + 1):
            for start in range(0, end + 1):
                prev = dp[stage - 1][start]
                if not math.isfinite(prev):
                    continue
                if stage not in (1, stages) and start == end:
                    continue
                load = base + segment_sum(start, end)
                candidate = max(prev, load)
                if candidate < dp[stage][end]:
                    dp[stage][end] = candidate
                    choice[stage][end] = start

    if not math.isfinite(dp[stages][total_layers]):
        return []

    lengths = [0] * stages
    idx = total_layers
    for stage in range(stages, 0, -1):
        split = choice[stage][idx]
        if split < 0:
            return []
        lengths[stage - 1] = idx - split
        idx = split

    return lengths


def estimate_stage_parameter_buckets(
    hf_conf: PretrainedConfig, tf_conf: TransformerConfig
) -> tuple[list[float], float, float]:
    # [FUNCTION DOC]
    # Estimate parameter counts. Determines layer count (tf num_layers, else hf
    # num_hidden_layers) and hidden size (hf, else tf); returns ([], 0.0, 0.0) if either
    # is missing/invalid. Reads FFN size, heads, KV heads, head_dim (hf head_dim, else
    # tf kv_channels, else hidden/heads) and bias/GLU flags with HF-then-Megatron
    # fallbacks. Each dense layer = Q/K/V/O projections + optional biases + two norm
    # vectors + MLP (up, optional gate, down, optional biases). Embedding = vocab*hidden
    # (+vocab if embedding_bias); output head = vocab*hidden, or equal to embedding when
    # tie_word_embeddings. For MoE (tf num_moe_experts), layers selected by
    # moe_layer_freq (int step -> every step-th layer starting at index step-1;
    # list/tuple -> truthy entries, asserted length = layers) use attention + router
    # (+expert bias) + local experts (experts / EP size, ceil) * expert MLP + optional
    # shared expert MLP. Returns (per-layer weights list, embedding params, output
    # params) as floats.
    # KNOWN ISSUE: the output projection is counted as hidden*hidden rather than
    # (heads*head_dim)*hidden; an int moe_layer_freq selects layers step-1, 2*step-1,
    # ... whereas Megatron's pattern marks layers with index divisible by the step (0,
    # step, ...); negative freq values are treated by absolute value. These only affect
    # load-balance estimates.
    total_layers = getattr(tf_conf, "num_layers", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        total_layers = getattr(hf_conf, "num_hidden_layers", 0)
    if not isinstance(total_layers, int) or total_layers <= 0:
        return ([], 0.0, 0.0)

    hidden_size = getattr(hf_conf, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(tf_conf, "hidden_size", None)
    if hidden_size is None or hidden_size <= 0:
        return ([], 0.0, 0.0)

    intermediate_size = getattr(
        hf_conf,
        "intermediate_size",
        getattr(tf_conf, "ffn_hidden_size", hidden_size * 4),
    )
    num_heads = getattr(
        hf_conf, "num_attention_heads", getattr(tf_conf, "num_attention_heads", 1)
    )
    num_kv_heads = getattr(
        hf_conf,
        "num_key_value_heads",
        getattr(tf_conf, "num_query_groups", num_heads),
    )
    head_dim = getattr(hf_conf, "head_dim", None)
    if head_dim is None:
        kv_channels = getattr(tf_conf, "kv_channels", None)
        if kv_channels:
            head_dim = kv_channels
        else:
            head_dim = hidden_size // max(num_heads, 1)

    attention_bias = bool(
        getattr(hf_conf, "attention_bias", False)
        or getattr(tf_conf, "add_bias_linear", False)
    )
    add_bias_linear = bool(getattr(tf_conf, "add_bias_linear", False))
    gated_linear_unit = bool(getattr(tf_conf, "gated_linear_unit", False))

    def mlp_params(intermediate: int | None) -> float:
        # [FUNCTION DOC]
        # HELPER: Parameter count of one MLP of the given intermediate size (0 if
        # falsy/non-positive): up and down projections, a gate projection when
        # gated_linear_unit, and biases when add_bias_linear.
        if not intermediate or intermediate <= 0:
            return 0.0
        proj = hidden_size * intermediate
        gate = hidden_size * intermediate if gated_linear_unit else 0.0
        down = intermediate * hidden_size
        bias = 0.0
        if add_bias_linear:
            bias = (2 if gated_linear_unit else 1) * intermediate + hidden_size
        return float(proj + gate + down + bias)

    q_params = hidden_size * (num_heads * head_dim)
    kv_proj = num_kv_heads * head_dim
    k_params = hidden_size * kv_proj
    v_params = hidden_size * kv_proj
    o_params = hidden_size * hidden_size
    attn_bias_params = 0.0
    if attention_bias:
        attn_bias_params = (num_heads * head_dim) + (2 * kv_proj) + hidden_size

    attn_and_norm = (
        q_params
        + k_params
        + v_params
        + o_params
        + attn_bias_params
        + (2.0 * hidden_size)
    )
    dense_mlp = mlp_params(int(intermediate_size))
    dense_layer_params = float(attn_and_norm + dense_mlp)

    vocab_size = getattr(hf_conf, "vocab_size", 0)
    embedding_params = float(vocab_size * hidden_size)
    if getattr(hf_conf, "embedding_bias", False):
        embedding_params += float(vocab_size)

    output_params = float(vocab_size * hidden_size)
    if getattr(hf_conf, "tie_word_embeddings", False):
        output_params = embedding_params

    layer_weights: list[float] = [dense_layer_params] * total_layers

    num_moe_experts = getattr(tf_conf, "num_moe_experts", None)
    if num_moe_experts:
        moe_hidden_size = getattr(tf_conf, "moe_ffn_hidden_size", None)
        if moe_hidden_size is None:
            moe_hidden_size = getattr(hf_conf, "moe_intermediate_size", None)
        if moe_hidden_size is None:
            moe_hidden_size = intermediate_size

        expert_parallel_size = getattr(tf_conf, "expert_model_parallel_size", 1) or 1
        num_local_experts = math.ceil(num_moe_experts / max(expert_parallel_size, 1))

        router_params = hidden_size * num_moe_experts
        if getattr(tf_conf, "moe_router_enable_expert_bias", False):
            router_params += num_moe_experts

        expert_params = mlp_params(int(moe_hidden_size)) * num_local_experts

        shared_size = getattr(tf_conf, "moe_shared_expert_intermediate_size", None)
        shared_params = mlp_params(int(shared_size)) if shared_size else 0.0

        moe_layer_params = float(
            attn_and_norm + router_params + expert_params + shared_params
        )

        moe_layer_indices: set[int] = set()
        freq = getattr(tf_conf, "moe_layer_freq", 1)
        if isinstance(freq, int):
            step = abs(freq)
            assert step >= 1
            for idx in range(step - 1, total_layers, step):
                moe_layer_indices.add(idx)
        elif isinstance(freq, list) or isinstance(freq, tuple):
            assert len(freq) == total_layers
            for idx, is_moe in enumerate(freq):
                if is_moe:
                    moe_layer_indices.add(idx)

        for idx in moe_layer_indices:
            layer_weights[idx] = moe_layer_params

    return (layer_weights, embedding_params, output_params)
