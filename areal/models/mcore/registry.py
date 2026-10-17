# [MODULE DOC]
# Megatron-Core model registry/factory used by the Megatron engine. Builds the (HF
# config, mcore TransformerConfig) pair and the mcore model chunks for a HF checkpoint,
# either through a bridge (mbridge or NVIDIA Megatron-Bridge) or via AReaL's own
# converters for Qwen3 dense and BailingMoe architectures. Also post-processes the
# output layer: a scalar ValueHead for critics, or for actors either AReaL's chunked
# vocab-parallel LM head or an in-place fp32 LM-head forward patch.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import types
from typing import Any

import torch
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig as MCoreDDPConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from transformers import AutoConfig, PretrainedConfig

from areal.api.cli_args import MegatronEngineConfig
from areal.engine.megatron_utils.deterministic import set_deterministic_algorithms
from areal.models.mcore.bailing_moe import (
    hf_to_mcore_config_bailing_moe,
    make_mcore_layer_specs_bailing_moe,
)
from areal.models.mcore.qwen3 import (
    hf_to_mcore_config_qwen3_dense,
    make_mcore_layer_specs_qwen3_dense,
)
from areal.models.mcore.vocab_parallel_head import (
    replace_output_layer_with_areal_lm_head,
)
from areal.utils import logging

logger = logging.getLogger("MCoreRegistry")


class ValueHead(torch.nn.Linear):
    # [CLASS DOC]
    # Linear layer (hidden -> output_size, default 1) used as critic value head in place
    # of the LM output layer. Weight initialised N(0, 0.02), optional bias zeroed. With
    # sequence parallelism the weight is marked sequence_parallel (so grads are
    # all-reduced across TP) and outputs are gathered along the sequence dim.
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        *,
        config: TransformerConfig,
        bias: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Creates the linear layer with the given sizes and bias flag, records
        # config.sequence_parallel, marks the weight sequence_parallel when enabled, and
        # initialises parameters in place.
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        # [FUNCTION DOC]
        # Applies the linear layer, casts to float32, all-gathers from the
        # sequence-parallel region if SP is enabled (no gradient reduce-scatter on
        # output grad), and returns (values, None) to mimic ColumnParallelLinear. weight
        # and runtime_gather_output are accepted for interface compatibility and
        # ignored.
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


def _replace_output_layer_with_value_head(
    model: GPTModel,
    tf_config: TransformerConfig,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Replaces model.output_layer with a ValueHead(hidden_size -> 1) cast to
    # tf_config.params_dtype and sets model.vocab_size to 1. Raises ValueError if the
    # model has no output_layer (e.g. non-post-process pipeline stage). Mutates the
    # model.
    """Replace model's output_layer with ValueHead.

    This function can be used on any GPTModel instance, whether created
    via mbridge or directly. After replacement:
    - model.output_layer becomes a ValueHead instance
    - model.vocab_size is set to 1

    Args:
        model: The GPTModel instance to modify
        tf_config: Transformer configuration containing hidden_size and SP settings
    """
    if not hasattr(model, "output_layer"):
        raise ValueError(
            "Model does not have output_layer. Ensure post_process=True when creating GPTModel."
        )

    dtype = tf_config.params_dtype

    model.output_layer = ValueHead(
        input_size=tf_config.hidden_size,
        output_size=1,
        config=tf_config,
        bias=False,
    ).to(dtype=dtype)

    model.vocab_size = 1


def _is_lm_head_module_name(name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if a module name is "output_layer"/"lm_head" or ends with
    # ".output_layer"/".lm_head".
    return name in ("output_layer", "lm_head") or name.endswith(
        (".output_layer", ".lm_head")
    )


def _call_with_optional_group(fn, tensor: torch.Tensor, group: Any | None):
    # [FUNCTION DOC]
    # HELPER: Calls fn(tensor, group=group), falling back to fn(tensor) if a TypeError
    # is raised (older megatron-core signatures without group).
    # KNOWN ISSUE: any TypeError raised inside fn itself is also swallowed and triggers
    # a second call without the group.
    try:
        return fn(tensor, group=group)
    except TypeError:
        return fn(tensor)


def _fp32_lm_head_forward_impl(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    sequence_parallel: bool,
    tp_group: Any | None = None,
    **_: Any,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Replacement for ColumnParallelLinear._forward_impl (keyword-only). If
    # sequence_parallel, all-gathers input along the sequence dim from the TP region
    # (with reduce-scatter in backward), passing tp_group when supported. Computes input
    # @ weight^T in float32 and adds float32 bias if given. Returns fp32 logits (local
    # vocab shard). Extra kwargs are ignored.
    """Run the lm-head projection in fp32.

    A bf16 head keeps the ``[tokens, vocab]`` logits and their gradient in bf16,
    which is the dominant rounding error in the loss for large vocabularies.
    Casting the operands here keeps the matmul and the bias add in fp32 while
    leaving the rest of the model in its configured dtype.

    Mirrors ``ColumnParallelLinear`` from megatron-core, verified against 0.17.0.
    The collective helpers gained a ``group`` keyword in newer releases, so the
    calls fall back to the positional form when it is absent.

    Upgrade considerations: this replaces the fused ``_forward_impl`` with a
    plain matmul, so the dgrad all-reduce comes from
    ``copy_to_tensor_model_parallel_region`` instead of the fused kernel -
    equivalent, not fused. If megatron-core changes the forward short-circuit
    conditions or the collective signatures, this reimplementation drifts
    silently; ``_enable_fp32_lm_head_forward`` therefore steps aside when the
    installed megatron-core exposes a native ``ColumnParallelLinearFP32``
    (absent as of 0.17.0).
    """
    if sequence_parallel:
        try:
            total_input = tensor_parallel.gather_from_sequence_parallel_region(
                input,
                tensor_parallel_output_grad=True,
                group=tp_group,
            )
        except TypeError:
            total_input = tensor_parallel.gather_from_sequence_parallel_region(
                input,
                tensor_parallel_output_grad=True,
            )
    else:
        total_input = input

    output = torch.matmul(total_input.float(), weight.t().float())
    if bias is not None:
        output = output + bias.float()
    return output


def _fp32_lm_head_forward(
    self,
    input_: torch.Tensor,
    weight=None,
    runtime_gather_output: bool | None = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Replacement forward bound to an LM-head module lacking _forward_impl,
    # mirroring ColumnParallelLinear.forward. Uses the supplied weight or self.weight
    # (RuntimeError if neither). Determines bias (added in the matmul unless
    # skip_bias_add) and tp_group. If none of async allreduce / sequence_parallel /
    # explicit_expert_comm is set, wraps input with copy_to_tensor_model_parallel_region
    # (identity forward, all-reduce in backward). Runs the fp32 impl, then gathers
    # output across TP if runtime_gather_output (kwarg overrides positional) or else
    # self.gather_output is true. Returns (output, output_bias) where output_bias is
    # self.bias only when skip_bias_add.
    if weight is None:
        weight = self.weight
    if weight is None:
        raise RuntimeError(
            "weight was not supplied to lm_head forward pass and "
            "skip_weight_param_allocation is True."
        )

    bias = self.bias if not getattr(self, "skip_bias_add", False) else None
    tp_group = getattr(self, "tp_group", None)

    if (
        getattr(self, "async_tensor_model_parallel_allreduce", False)
        or getattr(self, "sequence_parallel", False)
        or getattr(self, "explicit_expert_comm", False)
    ):
        input_parallel = input_
    else:
        input_parallel = _call_with_optional_group(
            tensor_parallel.copy_to_tensor_model_parallel_region, input_, tp_group
        )

    output_parallel = _fp32_lm_head_forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        sequence_parallel=getattr(self, "sequence_parallel", False),
        tp_group=tp_group,
    )

    runtime_gather_output = kwargs.get("runtime_gather_output", runtime_gather_output)
    gather_output = (
        getattr(self, "gather_output", False)
        if runtime_gather_output is None
        else runtime_gather_output
    )
    if gather_output:
        output = _call_with_optional_group(
            tensor_parallel.gather_from_tensor_model_parallel_region,
            output_parallel,
            tp_group,
        )
    else:
        output = output_parallel

    output_bias = self.bias if getattr(self, "skip_bias_add", False) else None
    return output, output_bias


def _enable_fp32_lm_head_forward(
    models: list[torch.nn.Module],
    *,
    enabled: bool,
) -> int:
    # [FUNCTION DOC]
    # HELPER: When enabled, walks each model chunk (unwrapping DDP via .module) for
    # modules named output_layer/lm_head. Skips modules already fp32 (native
    # ColumnParallelLinearFP32 or previously patched). Patches modules with
    # _forward_impl by replacing that attribute with the fp32 impl (saving the
    # original); otherwise, for modules with weight and forward, binds
    # _fp32_lm_head_forward as forward (saving the original). Marks patched modules.
    # Logs a warning listing patched modules, an info if all were already fp32, or a
    # warning if none were found. Returns the number of patched modules (0 when
    # disabled).
    if not enabled:
        return 0

    native_fp32_cls = getattr(tensor_parallel, "ColumnParallelLinearFP32", None)
    patched = []
    already_fp32 = []

    for model_idx, model in enumerate(models):
        module = model.module if isinstance(model, DDP) else model
        for name, submodule in module.named_modules():
            if not _is_lm_head_module_name(name):
                continue
            if (
                native_fp32_cls is not None and isinstance(submodule, native_fp32_cls)
            ) or type(submodule).__name__ == "ColumnParallelLinearFP32":
                already_fp32.append(f"model{model_idx}:{name}")
                continue
            if getattr(submodule, "_areal_fp32_lm_head_enabled", False):
                already_fp32.append(f"model{model_idx}:{name}")
                continue
            if hasattr(submodule, "_forward_impl"):
                setattr(
                    submodule,
                    "_areal_original_forward_impl",
                    submodule._forward_impl,
                )
                setattr(submodule, "_forward_impl", _fp32_lm_head_forward_impl)
                setattr(submodule, "_areal_fp32_lm_head_enabled", True)
                patched.append(f"model{model_idx}:{name}:{type(submodule).__name__}")
            elif all(hasattr(submodule, attr) for attr in ("weight", "forward")):
                setattr(submodule, "_areal_original_forward", submodule.forward)
                setattr(
                    submodule,
                    "forward",
                    types.MethodType(_fp32_lm_head_forward, submodule),
                )
                setattr(submodule, "_areal_fp32_lm_head_enabled", True)
                patched.append(f"model{model_idx}:{name}:{type(submodule).__name__}")

    if patched:
        logger.warning(
            "Enabled FP32 lm_head/output_layer forward for modules: %s",
            ", ".join(patched),
        )
    elif already_fp32:
        logger.info(
            "FP32 lm_head/output_layer is already enabled for modules: %s",
            ", ".join(already_fp32),
        )
    else:
        logger.warning(
            "enable_fp32_lm_head=True, but no output_layer/lm_head module was found "
            "on this model chunk. This is expected for non-post-process pipeline "
            "stages."
        )

    return len(patched)


def unwrap_to_gpt_model(model: torch.nn.Module) -> GPTModel:
    # [FUNCTION DOC]
    # Unwraps nested .module wrappers (DDP, Float16Module etc.) until a GPTModel is
    # found; for VLM wrappers returns .language_model if it is a GPTModel. Raises
    # TypeError otherwise.
    """Unwraps a model to the underlying GPTModel instance.

    Handles both plain GPTModel (possibly wrapped in DDP) and VLM models
    (e.g., Qwen2_5VLModel) where GPTModel lives at ``model.language_model``.
    """
    _model = model
    while not isinstance(_model, GPTModel) and hasattr(_model, "module"):
        _model = _model.module
    if isinstance(_model, GPTModel):
        return _model
    # VLM models wrap GPTModel as language_model (e.g., Qwen2_5VLModel)
    if hasattr(_model, "language_model") and isinstance(
        _model.language_model, GPTModel
    ):
        return _model.language_model
    raise TypeError(f"Model could not be unwrapped to GPTModel. Got {type(_model)}")


def _replace_actor_output_layers(
    models: list[GPTModel | DDP],
    *,
    enabled: bool,
    fp32_operands: bool = False,
) -> None:
    # [FUNCTION DOC]
    # HELPER: When enabled, replaces each chunk's GPT output layer with AReaL's
    # vocab-parallel LM head producing fp32 output, with fp32 operands if fp32_operands.
    # No-op otherwise.
    if not enabled:
        return
    for model in models:
        gpt_model = unwrap_to_gpt_model(model)
        replace_output_layer_with_areal_lm_head(
            gpt_model,
            fp32_output=True,
            fp32_operands=fp32_operands,
        )


def _configure_actor_output_layers(
    models: list[GPTModel | DDP],
    mcore_config: MegatronEngineConfig | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: No-op if mcore_config is None. If enable_chunked_logits, installs the
    # AReaL LM head (fp32 operands per enable_fp32_lm_head); otherwise applies the fp32
    # LM-head forward patch when enable_fp32_lm_head is set.
    if mcore_config is None:
        return
    if mcore_config.enable_chunked_logits:
        _replace_actor_output_layers(
            models,
            enabled=True,
            fp32_operands=mcore_config.enable_fp32_lm_head,
        )
    else:
        _enable_fp32_lm_head_forward(
            models,
            enabled=mcore_config.enable_fp32_lm_head,
        )


# Model registry for different architectures
def make_hf_and_mcore_config(
    hf_path: str,
    dtype: torch.dtype,
    bridge=None,
    bridge_type: str = "mbridge",
) -> tuple[PretrainedConfig, TransformerConfig]:
    # [FUNCTION DOC]
    # Returns (hf_config, TransformerConfig). With an mbridge bridge: the bridge's
    # hf_config (its _name_or_path set to hf_path) and bridge.config. With a
    # megatron-bridge bridge: the HF config from bridge.hf_pretrained (or the object
    # itself) and bridge.transformer_config. Otherwise loads the HF config with
    # AutoConfig (trust_remote_code=True; asserts exactly one architecture) and converts
    # it with the Qwen3-dense or BailingMoe converter using dtype; raises ValueError for
    # other architectures. Mutates the bridge's hf_config name.
    if bridge is not None and bridge_type == "mbridge":
        hf_config = bridge.hf_config
        hf_config._name_or_path = hf_path
        return hf_config, bridge.config
    elif bridge is not None and bridge_type == "megatron-bridge":
        hf_config = getattr(bridge.hf_pretrained, "config", bridge.hf_pretrained)
        if hasattr(hf_config, "_name_or_path"):
            hf_config._name_or_path = hf_path
        return hf_config, bridge.transformer_config
    else:
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=hf_path,
            trust_remote_code=True,
        )
        assert len(hf_config.architectures) == 1
        architecture = hf_config.architectures[0]
        if architecture == "Qwen3ForCausalLM":
            return hf_config, hf_to_mcore_config_qwen3_dense(hf_config, dtype)
        elif architecture in (
            "BailingMoeV2_5ForCausalLM",
            "BailingMoeLinearForCausalLM",
            "BailingHybridForCausalLM",
        ):
            return hf_config, hf_to_mcore_config_bailing_moe(hf_config, dtype)
        else:
            raise ValueError(
                f"Architecture not registered for config conversion: {architecture}."
            )


def make_mcore_layer_specs(hf_config: PretrainedConfig, tf_config: TransformerConfig):
    # [FUNCTION DOC]
    # Returns the mcore transformer layer spec (TransformerEngine-based) for the single
    # HF architecture: Qwen3 dense or BailingMoe variants. Raises ValueError for
    # unregistered architectures; asserts exactly one architecture.
    assert len(hf_config.architectures) == 1
    architecture = hf_config.architectures[0]
    if architecture == "Qwen3ForCausalLM":
        return make_mcore_layer_specs_qwen3_dense(tf_config, use_te=True)
    elif architecture in (
        "BailingMoeV2_5ForCausalLM",
        "BailingMoeLinearForCausalLM",
        "BailingHybridForCausalLM",
    ):
        return make_mcore_layer_specs_bailing_moe(tf_config, hf_config, use_te=True)
    else:
        raise ValueError(
            f"Architecture not registered for config conversion: {architecture}."
        )


def make_mcore_model(
    hf_config: PretrainedConfig,
    tf_config: TransformerConfig,
    mcore_config: MegatronEngineConfig | None = None,
    bridge: Bridge | Any | None = None,
    bridge_type: str = "mbridge",
    is_critic: bool = False,
    use_lora: bool = False,
) -> list[GPTModel | DDP]:
    # [FUNCTION DOC]
    # Builds the model chunks (list of GPTModel or DDP-wrapped models). mbridge path:
    # bridge.get_model with DDP/FSDP/precision options from mcore_config, then critic
    # value head replacement or actor output-layer configuration. megatron-bridge path:
    # creates a provider (no weight loading), sets TP/PP/VPP (VPP only if > 1)/CP/EP/ETP
    # sizes from Megatron parallel state, sequence parallel when TP > 1, pipeline dtype,
    # recompute settings, disables embedding/loss pipeline-split accounting, enforces
    # enable_mtp (ValueError if requested but missing; otherwise drops MTP layers with a
    # warning), disables gradient accumulation fusion for LoRA, forces
    # variable_seq_lengths, alltoall MoE dispatcher (warning that the configured value
    # is ignored), no batched p2p and overlap p2p only with VPP and PP > 1, mirrors
    # these into tf_config, applies deterministic settings if configured, finalises and
    # provides the distributed model (LoRA disables distributed optimizer and overlap
    # options), then applies critic/actor output-layer handling. Fallback (no bridge):
    # raises NotImplementedError for VPP > 1 or non-linear rope scaling; builds a single
    # GPTModel (pre/post process, untied embeddings, RoPE with hf rope_theta and linear
    # interpolation factor), applies output-layer handling and optionally wraps in DDP.
    # Returns a list of model chunks.
    # KNOWN ISSUE: mcore_config defaults to None but is dereferenced unconditionally in
    # the mbridge and megatron-bridge paths and for wrap_with_ddp in the fallback path,
    # so calling without it raises AttributeError.
    if bridge is not None and bridge_type == "mbridge":
        models = bridge.get_model(
            # TODO: Add DDP options when supporting training
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            ddp_config=dataclasses.asdict(mcore_config.ddp),
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            use_custom_fsdp=mcore_config.use_custom_fsdp,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_precision_aware_optimizer=mcore_config.use_precision_aware_optimizer,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    if bridge is not None and bridge_type == "megatron-bridge":
        provider = bridge.to_megatron_provider(load_weights=False)
        vpp_size = mcore_config.virtual_pipeline_parallel_size or 0

        provider.tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
        provider.pipeline_model_parallel_size = (
            mpu.get_pipeline_model_parallel_world_size()
        )
        provider.virtual_pipeline_model_parallel_size = (
            vpp_size if vpp_size > 1 else None
        )
        provider.context_parallel_size = mpu.get_context_parallel_world_size()
        provider.expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        provider.expert_tensor_parallel_size = (
            mpu.get_expert_tensor_parallel_world_size()
        )
        provider.sequence_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        provider.pipeline_dtype = tf_config.params_dtype

        provider.recompute_granularity = mcore_config.recompute_granularity
        provider.recompute_method = mcore_config.recompute_method
        provider.recompute_num_layers = mcore_config.recompute_num_layers
        provider.distribute_saved_activations = (
            mcore_config.distribute_saved_activations
        )
        provider.recompute_modules = mcore_config.recompute_modules

        provider.account_for_embedding_in_pipeline_split = False
        provider.account_for_loss_in_pipeline_split = False

        has_mtp = bool(getattr(provider, "mtp_num_layers", None))
        if mcore_config.enable_mtp:
            if not has_mtp:
                raise ValueError(
                    "megatron.enable_mtp=True but the model has no MTP layers."
                )
        elif has_mtp:
            logger.warning(
                "Dropping MTP head (mtp_num_layers=%s -> None); not used in RL and not "
                "exportable for Qwen3.6. Set megatron.enable_mtp=True to keep it.",
                provider.mtp_num_layers,
            )
            provider.mtp_num_layers = None

        # LoRA params are injected after model materialization and do not carry
        # Megatron main_grad buffers required by fused grad accumulation kernels.
        if use_lora:
            provider.gradient_accumulation_fusion = False

        # Keep these four flags aligned with mbridge base defaults.
        provider.variable_seq_lengths = True
        logger.warning(
            "Ignoring mcore_config.moe_token_dispatcher_type=%s for bridge_type='megatron-bridge'; "
            "using 'alltoall' and variable_seq_lengths=True.",
            mcore_config.moe_token_dispatcher_type,
        )
        provider.moe_token_dispatcher_type = "alltoall"
        provider.batch_p2p_comm = False
        provider.overlap_p2p_comm = (
            vpp_size > 1 and provider.pipeline_model_parallel_size > 1
        )

        # Aligning tf config settings with provider for consistency.
        tf_config.variable_seq_lengths = provider.variable_seq_lengths
        tf_config.moe_token_dispatcher_type = provider.moe_token_dispatcher_type
        tf_config.batch_p2p_comm = provider.batch_p2p_comm
        tf_config.overlap_p2p_comm = provider.overlap_p2p_comm

        # Megatron-Bridge creates a new provider instead of building from
        # ``tf_config`` directly. Apply the prebuild deterministic settings to
        # the actual provider so construction-time consumers (TP layers and TE
        # attention) see the same configuration.
        if mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(provider, prebuild=True)

        provider.finalize()

        ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
        if use_lora:
            ddp_config.use_distributed_optimizer = False
            ddp_config.overlap_grad_reduce = False
            ddp_config.overlap_param_gather = False

        models = provider.provide_distributed_model(
            ddp_config=ddp_config,
            fp16=tf_config.fp16,
            bf16=tf_config.bf16,
            use_megatron_fsdp=mcore_config.use_custom_fsdp,
            use_torch_fsdp2=mcore_config.use_torch_fsdp2,
            wrap_with_ddp=mcore_config.wrap_with_ddp,
            overlap_param_gather_with_optimizer_step=mcore_config.overlap_param_gather_with_optimizer_step,
        )
        models = list(models)

        if is_critic:
            for model in models:
                _model = unwrap_to_gpt_model(model)
                _replace_output_layer_with_value_head(_model, tf_config)
        else:
            _configure_actor_output_layers(models, mcore_config)

        return models

    else:
        if (
            mcore_config is not None
            and mcore_config.virtual_pipeline_parallel_size is not None
            and mcore_config.virtual_pipeline_parallel_size > 1
        ):
            raise NotImplementedError(
                "Virtual pipeline parallelism requires mbridge-backed models."
            )
        transformer_layer_spec = make_mcore_layer_specs(hf_config, tf_config)
        rope_scaling_args = {}
        if hf_config.rope_scaling is not None:
            if hf_config.rope_scaling["type"] != "linear":
                raise NotImplementedError(
                    f"Rope scaling type {hf_config.rope_scaling['type']} not supported yet."
                )
            rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling[
                "factor"
            ]

        model = GPTModel(
            config=tf_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=hf_config.vocab_size,
            max_sequence_length=hf_config.max_position_embeddings,
            pre_process=True,  # TODO: pipeline parallel
            post_process=True,  # TODO: pipeline parallel
            share_embeddings_and_output_weights=False,  # TODO: implement share output weights
            position_embedding_type="rope",
            rotary_base=hf_config.rope_theta,
            **rope_scaling_args,
            # vp_stage=None TODO: virtual pipeline parallel
        )

        # Replace output_layer with ValueHead for critic models
        if is_critic:
            _replace_output_layer_with_value_head(model, tf_config)
        else:
            _configure_actor_output_layers([model], mcore_config)

        if mcore_config.wrap_with_ddp:
            ddp_config = MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
            wrapped = DDP(
                config=tf_config,
                ddp_config=ddp_config,
                module=model,
                disable_bucketing=False,
            )
            return [wrapped]
        return [model]
