# [MODULE DOC]
# Parallelism setup for FSDPEngine: ParallelHelper validates DP/SP(Ulysses, stored as
# cp_size)/TP/EP sizes and builds the N-D device mesh plus flattened submeshes,
# apply_non_moe_tp applies a Megatron-style sequence-parallel tensor-parallel plan to HF
# decoder models (including VLM language models), and parallelize_model combines
# optional TP with FSDP2 sharding over the dp_sp submesh.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import PretrainedConfig

from areal.api import FSDPParallelStrategy
from areal.api.cli_args import FSDPWrapPolicy, TrainEngineConfig
from areal.engine.core.model import (
    is_gemma3_model,
    is_moe_model,
    is_qwen3_vl_model,
    is_valid_vision_model,
)
from areal.engine.fsdp_utils import apply_fsdp2
from areal.infra.platforms import current_platform
from areal.models.parallel_styles import ReplicateParallel

__all__ = ["ReplicateParallel", "ParallelHelper", "parallelize_model"]


@dataclass
class ParallelHelper:
    # [CLASS DOC]
    # Dataclass wrapping an FSDPParallelStrategy (_ps) and a lazily built world
    # DeviceMesh. Validates sizes on construction and exposes sizes, enabled flags,
    # process groups and derived factors. Note 'sp' here maps to the strategy's cp_size.
    _ps: FSDPParallelStrategy
    _world_mesh: DeviceMesh | None = None

    @classmethod
    def from_parallel_strategy(cls, fsdp_ps: FSDPParallelStrategy) -> "ParallelHelper":
        # [FUNCTION DOC]
        # Alternate constructor; asserts pp_size == 1 (no pipeline parallelism under
        # FSDP) and returns a helper for the strategy.
        assert fsdp_ps.pp_size == 1, "Pipeline parallelism is not supported in FSDP"

        return cls(_ps=fsdp_ps)

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns a compact string of dp, sp, tp, ep, etp and world_size.
        _ps = self._ps
        return f"(dp={_ps.dp_size}, sp={_ps.cp_size}, tp={_ps.tp_size}, ep={_ps.ep_size}, etp={_ps.etp_size}, world_size={_ps.world_size})"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Runs _validate after dataclass init.
        self._validate()

    def _validate(self):
        # [FUNCTION DOC]
        # HELPER: Asserts sp/tp/ep/etp >= 1; raises ValueError if dp*sp*tp !=
        # world_size. When ep > 1, asserts etp is tp or 1, and that ep divides properly:
        # with etp == tp, ep is a multiple of sp and divides dp*sp; with etp == 1, ep is
        # a multiple of sp*tp and divides dp*sp*tp.
        dp, sp, tp, ep, etp, world_size = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.ep_size,
            self._ps.etp_size,
            self._ps.world_size,
        )
        for d in (sp, tp, ep, etp):
            assert d >= 1, "Parallelism degree should be >= 1"

        if dp * sp * tp != world_size:
            raise ValueError(
                f"Invalid parallel dims: dp({dp}) * sp({sp}) * tp({tp}) != WORLD_SIZE({world_size})"
            )

        if ep > 1:
            assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"
            if etp == tp:
                # ep would borrow all sp and some dp degree
                assert ep % sp == 0 and (dp * sp) % ep == 0
            elif etp == 1:
                # ep would borrow all sp and tp and some dp degree
                assert ep % (sp * tp) == 0 and (dp * sp * tp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC]
        # Builds and returns the device mesh, dispatching to the EP or non-EP variant
        # depending on ep_size > 1. Side effect: creates process groups (collective
        # initialisation).
        if self._ps.ep_size > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # [FUNCTION DOC]
        # HELPER: Splits dp into dp_mod_ep x dp_in_ep so that ep = dp_in_ep*sp (etp ==
        # tp) or dp_in_ep*sp*tp (etp == 1), creates a 4-D mesh (dp_mod_ep, dp_in_ep, sp,
        # tp) on the platform device type and flattens submeshes named dp, dp_sp, sp_tp
        # and ep. Returns the root mesh.
        dp, sp, tp, ep, etp = (
            self._ps.dp_size,
            self._ps.cp_size,
            self._ps.tp_size,
            self._ps.ep_size,
            self._ps.etp_size,
        )

        # With ep, dp and ep are derived submeshes:
        # dp = dp_mod_ep * dp_in_ep
        if etp == tp:
            # ep = dp_in_ep * sp
            dp_mod_ep = dp * sp // ep
            dp_in_ep = ep // sp
        else:
            assert etp == 1
            # ep = dp_in_ep * sp * tp
            dp_mod_ep = dp * sp * tp // ep
            dp_in_ep = ep // (sp * tp)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp_mod_ep, dp_in_ep, sp, tp),
            mesh_dim_names=("dp_mod_ep", "dp_in_ep", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp_mod_ep, dp_in_ep, sp, tp
        #     sub  mesh: dp, dp_sp, sp_tp, ep
        mesh["dp_mod_ep", "dp_in_ep"]._flatten(mesh_dim_name="dp")
        mesh["dp_mod_ep", "dp_in_ep", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")
        ep_mesh_dim_names = ("dp_in_ep", "sp", "tp") if etp == 1 else ("dp_in_ep", "sp")
        mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        # [FUNCTION DOC]
        # HELPER: Creates a 3-D mesh (dp, sp, tp) and flattens the dp_sp and sp_tp
        # submeshes. Returns the root mesh.
        dp, sp, tp = (self._ps.dp_size, self._ps.cp_size, self._ps.tp_size)

        mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(dp, sp, tp),
            mesh_dim_names=("dp", "sp", "tp"),
        )

        # Create all the submesh here for process groups
        # Guaranteed dims:
        #     root mesh: dp, sp, tp
        #     sub  mesh: dp_sp, sp_tp
        mesh["dp", "sp"]._flatten(mesh_dim_name="dp_sp")
        mesh["sp", "tp"]._flatten(mesh_dim_name="sp_tp")

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        # [FUNCTION DOC]
        # Property: lazily builds and caches the device mesh.
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if dp_size > 1.
        return self._ps.dp_size > 1

    @property
    def sp_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if cp_size (sequence parallel) > 1.
        return self._ps.cp_size > 1

    @property
    def tp_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if tp_size > 1.
        return self._ps.tp_size > 1

    @property
    def ep_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if ep_size > 1.
        return self._ps.ep_size > 1

    @property
    def etp_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if etp_size > 1.
        return self._ps.etp_size > 1

    @property
    def dp_size(self) -> int:
        # [FUNCTION DOC]
        # Property: data-parallel size.
        return self._ps.dp_size

    @property
    def sp_size(self) -> int:
        # [FUNCTION DOC]
        # Property: sequence-parallel size (strategy cp_size).
        return self._ps.cp_size

    @property
    def tp_size(self) -> int:
        # [FUNCTION DOC]
        # Property: tensor-parallel size.
        return self._ps.tp_size

    @property
    def ep_size(self) -> int:
        # [FUNCTION DOC]
        # Property: expert-parallel size.
        return self._ps.ep_size

    @property
    def etp_size(self) -> int:
        # [FUNCTION DOC]
        # Property: expert tensor-parallel size.
        return self._ps.etp_size

    @property
    def dp_group(self) -> ProcessGroup:
        # [FUNCTION DOC]
        # Property: process group of the 'dp' mesh dimension (builds the mesh if
        # needed).
        return self.world_mesh["dp"].get_group()

    @property
    def sp_group(self) -> ProcessGroup:
        # [FUNCTION DOC]
        # Property: process group of the 'sp' mesh dimension.
        return self.world_mesh["sp"].get_group()

    @property
    def tp_group(self) -> ProcessGroup:
        # [FUNCTION DOC]
        # Property: process group of the 'tp' mesh dimension.
        return self.world_mesh["tp"].get_group()

    @property
    def gradient_div_factor(self) -> int:
        # [FUNCTION DOC]
        # Property: dp_size * cp_size, the gradient division factor kept consistent for
        # EP-sharded experts.
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self._ps.dp_size * self._ps.cp_size

    @property
    def context_and_model_parallel_size(self) -> int:
        # [FUNCTION DOC]
        # Property: cp_size * tp_size.
        return self._ps.cp_size * self._ps.tp_size

    @property
    def seq_len_divisor(self) -> int:
        # [FUNCTION DOC]
        # Property: tp_size * cp_size; sequence lengths must be divisible by it for SP
        # and Ulysses.
        # 1. Sequence Parallel requires that seq_len be divisible by TP degree.
        # 2. Ulysses Sequence Parallel requires that seq_len be divisible by SP degree.
        return self._ps.tp_size * self._ps.cp_size


def apply_non_moe_tp(
    model: nn.Module,
    model_config: PretrainedConfig,
    parallel_helper: ParallelHelper,
    tp_device_mesh: DeviceMesh,
):
    # [FUNCTION DOC]
    # Applies tensor + sequence parallelism in place to a HF causal LM (or critic /
    # VLM). Reads attention and KV head counts from model.config or its text_config and
    # raises ValueError if not divisible by tp size; raises RuntimeError if model.model
    # is missing. Plan for the inner model: embedding rowwise (output sharded on seq),
    # layernorms/final norm as SequenceParallel, attention input all-gathered, q/k/v
    # colwise, q/k norms replicated, o_proj rowwise with seq-sharded output; for non-MoE
    # models the MLP is likewise all-gathered, gate/up colwise, down rowwise; Gemma3
    # adds its extra feed-forward norms. Root plan: lm_head colwise from seq-sharded
    # input; a critic 'score' head replicated with replicated output. For vision models
    # the plan targets model.model.language_model, drops the embedding entry and instead
    # shards layer 0's input (Qwen3-VL also gets its deepstack hook patched for TP);
    # raises RuntimeError if language_model is missing. Otherwise parallelizes
    # model.model. Finally parallelizes the root. Returns None.
    # KNOWN ISSUE: for MoE models the expert MLPs are left unparallelized (the function
    # name reflects this) and the head-count check ignores model_config, reading
    # model.config instead.
    num_attention_heads: int
    num_key_value_heads: int
    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,  # type: ignore
            model.config.num_key_value_heads,  # type: ignore
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,  # type: ignore
            model.config.text_config.num_key_value_heads,  # type: ignore
        )

    tensor_parallel_size = parallel_helper.tp_size

    if (
        num_attention_heads % tensor_parallel_size != 0
        or num_key_value_heads % tensor_parallel_size != 0
    ):
        raise ValueError(
            f"num_attention_heads {num_attention_heads} and num_key_value_heads {num_key_value_heads} must be divisible by tensor_parallel_size {tensor_parallel_size}"
        )

    if not isinstance(model.model, nn.Module):
        raise RuntimeError("Model does not have the required submodule 'model'.")

    # For model or model.language_model
    model_tp_plan: dict[str, ParallelStyle] = {
        "embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "layers.*.input_layernorm": SequenceParallel(),
        # All-gather
        "layers.*.self_attn": PrepareModuleInput(
            input_kwarg_layouts={"hidden_states": Shard(1)},
            desired_input_kwarg_layouts={"hidden_states": Replicate()},
        ),
        "layers.*.self_attn.q_proj": ColwiseParallel(),
        "layers.*.self_attn.k_proj": ColwiseParallel(),
        "layers.*.self_attn.v_proj": ColwiseParallel(),
        # special q/k norm for qwen3
        "layers.*.self_attn.q_norm": ReplicateParallel(),
        "layers.*.self_attn.k_norm": ReplicateParallel(),
        # Reduce in RowwiseParallel, Scatter by Shard(1)
        "layers.*.self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1),
            use_local_output=False,
        ),
        "layers.*.post_attention_layernorm": SequenceParallel(),
        "norm": SequenceParallel(),
    }

    if not is_moe_model(model_config.model_type):
        model_tp_plan.update(
            {
                # All-gather
                "layers.*.mlp": PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                ),
                "layers.*.mlp.gate_proj": ColwiseParallel(),
                "layers.*.mlp.up_proj": ColwiseParallel(),
                # Reduce in RowwiseParallel, Scatter by Shard(1)
                "layers.*.mlp.down_proj": RowwiseParallel(
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            }
        )

    if is_gemma3_model(model_config.model_type):
        model_tp_plan.update(
            {
                "layers.*.pre_feedforward_layernorm": SequenceParallel(),
                "layers.*.post_feedforward_layernorm": SequenceParallel(),
            }
        )

    # For root module
    root_tp_plan: dict[str, ParallelStyle] = {}
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        # Implicitly all-gather in ColwiseParallel
        # Output is sharded on the last dimension (Shard(2))
        root_tp_plan["lm_head"] = ColwiseParallel(
            input_layouts=Shard(1),
        )
    if hasattr(model, "score") and isinstance(model.score, nn.Module):
        # For PPO's critic model's score layer:
        # 1. The input is sharded by sequence parallelism (Shard(1))
        # 2. `score` is a linear layer with replicated weights
        # 3. All-gather the output along the sequence dimension to get the full results
        root_tp_plan["score"] = ReplicateParallel(
            input_layout=Shard(1),
            desired_input_layout=Shard(1),
            output_layout=Replicate(),
        )

    if is_valid_vision_model(model_config.model_type):
        if isinstance(model.model.language_model, nn.Module):
            # For vision-language models, avoid sharding the embedding layer because
            # the visual components access it without tensor parallelism support.
            # Instead, configure the first transformer layer to handle input
            # sharding properly.
            model_tp_plan.pop("embed_tokens", None)
            model_tp_plan["layers.0"] = PrepareModuleInput(
                input_layouts=Replicate(),
                desired_input_layouts=Shard(1),
            )

            # For Qwen3 VL, patch _deepstack_process for TP
            if is_qwen3_vl_model(model_config.model_type):
                # NOTE: Lazy import to avoid ImportError when qwen3_vl model is not used.
                # transformers.models.qwen3_vl doesn't exist in all transformers versions,
                # so we only import it when actually needed for Qwen3 VL models.
                from areal.models.transformers.qwen3_vl import (
                    patch_qwen3_vl_deepstack_process_for_tp,
                )

                patch_qwen3_vl_deepstack_process_for_tp(model.model.language_model)

            parallelize_module(
                model.model.language_model,
                device_mesh=tp_device_mesh,
                parallelize_plan=model_tp_plan,
            )
        else:
            raise RuntimeError(
                "Vision model does not have the required submodule 'model.language_model'"
            )
    else:
        parallelize_module(
            model.model,
            device_mesh=tp_device_mesh,
            parallelize_plan=model_tp_plan,
        )

    parallelize_module(
        model,
        device_mesh=tp_device_mesh,
        parallelize_plan=root_tp_plan,
    )


def parallelize_model(
    model: nn.Module,
    config: TrainEngineConfig,
    model_config: PretrainedConfig,
    nd_device_mesh: DeviceMesh,
    parallel_helper: ParallelHelper,
    cpu_offload: CPUOffloadPolicy | None = None,
    wrap_policy: FSDPWrapPolicy | None = None,
):
    # [FUNCTION DOC]
    # Applies TP (via apply_non_moe_tp on the 'tp' submesh) when tp is enabled, then
    # FSDP2-shards the model on the 'dp_sp' submesh with a mixed-precision policy (param
    # dtype config.dtype, reduce dtype config.grad_reduce_dtype, casting forward
    # inputs), the given CPU offload policy, reshard_after_forward=True and the wrap
    # policy. Returns None; mutates model in place.
    tp_enabled = parallel_helper.tp_enabled

    if tp_enabled:
        apply_non_moe_tp(model, model_config, parallel_helper, nd_device_mesh["tp"])

    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, config.dtype),
        reduce_dtype=getattr(torch, config.grad_reduce_dtype),
        cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        # This dim is guaranteed to exist by FSDPParallelDims
        "mesh": nd_device_mesh["dp_sp"],
        "mp_policy": mixed_precision_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": True,
    }
    apply_fsdp2(model, fsdp_kwargs, wrap_policy)
