# [MODULE DOC]
# FSDP2-based training engine (FSDPEngine) implementing the TrainEngine API for HF
# models with N-D parallelism (DP + Ulysses SP + TP), optional LoRA, CPU offload, tree
# training, multimodal (VLM) inputs, and weight sync to rollout engines via NCCL/XCCL
# broadcast or disk. Also defines role-specific subclasses: PPO actor/critic, SFT LM,
# reward model and DPO engines, each delegating algorithm logic to areal.trainer
# helpers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import gc
import math
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.nn.functional as dist_F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from areal.api import (
    FinetuneSpec,
    FSDPParallelStrategy,
    InferenceEngine,
    ModelAllocation,
    ParallelStrategy,
    ParamSpec,
    SaveLoadMeta,
    TrainEngine,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import OptimizerConfig, PerfTracerConfig, TrainEngineConfig
from areal.api.io_struct import DeviceRuntimeInfo
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
    warmup_process_groups,
)
from areal.engine.core.model import (
    disable_dropout_in_model,
    is_gemma3_model,
    is_qwen3_5_model,
    is_qwen3_moe_model,
    is_qwen3_vl_model,
    is_qwen_vl_model,
    is_valid_vision_model,
)
from areal.engine.fsdp_utils import (
    fsdp2_load_full_state_dict,
    get_cosine_schedule_with_warmup,
)
from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.engine.fsdp_utils.grad import fsdp2_clip_grad_norm
from areal.engine.fsdp_utils.optimizer import AnyPrecisionAdamW, PerLayerOptimWrapper
from areal.engine.fsdp_utils.parallel import ParallelHelper, parallelize_model
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)
from areal.models.transformers.ulyssess_patch import apply_monkey_patch
from areal.models.tree_attn.functional import (
    _gather_packed_tree_logprobs,
    gather_packed_tree_logprobs_entropy,
    gather_packed_tree_vocab_stats,
    merge_packed_tree_results,
)
from areal.models.tree_attn.module import (
    build_tree_attn_kwargs,
    patch_fsdp_for_tree_training,
)
from areal.models.tree_attn.tree import TrieNode, build_packed_tree_batch
from areal.utils import (
    logging,
    name_resolve,
    names,
    perf_tracer,
    pkg_version,
    stats_tracker,
)
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    concat_batch,
    is_multi_modal_key,
    pack_tensor_dict,
    pad_mb_list,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_processor_and_tokenizer, load_hf_tokenizer
from areal.utils.lr_scheduler import get_num_warmup_steps
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.offload import is_tms_enabled, torch_memory_saver
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

if TYPE_CHECKING:
    from areal.api import Scheduler
    from areal.api.cli_args import DPOEngineConfig, PPOActorConfig, PPOCriticConfig


@dataclasses.dataclass
class FSDPTrainContext:
    # [CLASS DOC]
    # Dataclass carried through the forward/backward pipeline per micro-batch:
    # model_inputs (after Ulysses slicing), mb_input (original micro-batch), pad_length
    # (packing padding), ulysses_pad_size, and trie_node (tree training root or None).
    """Context passed through FSDP forward/backward pipeline.

    Attributes
    ----------
    model_inputs
        The prepared inputs passed to the model (may include Ulysses slicing).
    mb_input
        The original micro-batch dict before any Ulysses transformations.
    pad_length
        Number of padding tokens added for sequence packing.
    ulysses_pad_size
        Extra padding added for Ulysses sequence parallel alignment.
    trie_node
        The root TrieNode for tree training (if applicable).
    """

    model_inputs: dict[str, Any]
    mb_input: dict[str, Any]
    pad_length: int = 0
    ulysses_pad_size: int = 0
    trie_node: TrieNode | None = None

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns a shallow dict of the fields without recursing (avoids RecursionError
        # on TrieNode).
        """Shallow dict conversion (avoids ``dataclasses.asdict`` which would
        recurse into TrieNode and hit ``RecursionError``).
        """
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclasses.dataclass
class _PendingWeightUpdateBucket:
    # [CLASS DOC]
    # HELPER: Dataclass for an in-flight async broadcast bucket during weight sync:
    # communication work handles, the rollout engine's update future, the named tensors
    # kept alive until completion, and the optional side stream used.
    handles: list[Any]
    fut: Future[None]
    named_tensors: list[tuple[str, torch.Tensor]]
    stream: torch.cuda.Stream | None = None


_MULTIMODAL_FORWARD_KEYS = ("image_grid_thw", "pixel_values", "video_grid_thw")


def _is_multimodal_payload_key(key: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if the key is a multimodal key per areal.utils.data or one of
    # image_grid_thw, pixel_values, video_grid_thw.
    return is_multi_modal_key(key) or key in _MULTIMODAL_FORWARD_KEYS


def _drop_multimodal_payloads(data: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: Removes all multimodal payload keys from the dict in place.
    for key in list(data.keys()):
        if _is_multimodal_payload_key(key):
            data.pop(key, None)


def _prepare_multimodal_forward_inputs(
    mb: dict[str, Any],
    padded_mb: dict[str, Any],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Moves multimodal data to the forward micro-batch only. Pops
    # multi_modal_input from mb (or from padded_mb if absent in mb, and removes the
    # duplicate otherwise), concatenates each of
    # image_grid_thw/pixel_values/video_grid_thw across items along dim 0 into
    # padded_mb, then drops all multimodal keys from mb. Mutates both dicts.
    """Keep multimodal tensors only in the forward micro-batch.

    ``mb`` is used by loss and bookkeeping callbacks, while ``padded_mb`` is
    passed to the model forward. Large multimodal tensors such as
    ``pixel_values`` should not be duplicated on both sides.
    """

    multi_modal_input = mb.pop("multi_modal_input", None)
    if multi_modal_input is None:
        multi_modal_input = padded_mb.pop("multi_modal_input", None)
    else:
        padded_mb.pop("multi_modal_input", None)

    if multi_modal_input is not None:
        for key in _MULTIMODAL_FORWARD_KEYS:
            values = [item[key] for item in multi_modal_input if key in item]
            if values:
                padded_mb[key] = torch.cat(values, dim=0)

    _drop_multimodal_payloads(mb)


class FSDPEngine(TrainEngine):
    # [CLASS DOC]
    # TrainEngine backed by PyTorch FSDP2. Lifecycle: construct with TrainEngineConfig
    # (loads HF config), create_process_group (dist init, device mesh and groups),
    # initialize (build model, patches, sharding, optimizer), then train/eval/forward,
    # weight updates to a connected rollout engine, save/load, offload/onload and
    # destroy. Key state: model, optimizer, lr_scheduler, tokenizer/processor,
    # parallel_helper and dp/sp/mp groups, rank info, weight-update groups, offload
    # flags and an optional per-layer optimizer wrapper.
    def __init__(self, config: TrainEngineConfig):
        # [FUNCTION DOC]
        # Stores config, loads the HF AutoConfig from config.path (trust_remote_code),
        # flags vision models, and declares/initialises all state fields (no process
        # groups or model yet). Side effect: reads the model config from disk/hub.
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: ProcessorMixin | None = None
        self.model_config: PretrainedConfig
        self._version: int = 0

        self._initialized = False
        self.own_global_group = False
        self._cpu_group: dist.ProcessGroup
        self.weight_update_group_initialized = False
        self.weight_update_group_names: list[str] = []
        self.weight_update_groups: list = []
        self.weight_update_master_addr: str
        self.weight_update_master_port: int

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )
        self.is_vision_model = is_valid_vision_model(self.model_config.model_type)

        # FSDP-specific initialization
        self.cpu_offload: CPUOffloadPolicy | None = None

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.parallel_helper: ParallelHelper
        self.world_mesh: DeviceMesh

        self.dp_group: dist.ProcessGroup
        self.sp_group: dist.ProcessGroup
        self.mp_group: dist.ProcessGroup

        self.world_size: int
        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False
        self._offload_depth: int = 0
        self._per_layer_optim_wrapper: PerLayerOptimWrapper | None = None
        self.enable_tree_training: bool = self.config.enable_tree_training

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        experiment_name: str,
        trial_name: str,
        dp_size: int = 1,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        learning_rate: float | None = None,
        use_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: int = 16,
        **kwargs,
    ) -> FSDPEngine:
        # [FUNCTION DOC]
        # Convenience constructor: builds a TrainEngineConfig with path,
        # experiment/trial names, backend string 'fsdp:d<dp>t<tp>', dtype, an
        # OptimizerConfig only if learning_rate is given (else inference-only), LoRA
        # settings and extra kwargs, and returns cls(config).
        """Construct an FSDPEngine directly without assembling a TrainEngineConfig.

        Parameters
        ----------
        model : str
            Path to HuggingFace checkpoint or model.
        experiment_name: str
        trial_name: str
        dp_size: int
            Data parallel size, default 1.
        tp_size: int
            Tensor parallel size, default 1.
        dtype : str
            Parameter data type, default 'bfloat16'.
        learning_rate : float | None
            Learning rate. If None, no optimizer is created (inference-only).
        use_lora : bool
            Whether to use LoRA.
        """
        optimizer_config = (
            OptimizerConfig(lr=learning_rate) if learning_rate is not None else None
        )
        config = TrainEngineConfig(
            path=model,
            experiment_name=experiment_name,
            trial_name=trial_name,
            backend=f"fsdp:d{dp_size}t{tp_size}",
            dtype=dtype,
            optimizer=optimizer_config,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **kwargs,
        )
        return cls(config)

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        # [FUNCTION DOC]
        # Patches default PG timeouts, initialises the default process group if needed
        # (marking it owned), creates a gloo CPU group, resolves the parallel strategy
        # (argument, else parsed from config.backend, else default), converts it to FSDP
        # form, builds the ParallelHelper and world mesh, and sets dp/sp groups, the
        # sp_tp model-parallel group, rank, world size, dp_head (first rank of mp group)
        # and dp_rank. Creates the per-rank logger and warms up communicators. Side
        # effects: collectives, logging.
        patch_dist_group_timeout(DIST_GROUP_DEFAULT_TIMEOUT)

        backend = current_platform.communication_backend
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True

        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )

        # FSDP-specific process group setup
        if parallel_strategy is None:
            if self.config.backend:
                parallel_strategy = ModelAllocation.from_str(
                    self.config.backend
                ).parallel
            else:
                parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[FSDPEngine Rank {dist.get_rank()}]")

        parallel_strategy = self._make_parallel_strategy(parallel_strategy)

        self.parallel_helper = ParallelHelper.from_parallel_strategy(parallel_strategy)

        self.logger.info(
            f"Initializing device mesh with parallel dims {str(self.parallel_helper)}."
        )

        self.world_mesh = self.parallel_helper.world_mesh

        self.dp_group = self.world_mesh["dp"].get_group()
        self.sp_group = self.world_mesh["sp"].get_group()

        # Sequence and model parallel group (sp+tp)
        self.mp_group = self.world_mesh["sp_tp"].get_group()

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dp_head = dist.get_process_group_ranks(self.mp_group)[0]
        self.dp_rank = dist.get_rank(self.dp_group)

        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

        # Eagerly initialize HCCL/NCCL communicators for the subgroups so
        # that lazy init doesn't race with colocated engines (issue #1099).
        warmup_process_groups(self.dp_group, self.sp_group, self.mp_group)

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        # [FUNCTION DOC]
        # Builds the trainable model. Asserts addr is None and ft_spec given;
        # RuntimeError on torch < 2.4. Sets torch_memory_saver to preload mode if
        # enabled. Creates the device model, raises ValueError for tree training with sp
        # > 1, applies the Ulysses and tree-attention monkey patches, applies LoRA if
        # configured. For memory-efficient LLM loading or LoRA, rank 0 loads pretrained
        # weights (logging missing/unexpected keys) and takes the full state dict while
        # others use empty. Parallelizes with TP+FSDP2 (with CPU offload policy if
        # offload_params), broadcasts the full state if needed, creates optimizer and LR
        # scheduler, and if per_layer_optim_step is set (adam only, else ValueError)
        # wraps the optimizer in PerLayerOptimWrapper. Marks initialised.
        # Initialize distributed enviroments and load model.
        assert addr is None, "FSDPEngine does not support remote initialization."
        assert ft_spec is not None, "FSDPEngine requires FinetuneSpec to initialize."
        if pkg_version.is_version_less("torch", "2.4.0"):
            raise RuntimeError("areal only supports FSDP2, which requires torch>=2.4.0")

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        # Create device model
        self._create_device_model()

        if self.enable_tree_training and self.parallel_helper.sp_size > 1:
            raise ValueError(
                "Tree training currently cannot be enabled with sp_size > 1."
            )
        # Monkey patch: replace attention's forward() with Ulysses variant.
        apply_monkey_patch(
            model=self.model,
            ulysses_sp_size=self.parallel_helper.sp_size,
            shard_vision_across_sp=self.config.fsdp.shard_vision_across_sp,
        )
        # Monkey patch: replace attention's forward() with tree attention.
        patch_fsdp_for_tree_training(enable=self.enable_tree_training)

        if self.config.use_lora:
            self._apply_peft_wrapper()

        # sharding_strategy = ShardingStrategy.FULL_SHARD
        # Simple auto wrap policy
        self.cpu_offload = (
            CPUOffloadPolicy() if self.config.fsdp.offload_params else None
        )
        tik = time.perf_counter()

        full_state = None
        need_broadcast = False

        is_llm_cpu_load = (
            self.config.fsdp.memory_efficient_load
            and not self.config.init_from_scratch
            and not self.is_vision_model
        )

        if is_llm_cpu_load or self.config.use_lora:
            need_broadcast = True
            if dist.get_rank() == 0:
                if is_llm_cpu_load:
                    pretrained_state = get_state_dict_from_repo_id_or_path(
                        self.config.path
                    )
                    missing, unexpected = self.model.load_state_dict(
                        pretrained_state, strict=False
                    )
                    if missing:
                        self.logger.warning(
                            f"Missing keys when loading pretrained weights: {missing}"
                        )
                    if unexpected:
                        self.logger.warning(
                            f"Unexpected keys when loading pretrained weights: {unexpected}"
                        )
                    del pretrained_state
                    gc.collect()
                full_state = self.model.state_dict()
            else:
                full_state = {}

        # NOTE: This applies FSDP2 with N-D parallelism (DP+SP+TP)
        parallelize_model(
            self.model,
            config=self.config,
            model_config=self.model_config,
            nd_device_mesh=self.world_mesh,
            parallel_helper=self.parallel_helper,
            cpu_offload=self.cpu_offload,
            wrap_policy=self.config.fsdp.wrap_policy,
        )

        if need_broadcast:
            broadcast_tik = time.perf_counter()
            fsdp2_load_full_state_dict(
                self.model,
                full_state,
                self.cpu_offload,
                tie_word_embeddings=self.model_config.tie_word_embeddings,
            )
            self.logger.info(
                f"Broadcasting model weights took {time.perf_counter() - broadcast_tik:.2f} seconds"
            )

        self.logger.info(
            f"Applying FSDP2 with N-D parallelism for {time.perf_counter() - tik:.2f} seconds"
        )

        self._create_optimizer(ft_spec)

        if self.config.fsdp.per_layer_optim_step:
            if self.optimizer_config.type != "adam":
                raise ValueError(
                    f"per_layer_optim_step only supports 'adam' optimizer, got '{self.optimizer_config.type}'."
                )
            self._per_layer_optim_wrapper = PerLayerOptimWrapper(
                model=self.model,
                optimizer=self.optimizer,
                device_id=self.device,
                prefetch_layers=self.config.fsdp.optim_step_prefetch_layers,
            )

        self._initialized = True

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        # [FUNCTION DOC]
        # Property: the DP process group.
        return self.dp_group

    @property
    def data_parallel_rank(self) -> int:
        # [FUNCTION DOC]
        # Property: this rank's index in the DP group.
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        # [FUNCTION DOC]
        # Property: DP size.
        return self.parallel_helper.dp_size

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        # [FUNCTION DOC]
        # Property: the combined SP+TP group.
        return self.mp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        # [FUNCTION DOC]
        # Property: the gloo CPU group; asserts the engine is initialised.
        assert self._initialized
        return self._cpu_group

    def destroy(self):
        # [FUNCTION DOC]
        # Marks uninitialised, deletes optimizer, model and per-layer wrapper, runs GC
        # and empties the device cache. If this engine owns the global group, barriers
        # on the CPU group (failures logged and ignored), destroys all process groups
        # and clears ownership so repeated calls are safe.
        self._initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        if self._per_layer_optim_wrapper is not None:
            del self._per_layer_optim_wrapper
            self._per_layer_optim_wrapper = None
        gc.collect()
        current_platform.empty_cache()
        gc.collect()
        # NOTE: if `own_global_group` is true, we assume that
        # no communications are needed after `destroy`, so we
        # directly destroy all groups. Otherwise, process group
        # handles still exist and we expect another engine to
        # clean up these groups.
        if dist.is_initialized() and self.own_global_group:
            # Pre-destroy synchronization on a CPU (gloo) group so that all
            # ranks leave the NCCL collective phase together. Without this
            # barrier, rank-0 (which owns the TCPStore server) may exit
            # before peers finish their final NCCL abort, causing
            # HeartbeatMonitor background threads on other ranks to observe
            # "recvValue failed" on the already-closed store. This is
            # harmless but produces a noisy stderr backtrace at teardown.
            if getattr(self, "_cpu_group", None) is not None:
                try:
                    dist.barrier(group=self._cpu_group)
                except Exception as e:  # pragma: no cover - best-effort
                    self.logger.warning(
                        f"pre-destroy CPU barrier failed (ignored): {e}"
                    )
            dist.destroy_process_group()
            # Make destroy() idempotent: if the controller calls destroy
            # more than once (e.g. via cleanup hooks), the second call
            # must not try to destroy already-destroyed groups.
            self.own_global_group = False

    @property
    def initialized(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether initialize completed.
        return self._initialized

    def current_data_parallel_head(self) -> int:
        # [FUNCTION DOC]
        # Returns the global rank of this rank's DP head.
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if this rank is its DP head (first rank of its SP+TP group).
        return self.rank == self.dp_head

    def train(self, mode: bool = True):
        # [FUNCTION DOC]
        # Sets the model to train (or eval when mode=False) mode and returns self.
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # Connects a rollout InferenceEngine (warning if replacing a different one),
        # creates a DistRolloutCoordinator, initialises the XCCL weight-update groups
        # once when meta.type is 'xccl', then synchronises the device and barriers on
        # the CPU group.
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self.weight_update_group_initialized:
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Requires a connected rollout engine (RuntimeError otherwise) and delegates to
        # the coordinator's rollout_batch with the given data, workflow and grouping
        # options; returns the list of trajectory dicts.
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Requires a connected rollout engine and delegates to the coordinator's
        # prepare_batch (dataloader-driven async rollout with acceptance filter,
        # grouping, dynamic batch size, reward normalisation). Returns trajectories.
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def update_weights(self, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # Requires a connected rollout engine; within the offload-aware context
        # dispatches to distributed (xccl, asserting groups are initialised) or disk
        # weight update; raises ValueError for other types.
        self._check_rollout_engine_connected()
        with self._offload_aware_context():
            if meta.type == "xccl":
                assert self.weight_update_group_initialized
                self._update_weights_from_distributed(meta)
            elif meta.type == "disk":
                self._update_weights_from_disk(meta)
            else:
                raise ValueError(f"Unknown weight update type {meta.type}")

    def set_version(self, version: int):
        # [FUNCTION DOC]
        # Sets the model version counter.
        self._version = version

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # Returns the model version counter.
        return self._version

    def save(self, meta: SaveLoadMeta):
        # [FUNCTION DOC]
        # Within the offload-aware context saves weights as HF (with
        # tokenizer/processor) or DCP (optionally with optimizer); for HF format with
        # with_optim also saves the optimizer separately. Raises ValueError on unknown
        # formats.
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
            elif meta.weight_format == "dcp":
                self._save_to_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        # [FUNCTION DOC]
        # Within the offload-aware context loads HF weights or a DCP checkpoint
        # (optionally with optimizer); for HF with with_optim also loads the separately
        # saved optimizer state. Raises ValueError on unknown formats.
        with self._offload_aware_context():
            if meta.weight_format == "hf":
                self._load_model_from_hf(meta.path)
            elif meta.weight_format == "dcp":
                self._load_from_dcp(meta.path, meta.with_optim)
            else:
                raise ValueError(f"Unknown weight format {meta.weight_format}. ")

            if meta.with_optim and meta.weight_format == "hf":
                self._load_optimizer_state(meta.path)

            # Checkpoint load replaces optimizer state tensor objects, losing
            # pinning and normalization established by PerLayerOptimWrapper.__init__.
            if meta.with_optim and self._per_layer_optim_wrapper is not None:
                self._per_layer_optim_wrapper.refresh_states()

    @contextmanager
    def _offload_aware_context(self):
        # [FUNCTION DOC]
        # HELPER: Reentrant context manager. When the engine is offloaded, the outermost
        # entry onloads and the outermost exit offloads again (depth counter); otherwise
        # it just yields.
        """Temporarily onload parameters for offload-unsafe operations.

        Reentrant: nested calls increment depth; only the outermost
        call performs actual onload/offload transitions.
        """
        if not self.is_offload:
            yield
            return

        self._offload_depth += 1
        if self._offload_depth == 1:
            self.onload()
        try:
            yield
        finally:
            self._offload_depth -= 1
            if self._offload_depth == 0:
                self.offload()

    def optimizer_zero_grad(self):
        # [FUNCTION DOC]
        # Zeros optimizer gradients; asserts an optimizer exists.
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def set_lr(self, lr: float) -> None:
        # [FUNCTION DOC]
        # Sets lr on every optimizer param group (asserts optimizer exists). Note the LR
        # scheduler may overwrite it on its next step.
        assert self.optimizer is not None
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Clips gradients with fsdp2_clip_grad_norm over the dp_sp (FSDP) and tp groups
        # using optimizer_config.gradient_clipping. If the norm is non-finite, zeros
        # grads and skips the update; otherwise steps via the per-layer wrapper or the
        # plain optimizer (traced). Does not step the LR scheduler. Returns dict with
        # update_successful (1.0/0.0), grad_norm and current lr. Asserts optimizer,
        # config and scheduler exist.
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        grad_norm = fsdp2_clip_grad_norm(
            list(self.model.parameters()),
            max_norm=self.optimizer_config.gradient_clipping,
            fsdp_group=self.world_mesh["dp_sp"].get_group(),
            tp_group=self.world_mesh["tp"].get_group(),
            offload_params=self.config.fsdp.offload_params,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        elif self.config.fsdp.per_layer_optim_step:
            assert self._per_layer_optim_wrapper is not None
            with trace_scope("fsdp_engine.step"):
                self._per_layer_optim_wrapper.step()
            update_successful = True
        else:
            with trace_scope("fsdp_engine.step"):
                self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        # [FUNCTION DOC]
        # Advances the LR scheduler one step; asserts it exists.
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        # [FUNCTION DOC]
        # Iterates micro-batches: prepares model inputs and FSDPTrainContext, adds tree-
        # attention kwargs (prefixed 'tree_') when tree training is on, runs the model
        # forward, squeezes the batch dim of logits, removes the tree kwargs, calls
        # process_output_fn(logits, ctx dict), and runs backward on the returned loss
        # unless forward_only or the loss is None. Returns None; gradients accumulate
        # across micro-batches.
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            # Lazily create tree attention metadata just before forward.
            # The returned dict keys are prefixed with "tree_" to avoid collisions
            # with HuggingFace's own kwargs. The patched _tree_attn_fwd_func in
            # module_fsdp.py reads these keys from the **kwargs that transformers
            # forwards through.
            tree_attn_keys: list[str] = []
            if self.enable_tree_training and ctx.trie_node is not None:
                padded_size = mb_item.padded_to_length
                assert padded_size is not None
                tree_kwargs = build_tree_attn_kwargs(
                    ctx.trie_node, padded_size, self.device
                )
                inputs.update(tree_kwargs)
                tree_attn_keys = list(tree_kwargs.keys())

            with trace_scope("fsdp_engine.forward"):
                outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

            # Release tree attention metadata after forward pass
            for key in tree_attn_keys:
                del inputs[key]

            ctx_dict = ctx.to_dict()
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                with trace_scope("fsdp_engine.backward"):
                    loss.backward()

    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        # [FUNCTION DOC]
        # One training step: ensures ready (onload, set Ulysses group), zeros grads,
        # concatenates list inputs, splits into micro-batches on device, computes the
        # global loss weight over DP, runs forward/backward with per-micro-batch loss
        # computation (loss multiplied by dp_size to undo FSDP gradient averaging), then
        # optimizer_step. Returns the optimizer stats plus num_micro_batches.
        self._ensure_ready()
        self.optimizer_zero_grad()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward-backward using process_output_fn callback
        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            # [FUNCTION DOC]
            # HELPER: Rebuilds the context and returns the scaled loss from
            # _compute_logprobs_and_loss with loss_multiplier = dp_size.
            ctx = FSDPTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.parallel_helper.dp_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
        stats = self.optimizer_step()
        stats["num_micro_batches"] = len(mb_list.mbs)
        return stats

    @torch.no_grad()
    def eval_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        # [FUNCTION DOC]
        # No-grad evaluation: same micro-batching and loss-weight computation as
        # train_batch but forward only, collecting each micro-batch loss and returning
        # their sum all-reduced over DP (aggregate_eval_losses).
        self._ensure_ready()

        input_batched, _ = self._normalize_batch_input(input_)

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self.dp_group
        )

        # Step 3: Forward using process_output_fn callback, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            # [FUNCTION DOC]
            # HELPER: Computes the micro-batch loss, appends a detached copy to the
            # outer list and returns it.
            ctx = FSDPTrainContext(**ctx_dict)
            loss = self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate losses
        return aggregate_eval_losses(losses, self.dp_group)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        # [FUNCTION DOC]
        # No-grad inference returning per-token outputs (logprobs, values, or tree
        # results). For list input, infers output_seqlens from each item's
        # attention_mask length (ValueError if a given output_seqlens disagrees);
        # otherwise defaults to the packed sequence lengths. Micro-batches, runs forward
        # collecting _compute_forward_result outputs, then merges tree results or
        # reorders/pads outputs into a [batch, max_len] tensor. For list input, splits
        # the result back per trajectory.
        self._ensure_ready()

        input_batched, meta = self._normalize_batch_input(input_)

        # Step 1: Prepare sequence lengths
        if meta is not None:
            assert isinstance(input_, list)
            inferred_seqlens = [d["attention_mask"].shape[-1] for d in input_]
            if output_seqlens is not None and output_seqlens != inferred_seqlens:
                raise ValueError(
                    f"output_seqlens mismatch for list input: "
                    f"given {output_seqlens}, "
                    f"inferred {inferred_seqlens} from attention_mask shapes."
                )
            output_seqlens = inferred_seqlens
        cu_seqlens = pack_tensor_dict(input_batched)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_batched).to(self.device)

        # Step 3: Forward using process_output_fn callback, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            # [FUNCTION DOC]
            # HELPER: Computes the forward result for the micro-batch and appends it to
            # the outer list; returns None so no backward happens.
            ctx = FSDPTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate and reorder outputs
        if self.enable_tree_training:
            result = merge_packed_tree_results(outputs, batch_size)
        else:
            result = reorder_and_pad_outputs(
                outputs, output_seqlens, mb_list, aggregate_fn
            )

        if meta is None:
            return result
        return split_batch(result, meta)

    def get_lora_adapter_info(self) -> dict[str, list[int]]:
        # [FUNCTION DOC]
        # Returns {param name without the 'base_model.model.' prefix: shape list} for
        # trainable LoRA parameters, or {} when LoRA is disabled or there is no model.
        # Shapes are local (possibly sharded) shapes.
        """Return adapter parameter names and shapes (for generating synthetic checkpoints)."""
        import re

        if not self.config.use_lora or self.model is None:
            return {}
        result = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and "lora_" in name:
                clean_name = re.sub(r"^base_model\.model\.", "", name)
                result[clean_name] = list(param.shape)
        return result

    def export_stats(self) -> dict[str, float]:
        # [FUNCTION DOC]
        # Within the offload-aware context exports all stats_tracker statistics reduced
        # over the DP group; returns a dict of floats.
        with self._offload_aware_context():
            return stats_tracker.export_all(
                reduce_group=self.data_parallel_group,
            )

    def offload(self) -> None:
        # [FUNCTION DOC]
        # Offloads device memory with torch_memory_saver.pause after clearing caches,
        # then synchronises, barriers on the CPU group, logs device stats and sets
        # is_offload. Raises RuntimeError if TMS is not enabled.
        """Offload model memory to CPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """
        if not is_tms_enabled():
            raise RuntimeError(
                "torch_memory_saver requires `enable_offload=True` in yaml config."
            )

        self.get_device_stats().log("before offload model")

        # Use torch_memory_saver to pause CUDA memory
        current_platform.clear_memory()
        torch_memory_saver.pause()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")

        self.is_offload = True

    def onload(self) -> None:
        # [FUNCTION DOC]
        # Resumes memory with torch_memory_saver.resume, synchronises, barriers on the
        # CPU group, logs device stats and clears is_offload.
        """Onload model memory from CPU back to GPU using torch_memory_saver.

        Ref: https://github.com/THUDM/slime/blob/main/slime/backends/fsdp_utils/actor.py
        """

        torch_memory_saver.resume()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")

        self.is_offload = False

    def clear_batches(self, shard_ids: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Drops the given shard ids from this worker's RTensor fetch buffer; no-op when
        # none are passed (non-DP-head ranks).
        """Drain this worker's client-side RTensor fetch buffer.

        Called via RPC by ``TrainController.clear_batches`` at step end so
        cross-node consumer DP heads release cached tensors. See #1209.
        Non-DP-head ranks receive no positional args via
        ``_call_workers`` (see train_controller.py:575-577) — accept the
        no-args call and noop, since their ``_fetch_buffer`` is empty.
        """
        from areal.infra.rpc.rtensor import clear_fetch_buffer

        if shard_ids:
            clear_fetch_buffer(shard_ids)

    def fetch_buffer_stats(self) -> dict[str, int]:
        # [FUNCTION DOC]
        # Returns local RTensor fetch-buffer statistics.
        """Expose local fetch-buffer stats for post-step drain verification."""
        from areal.infra.rpc.rtensor import fetch_buffer_stats

        return fetch_buffer_stats()

    def get_device_stats(self) -> DeviceRuntimeInfo:
        # [FUNCTION DOC]
        # Returns current device memory/runtime info.
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        # [FUNCTION DOC]
        # Flushes the perf tracer for the given step (force to write regardless of
        # schedule).
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        # [FUNCTION DOC]
        # Configures the global perf tracer with config, rank and role unless already
        # configured.
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        # [FUNCTION DOC]
        # HELPER: Converts a generic ParallelStrategy into an FSDPParallelStrategy with
        # the same fields.
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def _create_llm_actor_or_critic(self):
        # [FUNCTION DOC]
        # HELPER: Builds a text model in optimizer_dtype storage with the configured
        # attention implementation: AutoModelForTokenClassification with one label for
        # critics, else AutoModelForCausalLM. Uses from_config (random/empty weights)
        # for init_from_scratch or memory-efficient loading, otherwise from_pretrained.
        # Returns the model.
        # Storage dtype = optimizer_dtype. FSDP2 MixedPrecisionPolicy
        # (configured in parallelize_model) will cast to self.config.dtype
        # in forward/backward.
        dtype = getattr(torch, self.config.optimizer_dtype)

        if self.config.is_critic:
            model_class = AutoModelForTokenClassification
            model_kwargs = {"num_labels": 1}
        else:
            model_class = AutoModelForCausalLM
            model_kwargs = {}

        common_kwargs = {
            "dtype": dtype,
            "attn_implementation": self.config.attn_impl,
        }
        model_kwargs.update(common_kwargs)

        if self.config.init_from_scratch or self.config.fsdp.memory_efficient_load:
            model = model_class.from_config(
                self.model_config,
                **model_kwargs,
            )
        else:
            model = model_class.from_pretrained(
                pretrained_model_name_or_path=self.config.path,
                trust_remote_code=True,
                **model_kwargs,
            )
        return model

    def _create_device_model(self):
        # [FUNCTION DOC]
        # HELPER: Sets device and NUMA affinity from LOCAL_RANK (KeyError if unset),
        # records self.device, chooses the loading device (meta on non-zero ranks / cpu
        # on rank 0 for memory-efficient LLM loading, cpu for memory-efficient VLM, else
        # the accelerator). For VLMs raises ValueError if compute dtype is float16 or
        # init_from_scratch is set, loads processor and tokenizer and the
        # AutoModelForImageTextToText; for LLMs loads the tokenizer and creates
        # actor/critic. Optionally disables dropout, enables non-reentrant gradient
        # checkpointing and HF kernels, logs timing and stores self.model.
        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        current_platform.set_numa_affinity(int(os.environ["LOCAL_RANK"]))
        if current_platform.device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(int(os.environ["LOCAL_RANK"]))

        # Load weights in optimizer_dtype (typically fp32) to maintain
        # master weights. FSDP2 MP casts to compute dtype on-the-fly.
        dtype = getattr(torch, self.config.optimizer_dtype)

        if self.config.fsdp.memory_efficient_load:
            # Only rank 0 loads on CPU; other ranks use meta device (zero memory)
            # to avoid CPU OOM when multiple workers share a node.
            # Weights are broadcast from rank 0 after FSDP sharding in initialize().
            # Note: meta device optimization only applies to LLM (not VLM), because
            # VLM uses from_pretrained() which doesn't support meta device context.
            if not self.is_vision_model and dist.get_rank() != 0:
                loading_device = "meta"
            else:
                loading_device = "cpu"
        else:
            loading_device = current_platform.device_type

        self.get_device_stats().log("before model creation/loading")

        # Note: VLMs often have vision_tower in fp32 already; loading whole
        # model in optimizer_dtype (fp32 default) is consistent.
        if self.is_vision_model:
            # Compute dtype (config.dtype) is what FSDP2 MP casts to in
            # forward/backward; storage dtype (optimizer_dtype) is restricted
            # to fp32/bf16 by config validation, so checking it would never
            # catch a float16 misconfiguration.
            if self._compute_dtype() == torch.float16:
                raise ValueError(
                    "Vision models do not support float16 dtype. Please use bfloat16."
                )
            if self.config.init_from_scratch:
                raise ValueError(
                    "Vision models do not support initialization from scratch. Please use a pretrained model."
                )
            self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
                self.config.path
            )

            tik = time.perf_counter()
            # VLM: Use from_pretrained() on loading_device.
            with torch.device(loading_device):
                model = AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=self.config.path,
                    trust_remote_code=True,
                    dtype=dtype,
                    attn_implementation=self.config.attn_impl,
                )
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)
        else:
            self.tokenizer = load_hf_tokenizer(self.config.path)
            self.processor = None
            tik = time.perf_counter()
            # LLM: Decide between from_config() or from_pretrained() based on memory_efficient_load
            with torch.device(loading_device):
                model = self._create_llm_actor_or_critic()
                if self.config.disable_dropout:
                    disable_dropout_in_model(model)

        self.get_device_stats().log("after model creation/loading")

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.config.use_kernels:
            model.use_kernels = True
        self.logger.info(
            f"Model creation and loading time: {time.perf_counter() - tik:.2f}s"
        )
        self.model = model

    def _apply_peft_wrapper(self):
        # [FUNCTION DOC]
        # HELPER: Wraps the model with a PEFT LoRA adapter (causal-LM task, rank/alpha
        # from config, 'all-linear' targets unless specific modules are given, no bias),
        # enabling input grads and keeping adapter dtype unchanged. Rank 0 prints
        # trainable parameters. Raises NotImplementedError for non-'lora' peft types.
        # KNOWN ISSUE: task type is always CAUSAL_LM, even for critic (token-
        # classification) models.
        config = self.config
        if not config.target_modules or config.target_modules == ["all-linear"]:
            target_modules = "all-linear"
        else:
            target_modules = config.target_modules
        peft_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": target_modules,
            "bias": "none",
        }
        if self.config.peft_type == "lora":
            peft_config = LoraConfig(**peft_config)
        else:
            raise NotImplementedError()

        self.model.enable_input_require_grads()
        # autocast_adapter_dtype=False: with optimizer_dtype=float32, LoRA
        # adapter weights are stored in fp32 alongside the base model.
        # FSDP2 MixedPrecisionPolicy (param_dtype=config.dtype) casts both
        # base and adapter params to compute dtype during forward/backward.
        # The fp32 adapter path is not exercised by tests/test_fsdp_optimizer_dtype.py
        # (full-model only); LoRA + fp32 master is a known untested combination.
        self.model = get_peft_model(
            self.model,
            peft_config,
            autocast_adapter_dtype=False,
        )

        if self.rank == 0:
            self.model.print_trainable_parameters()

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op without an optimizer config. Asserts type is adam, adam_bf16 or
        # sgd, logging warnings for risky combinations (SGD, adam_bf16 without bf16
        # storage, adam with bf16 storage). Creates AdamW (fused unless VLM with TP),
        # AnyPrecisionAdamW with bf16 states, or SGD. Builds the LR scheduler (cosine
        # with min ratio, linear or constant, all with warmup computed from the finetune
        # spec; ValueError otherwise) and logs timing.
        if self.optimizer_config is None:
            return
        assert self.model is not None
        # Set up optimizer
        tik = time.perf_counter()
        assert self.optimizer_config.type in [
            "adam",
            "adam_bf16",
            "sgd",
        ], "Only adam/adam_bf16/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                "Using SGD with FSDP may be less stable. Consider using "
                "the 'adam' (AdamW) optimizer for improved stability."
            )
        elif self.optimizer_config.type == "adam_bf16":
            if self.config.optimizer_dtype != "bfloat16":
                # __post_init__ canonicalizes bf16 → bfloat16; check canonical value only.
                self.logger.warning(
                    "adam_bf16 is intended to be paired with optimizer_dtype="
                    "'bfloat16' for memory savings (m/v in bf16, Kahan summation "
                    "for fp32-equivalent updates). Current optimizer_dtype=%s — "
                    "you may want to use the standard 'adam' optimizer instead.",
                    self.config.optimizer_dtype,
                )
        elif (
            self.optimizer_config.type == "adam"
            and self.config.optimizer_dtype == "bfloat16"
        ):
            self.logger.warning(
                "optimizer.type='adam' with optimizer_dtype='bfloat16' is the "
                "exact configuration that triggers issue #1292 (silent loss "
                "plateau ~3x higher than fp32 master weights). torch.optim.AdamW "
                "will create bf16 exp_avg/exp_avg_sq inheriting from "
                "model.parameters(). For memory savings, switch to "
                "optimizer.type='adam_bf16' (uses Kahan summation). "
                "For default behavior, leave optimizer_dtype='float32'."
            )
        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay
        beta1 = self.optimizer_config.beta1
        beta2 = self.optimizer_config.beta2
        eps = self.optimizer_config.eps
        if self.optimizer_config.type == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                # VLM with tensor parallelism is incompatible with fused AdamW
                fused=not (self.is_vision_model and self.parallel_helper.tp_enabled),
            )
        elif self.optimizer_config.type == "adam_bf16":
            self.optimizer = AnyPrecisionAdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                eps=eps,
                momentum_dtype="bfloat16",
                variance_dtype="bfloat16",
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        total_train_steps = ft_spec.total_train_steps
        num_warmup_steps = get_num_warmup_steps(
            self.optimizer_config,
            total_train_steps,
        )

        if self.optimizer_config.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
                min_lr_ratio=self.optimizer_config.min_lr_ratio,
            )
        elif self.optimizer_config.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                total_train_steps,
            )
        elif self.optimizer_config.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Unknown lr scheduler type {self.optimizer_config.lr_scheduler_type}"
            )
        self.logger.info(f"Create optimizer time: {time.perf_counter() - tik}")

    def _check_rollout_engine_connected(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError unless both rollout engine and coordinator are
        # set.
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Onloads if offloaded and, when SP > 1, sets the global Ulysses SP
        # group.
        if self.is_offload:
            self.onload()

        if self.parallel_helper.sp_size > 1:
            set_ulysses_sequence_parallel_group(self.sp_group)

    @staticmethod
    def _normalize_batch_input(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> tuple[dict[str, Any], Any | None]:
        # [FUNCTION DOC]
        # HELPER: Static. For list input returns concat_batch's (batched dict, split
        # meta); for a dict returns (dict, None).
        """Normalize list/dict batch input to a single batched dict.

        Returns ``(batched_input, meta)`` where ``meta`` is non-None only when
        input is list-based and can be used to split forward outputs back into
        per-trajectory results.
        """
        if isinstance(input_, list):
            return concat_batch(input_)
        return input_, None

    def _get_model_name_parameters(
        self, meta: WeightUpdateMeta
    ) -> Iterator[tuple[str, nn.Parameter]]:
        # [FUNCTION DOC]
        # HELPER: Yields (name, param) pairs renamed to the rollout engine's naming.
        # Qwen-VL: for SGLang strips the first 'language_model.' and drops a leading
        # 'model.' for visual params; for vLLM strips the first 'model.' and remaps
        # language_model.* to language_model.model.* and lm_head.* to
        # language_model.lm_head.*. Gemma3 VLM uses the vLLM-like scheme. Other models
        # yield names unchanged.
        name_params_iterator = self.model.named_parameters()
        if self.is_vision_model and is_qwen_vl_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                if meta.gen_allocation.backend == "sglang":
                    # SGLang 0.5.9 branch
                    # LLM part: "model.language_model.norm.weight" -> "model.norm.weight"
                    # Vision part: "model.visual.blocks.5.mlp.gate_proj.weight" -> "visual.blocks.5.mlp.gate_proj.weight"
                    new_name = name.replace("language_model.", "", 1)
                    if new_name.startswith("model.visual."):
                        new_name = new_name.replace("model.", "", 1)
                    yield new_name, value
                    continue
                # vLLM 0.17.0 branch
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = f"language_model.{new_name}"
                yield new_name, value
        elif self.is_vision_model and is_gemma3_model(self.model_config.model_type):
            for name, value in name_params_iterator:
                new_name = name.replace("model.", "", 1)
                if new_name.startswith("language_model."):
                    new_name = new_name.replace(
                        "language_model.", "language_model.model.", 1
                    )
                elif new_name.startswith("lm_head."):
                    new_name = new_name.replace(
                        "lm_head.", "language_model.lm_head.", 1
                    )
                yield new_name, value
        else:
            yield from name_params_iterator

    def _compute_dtype(self) -> torch.dtype:
        # [FUNCTION DOC]
        # HELPER: Returns the torch dtype named by config.dtype.
        """Resolve config.dtype to torch.dtype (canonicalized in TrainEngineConfig)."""
        return getattr(torch, self.config.dtype)

    def _cast_to_compute_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Casts floating-point tensors whose dtype differs from the compute
        # dtype to it (e.g. fp32 master to bf16); returns others unchanged.
        """Cast fp32 master storage to compute dtype for export / rollout sync.

        When optimizer_dtype=float32 (the fp32 master weights default), the
        underlying storage is fp32. HF export and xccl weight sync to rollout
        engines (SGLang/vLLM) must cast back to compute dtype (typically bf16)
        so deployment artefacts and broadcast bandwidth stay unchanged.
        No-op when storage already matches compute dtype.
        """
        compute_dtype = self._compute_dtype()
        if tensor.is_floating_point() and tensor.dtype != compute_dtype:
            return tensor.to(compute_dtype)
        return tensor

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Returns the unsharded tensor for a parameter: full_tensor() for device
        # DTensors, rebuilding a DTensor from the local shard for CPU-offloaded DTensors
        # (gathers onto the shard's device), or moves plain CPU tensors to the
        # accelerator. Collective for DTensors.
        """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors."""
        tensor = param.data
        if isinstance(tensor, DTensor):
            # For non-offloaded DTensor, directly call full_tensor()
            if tensor.device.type != "cpu":
                return tensor.full_tensor()

            # Handle CPU offloaded DTensor: reconstruct DTensor from local tensor
            temp_dtensor = DTensor.from_local(
                tensor.to_local(),
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
            )
            return temp_dtensor.full_tensor()
        else:
            if tensor.device.type == "cpu":
                tensor = tensor.to(current_platform.device_type)
            return tensor

    def _update_bucket_weights_from_distributed_async(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
        stream: torch.cuda.Stream | None = None,
    ) -> _PendingWeightUpdateBucket | None:
        # [FUNCTION DOC]
        # HELPER: Rank-0 only. Sends one bucket of (name, full tensor) pairs to the
        # rollout engine. Returns None for an empty bucket. Builds ParamSpecs (name,
        # shape, dtype string) and, for LoRA, attaches the PEFT config to meta. With
        # multiple per-PP groups, for each group asks the rollout engine to receive
        # (meta copy with that group name), broadcasts every tensor synchronously from
        # rank 0 and waits for the engine, returning an already-completed bucket. With a
        # single group, asks the engine to receive, then issues async broadcasts (on the
        # given side stream after it waits on the current stream, if provided) and
        # returns a pending bucket with handles, future and tensors.
        # Early exit when chunk size is relatively small
        if not named_tensors:
            return None

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        if self.config.use_lora:
            if not self.config.target_modules or self.config.target_modules == [
                "all-linear"
            ]:
                target_modules = "all-linear"
            else:
                target_modules = self.config.target_modules

            meta.peft_config = {
                "r": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "target_modules": target_modules,
                "bias": "none",
            }

        if len(self.weight_update_groups) > 1:
            # Per-PP-rank groups: broadcast to each group sequentially.
            # Each group's NCCL broadcast must complete before the next
            # starts, because sglang's PP event loop processes requests
            # serially within each PP rank.
            for group_name, group in zip(
                self.weight_update_group_names, self.weight_update_groups
            ):
                pp_meta = copy.copy(meta)
                pp_meta.nccl_group_name = group_name
                fut = self.rollout_engine.update_weights_from_distributed(
                    pp_meta, param_specs
                )
                for _, tensor in named_tensors:
                    dist.broadcast(tensor, src=0, group=group, async_op=False)
                fut.result()
            # Return a no-op bucket (all work is already done).
            _done_fut: Future = Future()
            _done_fut.set_result(None)
            return _PendingWeightUpdateBucket(
                handles=[],
                fut=_done_fut,
                named_tensors=named_tensors,
                stream=stream,
            )

        # Single group: original async path
        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            context = torch.cuda.stream(stream)
        else:
            context = nullcontext()

        with context:
            for _, tensor in named_tensors:
                handles.append(
                    dist.broadcast(
                        tensor,
                        src=0,
                        group=self.weight_update_groups[0],
                        async_op=True,
                    )
                )

        return _PendingWeightUpdateBucket(
            handles=handles,
            fut=fut,
            named_tensors=named_tensors,
            stream=stream,
        )

    def _wait_pending_weight_update_bucket(
        self, pending_bucket: _PendingWeightUpdateBucket | None
    ):
        # [FUNCTION DOC]
        # HELPER: No-op for None; otherwise waits on all broadcast handles, then the
        # rollout engine's future, and clears the bucket's tensor list to free memory.
        if pending_bucket is None:
            return

        for handle in pending_bucket.handles:
            handle.wait()

        pending_bucket.fut.result()
        pending_bucket.named_tensors.clear()

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
        # [FUNCTION DOC]
        # HELPER: Synchronous variant: sends a bucket and waits for it to complete.
        pending_bucket = self._update_bucket_weights_from_distributed_async(
            meta, named_tensors
        )
        self._wait_pending_weight_update_bucket(pending_bucket)

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # HELPER: Sets TORCHELASTIC_USE_AGENT_STORE=False so a new TCP store can be
        # created. If the inference allocation has PP > 1, delegates to per-PP group
        # creation. Otherwise records group name 'update_weight_group', fills meta with
        # this host's IP and a free port (on all ranks), and on rank 0 asks the rollout
        # engine to join, creates a custom process group of size gen_world_size + 1
        # (trainer rank 0) via TCP rendezvous, stores it, and waits for the engine.
        # Asserts meta.type is 'xccl' and gen_allocation exists. Side effects: env
        # mutation, network, logging.
        assert meta.type == "xccl"
        gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1

        # NOTE: Processes launched with torchrun will set the following env var
        # to True, which blocks creating another TCP store for weight update.
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

        if gen_pp_size > 1:
            # When the inference side uses PP > 1, we MUST create per-PP-rank
            # NCCL groups (one per PP stage).  sglang's PP scheduler event loop
            # processes requests at PP rank 0 BEFORE forwarding them to PP
            # rank 1.  If a single NCCL group spans all PP ranks, the TCP
            # rendezvous at PP rank 0 blocks forever because PP rank 1 never
            # receives the init request -> deadlock.  Per-PP-rank groups avoid
            # this: each group only requires the trainer + one PP stage's TP
            # workers, so the rendezvous can complete within one PP stage.
            self.logger.info(
                f"gen_pp_size={gen_pp_size} > 1: creating per-PP-rank "
                f"weight update groups to avoid PP event-loop deadlock."
            )
            self._init_per_pp_weight_update_groups(meta, gen_pp_size)
        else:
            # PP == 1: single group spanning all inference workers.
            group_name = "update_weight_group"
            self.weight_update_group_names = [group_name]

            meta.nccl_master_address = self.weight_update_master_addr = gethostip()
            meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[
                0
            ]
            meta.nccl_group_name = group_name

            if dist.get_rank() == 0:
                assert meta.gen_allocation is not None
                gen_world_size = meta.gen_allocation.parallel.world_size

                fut = self.rollout_engine.init_weights_update_group(meta)

                init_method = (
                    f"tcp://{format_host_for_url(meta.nccl_master_address)}"
                    f":{meta.nccl_master_port}"
                )
                self.logger.info(
                    f"Initializing weight update group: type={meta.type} "
                    f"init_method={init_method} group={group_name} "
                    f"world_size={gen_world_size + 1}"
                )
                pg = init_custom_process_group(
                    backend=current_platform.communication_backend,
                    world_size=gen_world_size + 1,
                    init_method=init_method,
                    rank=0,
                    group_name=group_name,
                    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                )
                self.weight_update_groups = [pg]
                self.weight_update_group = pg

                fut.result()
                self.logger.info(f"Weight update group '{group_name}' initialized.")

    def _init_per_pp_weight_update_groups(
        self, meta: WeightUpdateMeta, gen_pp_size: int
    ):
        # [FUNCTION DOC]
        # HELPER: On rank 0, for each inference PP stage creates group
        # 'update_weight_group_<pp>' of size (gen_world_size / pp) + 1 with its own
        # host/free port, asking the rollout engine to join each before rendezvous and
        # waiting on it; remembers stage 0's address/port and sets weight_update_group
        # to the first group. Other ranks only record group names and empty
        # address/port. Asserts gen_allocation exists.
        """Create one NCCL weight-update group per inference PP stage.

        The group name carries a PP-rank suffix (e.g. ``update_weight_group_0``)
        which the inference side uses (Scenario 2 in ``sglang_remote.py``) to
        route the request to the correct PP stage's workers.
        """
        assert meta.gen_allocation is not None
        gen_world_size = meta.gen_allocation.parallel.world_size
        per_pp_world_size = gen_world_size // gen_pp_size

        self.weight_update_group_names = []
        self.weight_update_groups = []

        if dist.get_rank() == 0:
            for pp_rank in range(gen_pp_size):
                group_name = f"update_weight_group_{pp_rank}"
                self.weight_update_group_names.append(group_name)

                pp_meta = copy.copy(meta)
                pp_meta.nccl_master_address = gethostip()
                pp_meta.nccl_master_port = find_free_ports(1)[0]
                pp_meta.nccl_group_name = group_name

                if pp_rank == 0:
                    self.weight_update_master_addr = pp_meta.nccl_master_address
                    self.weight_update_master_port = pp_meta.nccl_master_port

                self.logger.info(
                    f"Initializing per-PP weight update group: "
                    f"pp_rank={pp_rank} group={group_name} "
                    f"world_size={per_pp_world_size + 1}"
                )

                fut = self.rollout_engine.init_weights_update_group(pp_meta)

                init_method = (
                    f"tcp://{format_host_for_url(pp_meta.nccl_master_address)}"
                    f":{pp_meta.nccl_master_port}"
                )
                pg = init_custom_process_group(
                    backend=current_platform.communication_backend,
                    world_size=per_pp_world_size + 1,
                    init_method=init_method,
                    rank=0,
                    group_name=group_name,
                    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                )
                self.weight_update_groups.append(pg)

                fut.result()
                self.logger.info(
                    f"Per-PP weight update group '{group_name}' "
                    f"(pp_rank={pp_rank}) initialized."
                )

            # Backward compat: set single-group attribute to first group
            self.weight_update_group = self.weight_update_groups[0]
        else:
            # Non rank-0 FSDP ranks do not participate in NCCL groups
            for pp_rank in range(gen_pp_size):
                self.weight_update_group_names.append(f"update_weight_group_{pp_rank}")
            self.weight_update_master_addr = ""
            self.weight_update_master_port = 0

    @trace_perf("fsdp_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # HELPER: Broadcast-based weight sync (traced). Refreshes meta's NCCL
        # address/port/group name, rank 0 pauses generation, all ranks barrier. Iterates
        # (renamed) parameters, LoRA-trainable ones only when LoRA is on; every rank
        # gathers the full tensor (collective), but only rank 0 casts it to compute
        # dtype and accumulates it into a bucket up to weight_chunked_mem_mb, sending
        # full buckets asynchronously on a CUDA side stream while keeping at most one
        # bucket in flight; the remainder is sent synchronously and any pending bucket
        # is drained in a finally block. Then barrier, rank 0 resumes generation, device
        # sync and a final barrier.
        """Broadcast parameters with single-pending-bucket pipelining."""

        # Reset weight weight meta with local info
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = (
            self.weight_update_group_names[0]
            if self.weight_update_group_names
            else "update_weight_group"
        )

        main_rank = dist.get_rank() == 0
        if main_rank:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        broadcast_stream = None

        if (
            main_rank
            and current_platform.device_type == "cuda"
            and torch.cuda.is_available()
        ):
            broadcast_stream = torch.cuda.Stream()

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []
        pending_bucket: _PendingWeightUpdateBucket | None = None

        if self.config.use_lora:
            # For LoRA, only iterate over trainable LoRA parameters
            param_iterator = (
                (name, param)
                for name, param in self._get_model_name_parameters(meta)
                if param.requires_grad
            )
        else:
            # For full model, iterate over all parameters
            param_iterator = self._get_model_name_parameters(meta)

        try:
            for name, param in param_iterator:
                # Ranks other than 0 only help to get the full tensor
                # (DTensor.full_tensor() is a collective; all ranks must
                # call _get_full_tensor). Only rank 0 broadcasts to the
                # rollout engine, so casting is main-rank-only by design.
                tensor = self._get_full_tensor(param)
                if not main_rank:
                    continue
                # Cast fp32 master storage to compute dtype before broadcast.
                # Rollout engines (SGLang/vLLM) expect compute dtype (bf16).
                tensor = self._cast_to_compute_dtype(tensor)

                tensor_size = tensor.numel() * tensor.element_size()
                bucket_overflow = (
                    buffer_size > 0
                    and tensor_size + buffer_size > weight_chunked_mem_size
                )
                if bucket_overflow:
                    # Only middle buckets need drain+align before the next all-gather.
                    if pending_bucket is not None:
                        self._wait_pending_weight_update_bucket(pending_bucket)
                        pending_bucket = None

                    pending_bucket = self._update_bucket_weights_from_distributed_async(
                        meta,
                        named_tensors,
                        stream=broadcast_stream,
                    )

                    named_tensors = []
                    buffer_size = 0

                buffer_size += tensor_size
                named_tensors.append((name, tensor))

            if pending_bucket:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None

            # Process remaining parameters
            if buffer_size > 0:
                self._update_bucket_weights_from_distributed(meta, named_tensors)
        finally:
            if main_rank and pending_bucket is not None:
                self._wait_pending_weight_update_bucket(pending_bucket)
                pending_bucket = None

        dist.barrier(group=self.cpu_group)
        if main_rank:
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("fsdp_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # HELPER: Disk-based weight sync (traced). Rank 0 pauses generation and asks the
        # rollout engine to load from meta.path (asynchronously); all ranks save the HF
        # model to meta.path (asserts it is set). Rank 0 then (re)publishes a
        # name_resolve key for this experiment/trial/version with the current timestamp
        # (TTL 120 s) so the engine can see the save finished, waits for the engine and
        # resumes generation. Ends with device sync and CPU barrier.
        fut = Future()

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
        self._save_model_to_hf(meta.path, self.tokenizer, self.processor)
        # dist.barrier() are called when _save_model_to_hf finished

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            try:
                name_resolve.delete(update_name)
            except Exception:
                pass
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )

            fut.result()
            self.rollout_engine.continue_generation()

        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError if no model; creates the directory and saves
        # either the LoRA adapter only or the full model with tokenizer/processor.
        """Save model in HuggingFace format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        if self.config.use_lora:
            self._save_lora_to_hf(path)
        else:
            self._save_full_model_to_hf(path, tokenizer, processor)

    def _save_lora_to_hf(self, path: str):
        # [FUNCTION DOC]
        # HELPER: Gathers each trainable LoRA parameter individually (full_tensor
        # collective for DTensors); rank 0 renames keys to PEFT-standard (drops the
        # '.default' adapter segment), casts floats to compute dtype on CPU, writes
        # adapter_model.safetensors and the adapter config. Ends with a CPU-group
        # barrier.
        """Save only LoRA adapter weights without gathering full model state.

        Iterates adapter parameters and unshards them individually to avoid
        allocating the full base model state dict (which would OOM).
        """
        import re

        from safetensors.torch import save_file
        from torch.distributed.tensor import DTensor

        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)

        adapter_state = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad or "lora_" not in name:
                continue

            if isinstance(param.data, DTensor):
                full_param = param.data.full_tensor()
            else:
                full_param = param.data

            if dist.get_rank() == 0:
                # Emit PEFT-serving-standard keys. Drop the active-adapter
                # segment (".default") so names match what
                # PeftModel.save_pretrained produces
                # (e.g. "...down_proj.lora_A.weight"). Keeping ".default"
                # makes vLLM's parse_fine_tuned_lora_name reject the adapter
                # with "unsupported LoRA weight" on disk-mode load.
                clean_name = re.sub(r"\.default\.(weight|bias)$", r".\1", name)
                adapter_state[clean_name] = (
                    self._cast_to_compute_dtype(full_param.cpu())
                    if full_param.is_floating_point()
                    else full_param.cpu()
                )

        if dist.get_rank() == 0:
            save_file(adapter_state, os.path.join(path, "adapter_model.safetensors"))
            # Save adapter config
            self.model.peft_config["default"].save_pretrained(path)

        dist.barrier(group=self.cpu_group)

    def _save_full_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        # [FUNCTION DOC]
        # HELPER: Gathers the full model state dict offloaded to CPU (collective), rank
        # 0 casts float tensors to compute dtype and saves the model with
        # save_pretrained plus the HF config, tokenizer and processor if given. Ends
        # with a CPU-group barrier.
        """Save full model weights."""
        # FSDP2 checkpoint saving
        # Get full state dict with FSDP2
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self.model, options=options)

        # Cast weights to compute dtype before HF export. When
        # optimizer_dtype=float32 (default for fp32 master weights), the
        # underlying storage is fp32, but downstream consumers
        # (rollout engines, HF users) expect compute dtype (bfloat16).
        if dist.get_rank() == 0:
            state_dict = {
                k: self._cast_to_compute_dtype(v) if v.is_floating_point() else v
                for k, v in state_dict.items()
            }

        # save huggingface model on rank 0
        if dist.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            if tokenizer is not None:
                tokenizer.save_pretrained(path)
            if processor is not None:
                processor.save_pretrained(path)
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str):
        # [FUNCTION DOC]
        # HELPER: Rank 0 reads the full state dict from the path or hub repo; all ranks
        # then load/broadcast it into the sharded model via fsdp2_load_full_state_dict.
        """Load model from HuggingFace format."""
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            self.cpu_offload,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(
        self,
        path: str,
        with_optim: bool,
    ):
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError if no model; creates the directory and saves a
        # DCPState (model plus optimizer when with_optim) under key 'dcp' with
        # torch.distributed.checkpoint.save (collective, sharded files).
        """Save model in PyTorch Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        os.makedirs(path, exist_ok=True)

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self, path: str, with_optim: bool):
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError if no model; loads a DCPState (model plus optional
        # optimizer) from the DCP checkpoint at path, in place.
        """Load model from Distributed Checkpoint (DCP) format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")

        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=path,
        )

    def _save_optimizer_state(self, path: str):
        # [FUNCTION DOC]
        # HELPER: Each rank saves its local optimizer state dict with torch.save to
        # optim_world_size_<W>_rank_<R>.pt in path, then barriers. Asserts optimizer and
        # distributed init.
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(group=self.cpu_group)

    def _load_optimizer_state(self, path: str):
        # [FUNCTION DOC]
        # HELPER: Each rank loads its optim_world_size_<W>_rank_<R>.pt shard
        # (weights_only=False, so pickle execution is possible) into the optimizer, then
        # barriers. Requires the same world size as when saved.
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(group=self.cpu_group)

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        # [FUNCTION DOC]
        # HELPER: Turns a padded batch dict (requires attention_mask and input_ids;
        # shallow-copied) into a MicroBatchList on CPU. Tree training: builds packed
        # trie micro-batches synchronised over DP (size aligned to tp*sp) and returns
        # early. Otherwise computes position ids: for Qwen-VL casts input_ids to int64,
        # concatenates image/video grid sizes from multi_modal_input and calls the
        # model's 3-D (mRoPE) position-id routine (requires mm_token_type_ids),
        # permuting dims; other models get standard position ids. Splits by mb_spec,
        # packs each micro-batch, pads (to maximum optionally), logs token counts, adds
        # a batch dim, and for Qwen-VL restores the [3, 1, seq] position layout. For
        # each micro-batch and its padded twin sets integer max_length_q/k/max_seqlen,
        # cu_seq_lens_q/k, use_cache=False, attention_mask None (Qwen3 MoE/VL/3.5) or an
        # empty full/sliding dict, and moves multimodal payloads into the padded
        # (forward) batch only. Finally drops multimodal payloads from the list's data.
        # Returns the MicroBatchList.
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        # Tree training path
        if self.enable_tree_training:
            mb_list = build_packed_tree_batch(
                input_,
                mb_spec=self.config.mb_spec,
                pad_to_maximum=self.config.pad_to_maximum,
                dp_group=self.data_parallel_group,
                parallel_size=self.parallel_helper.tp_size
                * self.parallel_helper.sp_size,
            )
            self.logger.info(
                f"Packed tree #microbatch: {len(mb_list)}, microbatch #tokens: {mb_list.group_lens}, "
                f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}."
            )
            return mb_list

        if is_qwen_vl_model(self.model_config.model_type):
            attn_mask = input_["attention_mask"]
            input_ids = input_["input_ids"]
            # NOTE: Qwen-VL get_rope_index performs indexed assignment where
            # source positions are int64 and position_ids inherits input_ids.dtype.
            # Ensure input_ids uses int64 so destination/source dtypes align and
            # avoid "Index put requires the source and destination dtypes match".
            if input_ids.dtype != torch.long:
                input_ids = input_ids.to(torch.long)
                input_["input_ids"] = input_ids
            image_grid_thw = None
            video_grid_thw = None
            if "multi_modal_input" in input_:
                multi_modal_input = input_["multi_modal_input"]
                image_grid_thw_list = [
                    m["image_grid_thw"]
                    for m in multi_modal_input
                    if "image_grid_thw" in m
                ]
                if image_grid_thw_list:
                    image_grid_thw = torch.cat(image_grid_thw_list)
                video_grid_thw_list = [
                    m["video_grid_thw"]
                    for m in multi_modal_input
                    if "video_grid_thw" in m
                ]
                if video_grid_thw_list:
                    video_grid_thw = torch.cat(video_grid_thw_list)

            position_ids = self.model.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attn_mask,
                mm_token_type_ids=input_["mm_token_type_ids"],
                inputs_embeds=None,
                past_key_values=None,
            )
            position_ids = torch.einsum("ijk->jki", position_ids)
            input_["position_ids"] = position_ids
        else:
            input_ = amend_position_ids(input_)

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        self.logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}"
        )
        mb_list = unsqueeze_mb_list(mb_list)
        if is_qwen_vl_model(self.model_config.model_type):
            assert mb_list.padded_mbs is not None
            for mb in mb_list.padded_mbs:
                mb["position_ids"] = torch.einsum("ijk->kij", mb["position_ids"])

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
            mb["max_length_q"] = mb["max_length_k"] = mb["max_seqlen"] = int(
                mb["max_seqlen"]
            )
            padded_mb["max_length_q"] = padded_mb["max_length_k"] = padded_mb[
                "max_seqlen"
            ] = int(padded_mb["max_seqlen"])
            mb["cu_seq_lens_q"] = mb["cu_seq_lens_k"] = mb["cu_seqlens"]
            padded_mb["cu_seq_lens_q"] = padded_mb["cu_seq_lens_k"] = padded_mb[
                "cu_seqlens"
            ]
            mb["use_cache"] = False
            padded_mb["use_cache"] = False
            if (
                is_qwen3_moe_model(self.model_config.model_type)
                or is_qwen3_vl_model(self.model_config.model_type)
                or is_qwen3_5_model(self.model_config.model_type)
            ):
                mb["attention_mask"] = None
                padded_mb["attention_mask"] = None
            else:
                mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
                padded_mb["attention_mask"] = dict(
                    full_attention=None, sliding_attention=None
                )
            _prepare_multimodal_forward_inputs(mb, padded_mb)
        _drop_multimodal_payloads(mb_list.data)
        return mb_list

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], FSDPTrainContext]:
        # [FUNCTION DOC]
        # HELPER: Builds model inputs and an FSDPTrainContext for one micro-batch. With
        # Ulysses SP > 1: pads (VLM, no slicing since vision is handled separately) or
        # pads and slices (LLM) input_ids/position_ids to the SP size, makes position
        # ids contiguous and builds inputs via ulysses_prepare_inputs (which adds rolled
        # labels). Otherwise copies the padded micro-batch and pops trie_node. Always
        # removes turn_ids. Returns (inputs, ctx with original micro-batch, padding
        # length, Ulysses pad size and trie node).
        """Prepare micro-batch inputs with Ulysses sequence parallel handling.

        This method handles Ulysses SP padding and slicing, returning both
        the prepared model inputs and a context object for later processing.
        """
        trie_node = None
        if self.parallel_helper.sp_size > 1:
            input_ids = mb_item.padded_mb["input_ids"]
            position_ids = mb_item.padded_mb.get("position_ids", None)

            if self.is_vision_model:
                (
                    ulysses_input_ids,
                    ulysses_position_ids,
                    ulysses_pad_size,
                ) = ulysses_pad(
                    input_ids, position_ids, sp_size=self.parallel_helper.sp_size
                )
            else:
                # Pad and slice the inputs
                (
                    ulysses_input_ids,
                    ulysses_position_ids,
                    ulysses_pad_size,
                ) = ulysses_pad_and_slice_inputs(
                    input_ids,
                    position_ids,
                    sp_size=self.parallel_helper.sp_size,
                )
            if (
                ulysses_position_ids is not None
                and not ulysses_position_ids.is_contiguous()
            ):
                ulysses_position_ids = ulysses_position_ids.contiguous()

            inputs = ulysses_prepare_inputs(
                mb_item.padded_mb,
                ulysses_input_ids,
                ulysses_position_ids,
                self.parallel_helper.sp_size,
            )
        else:
            inputs = dict(mb_item.padded_mb)
            trie_node = inputs.pop("trie_node", None)
            ulysses_pad_size = 0

        inputs.pop("turn_ids", None)

        ctx = FSDPTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
            pad_length=mb_item.padding_length,
            ulysses_pad_size=ulysses_pad_size,
            trie_node=trie_node,
        )
        return inputs, ctx

    def _sp_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Differentiable all-gather over the SP group, concatenated along the
        # last dim.
        gathered = dist_F.all_gather(tensor, group=self.sp_group)
        return torch.cat(gathered, dim=-1)

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Returns per-token min and max logits over the vocab (detached,
        # float32), all-gathered across SP and trimmed of Ulysses padding when SP > 1.
        # KNOWN ISSUE: with TP the logits are vocab-sharded, so these are per-shard
        # extremes, not global ones.
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        if self.parallel_helper.sp_size > 1:
            vocab_min_logits = self._sp_all_gather(vocab_min_logits)
            vocab_max_logits = self._sp_all_gather(vocab_max_logits)
            if ulysses_pad_size > 0:
                vocab_min_logits = vocab_min_logits[:-ulysses_pad_size]
                vocab_max_logits = vocab_max_logits[:-ulysses_pad_size]
        return vocab_min_logits, vocab_max_logits

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Uses rolled_input_ids (Ulysses) or input_ids rolled left by one as
        # labels (squeezing a leading batch dim), computes temperature-scaled token
        # logprobs and entropy (TP-aware, chunked), then gathers across SP and trims
        # Ulysses padding. Returns (logprobs, entropy).
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
            chunk_size=self.config.logprobs_chunk_size,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            entropy = self._sp_all_gather(entropy)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
                entropy = entropy[:-ulysses_pad_size]
        return logprobs, entropy

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Same as _compute_logprobs_entropy but returns only logprobs.
        # Try to get rolled_input_ids (if Ulysses SP is enabled)
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        # inputs (padded_mbs) has batch dim (1, seq_len), squeeze to match logits (seq_len,)
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
            tp_group=self.parallel_helper.tp_group
            if self.parallel_helper.tp_size > 1
            else None,
            chunk_size=self.config.logprobs_chunk_size,
        )
        if self.parallel_helper.sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
        return logprobs

    def _compute_values(
        self,
        values: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Gathers critic values across SP and trims Ulysses padding when SP > 1;
        # returns values.
        if self.parallel_helper.sp_size > 1:
            values = self._sp_all_gather(values)
            if ulysses_pad_size > 0:
                values = values[:-ulysses_pad_size]
        return values

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Computes the scaled micro-batch loss. If the micro-batch's loss weight
        # is 0 returns zero loss connected to logits. Raises NotImplementedError for
        # critic + tree training. Actor: with tree training returns zero loss for empty
        # tries, else unpacks vocab stats and logprobs/entropy from the packed tree;
        # without it computes logprobs/entropy and vocab min/max and trims packing
        # padding; calls loss_fn(logprobs, entropy, mb_input, vocab_min_logits=...,
        # vocab_max_logits=...). Critic: computes values (trimmed) and calls
        # loss_fn(values, mb_input). Returns loss * local_weight / total_loss_weight *
        # loss_multiplier.
        """Compute logprobs/entropy and return scaled loss."""
        local_weight = loss_weight_fn(ctx.mb_input)
        if local_weight == 0:
            return logits.mean() * 0.0

        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                # Handle dummy trie (empty tree for DP synchronization)
                # When trie has no sequences, return zero loss with grad connection
                if ctx.trie_node is None or not ctx.trie_node.all_sequence_ids:
                    # Return zero loss that maintains gradient connection to logits
                    # This ensures backward() works correctly for FSDP synchronization
                    return logits.mean() * 0.0

                # For tree training, use gather_packed_tree_vocab_stats to properly
                # unpack vocab stats from tree structure back to per-sequence format.
                # This is necessary because the logits are in packed tree format where
                # multiple sequences share prefix positions.
                vocab_min_logits, vocab_max_logits = gather_packed_tree_vocab_stats(
                    logits, ctx.trie_node
                )
                logprobs, entropy = gather_packed_tree_logprobs_entropy(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                    chunk_size=self.config.logprobs_chunk_size,
                )
            else:
                logprobs, entropy = self._compute_logprobs_entropy(
                    logits, ctx.model_inputs, ctx.ulysses_pad_size
                )
                vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                    logits, ctx.ulysses_pad_size
                )
                if ctx.pad_length > 0:
                    logprobs = logprobs[: -ctx.pad_length]
                    entropy = entropy[: -ctx.pad_length]
                    vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                    vocab_max_logits = vocab_max_logits[: -ctx.pad_length]
            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = local_weight / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Forward output for one micro-batch: for actors, tree-unpacked per-
        # sequence logprobs dict (tree training) or logprobs trimmed of padding; for
        # critics, values trimmed of padding. Raises NotImplementedError for critic +
        # tree training.
        """Compute forward output (logprobs or values)."""
        if self.config.is_critic and self.enable_tree_training:
            raise NotImplementedError(
                "Tree training with critic model is not supported yet."
            )
        if not self.config.is_critic:
            if self.enable_tree_training:
                result = _gather_packed_tree_logprobs(
                    logits,
                    ctx.trie_node,
                    ctx.mb_input["input_ids"],
                    temperature=self.config.temperature,
                    tp_group=self.parallel_helper.tp_group
                    if self.parallel_helper.tp_size > 1
                    else None,
                    chunk_size=self.config.logprobs_chunk_size,
                )
                return result
            result = self._compute_logprobs(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
        else:
            result = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result


# =============================================================================
# Algorithm-specific FSDP Engines
# =============================================================================


class FSDPPPOActor(FSDPEngine):
    # [CLASS DOC]
    # FSDPEngine specialised as a PPO actor; delegates logp, advantage and PPO update
    # logic to an areal PPOActor helper bound to this engine, and adds a plain-list SFT
    # training entry.
    """PPO Actor implementation using FSDP backend."""

    def __init__(self, config: PPOActorConfig):
        # [FUNCTION DOC]
        # Initialises the engine and creates the PPOActor helper.
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> list[torch.Tensor] | None:
        # [FUNCTION DOC]
        # No-grad; forwards to PPOActor.compute_logp.
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # No-grad; forwards to PPOActor.compute_advantages.
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Forwards to PPOActor.ppo_update (one or more PPO minibatch updates).
        self.actor.ppo_update(*args, **kwargs)

    def sft_train_batch(self, data: list) -> dict:
        # [FUNCTION DOC]
        # Converts plain-list items (input_ids, attention_mask, loss_mask, cu_seqlens)
        # into tensors with a batch dim and runs train_batch with a summed next-token
        # NLL loss masked by loss_mask shifted left, weighted by the number of target
        # tokens. Returns train stats.
        import torch

        # Convert plain-list inputs (sent without tensors to avoid RPC partitioning)
        # Engine expects 2D tensors [batch, seqlen]
        tensor_data = []
        for item in data:
            tensor_data.append(
                {
                    "input_ids": torch.tensor(
                        item["input_ids"], dtype=torch.long
                    ).unsqueeze(0),
                    "attention_mask": torch.tensor(
                        item["attention_mask"], dtype=torch.float32
                    ).unsqueeze(0),
                    "loss_mask": torch.tensor(
                        item["loss_mask"], dtype=torch.float32
                    ).unsqueeze(0),
                    "cu_seqlens": torch.tensor(item["cu_seqlens"], dtype=torch.int32),
                }
            )

        def sft_loss_fn(logprobs, entropy, mb_input, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Returns the negative sum of logprobs[:-1] weighted by
            # loss_mask[1:] (all tokens if no mask).
            # logprobs[i] = log P(token[i+1] | context), so apply loss_mask shifted left
            loss_mask = mb_input.get("loss_mask", None)
            if loss_mask is not None:
                if loss_mask.ndim == 2:
                    loss_mask = loss_mask.squeeze(0)
                # loss_mask[i] marks token i as target; logprobs[i-1] predicts token i
                mask = loss_mask[1:]
                lp = logprobs[:-1]
                return -(lp * mask).sum()
            return -logprobs[:-1].sum()

        def sft_loss_weight_fn(input_data):
            # [FUNCTION DOC]
            # HELPER: Returns the count of target tokens (sum of loss_mask[1:]) or
            # sequence length minus one when there is no mask.
            loss_mask = input_data.get("loss_mask", None)
            if loss_mask is not None:
                if loss_mask.ndim == 2:
                    loss_mask = loss_mask.squeeze(0)
                return loss_mask[1:].sum()
            return torch.tensor(
                input_data["input_ids"].shape[-1] - 1, dtype=torch.float32
            )

        return self.train_batch(
            tensor_data, loss_fn=sft_loss_fn, loss_weight_fn=sft_loss_weight_fn
        )

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        # [FUNCTION DOC]
        # Class method returning a PPOActorControllerV2 when config._version is 'v2',
        # else a PPOActorController, both wrapping this engine class with the scheduler.
        if config._version == "v2":
            from areal.trainer.ppo.actor import PPOActorControllerV2

            return PPOActorControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPPPOCritic(FSDPEngine):
    # [CLASS DOC]
    # FSDPEngine specialised as a PPO critic, delegating to an areal PPOCritic helper.
    """PPO Critic implementation using FSDP backend."""

    def __init__(self, config: PPOCriticConfig):
        # [FUNCTION DOC]
        # Initialises the engine and creates the PPOCritic helper.
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        # [FUNCTION DOC]
        # No-grad; forwards to PPOCritic.compute_values.
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Forwards to PPOCritic.ppo_update.
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOCriticConfig, scheduler: Scheduler):
        # [FUNCTION DOC]
        # Returns PPOCriticControllerV2 for v2 configs, else PPOCriticController.
        if config._version == "v2":
            from areal.trainer.ppo.critic import PPOCriticControllerV2

            return PPOCriticControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPLMEngine(FSDPEngine):
    # [CLASS DOC]
    # FSDPEngine for supervised LM fine-tuning, delegating to an areal LMEngine helper.
    """Language model engine for SFT using FSDP backend."""

    def __init__(self, config: TrainEngineConfig):
        # [FUNCTION DOC]
        # Initialises the engine and creates the LMEngine helper.
        from areal.trainer.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        # [FUNCTION DOC]
        # Forwards to LMEngine.train_lm and returns its stats.
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        # [FUNCTION DOC]
        # Forwards to LMEngine.evaluate_lm.
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        # [FUNCTION DOC]
        # Returns LMControllerV2 for v2 configs, else LMController.
        if config._version == "v2":
            from areal.trainer.sft.lm_engine import LMControllerV2

            return LMControllerV2(
                train_engine=cls,
                config=config,
                scheduler=scheduler,
            )

        from areal.trainer.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPRWEngine(FSDPEngine):
    # [CLASS DOC]
    # FSDPEngine for reward-model training on paired data, delegating to an areal
    # RWEngine helper; forces micro-batch granularity 2 so pairs stay together.
    """Reward model engine using FSDP backend."""

    def __init__(self, config: TrainEngineConfig):
        # [FUNCTION DOC]
        # Initialises the engine and RWEngine helper; if mb_spec.granularity != 2, logs
        # a warning and replaces config with a deep copy whose granularity is 2.
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            logger = logging.getLogger("RWEngine")
            logger.warning("mb_spec.granularity must be 2 for reward modeling")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_rw(self, data):
        # [FUNCTION DOC]
        # Forwards to RWEngine.train_rw.
        return self.rw_engine.train_rw(data)

    def evaluate_rw(self, data):
        # [FUNCTION DOC]
        # Forwards to RWEngine.evaluate_rw.
        return self.rw_engine.evaluate_rw(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        # [FUNCTION DOC]
        # Returns RWControllerV2 for v2 configs, else RWController.
        if config._version == "v2":
            from areal.trainer.rw.rw_engine import RWControllerV2

            return RWControllerV2(train_engine=cls, config=config, scheduler=scheduler)

        from areal.trainer.rw.rw_engine import RWController

        return RWController(train_engine=cls, config=config, scheduler=scheduler)


class FSDPDPOEngine(FSDPEngine):
    # [CLASS DOC]
    # FSDPEngine for DPO training, delegating to an areal DPOEngine helper; forces
    # micro-batch granularity 2 so chosen/rejected pairs stay together.
    """DPO training engine using FSDP backend."""

    def __init__(self, config: DPOEngineConfig):
        # [FUNCTION DOC]
        # Initialises the engine and DPOEngine helper; if granularity != 2, warns and
        # deep-copies config with granularity 2.
        from copy import deepcopy

        from areal.trainer.dpo.dpo_engine import DPOEngine

        super().__init__(config)
        self.dpo_engine = DPOEngine(self)
        if self.config.mb_spec.granularity != 2:
            dpo_logger = logging.getLogger("DPOEngine")
            dpo_logger.warning("mb_spec.granularity must be 2 for DPO training")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_dpo(self, data):
        # [FUNCTION DOC]
        # Forwards to DPOEngine.train_dpo.
        return self.dpo_engine.train_dpo(data)

    def evaluate_dpo(self, data):
        # [FUNCTION DOC]
        # Forwards to DPOEngine.evaluate_dpo.
        return self.dpo_engine.evaluate_dpo(data)

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        # [FUNCTION DOC]
        # Forwards to DPOEngine.compute_logp (reference/policy logprobs per sample).
        return self.dpo_engine.compute_logp(data)

    @classmethod
    def as_controller(
        cls,
        config: DPOEngineConfig,
        scheduler: Scheduler,
    ):
        # [FUNCTION DOC]
        # Returns DPOControllerV2 for v2 configs, else DPOController.
        if config._version == "v2":
            from areal.trainer.dpo.dpo_engine import DPOControllerV2

            return DPOControllerV2(train_engine=cls, config=config, scheduler=scheduler)

        from areal.trainer.dpo.dpo_engine import DPOController

        return DPOController(train_engine=cls, config=config, scheduler=scheduler)
