# [MODULE DOC]
# Weight synchronization from an ArchonEngine (training) to a rollout InferenceEngine.
# Two transports: (1) "xccl" - a dedicated NCCL/XCCL process group between a training
# rank (rank 0 of the group) and the inference workers, over which full (unsharded)
# parameters, converted to HF names, are broadcast in memory-bounded buckets; and
# (2) disk - save an HF checkpoint and tell inference to reload it. For SGLang with
# inference PP > 1, one group per PP stage is created (Megatron-style), each owned by the
# matching training PP-stage head. Mutable per-engine state lives in WeightSyncState.
# KNOWN ISSUE: Uses stdlib logging.getLogger(__name__) instead of areal.utils.logging with
# a PascalCase name as required by the project style rules.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import Future
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from areal.api import ParamSpec, WeightUpdateMeta
from areal.engine.core.distributed import init_custom_process_group
from areal.experimental.engine.archon_checkpoint import save_model_to_hf
from areal.infra.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.lock import DistributedLock
from areal.utils.network import find_free_ports, format_host_for_url, gethostip
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api import InferenceEngine
    from areal.experimental.engine.archon_engine import ArchonEngine

logger = logging.getLogger(__name__)


class WeightSyncState:
    # [CLASS DOC]
    # Plain mutable container for weight-update group state. Single-group fields: group_name
    # (default "update_weight_group_{pp_rank}"), master_addr/port, group, group_initialized.
    # Multi-group fields (SGLang PP > 1): group_names, groups, master_addrs, master_ports.
    """State container for weight synchronization.

    Attributes:
        group_initialized: Whether the weight update group has been initialized.
        group_name: Name of the NCCL group for weight updates (single-group mode).
        master_addr: Master address for TCP store initialization (single-group mode).
        master_port: Master port for TCP store initialization (single-group mode).
        group: The distributed process group for weight updates (single-group mode).
        group_names: List of group names for per-PP-rank groups (multi-group mode).
        groups: List of process groups for per-PP-rank groups (multi-group mode).
        master_addrs: List of master addresses for per-PP-rank groups (multi-group mode).
        master_ports: List of master ports for per-PP-rank groups (multi-group mode).
    """

    def __init__(self, pp_rank: int):
        # [FUNCTION DOC]
        # Initializes all fields to empty/default values; pp_rank only shapes the default name.
        self.group_initialized: bool = False
        self.group_name: str = f"update_weight_group_{pp_rank}"
        self.master_addr: str = ""
        self.master_port: int = 0
        self.group: dist.ProcessGroup | None = None
        # Multi-group state for sglang PP > 1
        self.group_names: list[str] = []
        self.groups: list[dist.ProcessGroup] = []
        self.master_addrs: list[str] = []
        self.master_ports: list[int] = []


def init_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # Asserts meta.type is "xccl", sets env TORCHELASTIC_USE_AGENT_STORE="False" so a new TCP
    # store can be created, derives gen_pp_size and backend from meta.gen_allocation (1/None
    # if absent), then calls the per-PP initializer for sglang with gen_pp_size > 1 and the
    # single-group initializer otherwise. Marks state.group_initialized True. Side effects:
    # env mutation, network/process-group creation, RPC to inference engine.
    """Initialize the weight update process group for XCCL synchronization.

    When the SGLang inference engine uses pipeline parallelism (gen_pp_size > 1),
    creates per-PP-rank NCCL groups so that each sglang PP stage receives only
    the parameters it owns; this requires train_pp_size == gen_pp_size. In all
    other cases -- gen_pp_size == 1, or vLLM with any gen_pp_size -- falls back
    to the single-group path, which spans all inference workers and supports
    differing training/inference PP sizes.
    """
    assert meta.type == "xccl"

    # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1
    gen_backend = meta.gen_allocation.backend if meta.gen_allocation else None
    logger.debug(
        f"[ArchonWeightSync] init_weight_update_group called: "
        f"gen_pp_size={gen_pp_size}, gen_backend={gen_backend}, "
        f"is_pp_head={engine.is_pipeline_parallel_head()}, "
        f"group_name={state.group_name}"
    )

    # The per-PP-rank path is specific to SGLang, whose rollout-side request
    # forms one NCCL group per PP stage and therefore requires
    # train_pp_size == gen_pp_size. vLLM joins a single flat group spanning all
    # inference workers, so it supports differing training/inference PP sizes.
    if gen_backend == "sglang" and gen_pp_size > 1:
        _init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)
    else:
        _init_single_weight_update_group(state, meta, engine)

    state.group_initialized = True
    logger.debug(
        f"[ArchonWeightSync] Weight update group(s) initialized successfully. "
        f"single_group={state.group is not None}, "
        f"multi_groups={len(state.groups)}"
    )


def _init_single_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # HELPER: On every rank picks this host's IP and a free port, stores them in state and in
    # meta (also group name). Only the PP head rank (asserts gen_allocation set), under the
    # engine lock, asks the rollout engine to join the group asynchronously, then creates a
    # custom process group (platform comm backend, world size = gen world size + 1, rank 0,
    # TCP init method, default timeout) and waits for the rollout future.
    # KNOWN ISSUE: Non-head ranks also call find_free_ports/gethostip and overwrite meta with
    # their own unused address; harmless only if meta is per-rank.
    """Original single-group initialization path (gen_pp_size == 1)."""
    state.master_addr = gethostip()
    state.master_port = find_free_ports(1)[0]

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    if engine.is_pipeline_parallel_head():
        assert meta.gen_allocation is not None

        with engine.engine_lock:
            fut = engine.rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            init_method = f"tcp://{format_host_for_url(meta.nccl_master_address)}:{meta.nccl_master_port}"
            engine.logger.info(
                f"[ArchonWeightSync] Initializing single weight update group: "
                f"type={meta.type}, init_method={init_method}, "
                f"group={meta.nccl_group_name}, world_size={gen_world_size + 1}"
            )
            state.group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=gen_world_size + 1,
                init_method=init_method,
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()
            engine.logger.debug(
                f"[ArchonWeightSync] Single weight update group initialized: "
                f"group={meta.nccl_group_name}"
            )


def _init_per_pp_weight_update_groups(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
    gen_pp_size: int,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Requires train pp size == gen_pp_size (ValueError otherwise). Non-PP-head ranks
    # just record all group names, set group_name for their pp rank, clear addr/port and
    # return. The head of training PP rank k creates group "update_weight_group_k" with world
    # size gen_world_size / gen_pp_size + 1 at a fresh host/port: it copies meta with these
    # fields, asks the rollout engine to join (async), creates the process group as rank 0 under
    # the engine lock, waits, and stores the group in both the list fields (one entry each) and
    # the single-group alias fields.
    # KNOWN ISSUE: Non-head ranks get group_names listing every stage, while heads get only
    # their own; code iterating group_names sees inconsistent contents across ranks.
    """Create one NCCL weight-update group per inference PP stage.

    Mirrors the Megatron pattern: each training PP-stage head
    (``dp=cp=tp=0``, ``pp_rank=k``) creates ONLY its own
    ``update_weight_group_{k}`` group, joined by the inference workers at
    sglang PP rank ``k`` plus this single training rank as ``rank=0``.

    Requirements:
        * ``train_pp_size == gen_pp_size``: every sglang PP stage has a
          matching training PP-stage head that holds the parameters for
          that stage.
    """
    assert meta.gen_allocation is not None

    train_pp_size = engine.parallel_dims.pp
    if train_pp_size != gen_pp_size:
        raise ValueError(
            f"Archon train_pp_size={train_pp_size} != gen_pp_size="
            f"{gen_pp_size}; train_pp_size == gen_pp_size is required for "
            f"per-PP-rank weight sync."
        )

    train_pp_rank = engine.pipeline_parallel_rank
    if not engine.is_pipeline_parallel_head():
        for pp_rank in range(gen_pp_size):
            state.group_names.append(f"update_weight_group_{pp_rank}")
        state.group_name = f"update_weight_group_{train_pp_rank}"
        state.master_addr = ""
        state.master_port = 0
        logger.debug(
            "[ArchonWeightSync] Non-PP-head rank (train_pp_rank=%d); "
            "skipping per-PP group creation.",
            train_pp_rank,
        )
        return

    gen_world_size = meta.gen_allocation.parallel.world_size
    # per_pp_world_size = workers at one PP stage = n_servers * tp_size.
    per_pp_world_size = gen_world_size // gen_pp_size

    pp_rank = train_pp_rank
    group_name = f"update_weight_group_{pp_rank}"
    master_addr = gethostip()
    master_port = find_free_ports(1)[0]
    world_size = per_pp_world_size + 1  # +1 for this training-PP-stage head

    # Populate both the single-group aliases AND the per-stage lists with
    # exactly one entry: the group this stage head owns. Destroy/cleanup
    # paths can iterate over ``state.group_names`` uniformly across single-
    # group and per-stage modes.
    state.group_names = [group_name]
    state.master_addrs = [master_addr]
    state.master_ports = [master_port]

    pp_meta = copy.copy(meta)
    pp_meta.nccl_master_address = master_addr
    pp_meta.nccl_master_port = master_port
    pp_meta.nccl_group_name = group_name

    init_method = f"tcp://{format_host_for_url(master_addr)}:{master_port}"
    engine.logger.info(
        f"[ArchonWeightSync] Initializing per-PP group: pp_rank={pp_rank}, "
        f"group={group_name}, init_method={init_method}, "
        f"world_size={world_size}"
    )

    with engine.engine_lock:
        fut = engine.rollout_engine.init_weights_update_group(pp_meta)

        group = init_custom_process_group(
            backend=current_platform.communication_backend,
            world_size=world_size,
            init_method=init_method,
            rank=0,
            group_name=group_name,
            timeout=DIST_GROUP_DEFAULT_TIMEOUT,
        )
        state.groups = [group]

        fut.result()

    engine.logger.debug(
        f"[ArchonWeightSync] Per-PP group initialized: pp_rank={pp_rank}, "
        f"group={group_name}"
    )

    # Single-group aliases point at this stage's own group (for backward
    # compatibility with single-group code paths).
    state.group_name = group_name
    state.master_addr = master_addr
    state.master_port = master_port
    state.group = group


def _get_full_tensor(param: nn.Parameter) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns the full, unsharded tensor for a parameter on the accelerator. DTensors
    # on device are gathered with full_tensor(); CPU-offloaded DTensors are rebuilt from their
    # local shard with the same mesh/placements and gathered (collective). Plain CPU tensors
    # are moved to the platform device; other tensors returned as-is.
    # KNOWN ISSUE: For a CPU DTensor the rebuilt DTensor still holds a CPU local shard, so the
    # gather runs on CPU tensors and the result is not moved to the accelerator.
    """Get full tensor from a parameter, handling DTensor and CPU offload."""
    tensor = param.data
    if isinstance(tensor, DTensor):
        if tensor.device.type != "cpu":
            return tensor.full_tensor()

        return DTensor.from_local(
            tensor.to_local(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        ).full_tensor()
    else:
        if tensor.device.type == "cpu":
            tensor = tensor.to(current_platform.device_type)
        return tensor


@trace_perf("archon_engine.update_weights_from_distributed", category="comm")
def update_weights_from_distributed(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # Perf-traced. Rank 0 pauses rollout generation; cpu_group barrier; if gen_pp_size > 1
    # uses the per-stage path, otherwise writes state's addr/port/group name into meta and
    # uses the single-group path; barrier; rank 0 resumes generation; device synchronize and a
    # final barrier. Must be called collectively by all training ranks.
    # KNOWN ISSUE: Branches on gen_pp_size alone, but init used the single-group path for vLLM
    # with gen_pp_size > 1. In that case state.groups is empty, so the per-stage path sends no
    # weights at all.
    """Update weights by broadcasting from training engine to inference engine.

    When the inference side uses pipeline parallelism (gen_pp_size > 1), each
    training PP-stage head broadcasts only its own stage's parameters via its
    own ``update_weight_group_{pp_rank}`` group (Megatron-style). Otherwise
    uses the original single-group path.
    """
    assert engine.rollout_engine is not None

    if dist.get_rank() == 0:
        engine.rollout_engine.pause_generation()

    dist.barrier(group=engine.cpu_group)

    gen_pp_size = meta.gen_allocation.parallel.pp_size if meta.gen_allocation else 1
    if gen_pp_size > 1:
        logger.debug(
            f"[ArchonWeightSync] update_weights: per-PP-stage mode, "
            f"train_pp_rank={engine.pipeline_parallel_rank}, "
            f"owned_groups={len(state.groups)}"
        )
        _update_weights_per_stage(state, meta, engine)
    else:
        logger.debug("[ArchonWeightSync] update_weights: single-group mode")
        meta.nccl_master_address = state.master_addr
        meta.nccl_master_port = state.master_port
        meta.nccl_group_name = state.group_name
        _update_weights_single_group(state, meta, engine)

    dist.barrier(group=engine.cpu_group)

    if dist.get_rank() == 0:
        engine.rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=engine.cpu_group)


def _update_weights_single_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Iterates all model parameters; every rank materializes the full tensor (needed
    # for the collective gather) but only the PP head continues. Each param is converted to HF
    # (name, tensor) pairs via the engine's state_dict_adapter if present. Pairs are batched
    # into buckets bounded by meta.weight_chunked_mem_mb (MiB); a full bucket is flushed with
    # _update_bucket_weights before adding. The trailing bucket is flushed at the end.
    """Original single-group weight update path."""
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)

        if not engine.is_pipeline_parallel_head():
            continue

        if engine.state_dict_adapter is not None:
            hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
        else:
            hf_pairs = [(name, tensor)]

        for hf_name, hf_tensor in hf_pairs:
            tensor_size = hf_tensor.numel() * hf_tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                _update_bucket_weights(
                    state,
                    meta,
                    engine.rollout_engine,
                    engine.engine_lock,
                    named_tensors,
                )
                buffer_size = 0
                named_tensors = []

            named_tensors.append((hf_name, hf_tensor))
            buffer_size += tensor_size

    if named_tensors:
        _update_bucket_weights(
            state, meta, engine.rollout_engine, engine.engine_lock, named_tensors
        )


def _update_weights_per_stage(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Same bucketing as the single-group path but only for ranks that are PP heads and
    # own a group; they broadcast this stage's parameters through their per-stage group using
    # a copy of meta carrying that group's addr/port/name. Non-owners still gather full
    # tensors (collectives) but send nothing. Logs the tensor count at DEBUG.
    """Per-PP-stage weight update path for sglang PP > 1.

    Each training PP-stage head broadcasts ONLY the parameters that live on
    its own stage, through its own ``update_weight_group_{pp_rank}`` group.
    Non-PP-head ranks (dp>0, tp>0, cp>0) own no group and skip this path.

    This mirrors the Megatron weight-sync layout: the inference side at PP
    rank ``k`` receives parameters from training PP rank ``k`` only.
    """
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024

    is_pp_head = engine.is_pipeline_parallel_head() and bool(state.groups)

    if is_pp_head:
        group = state.groups[0]
        group_name = state.group_names[0]
        master_addr = state.master_addrs[0]
        master_port = state.master_ports[0]

        pp_meta = copy.copy(meta)
        pp_meta.nccl_master_address = master_addr
        pp_meta.nccl_master_port = master_port
        pp_meta.nccl_group_name = group_name

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []
    total_tensors = 0

    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)

        if not is_pp_head:
            continue

        if engine.state_dict_adapter is not None:
            hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
        else:
            hf_pairs = [(name, tensor)]

        for hf_name, hf_tensor in hf_pairs:
            tensor_size = hf_tensor.numel() * hf_tensor.element_size()

            if tensor_size + buffer_size > weight_chunked_mem_size:
                _update_bucket_weights_multi_group(
                    group,
                    pp_meta,
                    engine.rollout_engine,
                    engine.engine_lock,
                    named_tensors,
                )
                buffer_size = 0
                named_tensors = []

            named_tensors.append((hf_name, hf_tensor))
            buffer_size += tensor_size
            total_tensors += 1

    if named_tensors:
        _update_bucket_weights_multi_group(
            group,
            pp_meta,
            engine.rollout_engine,
            engine.engine_lock,
            named_tensors,
        )

    if is_pp_head:
        logger.debug(
            f"[ArchonWeightSync] Per-stage broadcast finished: "
            f"group={group_name}, tensors={total_tensors}"
        )


def _update_bucket_weights(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    engine_lock: DistributedLock,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    # [FUNCTION DOC]
    # HELPER: No-op for an empty bucket. Under the engine lock, builds ParamSpecs (name,
    # shape, dtype string without the "torch." prefix), asks the rollout engine to receive
    # them (async future), issues async broadcasts from src 0 over state.group for each tensor,
    # waits for all, waits for the future, and clears the passed list in place.
    """Broadcast a bucket of weights to the inference engine (single-group mode)."""
    if not named_tensors:
        return

    with engine_lock:
        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        assert state.group is not None
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(tensor, src=0, group=state.group, async_op=True)
            )
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()


def _update_bucket_weights_multi_group(
    group: dist.ProcessGroup,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    engine_lock: DistributedLock,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Same as _update_bucket_weights but broadcasts over the explicitly given group.
    """Broadcast a bucket of weights to one per-PP-rank group (multi-group mode)."""
    if not named_tensors:
        return

    with engine_lock:
        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        fut = rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, tensor in named_tensors:
            handles.append(dist.broadcast(tensor, src=0, group=group, async_op=True))
        for handle in handles:
            handle.wait()

        fut.result()

        named_tensors.clear()


@trace_perf("archon_engine.update_weights_from_disk", category="io")
def update_weights_from_disk(
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    # [FUNCTION DOC]
    # Perf-traced. Rank 0 pauses generation and asks the rollout engine to load from
    # meta.path (async). All ranks save the model as an HF checkpoint to meta.path (asserted
    # non-None). Rank 0 then publishes a timestamp under the name_resolve key for this
    # experiment/trial/version (keepalive 120s), waits for the reload, and resumes
    # generation. Ends with device synchronize and cpu_group barrier.
    """Update weights by saving to disk and loading in inference engine."""
    fut: Future | None = None

    if dist.get_rank() == 0:
        engine.rollout_engine.pause_generation()
        fut = engine.rollout_engine.update_weights_from_disk(meta)

    assert meta.path is not None
    save_model_to_hf(engine, meta.path, engine.tokenizer, None)

    if dist.get_rank() == 0:
        update_name = names.update_weights_from_disk(
            engine.config.experiment_name,
            engine.config.trial_name,
            engine.get_version(),
        )
        name_resolve.add(
            update_name, str(datetime.now().timestamp()), keepalive_ttl=120
        )

        assert fut is not None
        fut.result()
        engine.rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=engine.cpu_group)
