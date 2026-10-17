# [MODULE DOC]
# AWEX training-side adapter for FSDP2 engines. Exposes per-rank parameter shard
# metadata (DTensor placements -> awex ParameterShardMeta), builds a local awex
# TransferPlan from metadata published in the gateway KV store, creates dedicated
# NCCL+Gloo process groups, and pushes local shards directly to inference ranks via NCCL
# P2P sends.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# pyright: reportMissingImports=false
import os
from typing import TYPE_CHECKING

import torch
from awex.meta.weight_meta import (
    ParameterMeta,
    ParameterReplicaMeta,
    ParameterShardMeta,
)
from awex.sharding.param_sharding import ShardingType
from awex.sharding.rank_info import RankInfo
from awex.transfer.nccl_comm import batch_send_recv, nccl_build_send_ops
from awex.transfer.transfer_plan import TransferPlan, TransferPlanBuilder
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from areal.engine.core.model import is_qwen_vl_model
from areal.utils import logging
from areal.v2.weight_update.awex import (
    awex_wu_use_group,
    fetch_kv_metadata,
)
from areal.v2.weight_update.nccl_group import (
    init_weights_update_group,
    setup_batch_isend_irecv,
)
from areal.v2.weight_update.training_adapter import (
    AwexTrainingAdapter,
)

if TYPE_CHECKING:
    from areal.engine.fsdp_engine import FSDPEngine

logger = logging.getLogger("AwexFSDPAdapter")


class AwexFSDPAdapter(AwexTrainingAdapter):
    # [CLASS DOC]
    # Implements AwexTrainingAdapter for an FSDPEngine. State: the wrapped engine, the
    # transfer plan, NCCL and Gloo weight-update groups, and this rank's global transfer
    # rank (all None until init_weight_update_group).
    """Awex training adapter wrapping FSDPEngine for shard-direct NCCL P2P updates."""

    def __init__(self, engine: FSDPEngine):
        # [FUNCTION DOC]
        # Stores the engine and initializes plan/group/rank fields to None. No side
        # effects.
        self._engine = engine
        self._transfer_plan: TransferPlan | None = None
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._transfer_rank: int | None = None

    @property
    def parallelism_strategy(self) -> dict:
        # [FUNCTION DOC]
        # Returns a dict describing parallelism: world_size, tp_size (size of the
        # 'sp_tp' mesh dim or 1), pp_size=1, dp_size from the engine, ep_size=1,
        # dp_replicated=False.
        mesh = self._engine.world_mesh
        dim_names = tuple(mesh.mesh_dim_names or ())
        tp_size = mesh.size(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 1

        return {
            "world_size": self._engine.world_size,
            "tp_size": tp_size,
            "pp_size": 1,
            "dp_size": self._engine.data_parallel_world_size,
            "ep_size": 1,
            "dp_replicated": False,
        }

    @property
    def _tie_word_embeddings(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the model config has tie_word_embeddings set (default False).
        return getattr(self._engine.model_config, "tie_word_embeddings", False)

    def get_weight_metadata(self) -> list[ParameterMeta]:
        # [FUNCTION DOC]
        # Builds one awex ParameterMeta per model parameter (HF-renamed via
        # _to_hf_name), skipping lm_head.weight when embeddings are tied. DTensor params
        # get shard meta derived from placements; plain tensors get an unsharded meta.
        # Global shape/numel/dtype come from the (DTensor global) tensor; a single
        # replica containing the single local shard is attached. Returns the list.
        rank_info = self._build_rank_info()
        metadata: list[ParameterMeta] = []

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if self._tie_word_embeddings and name == "lm_head.weight":
                continue
            tensor = param.data
            if isinstance(tensor, DTensor):
                shard_meta = self._extract_dtensor_shard_meta(name, tensor, rank_info)
                global_shape = tuple(tensor.shape)
                global_numel = int(tensor.numel())
                dtype = tensor.dtype
            else:
                shard_meta = self._extract_plain_shard_meta(name, tensor, rank_info)
                global_shape = tuple(tensor.shape)
                global_numel = int(tensor.numel())
                dtype = tensor.dtype

            replica = ParameterReplicaMeta(shards=[shard_meta])
            metadata.append(
                ParameterMeta(
                    name=name,
                    global_numel=global_numel,
                    global_shape=global_shape,
                    dtype=dtype,
                    shards=[shard_meta],
                    replicas=[replica],
                )
            )

        return metadata

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Returns a mapping HF-name -> local tensor for every parameter (minus tied
        # lm_head), optionally filtered to required_names (None/empty means all). For
        # DTensors returns the underlying local shard (no copy).
        required = set(required_names) if required_names else None
        local_params: dict[str, torch.Tensor] = {}

        for raw_name, param in self._engine.model.named_parameters():
            name = self._to_hf_name(raw_name)
            if self._tie_word_embeddings and name == "lm_head.weight":
                continue
            if required is not None and name not in required:
                continue

            tensor = param.data
            if isinstance(tensor, DTensor):
                local_params[name] = tensor._local_tensor
            else:
                local_params[name] = tensor

        return local_params

    def save_parameters(self, save_path: str, names: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Saves this rank's local shards (optionally filtered by names) to save_path via
        # torch.save after detaching and cloning to CPU. Side effect: file write.
        params = self.get_local_shard_parameters(names)
        cpu_params = {k: v.detach().cpu().clone() for k, v in params.items()}
        torch.save(cpu_params, save_path)

    def init_weight_update_group(
        self,
        pair_name: str,
        master_addr: str,
        master_port: int,
        transfer_rank: int,
        world_size: int,
        kv_store_url: str,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
    ) -> None:
        # [FUNCTION DOC]
        # Sets the transfer rank, fetches inference/training metadata for pair_name from
        # the KV store, builds a local TransferPlan with awex TransferPlanBuilder, sets
        # TORCHELASTIC_USE_AGENT_STORE=False in the environment, then creates an NCCL
        # group 'awex_<pair>' and a Gloo group 'awex_<pair>_gloo' over the given master
        # addr/port, rank and world_size. Logs on success.
        self._transfer_rank = transfer_rank

        infer_meta, train_meta = fetch_kv_metadata(kv_store_url, pair_name)

        builder = TransferPlanBuilder(
            infer_world_size=infer_world_size,
            train_world_size=train_world_size,
            num_infer_engines=num_engines,
        )
        self._transfer_plan = builder.build_local_transfer_plan(
            infer_meta, train_meta, global_transfer_rank=transfer_rank
        )

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        self._weights_update_group = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}",
            role="training",
        )
        self._weights_update_group_gloo = init_weights_update_group(
            master_address=master_addr,
            master_port=master_port,
            rank=transfer_rank,
            world_size=world_size,
            group_name=f"awex_{pair_name}_gloo",
            backend="gloo",
            role="training",
        )
        logger.info(
            "Initialized AWEX weight update groups for pair=%s role=training "
            "rank=%s world_size=%s nccl=awex_%s gloo=awex_%s_gloo",
            pair_name,
            transfer_rank,
            world_size,
            pair_name,
            pair_name,
        )

    def execute_weight_update(self, version: int) -> None:
        # [FUNCTION DOC]
        # Performs one weight push: version is ignored. Raises RuntimeError if plan,
        # NCCL group, Gloo group or transfer rank are not initialized. Builds NCCL send
        # ops for all local shards from the plan, executes them blocking (grouped per
        # awex_wu_use_group()), then barriers on the Gloo group.
        del version
        if self._transfer_plan is None:
            raise RuntimeError("Transfer plan is not initialized")
        if self._weights_update_group is None:
            raise RuntimeError("Weight update group is not initialized")
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")
        if self._transfer_rank is None:
            raise RuntimeError("Transfer rank is not initialized")

        params = self.get_local_shard_parameters()
        send_ops, _, _ = nccl_build_send_ops(
            params,
            self._transfer_plan,
            self._weights_update_group,
            copy_rank=self._transfer_rank,
        )
        batch_send_recv(
            send_ops=send_ops,
            recv_ops=[],
            blocking=True,
            use_group=awex_wu_use_group(),
        )
        torch.distributed.barrier(group=self._weights_update_group_gloo)

    def batch_isend_irecv(self, **kwargs) -> None:
        # [FUNCTION DOC]
        # Runs the awex batch_isend_irecv setup/test helper on the NCCL group with this
        # rank, passing world_size from kwargs (default 0), the Gloo group as barrier,
        # and the remaining kwargs (excluding world_size/barrier_group). Raises
        # RuntimeError if the Gloo group is missing.
        if self._weights_update_group_gloo is None:
            raise RuntimeError("Gloo weight update group is not initialized")
        setup_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("world_size", "barrier_group")
        }
        setup_batch_isend_irecv(
            self._weights_update_group,
            self._transfer_rank,
            kwargs.get("world_size", 0),
            barrier_group=self._weights_update_group_gloo,
            **setup_kwargs,
        )

    def teardown_weight_update_group(self) -> None:
        # [FUNCTION DOC]
        # Destroys the NCCL and Gloo groups if present and torch.distributed is
        # initialized, then resets groups, plan and transfer rank to None.
        if (
            self._weights_update_group is not None
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group(self._weights_update_group)
        if (
            self._weights_update_group_gloo is not None
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group(self._weights_update_group_gloo)
        self._weights_update_group = None
        self._weights_update_group_gloo = None
        self._transfer_plan = None
        self._transfer_rank = None

    def _to_hf_name(self, name: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Maps an engine parameter name to the HF checkpoint name. For Qwen-VL
        # vision models strips a leading 'model.model.' to 'model.' and then
        # 'model.visual.' to 'visual.'; otherwise returns the name unchanged.
        if self._engine.is_vision_model and is_qwen_vl_model(
            self._engine.model_config.model_type
        ):
            new_name = name
            if new_name.startswith("model.model."):
                new_name = new_name.replace("model.model.", "model.", 1)
            if new_name.startswith("model.visual."):
                new_name = new_name.replace("model.", "", 1)
            return new_name
        return name

    def _build_rank_info(self) -> RankInfo:
        # [FUNCTION DOC]
        # HELPER: Builds an awex RankInfo for this training rank: tp size/rank from mesh
        # dim 'sp_tp', cp size/rank from 'sp', dp from engine, pp/ep fixed to single,
        # local_rank from LOCAL_RANK env (fallback engine rank), engine_rank=0,
        # is_infer=False, cp_mode='none'.
        mesh = self._engine.world_mesh
        dim_names = tuple(mesh.mesh_dim_names or ())

        tp_size = mesh.size(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 1
        tp_rank = (
            mesh.get_local_rank(dim_names.index("sp_tp")) if "sp_tp" in dim_names else 0
        )
        cp_size = mesh.size(dim_names.index("sp")) if "sp" in dim_names else 1
        cp_rank = mesh.get_local_rank(dim_names.index("sp")) if "sp" in dim_names else 0
        local_rank = int(os.environ.get("LOCAL_RANK", self._engine.rank))

        return RankInfo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=0,
            pp_size=1,
            dp_size=self._engine.data_parallel_world_size,
            dp_rank=self._engine.dp_rank,
            ep_rank=0,
            ep_size=1,
            ep_tp_rank=0,
            ep_tp_size=1,
            attn_tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_dp_rank=self._engine.dp_rank,
            world_size=self._engine.world_size,
            global_rank=self._engine.rank,
            local_rank=local_rank,
            engine_rank=0,
            is_infer=False,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_mode="none",
        )

    @staticmethod
    def _compute_dtensor_offset(dtensor: DTensor) -> tuple[int, ...]:
        # [FUNCTION DOC]
        # HELPER: Computes the global offset of this rank's local DTensor shard by
        # walking placements; for each Shard placement divides the remaining dim size by
        # the mesh dim size and adds coordinate*chunk. Returns a tuple per dim.
        # KNOWN ISSUE: uses floor division, so uneven shards (DTensor pads/uneven
        # splits) produce wrong offsets for the last ranks.
        global_shape = tuple(dtensor.shape)
        placements = dtensor.placements
        mesh = dtensor.device_mesh

        offset = [0] * len(global_shape)
        remaining_shape = list(global_shape)

        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                mesh_size = mesh.size(mesh_dim)
                chunk_size = remaining_shape[shard_dim] // mesh_size
                coord = mesh.get_local_rank(mesh_dim)
                offset[shard_dim] += coord * chunk_size
                remaining_shape[shard_dim] = chunk_size

        return tuple(offset)

    @staticmethod
    def _extract_dtensor_sharding(dtensor: DTensor) -> tuple[int, int]:
        # [FUNCTION DOC]
        # HELPER: Aggregates Shard placements into total shard counts per tensor dim and
        # returns (dim with the most shards, its count); (0, 1) if not sharded.
        shard_info: dict[int, int] = {}
        for mesh_dim, placement in enumerate(dtensor.placements):
            if isinstance(placement, Shard):
                dim = placement.dim
                mesh_size = dtensor.device_mesh.size(mesh_dim)
                shard_info[dim] = shard_info.get(dim, 1) * mesh_size

        if not shard_info:
            return 0, 1

        primary_dim = max(shard_info.items(), key=lambda item: item[1])[0]
        return primary_dim, shard_info[primary_dim]

    def _extract_dtensor_shard_meta(
        self,
        name: str,
        dtensor: DTensor,
        rank_info: RankInfo,
    ) -> ParameterShardMeta:
        # [FUNCTION DOC]
        # HELPER: Builds a ParameterShardMeta for a DTensor using its local tensor's
        # shape/numel/dtype, rank_info fields, offset from _compute_dtensor_offset, and
        # TP_SHARDING if num_shards>1 else NO_SHARDING.
        local_tensor = dtensor._local_tensor
        sharding_dim, num_shards = self._extract_dtensor_sharding(dtensor)
        sharding_type = (
            ShardingType.TP_SHARDING if num_shards > 1 else ShardingType.NO_SHARDING
        )

        return ParameterShardMeta(
            tp_rank=rank_info.tp_rank,
            attn_tp_rank=rank_info.attn_tp_rank,
            pp_rank=rank_info.pp_rank,
            ep_rank=rank_info.ep_rank,
            ep_tp_rank=rank_info.ep_tp_rank,
            global_rank=rank_info.global_rank,
            world_size=rank_info.world_size,
            engine_rank=rank_info.engine_rank,
            cp_rank=rank_info.cp_rank,
            cp_size=rank_info.cp_size,
            cp_mode=rank_info.cp_mode,
            name=name,
            shape=tuple(local_tensor.shape),
            numel=int(local_tensor.numel()),
            dtype=local_tensor.dtype,
            global_offset=self._compute_dtensor_offset(dtensor),
            sharding_type=sharding_type,
            num_shards=num_shards,
            sharding_dim=sharding_dim,
        )

    def _extract_plain_shard_meta(
        self,
        name: str,
        tensor: torch.Tensor,
        rank_info: RankInfo,
    ) -> ParameterShardMeta:
        # [FUNCTION DOC]
        # HELPER: Builds an unsharded ParameterShardMeta for a plain tensor (zero
        # offset, NO_SHARDING, num_shards=1, sharding_dim=0) using rank_info fields.
        return ParameterShardMeta(
            tp_rank=rank_info.tp_rank,
            attn_tp_rank=rank_info.attn_tp_rank,
            pp_rank=rank_info.pp_rank,
            ep_rank=rank_info.ep_rank,
            ep_tp_rank=rank_info.ep_tp_rank,
            global_rank=rank_info.global_rank,
            world_size=rank_info.world_size,
            engine_rank=rank_info.engine_rank,
            cp_rank=rank_info.cp_rank,
            cp_size=rank_info.cp_size,
            cp_mode=rank_info.cp_mode,
            name=name,
            shape=tuple(tensor.shape),
            numel=int(tensor.numel()),
            dtype=tensor.dtype,
            global_offset=tuple([0] * len(tuple(tensor.shape))),
            sharding_type=ShardingType.NO_SHARDING,
            num_shards=1,
            sharding_dim=0,
        )
