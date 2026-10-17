# [MODULE DOC]
# Structural typing Protocol (runtime_checkable) defining the training-side AWEX weight-
# update adapter interface used by the worker HTTP endpoints and the gateway. Methods
# are abstract (bodies are '...').
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AwexTrainingAdapter(Protocol):
    # [CLASS DOC]
    # Protocol that training adapters (e.g. AwexFSDPAdapter/AwexMegatronAdapter)
    # implement: parallelism reporting, metadata export, local shard access, NCCL group
    # init/teardown, P2P update execution, colocated IPC update, and memory
    # release/resume.
    """Protocol for training-side weight update adapters."""

    @property
    def parallelism_strategy(self) -> dict:
        # [FUNCTION DOC]
        # Abstract: returns dict with world_size and tp/pp/dp/ep sizes (implementations
        # may add keys).
        """Report parallelism strategy.

        Returns dict with world_size, tp_size, pp_size, dp_size, ep_size.
        """
        ...

    def get_weight_metadata(self) -> list:
        # [FUNCTION DOC]
        # Abstract: returns list of awex ParameterMeta describing this worker's local
        # shards.
        """Extract this worker's parameter shard metadata in awex format.

        Returns list[ParameterMeta].
        """
        ...

    def get_local_shard_parameters(
        self, required_names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Abstract: returns HF-named local shard tensors, optionally filtered to
        # required_names.
        """Return local shard tensors in canonical HF naming."""
        ...

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
        # Abstract: fetch peer metadata from the KV store, build local transfer plan,
        # join the NCCL group for pair_name.
        """Pull peer meta from KV store, build local send plan, join NCCL group."""
        ...

    def execute_weight_update(self, version: int) -> None:
        # [FUNCTION DOC]
        # Abstract: run the cached P2P plan for version.
        """Execute cached local P2P send plan."""
        ...

    def batch_isend_irecv(self, **kwargs) -> None:
        # [FUNCTION DOC]
        # Abstract: run a batch isend/irecv connectivity test on the group.
        """Execute awex batch P2P send/recv operations."""
        ...

    def teardown_weight_update_group(self) -> None:
        # [FUNCTION DOC]
        # Abstract: destroy groups and clear cached state.
        """Destroy NCCL group and clear cached state."""
        ...

    def init_colocate_weight_update(
        self,
        pair_name: str,
        kv_store_url: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        num_engines: int,
        master_port: int,
        admin_api_key: str = "areal-admin-key",
        timeout_s: float = 120.0,
    ) -> None:
        # [FUNCTION DOC]
        # Abstract: prepare colocated (shared-GPU) weight transfer.
        """Register device info in KV store for colocated weight transfer."""
        ...

    def execute_colocate_weight_update(self, version: int) -> None:
        # [FUNCTION DOC]
        # Abstract: perform one colocated update for version.
        """Serialize weights via IPC and put to KV store."""
        ...

    def release_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Abstract: release GPU memory for the given tags.
        """Release GPU memory (optimizer/weights) for colocated mode."""
        ...

    def resume_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Abstract: resume GPU memory for the given tags.
        """Resume GPU memory occupation."""
        ...
