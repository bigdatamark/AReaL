# [MODULE DOC]
# Single-controller for distributed training workers. The TrainController creates one
# training-engine worker per GPU rank through a Scheduler, sets torch.distributed env
# vars, initializes engines remotely, and exposes a TrainEngine-like API whose calls fan
# out over RPC: tensor-like inputs (lists of trajectory dicts containing torch tensors
# or RTensors) are partitioned across data-parallel groups by balanced token count and
# sent only to DP-head workers, other inputs are replicated, and results are merged back
# into the original order. Also bridges to a RolloutController for weight sync and batch
# preparation, and clears distributed RTensor batches.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    FinetuneSpec,
    Job,
    ParallelStrategy,
    SaveLoadMeta,
    Scheduler,
    TrainEngine,
    WeightUpdateMeta,
    Worker,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
from areal.infra.rpc.rtensor import RTensor, flatten_shard_ids
from areal.infra.utils.concurrent import run_async_task
from areal.utils import logging, stats_tracker
from areal.utils.data import make_dummy_eval_item
from areal.utils.network import find_free_ports
from areal.utils.seqpack import balanced_greedy_partition

from .rollout_callback import RolloutCallback
from .rollout_controller import RolloutController

logger = logging.getLogger("TrainController")


def _find_in_structure(obj: Any, type_: type) -> Any | None:
    # [FUNCTION DOC]
    # HELPER: Depth-first search through nested dicts, lists and tuples; returns the
    # first object that is an instance of type_, or None.
    """Find first instance of type_ in a nested structure."""
    if isinstance(obj, type_):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            result = _find_in_structure(v, type_)
            if result is not None:
                return result
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = _find_in_structure(item, type_)
            if result is not None:
                return result
    return None


def _is_tensor_like(obj: Any) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if obj (possibly nested) contains a torch Tensor or an RTensor.
    """Check if obj contains tensors or rtensors."""
    return (
        _find_in_structure(obj, torch.Tensor) is not None
        or _find_in_structure(obj, RTensor) is not None
    )


def _item_weight(d: dict[str, Any]) -> int:
    # [FUNCTION DOC]
    # HELPER: Token weight of one trajectory dict for load balancing: sum of the
    # attention_mask if it is a local tensor (forces a GPU-CPU sync if on GPU), element
    # count of a remote attention_mask's local data, otherwise the element count of the
    # first RTensor value or of the first local tensor with at least 2 dims; defaults to
    # 1.
    attn_mask = d.get("attention_mask")
    if isinstance(attn_mask, torch.Tensor):
        return int(attn_mask.sum().item())
    if isinstance(attn_mask, RTensor):
        return attn_mask.data.numel()
    # Fallback: first tensor's numel
    for v in d.values():
        if isinstance(v, RTensor):
            return v.data.numel()
        if isinstance(v, torch.Tensor) and v.ndim >= 2:
            return v.numel()
    return 1


def _dispatch_tensors(
    item_list: list[dict[str, Any]],
    dp_size: int,
    group_size: int = 1,
) -> tuple[list[list[dict[str, Any]]], list[list[int]]]:
    # [FUNCTION DOC]
    # HELPER: Partitions item_list across dp_size groups. Items are bundled into atomic
    # consecutive groups of group_size (never split; e.g. chosen/rejected pairs), group
    # weights are summed token weights, and a balanced greedy partition into dp_size
    # bins is computed. Returns (per-DP lists of items, per-DP lists of original item
    # indices). Raises ValueError if the item count is not divisible by group_size;
    # asserts every shard is a multiple of group_size.
    """Partition trajectories across DP groups by balanced token count.

    Args:
        group_size: number of consecutive items that form an atomic dispatch
            unit (e.g. 2 for chosen/rejected RW pairs).  Groups are never
            split across DP ranks.  ``group_size=1`` degenerates to per-item
            partitioning.
    """
    n = len(item_list)
    if n % group_size != 0:
        raise ValueError(
            f"item count ({n}) must be divisible by group_size ({group_size})"
        )

    token_weights = [_item_weight(d) for d in item_list]
    n_groups = n // group_size

    group_weights = [
        sum(token_weights[g * group_size + k] for k in range(group_size))
        for g in range(n_groups)
    ]

    gpart = balanced_greedy_partition(group_weights, K=dp_size)

    group_indices: list[list[int]] = []
    splits: list[list[dict[str, Any]]] = []
    for gidxs in gpart:
        item_idxs: list[int] = []
        items: list[dict[str, Any]] = []
        for g in gidxs:
            for k in range(group_size):
                idx = g * group_size + k
                item_idxs.append(idx)
                items.append(item_list[idx])
        group_indices.append(item_idxs)
        splits.append(items)

    assert all(len(s) % group_size == 0 for s in splits), (
        f"Post-dispatch invariant violated: shard sizes "
        f"{[len(s) for s in splits]} not all divisible by group_size={group_size}"
    )
    return splits, group_indices


def _pad_eval_batch(
    args: tuple[Any, ...], dp_size: int, group_size: int = 1
) -> tuple[Any, ...]:
    # [FUNCTION DOC]
    # HELPER: For evaluation dispatch, finds the first positional argument that is a
    # non-empty tensor-like list and pads it with dummy items (built from the first
    # item, with zero masks) up to a multiple of dp_size times group_size, logging at
    # INFO. Returns the args as a new tuple; other args are unchanged.
    """Pad the first tensor-like arg to a multiple of ``dp_size * group_size``.

    Called before dispatch for explicit evaluation controller paths so that
    ``balanced_greedy_partition`` always receives a divisible input.
    Dummy items have zero attention/loss masks and contribute nothing
    to metrics or loss.
    """
    result = list(args)
    pad_target = dp_size * group_size
    for i, arg in enumerate(result):
        if isinstance(arg, list) and arg and _is_tensor_like(arg):
            n = len(arg)
            pad_count = (-n) % pad_target
            if pad_count > 0:
                padded = list(arg)
                template = arg[0]
                padded.extend(make_dummy_eval_item(template) for _ in range(pad_count))
                result[i] = padded
                logger.info(
                    f"Eval dispatch: padded {pad_count} dummy items "
                    f"(total {len(padded)}) for dp_size={dp_size}"
                )
            break  # only pad the first tensor-like arg
    return tuple(result)


def _merge_tensors(
    results: list[Any], group_indices: list[list[int]]
) -> list[Any] | None:
    # [FUNCTION DOC]
    # HELPER: Merges per-DP results back to original order. Returns None if every group
    # result is None. A non-list group result is broadcast to every index of that group.
    # Asserts each group result length matches its index list, and places each item at
    # its original index. Returns the reordered list.
    """Flatten per-DP-group results and reorder to original trajectory order."""
    if all(r is None for r in results):
        return None

    n_total = sum(len(g) for g in group_indices)
    reordered: list[Any] = [None] * n_total
    for group_result, indices in zip(results, group_indices):
        if not isinstance(group_result, list):
            group_result = [group_result] * len(indices)
        assert len(group_result) == len(indices), (
            f"DP group returned {len(group_result)} results but expected {len(indices)}"
        )
        for result_item, orig_idx in zip(group_result, indices):
            reordered[orig_idx] = result_item
    return reordered


class TrainController:
    # [CLASS DOC]
    # Controller presenting a TrainEngine-like interface over remote workers. State:
    # engine class, TrainEngineConfig, scheduler, parsed train allocation (parallel
    # strategy), worker list, per-worker DP-head flags (only DP heads receive data;
    # other ranks get it by in-engine broadcast), worker role, whether this controller
    # created its own local process group, master address/port for torch.distributed,
    # and the connected RolloutController (None until connect_engine).
    """Controller for managing distributed training across multiple workers.

    This class orchestrates the lifecycle of training workers, handles data
    distribution across data-parallel groups, and provides a unified interface
    for training operations. It manages worker creation, engine initialization,
    and coordinates method calls across distributed workers.

    The controller automatically handles:
    - Worker creation and lifecycle management via scheduler
    - Data splitting across data-parallel groups
    - Result merging from multiple workers
    - Distributed training configuration (MASTER_ADDR, MASTER_PORT)
    """

    def __init__(
        self,
        train_engine: type[TrainEngine],
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ):
        # [FUNCTION DOC]
        # Stores engine class, config and scheduler, parses config.backend into a
        # ModelAllocation, and initializes empty worker lists, default role "default",
        # no own process group and no rollout. No I/O.
        self.train_engine = train_engine
        self.config = config
        self.scheduler = scheduler

        # Parse allocation from config.backend
        self.train_alloc = ModelAllocation.from_str(config.backend)

        self.workers: list[Worker] = []
        # Boolean list indicating which workers are data-parallel heads
        # Only DP head workers receive data slices; others get data via broadcast
        self.workers_is_dp_head: list[bool] = []

        self._worker_role: str = "default"
        self._own_process_group = False

        self.rollout: RolloutController = None

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        # [FUNCTION DOC]
        # If torch.distributed is not initialized in the controller process, creates a
        # single-process gloo group (rank 0, world size 1) on a free localhost TCP port
        # and records that it owns it. parallel_strategy is ignored. Real worker process
        # groups are created in initialize.
        # KNOWN ISSUE: the docstring calls this a dummy placeholder, but it does
        # initialize a local gloo process group.
        """Placeholder method for process group creation.

        This is a dummy method maintained for API compatibility. The actual
        process group creation happens during `initialize()` when engines are
        initialized on workers.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy | None, optional
            Parallel strategy configuration (currently unused), by default None
        """
        if not dist.is_initialized():
            port = find_free_ports(1)[0]
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://localhost:{port}",
                rank=0,
                world_size=1,
            )
            self._own_process_group = True

    @property
    def parallel_strategy(self) -> ParallelStrategy:
        # [FUNCTION DOC]
        # Property: the parallel strategy from the parsed backend allocation.
        """Parallel strategy derived from the parsed backend allocation."""
        return self.train_alloc.parallel

    @property
    def data_parallel_rank(self) -> int:
        # [FUNCTION DOC]
        # Property: always 0 (the controller acts as a single DP rank).
        return 0

    @property
    def data_parallel_world_size(self) -> int:
        # [FUNCTION DOC]
        # Property: always 1.
        return 1

    def is_data_parallel_head(self) -> bool:
        # [FUNCTION DOC]
        # Always True (the controller is its own DP head).
        return True

    @property
    def cpu_group(self):
        # [FUNCTION DOC]
        # Property: always None (no CPU process group on the controller).
        return None

    def initialize(
        self,
        role: str,
        ft_spec: FinetuneSpec,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Creates and initializes the training fleet. Inputs: role (worker role name),
        # ft_spec (FinetuneSpec), kwargs forwarded to each engine's initialize. Builds a
        # Job with world_size replicas using the configured scheduling specs and
        # strategy, creates and fetches workers, takes MASTER_ADDR from rank 0's IP and
        # MASTER_PORT from its second engine port (or second worker port if none),
        # creates engines (with env vars) and initializes them synchronously, then
        # queries which workers are DP heads. Logs progress.
        """Initialize environments for distributed training and load models.

        Parameters
        ----------
        role : str
            Role identifier for the workers
        ft_spec : FinetuneSpec
            Finetune specification for model initialization
        **kwargs
            Additional keyword arguments passed to engine initialization
        """
        # Store configuration
        self._worker_role = role

        world_size = self.train_alloc.parallel.world_size

        # Create job specification for scheduler
        # Convert scheduling_spec tuple to list for scheduler compatibility
        # The scheduler will handle task replication across workers if needed
        job = Job(
            replicas=world_size,
            tasks=list(self.config.scheduling_spec),
            scheduling_strategy=self.config.scheduling_strategy,
            role=self._worker_role,
        )

        # Create workers via scheduler
        logger.info("Creating workers via scheduler...")
        worker_ids = self.scheduler.create_workers(job=job)
        logger.info(f"Workers created: {worker_ids}")

        # Wait for workers to be ready
        logger.info("Waiting for workers to be ready...")
        self.workers = self.scheduler.get_workers(role=job.role)
        logger.info(f"Workers ready: {[w.id for w in self.workers]}")

        # Determine distributed training master address and port from rank 0 worker
        # These are used for PyTorch distributed initialization across workers
        # Prefer engine_ports[1] if available, fallback to worker_ports[1]
        rank0_worker = self.workers[0]
        if rank0_worker.engine_ports:
            self._master_port = int(rank0_worker.engine_ports[1])
        else:
            self._master_port = int(rank0_worker.worker_ports[1])
        self._master_addr = rank0_worker.ip

        logger.info(
            f"Distributed training: MASTER_ADDR={self._master_addr}, MASTER_PORT={self._master_port}"
        )

        # Construct engine class import path for dynamic loading on workers
        # Workers will import and instantiate the engine class using this path
        engine_class = self.train_engine

        # Create and initialize engines on workers
        run_async_task(
            self._async_create_engines,
            f"{engine_class.__module__}.{engine_class.__name__}",
        )
        run_async_task(self._async_initialize_engines, ft_spec, **kwargs)

        # Identify DP head workers
        self._identify_dp_heads()
        logger.info("TrainController initialization complete")

    def _engine_name(self, rank: int) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns the engine name "<role>/<rank>".
        """Generate engine name for a worker rank.

        Engine names follow the "role/index" format (e.g., "actor/0", "ref/1").
        """
        return f"{self._worker_role}/{rank}"

    async def _async_create_engines(self, engine: str):
        # [FUNCTION DOC]
        # HELPER: For every worker concurrently, sets RANK, WORLD_SIZE, MASTER_ADDR,
        # MASTER_PORT and LOCAL_RANK=0 (one GPU per process) env vars and then creates
        # the engine (by import path string) with the config.
        """Create engine instances on all workers. Sets distributed env vars before creation."""
        logger.info("Creating engines on workers...")

        async def _setup_worker(worker: Worker, rank: int):
            # [FUNCTION DOC]
            # HELPER: Sets the distributed env for one worker and creates its engine.
            env = {
                "RANK": str(rank),
                "WORLD_SIZE": str(len(self.workers)),
                "MASTER_ADDR": str(self._master_addr),
                "MASTER_PORT": str(self._master_port),
                "LOCAL_RANK": "0",  # NOTE: local rank is always 0 while each process use only one GPU
            }
            await self.scheduler.set_worker_env(worker.id, env)
            await self.scheduler.create_engine(
                worker_id=worker.id,
                engine=engine,
                engine_name=self._engine_name(rank),
                config=self.config,
            )

        tasks = [
            _setup_worker(worker, rank) for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("Engines created on all workers!")

    async def _async_initialize_engines(self, ft_spec: FinetuneSpec, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Two phases over all workers concurrently: create_process_group with
        # the parallel strategy, then initialize with ft_spec and kwargs (model load,
        # optimizer setup).
        """Initialize engines: create process groups, then load models and setup optimizers."""
        logger.info("Calling engine initialization...")
        # Phase 1: Create process groups for distributed training
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="create_process_group",
                engine_name=self._engine_name(rank),
                parallel_strategy=self.parallel_strategy,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        # Phase 2: Initialize engines (load models, setup optimizers, etc.)
        tasks = [
            self.scheduler.async_call_engine(
                worker_id=worker.id,
                method="initialize",
                engine_name=self._engine_name(rank),
                ft_spec=ft_spec,
                **kwargs,
            )
            for rank, worker in enumerate(self.workers)
        ]
        await asyncio.gather(*tasks)
        logger.info("All engines are initialized!")

    def _identify_dp_heads(self):
        # [FUNCTION DOC]
        # HELPER: Queries is_data_parallel_head on every engine and stores the boolean
        # list in workers_is_dp_head.
        """Query workers to identify DP heads. Stores result in self.workers_is_dp_head."""
        logger.info("Identifying DP head workers...")

        async def _get_dp_head():
            # [FUNCTION DOC]
            # HELPER: Async gather of the is_data_parallel_head calls.
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="is_data_parallel_head",
                    engine_name=self._engine_name(rank),
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        self.workers_is_dp_head = run_async_task(_get_dp_head)

    def destroy(self):
        # [FUNCTION DOC]
        # Tears down: if workers exist, calls destroy on every engine concurrently
        # (exceptions collected and logged as warnings per rank; other errors logged),
        # then deletes workers in reverse rank order (so rank 0, the TCPStore owner,
        # dies last), clears worker lists, and destroys the controller's own local
        # process group if it created one.
        """Destroy the controller and release GPU memory of models.

        Cleans up all resources including workers, engines, and internal state.

        The teardown order is carefully chosen to avoid a noisy
        ``TCPStore.recvValue failed`` warning from NCCL's HeartbeatMonitor
        on non-zero ranks:

        1. Remote engines' ``destroy()`` runs first so that every rank calls
           ``dist.destroy_process_group()`` after a CPU barrier. This
           guarantees all ranks finish NCCL abort together before any store
           shuts down.
        2. Workers are killed in reverse rank order so that rank-0 (owner
           of the global TCPStore server) receives SIGTERM last. This
           avoids the short window where non-zero ranks' HeartbeatMonitor
           threads poll a store whose TCP listener has already been closed.
        """
        logger.info("Destroying TrainController...")

        # First destroy engines to release GPU memory
        if self.workers:
            logger.info("Destroying engines on all workers...")
            try:

                async def _destroy_all_engines():
                    # [FUNCTION DOC]
                    # HELPER: Concurrently calls destroy on every engine, returning
                    # results or exceptions.
                    tasks = [
                        self.scheduler.async_call_engine(
                            worker_id=worker.id,
                            method="destroy",
                            engine_name=self._engine_name(rank),
                        )
                        for rank, worker in enumerate(self.workers)
                    ]
                    return await asyncio.gather(*tasks, return_exceptions=True)

                results = run_async_task(_destroy_all_engines)
                # Surface per-worker failures instead of silently swallowing them.
                for rank, res in enumerate(results or []):
                    if isinstance(res, BaseException):
                        logger.warning(
                            f"Engine destroy on rank {rank} raised "
                            f"{type(res).__name__}: {res}"
                        )
                logger.info("Engines destroyed")
            except Exception as e:
                logger.error(f"Error destroying engines: {e}")

        # Then delete workers via scheduler. Pass reverse_order=True so
        # that rank-0 (TCPStore owner) is killed last. All in-tree
        # Scheduler implementations (Local/Ray/Slurm) accept this kwarg;
        # third-party subclasses that override ``delete_workers`` must
        # adopt the same signature.
        try:
            logger.info("Deleting all workers (reverse rank order)...")
            self.scheduler.delete_workers(role=self._worker_role, reverse_order=True)
            logger.info("Workers deleted")
        except Exception as e:
            logger.error(f"Error deleting workers: {e}")

        # Clear worker lists
        self.workers.clear()
        self.workers_is_dp_head.clear()

        if dist.is_initialized() and self._own_process_group:
            dist.destroy_process_group()
        logger.info("TrainController destroyed")

    def _custom_function_call(
        self,
        method: str,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Synchronous generic RPC: prepares DP dispatch of args/kwargs (popping
        # an optional group_size kwarg), calls method on all workers with optional
        # rpc_meta, and collects results (merged list for tensor dispatch, first DP
        # head's result otherwise).
        """Dispatch method call to workers via the appropriate path."""
        dp_args, dp_kwargs, group_indices = self._prepare_dispatch(*args, **kwargs)
        results = run_async_task(
            self._call_workers, method, dp_args, dp_kwargs, rpc_meta=rpc_meta
        )
        return self._collect_results(results, group_indices)

    async def _async_custom_function_call(
        self,
        method: str,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Async counterpart of _custom_function_call.
        """Async version of _custom_function_call."""
        dp_args, dp_kwargs, group_indices = self._prepare_dispatch(*args, **kwargs)
        results = await self._call_workers(
            method, dp_args, dp_kwargs, rpc_meta=rpc_meta
        )
        return self._collect_results(results, group_indices)

    def _pad_eval_dispatch_args(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        group_size: int,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Pads the first tensor-like positional arg to a multiple of dp_size
        # times group_size (eval path) and returns (padded args, a shallow copy of
        # kwargs).
        """Pad eval batches for explicit algorithm-level evaluation dispatch."""
        kwargs = dict(kwargs)
        args = _pad_eval_batch(
            args, self.parallel_strategy.dp_size, group_size=group_size
        )
        return args, kwargs

    def _prepare_dispatch(
        self, *args, **kwargs
    ) -> tuple[list[list[Any]], dict[str, list[Any]], list[list[int]] | None]:
        # [FUNCTION DOC]
        # HELPER: Pops group_size (default 1) from kwargs; if any arg or kwarg is
        # tensor-like, partitions inputs across DP groups, otherwise replicates them.
        # Returns (per-arg lists of DP slices, per-kwarg lists of DP slices, group
        # indices or None).
        """Route to tensor or scalar dispatch based on input type.

        Returns (dp_split_args, dp_split_kwargs, group_indices).
        group_indices is non-None only for tensor dispatches.
        """
        group_size = kwargs.pop("group_size", 1)
        if _is_tensor_like(args) or _is_tensor_like(kwargs):
            return self._partition_inputs(group_size, *args, **kwargs)
        return self._replicate_inputs(*args, **kwargs)

    def _partition_inputs(
        self, group_size: int, /, *args, **kwargs
    ) -> tuple[list[list[Any]], dict[str, list[Any]], list[list[int]]]:
        # [FUNCTION DOC]
        # HELPER: Splits tensor-like args/kwargs across DP groups. The first tensor-like
        # value determines the partition (via balanced token dispatch); later tensor-
        # like values are split with the same indices (so they must be aligned lists);
        # non-tensor values are replicated dp_size times. Asserts at least one value was
        # tensor-like. Returns (dp args, dp kwargs, group indices).
        """Partition tensor args across DP groups; replicate others."""
        dp_size = self.parallel_strategy.dp_size
        group_indices: list[list[int]] | None = None

        def _split(item: Any) -> list[Any]:
            # [FUNCTION DOC]
            # HELPER: Splits one value per the rules above, recording group indices on
            # first use.
            nonlocal group_indices
            if _is_tensor_like(item):
                if group_indices is None:
                    splits, group_indices = _dispatch_tensors(
                        item, dp_size, group_size=group_size
                    )
                    return splits
                return [[item[i] for i in idxs] for idxs in group_indices]
            return [item] * dp_size

        dp_args = [_split(a) for a in args]
        dp_kwargs = {k: _split(v) for k, v in kwargs.items()}
        assert group_indices is not None
        return dp_args, dp_kwargs, group_indices

    def _replicate_inputs(
        self, *args, **kwargs
    ) -> tuple[list[list[Any]], dict[str, list[Any]], None]:
        # [FUNCTION DOC]
        # HELPER: Replicates every arg and kwarg dp_size times; returns group indices
        # None.
        """Replicate all args to every DP group."""
        dp_size = self.parallel_strategy.dp_size
        dp_args = [[a] * dp_size for a in args]
        dp_kwargs = {k: [v] * dp_size for k, v in kwargs.items()}
        return dp_args, dp_kwargs, None

    async def _call_workers(
        self,
        method: str,
        dp_split_args: list[list[Any]],
        dp_split_kwargs: dict[str, list[Any]],
        rpc_meta: dict[str, Any] | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Calls method on every worker concurrently. DP-head workers receive the
        # next DP slice of each arg/kwarg in order; non-head workers receive no data
        # args. Passes rpc_meta through. Returns the list of per-worker results.
        """Send dispatched inputs to workers. DP heads get slices, others empty."""
        tasks = []
        dp_idx = 0
        for idx, worker in enumerate(self.workers):
            if self.workers_is_dp_head[idx]:
                worker_args = [splits[dp_idx] for splits in dp_split_args]
                worker_kwargs = {
                    k: splits[dp_idx] for k, splits in dp_split_kwargs.items()
                }
                dp_idx += 1
            else:
                worker_args = []
                worker_kwargs = {}

            tasks.append(
                self.scheduler.async_call_engine(
                    worker.id,
                    method,
                    self._engine_name(idx),
                    *worker_args,
                    rpc_meta=rpc_meta,
                    **worker_kwargs,
                )
            )
        return await asyncio.gather(*tasks)

    def _collect_results(
        self, results: list[Any], group_indices: list[list[int]] | None
    ) -> Any:
        # [FUNCTION DOC]
        # HELPER: Keeps only DP-head results; returns them merged into original order if
        # group_indices is given, else the first DP head's result.
        """Filter to DP heads, then reorder (tensor) or merge (scalar)."""
        results = [r for idx, r in enumerate(results) if self.workers_is_dp_head[idx]]
        if group_indices is not None:
            return _merge_tensors(results, group_indices)
        return results[0]

    def connect_engine(self, rollout: RolloutController, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # Connects a RolloutController for weight updates: warns if replacing a
        # different one, stores it, and sends each training engine a RolloutCallback
        # pointing at the rollout controller's callback address, together with meta, via
        # a connect_engine RPC.
        if self.rollout is not None and self.rollout != rollout:
            logger.warning(
                f"Connected rollout controller changed from {self.rollout} to {rollout}."
            )
        self.rollout = rollout

        # Register a callback engine on train engines
        # RolloutCallback is a dataclass and can be serialized
        engine = RolloutCallback(controller_addr=rollout.callback_addr)
        self._custom_function_call("connect_engine", engine=engine, meta=meta)

    def export_stats(self):
        # [FUNCTION DOC]
        # Returns the controller-local stats tracker export updated with the stats from
        # the first DP head (workers are assumed to have already all-reduced their
        # stats).
        """Export training statistics from all workers.

        Collects statistics from all workers. The statistics are assumed to be
        already aggregated and synchronized (e.g., via all-reduce operations),
        so only the first result is returned.

        Returns
        -------
        dict[str, Any]
            Training statistics dictionary
        """
        # Statistics have been aggregated and synchronized across workers
        # All results should be identical, so return the first one
        stats = stats_tracker.export_all()
        stats.update(self._custom_function_call("export_stats"))
        return stats

    # ==================== ENGINE RPC WRAPPERS ====================
    # Note: Methods like train_batch, forward, etc. are not implemented here.
    # They are expected to be called directly via _custom_function_call in
    # specific training scenarios (PPO, SFT, etc.) where the appropriate
    # loss functions and data processing are handled.
    def train(self, mode: bool = True):
        # [FUNCTION DOC]
        # RPC: sets training mode (mode True) or eval mode on all engines; returns self
        # for chaining.
        """Set the engine to training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the engine to training mode, by default True

        Returns
        -------
        TrainController
            Returns self for method chaining
        """
        self._custom_function_call("train", mode)
        return self

    def eval(self):
        # [FUNCTION DOC]
        # Equivalent to train(False); returns self.
        """Set the engine to evaluation mode.

        This is a convenience method that calls `self.train(False)`.

        Returns
        -------
        TrainController
            Returns self for method chaining
        """
        return self.train(False)

    def set_version(self, version: int):
        # [FUNCTION DOC]
        # RPC: sets the weight version on all engines.
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        self._custom_function_call("set_version", version)

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # RPC: returns the weight version reported by the first DP head.
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        return self._custom_function_call("get_version")

    def get_lora_adapter_info(self) -> dict[str, list[int]]:
        # [FUNCTION DOC]
        # RPC: fetches LoRA adapter info. Returns results[0] if the RPC result is
        # truthy, else an empty dict.
        # KNOWN ISSUE: a non-tensor RPC already returns the first DP head's result (a
        # dict), so indexing it with 0 looks up key 0 in that dict and will likely raise
        # KeyError.
        """Get LoRA adapter parameter names and shapes from worker rank 0."""
        results = self._custom_function_call("get_lora_adapter_info")
        if results:
            return results[0]
        return {}

    def save(self, meta: SaveLoadMeta):
        # [FUNCTION DOC]
        # RPC: saves model/optimizer state according to meta on all engines.
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        self._custom_function_call("save", meta)

    def load(self, meta: SaveLoadMeta):
        # [FUNCTION DOC]
        # RPC: loads model/optimizer state according to meta on all engines.
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        self._custom_function_call("load", meta)

    def init_awex_adapter(self, meta_server_addr: str | None = None):
        # [FUNCTION DOC]
        # RPC: creates the awex weight-sync adapter on all engines with an optional
        # meta-server address.
        """Create awex adapter early for selective memory management."""
        self._custom_function_call(
            "init_awex_adapter", meta_server_addr=meta_server_addr
        )

    def step_lr_scheduler(self):
        # [FUNCTION DOC]
        # RPC: steps the learning-rate scheduler on all engines.
        """Step the learning rate scheduler.

        Since PPO uses minibatch updates, this method should be called periodically
        (e.g., once per PPO step). It is separated from train_batch to allow
        for more flexible learning rate scheduling.
        """
        self._custom_function_call("step_lr_scheduler")

    def update_weights(self, meta: WeightUpdateMeta):
        # [FUNCTION DOC]
        # Requires a connected rollout controller (RuntimeError otherwise), then RPC
        # update_weights with meta on all engines, which push weights to the inference
        # side.
        self._check_rollout_engine_connected()
        self._custom_function_call("update_weights", meta=meta)

    def offload(self) -> None:
        # [FUNCTION DOC]
        # RPC: offloads model parameters to CPU on all engines.
        """Offload model parameters to CPU across all train workers."""
        self._custom_function_call("offload")

    def onload(self) -> None:
        # [FUNCTION DOC]
        # RPC: onloads model parameters back to GPU on all engines.
        """Onload model parameters to GPU across all train workers."""
        self._custom_function_call("onload")

    def get_device_stats(self):
        # [FUNCTION DOC]
        # RPC: returns device memory stats from the first DP head.
        return self._custom_function_call("get_device_stats")

    def start_memory_profile(self, max_entries: int = 100000):
        # [FUNCTION DOC]
        # RPC: starts CUDA memory history recording on all engines with max_entries;
        # returns the first DP head's result.
        return self._custom_function_call("start_memory_profile", max_entries)

    def stop_memory_profile(self, snapshot_dir: str):
        # [FUNCTION DOC]
        # RPC: stops memory profiling and dumps snapshots into snapshot_dir; returns the
        # first DP head's result.
        return self._custom_function_call("stop_memory_profile", snapshot_dir)

    def config_perf_tracer(self, config: PerfTracerConfig, role: str) -> None:
        # [FUNCTION DOC]
        # Configures the performance tracer on every worker concurrently with its rank,
        # role and config (called on all workers, not only DP heads).
        async def _call():
            # [FUNCTION DOC]
            # HELPER: Async gather of per-worker config_perf_tracer calls.
            tasks = [
                self.scheduler.async_call_engine(
                    worker_id=worker.id,
                    method="config_perf_tracer",
                    engine_name=self._engine_name(rank),
                    rank=rank,
                    role=role,
                    config=config,
                )
                for rank, worker in enumerate(self.workers)
            ]
            return await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        # [FUNCTION DOC]
        # RPC: asks engines to save perf traces for an optional step, optionally forced.
        self._custom_function_call("save_perf_tracer", step=step, force=force)

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Delegates to the connected RolloutController's prepare_batch with the same
        # arguments and returns its trajectories.
        # KNOWN ISSUE: does not check that a rollout controller is connected, so it
        # raises AttributeError on None instead of the clearer RuntimeError.
        return self.rollout.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Delegates to the connected RolloutController's rollout_batch and returns its
        # trajectories (no connection check either).
        return self.rollout.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def _check_rollout_engine_connected(self):
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError if no rollout controller is connected.
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    async def _async_clear_batches(self, *targets: dict[str, RTensor]):
        # [FUNCTION DOC]
        # HELPER: Groups the shard IDs referenced by the given RTensor batch dicts by
        # storage node and concurrently sends a clear request to each node; exceptions
        # are swallowed. No-op if there are no shards.
        """Extract shard IDs and clear tensors on each worker.

        HTTP DELETEs to each storage node's ``/data/clear`` — this evicts
        ``_storage`` (mandatory, otherwise HTTP storage grows unboundedly)
        and, via :func:`rtensor.remove`, also pops the storage owner's own
        ``_fetch_buffer`` (covers storage-owner-as-consumer). See #1209.
        """
        shards_by_node = RTensor.collect_shards(targets)

        if not shards_by_node:
            return

        await asyncio.gather(
            *[RTensor.clear_node(addr, sids) for addr, sids in shards_by_node.items()],
            return_exceptions=True,
        )

    def clear_batches(self, *targets: dict[str, RTensor]):
        # [FUNCTION DOC]
        # Frees distributed batch shards. First clears storage on each storage node;
        # then, if any shard IDs exist, sends the flat list of IDs (not tensor-like, so
        # replicated) to every DP head with broadcast disabled so each drains its local
        # fetch buffer. Finally, best-effort, queries fetch_buffer_stats from DP head 0
        # and logs a WARNING if entries remain (possible leak) or DEBUG if clean; an RPC
        # failure there is only logged at DEBUG.
        """Clear distributed batch shards from workers to free memory.

        Two fan-outs — see areal-project/AReaL#1209:

        1. ``_async_clear_batches``: HTTP DELETE to each storage node,
           dropping ``_storage`` entries (and the owner's ``_fetch_buffer``
           via :func:`rtensor.remove`).
        2. Replicated RPC to every DP head so cross-node consumer workers
           drain their local ``_fetch_buffer``. Payload is a flat
           ``list[str]`` of shard IDs — sending IDs (not RTensors)
           side-steps the RPC's ``localize`` pass (no RTensor → no
           re-fetch), and ``_is_tensor_like(list[str]) == False`` routes
           dispatch through ``_replicate_inputs`` so every head sees the
           full sid set.

        After the second fan-out, a ``fetch_buffer_stats`` RPC logs the
        drain result — WARNING on leak, DEBUG when clean.
        """
        run_async_task(self._async_clear_batches, *targets)
        sids = flatten_shard_ids(targets)
        if not sids:
            return
        # broadcast=False → purely local per-head op (no NCCL collective).
        # list[str] is not tensor-like → _replicate_inputs copies the full
        # sid set to every DP head.
        self._custom_function_call("clear_batches", sids, rpc_meta={"broadcast": False})
        # Always observe post-drain state. _custom_function_call returns
        # the first DP head's stats (scalar dispatch collapses via
        # _collect_results[0]); all heads are symmetric in steady state,
        # so head 0 is a sufficient leak signal. Best-effort: an RPC
        # failure here is observability-only and must not break training.
        try:
            stats = self._custom_function_call(
                "fetch_buffer_stats", rpc_meta={"broadcast": False}
            )
        except Exception as e:
            logger.debug(
                "fetch_buffer_stats RPC failed (observability only, role=%s): %s",
                self._worker_role,
                e,
            )
            return
        n_entries = stats.get("num_entries", 0) if isinstance(stats, dict) else 0
        if n_entries > 0:
            logger.warning(
                "clear_batches: _fetch_buffer non-empty on DP head 0 "
                "(role=%s, num_entries=%d) — possible leak, see #1209",
                self._worker_role,
                n_entries,
            )
        else:
            logger.debug(
                "clear_batches: _fetch_buffer drained on DP head 0 (role=%s)",
                self._worker_role,
            )
