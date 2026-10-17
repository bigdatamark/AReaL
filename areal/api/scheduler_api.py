# [MODULE DOC]
# Scheduler contract: the Worker and Job dataclasses describing scheduled processes, and
# the abstract Scheduler interface implemented by local/Ray/Slurm schedulers to create,
# query, fork and delete workers, create engines on them, set worker env, and RPC into
# engine methods (sync and async).
# SPDX-License-Identifier: Apache-2.0

import abc
from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
)


@dataclass
class Worker:
    # [CLASS DOC]
    # Dataclass for a running worker: id (e.g. "rollout/0"), ip, worker_ports (ports for
    # worker RPC) and engine_ports (ports for engine servers); port lists are strings
    # and default to empty.
    """Represents a worker process in the distributed system.

    Attributes
    ----------
    id : str
        Unique identifier for the worker (e.g., "rollout/0", "actor/1")
    ip : str
        IP address where the worker is running
    worker_ports : list[str]
        List of port numbers allocated to this worker for worker communication
    engine_ports : list[str]
        List of port numbers allocated to this worker for engine communication
    """

    id: str
    ip: str
    worker_ports: list[str] = field(default_factory=list)
    engine_ports: list[str] = field(default_factory=list)


@dataclass
class Job:
    # [CLASS DOC]
    # Dataclass describing a worker-group request: role name, replicas count (default
    # 0), tasks (per-replica SchedulingSpec list describing resources/commands, default
    # empty) and scheduling_strategy (SchedulingStrategy, default-constructed, e.g.
    # colocation constraints).
    role: str
    replicas: int = 0
    tasks: list[SchedulingSpec] = field(default_factory=list)
    scheduling_strategy: SchedulingStrategy = field(default_factory=SchedulingStrategy)


class Scheduler(abc.ABC):
    # [CLASS DOC]
    # Abstract scheduler managing worker processes/containers, resource allocation,
    # remote engine instances and RPC calls. All methods are abstract and raise
    # NotImplementedError in the base.
    """Abstract base class for schedulers that manage distributed worker processes.

    A scheduler is responsible for creating and managing worker processes/containers,
    allocating resources (GPUs, ports, memory), creating and managing engine instances
    on workers, and facilitating RPC calls to engine methods.
    """

    @property
    @abc.abstractmethod
    def n_gpus_per_node(self) -> int:
        # [FUNCTION DOC]
        # Abstract property: number of GPUs per node configured for this scheduler
        # (int).
        """Return the number of GPUs per node configured for this scheduler."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        # [FUNCTION DOC]
        # Abstract. Launches job.replicas workers for job.role per its tasks/strategy;
        # returns the list of created worker IDs. Implementations may raise
        # WorkerCreationError or ValueError.
        """Create and start worker processes for a specific role.

        Parameters
        ----------
        job : Job
            Job configuration specifying replicas, resources, and scheduling strategy
        *args
            Additional positional arguments (implementation-specific)
        **kwargs
            Additional keyword arguments (implementation-specific)

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["rollout/0", "rollout/1"])

        Raises
        ------
        WorkerCreationError
            If worker creation fails
        ValueError
            If scheduler_config is invalid
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_workers(self, role: str, timeout: int | None = None) -> list[Worker]:
        # [FUNCTION DOC]
        # Abstract. Blocks until all workers of role are ready for RPC (timeout seconds,
        # None = implementation default) and returns their Worker records. May raise
        # WorkerNotFoundError, WorkerFailedError, WorkerTimeoutError.
        """Wait for workers to be ready and return their information.

        This method blocks until all workers for the specified role are ready
        to accept RPC requests, or until the timeout is reached.

        Parameters
        ----------
        role : str
            Role name to query (e.g., "rollout", "actor")
        timeout : int, optional
            Maximum time to wait in seconds. None means use the default timeout

        Returns
        -------
        list[Worker]
            List of Worker objects containing worker ID, IP address, and allocated ports

        Raises
        ------
        WorkerNotFoundError
            If no workers exist for the specified role
        WorkerFailedError
            If any worker process has failed
        WorkerTimeoutError
            If timeout is exceeded while waiting for workers
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        # [FUNCTION DOC]
        # Abstract. Gracefully stops workers of role (all roles if None) and frees
        # resources; must tolerate already-stopped workers. reverse_order=True kills
        # highest ranks first so rank 0 (TCPStore owner) dies last. May raise
        # WorkerNotFoundError for unknown role.
        """Stop and clean up worker processes.

        Parameters
        ----------
        role : str, optional
            Specific role to delete. If None, all workers are deleted
        reverse_order : bool, optional
            If True, terminate workers in reverse order of their IDs so that
            rank-0 (which typically owns the global TCPStore server) is the
            last one to be killed. This helps avoid a noisy
            ``TCPStore.recvValue failed`` warning emitted by NCCL's
            HeartbeatMonitor background thread on non-zero ranks during
            teardown. Implementations that tear down all workers as a single
            atomic operation (e.g. ``scancel`` for Slurm) may safely ignore
            this argument. Defaults to False for backward compatibility.

        Raises
        ------
        WorkerNotFoundError
            If the specified role doesn't exist

        Notes
        -----
        This method should gracefully terminate workers and clean up resources.
        It should not raise an exception if workers have already stopped.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def fork_workers(
        self,
        role: str,
        target_role: str,
        command: str | None = None,
    ) -> list[str]:
        # [FUNCTION DOC]
        # Abstract. Spawns new workers under role, one colocated with each existing
        # target_role worker, optionally running a custom module command instead of the
        # default RPC server. Returns new worker IDs; may raise WorkerCreationError /
        # WorkerNotFoundError.
        """Fork new worker processes from existing workers.

        Creates new worker processes by forking from existing workers of the target role.
        The forked workers are colocated on the same nodes as their target workers.

        Parameters
        ----------
        role : str
            Role name for the new forked workers (e.g., "proxy")
        target_role : str
            Role of existing workers to fork from (e.g., "rollout")
        command : str, optional
            Custom module path to run instead of the default rpc_server.
            If specified, the forked process runs this module (e.g.,
            "areal.experimental.openai.proxy.proxy_rollout_server").

        Returns
        -------
        list[str]
            List of worker IDs created (e.g., ["proxy/0", "proxy/1"])

        Raises
        ------
        WorkerCreationError
            If fork operation fails
        WorkerNotFoundError
            If target_role workers don't exist
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def create_engine(
        self,
        worker_id: str,
        engine: str,
        engine_name: str | None = None,
        *args,
        **kwargs,
    ) -> Any:
        # [FUNCTION DOC]
        # Abstract coroutine. Instantiates the engine class given by import path engine
        # on worker worker_id under engine_name (defaults to worker_id; allows multiple
        # colocated engines), passing args/kwargs to its constructor. Returns the
        # initialization result; may raise WorkerNotFoundError, WorkerFailedError,
        # EngineCreationError.
        """Create an engine instance on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.fsdp_engine.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Parameters
        ----------
        worker_id : str
            ID of the worker to create the engine on (e.g., "rollout/0")
        engine : str
            Import path to the engine class (e.g., "areal.engine.fsdp_engine.FSDPPPOActor")
        engine_name : str, optional
            Unique name for this engine instance (e.g., "actor/0", "ref/0").
            Defaults to worker_id if not specified. This allows multiple engines
            per worker when using colocation.
        *args
            Positional arguments passed to engine initialization
        **kwargs
            Keyword arguments passed to engine initialization

        Returns
        -------
        Any
            Result from engine initialization

        Raises
        ------
        WorkerNotFoundError
            If the specified worker doesn't exist
        WorkerFailedError
            If the worker process has failed
        EngineCreationError
            If engine creation or initialization fails
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def set_worker_env(self, worker_id: str, env: dict[str, str]) -> None:
        # [FUNCTION DOC]
        # Abstract coroutine. Sets the given environment variables on the worker process
        # before engine creation. Returns None.
        """Set environment variables on a worker before engine creation.

        Parameters
        ----------
        worker_id : str
            ID of the worker to configure (e.g., "rollout/0")
        env : dict[str, str]
            Mapping of environment variable names to their values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        # [FUNCTION DOC]
        # Abstract. Synchronously RPCs method on the named engine (default worker_id) of
        # worker worker_id with args/kwargs and optional rpc_meta, returning the method
        # result. May raise WorkerNotFoundError, WorkerFailedError, EngineCallError.
        """Call a method on an engine instance running on a worker (data plane operation).

        This is the synchronous version. Use `async_call_engine` for async operations.

        Parameters
        ----------
        worker_id : str
            ID of the worker hosting the engine (e.g., "rollout/0")
        method : str
            Name of the method to call on the engine
        engine_name : str, optional
            Name of the engine to call (e.g., "actor/0", "ref/0").
            Defaults to worker_id if not specified.
        *args
            Positional arguments to pass to the method
        rpc_meta : dict[str, Any] | None, optional
            RPC metadata, by default None
        **kwargs
            Keyword arguments to pass to the method

        Returns
        -------
        Any
            Result from the engine method call

        Raises
        ------
        WorkerNotFoundError
            If the specified worker doesn't exist
        WorkerFailedError
            If the worker process has failed
        EngineCallError
            If the method call fails
        """
        raise NotImplementedError()

    @abc.abstractmethod
    async def async_call_engine(
        self,
        worker_id: str,
        method: str,
        engine_name: str | None = None,
        *args,
        rpc_meta: dict[str, Any] | None = None,
        **kwargs,
    ) -> Any:
        # [FUNCTION DOC]
        # Abstract coroutine. Async counterpart of call_engine with identical inputs,
        # outputs and errors.
        """Async version of call_engine for calling engine methods asynchronously.

        This is useful for concurrent operations or when integrating with async frameworks.

        Parameters
        ----------
        worker_id : str
            ID of the worker hosting the engine (e.g., "rollout/0")
        method : str
            Name of the method to call on the engine
        engine_name : str, optional
            Name of the engine to call (e.g., "actor/0", "ref/0").
            Defaults to worker_id if not specified.
        *args
            Positional arguments to pass to the method
        rpc_meta : dict[str, Any] | None, optional
            RPC metadata, by default None
        **kwargs
            Keyword arguments to pass to the method

        Returns
        -------
        Any
            Result from the engine method call

        Raises
        ------
        WorkerNotFoundError
            If the specified worker doesn't exist
        WorkerFailedError
            If the worker process has failed
        EngineCallError
            If the method call fails
        """
        raise NotImplementedError()
