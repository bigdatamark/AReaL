# [MODULE DOC]
# GatewayTrainController: client-side controller (v2) that stands up the serverless
# training service through a Scheduler and then mimics the TrainController / train
# engine interface by calling the gateway over HTTP. Launch sequence: one RPC guard per
# GPU rank via the scheduler, NCCL env on each guard, one forked train worker per guard,
# engine creation/process group/initialize on every worker, then router, data proxy and
# gateway forked on guard 0, and finally registration of the data proxy in the router
# under a generated API key. Also wires RL weight updates to a v2 rollout controller
# through a WeightUpdateController.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import traceback
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp

from areal.infra.utils.concurrent import get_executor, run_async_task
from areal.infra.utils.http import create_httpx_client
from areal.utils import logging
from areal.utils.network import format_hostport, gethostip

if TYPE_CHECKING:
    from areal.api import ParallelStrategy, TrainEngine
    from areal.api.cli_args import TrainEngineConfig
    from areal.api.io_struct import FinetuneSpec
    from areal.api.scheduler_api import Scheduler, Worker

logger = logging.getLogger("GatewayTrainController")


class GatewayTrainController:
    # [CLASS DOC]
    # Controller holding: engine class (type or import path), TrainEngineConfig,
    # scheduler, parsed ModelAllocation (parallel strategy, world size), service
    # addresses (gateway, router, data proxy "model", workers, guards), forked-service
    # and scheduler role lists for cleanup, the generated api_key, a locally tracked
    # weight version, an optional rollout controller and weight-update controller, a
    # lazily created per-event-loop httpx client, and background-initialization state
    # (future, lock, workers-ready and shutdown events). Exposes duck-typed train-engine
    # properties reporting a single DP rank with no CPU group.
    _GUARD_SUFFIX = "-guard"

    def __init__(
        self,
        train_engine: type[TrainEngine] | str,
        config: TrainEngineConfig,
        scheduler: Scheduler,
    ) -> None:
        # [FUNCTION DOC]
        # Stores arguments, parses config.backend into a ModelAllocation and initializes
        # all state described in the class doc (no I/O).
        from areal.api.alloc_mode import ModelAllocation

        self.train_engine = train_engine
        self.scheduler = scheduler
        self.config = config
        self.train_alloc = ModelAllocation.from_str(config.backend)
        self.api_key: str | None = None
        self._gateway_addr: str = ""
        self._router_addr: str = ""
        self._model_addr: str = ""
        self._worker_addrs: list[str] = []
        self._guard_addrs: list[str] = []
        self._forked_services: list[tuple[str, str, int]] = []
        self._service_roles: list[str] = []
        self._role: str = ""
        self._parallel_strategy = self.train_alloc.parallel
        self._own_process_group = False
        self.rollout: Any | None = None
        self._weight_update_ctrl: Any | None = None

        # Version management
        self._version_lock = Lock()
        self._version = 0

        # Shared HTTP client (lazy, per-event-loop)
        self._async_client: Any | None = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None

        # Pipelined initialization state
        self._init_future: concurrent.futures.Future | None = None
        self._init_lock = threading.Lock()
        self._workers_ready = threading.Event()
        self._shutdown_requested = threading.Event()

    # -- Initialize --------------------------------------------------------

    def initialize(
        self,
        role: str,
        ft_spec: FinetuneSpec | None = None,
        *,
        wait: bool = False,
        **kwargs: Any,
    ) -> concurrent.futures.Future | None:
        # [FUNCTION DOC]
        # Starts background initialization for role on the "ctrl_init" executor and
        # blocks until guards are ready (TimeoutError after
        # config.workers_ready_timeout). Re-raises a background failure if the future
        # already finished. With wait=True blocks until full initialization completes
        # and returns None; otherwise returns the future. Raises RuntimeError if a
        # previous initialization future is still pending. Extra kwargs (e.g. addr) are
        # forwarded to engine initialize.
        if self._init_future is not None:
            raise RuntimeError(
                "initialize() called while a previous initialization is in progress"
            )

        self._role = role

        self._workers_ready.clear()
        self._shutdown_requested.clear()
        self._init_future = get_executor("ctrl_init").submit(
            self._guarded_bg_initialize, role, ft_spec, **kwargs
        )

        ready_timeout = self.config.workers_ready_timeout
        if not self._workers_ready.wait(timeout=ready_timeout):
            raise TimeoutError(f"Worker creation timed out after {ready_timeout}s")
        if self._init_future.done():
            self._init_future.result()

        if wait:
            self._ensure_initialized()
            return None
        return self._init_future

    def _guarded_bg_initialize(self, *args: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs _bg_initialize and, on any exception, sets the workers-ready
        # event before re-raising so initialize() does not wait for the full timeout.
        """Ensure _workers_ready is signaled even if _bg_initialize fails."""
        try:
            self._bg_initialize(*args, **kwargs)
        except BaseException:
            self._workers_ready.set()
            raise

    def _bg_initialize(
        self, role: str, ft_spec: FinetuneSpec | None = None, **kwargs: Any
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs the async initialization to completion via run_async_task and
        # logs the resulting role, api_key and gateway address unless shutdown was
        # requested.
        # KNOWN ISSUE: logs the generated API key in clear text.
        run_async_task(self._async_initialize, role, ft_spec, **kwargs)
        if self._shutdown_requested.is_set():
            return
        logger.info(
            "GatewayTrainController initialized (role=%s, api_key=%s, gateway=%s)",
            role,
            self.api_key,
            self._gateway_addr,
        )

    def _ensure_initialized(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If a background initialization is pending, waits for it (up to
        # setup_timeout, re-raising its exception) under a lock and clears the future.
        if self._init_future is None:
            return
        with self._init_lock:
            future = self._init_future
            if future is None:
                return
            future.result(timeout=self.config.setup_timeout)
            self._init_future = None

    async def _get_async_client(self):
        # [FUNCTION DOC]
        # HELPER: Returns an httpx async client bound to the current event loop (timeout
        # setup_timeout), recreating it (and closing the old one best-effort) when
        # called from a different loop.
        current_loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_client_loop is not current_loop:
            old = self._async_client
            self._async_client = create_httpx_client(timeout=self.config.setup_timeout)
            self._async_client_loop = current_loop
            if old is not None:
                try:
                    await old.aclose()
                except Exception:
                    pass
        return self._async_client

    async def _async_initialize(
        self,
        role: str,
        ft_spec: FinetuneSpec | None = None,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Performs the full launch sequence. Requires exactly one
        # scheduling_spec (ValueError otherwise). Creates world_size guard workers under
        # role "<role>-guard" running the training guard module, waits for them, sets
        # the workers-ready event; allocates the NCCL master port on guard 0 and sets
        # RANK/LOCAL_RANK=0/WORLD_SIZE/ MASTER_ADDR/MASTER_PORT on every guard; forks
        # one train worker per guard; creates the engine (config passed as init kwarg)
        # on all workers, then calls create_process_group with the parallel strategy and
        # initialize with addr and ft_spec; forks router, data proxy (all worker addrs)
        # and gateway (router addr, request_timeout as forward timeout) on guard 0,
        # waiting for each /health; generates api_key "ak-<role>-<12 hex>" and registers
        # the data proxy in the router. Checks the shutdown flag between stages and
        # returns early if set. On any exception logs, rolls back via
        # _cleanup_runtime_state and re-raises.
        from dataclasses import asdict

        from areal.api.cli_args import SchedulingSpec
        from areal.api.scheduler_api import Job

        cfg = self.config

        world_size = self.train_alloc.parallel.world_size

        try:
            # ==============================================================
            # Step 0: Create world_size guards via scheduler (one per GPU rank)
            # ==============================================================
            # Each guard is allocated a GPU by the scheduler (like TrainController
            # workers). Forked workers inherit the guard's GPU environment.
            if len(cfg.scheduling_spec) != 1:
                raise ValueError(
                    "GatewayTrainController (controller v2) requires exactly "
                    "one scheduling_spec. Legacy 2-spec worker/engine layouts "
                    "are only supported by TrainController (controller v1)."
                )

            guard_spec = SchedulingSpec(**asdict(cfg.scheduling_spec[0]))
            guard_spec.cmd = "python -m areal.v2.training_service.guard"

            guard_role = f"{role}{self._GUARD_SUFFIX}"
            guard_job = Job(
                replicas=world_size,
                tasks=[guard_spec],
                scheduling_strategy=cfg.scheduling_strategy,
                role=guard_role,
            )
            await asyncio.to_thread(self.scheduler.create_workers, job=guard_job)
            self._service_roles.append(guard_role)
            guard_workers = await asyncio.to_thread(
                self.scheduler.get_workers,
                role=guard_role,
                timeout=int(self.config.setup_timeout),
            )
            logger.info("Guards ready: %s", [w.id for w in guard_workers])

            self._workers_ready.set()

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 1: Allocate master addr/port for NCCL rendezvous
            # ==============================================================
            guard_addr_0 = f"http://{format_hostport(guard_workers[0].ip, int(guard_workers[0].worker_ports[0]))}"
            master_addr = guard_workers[0].ip

            # Persist guard addresses so connect_engine() can allocate
            # ports later (e.g. for the weight-update NCCL group).
            def _guard_addr(worker: Worker) -> str:
                # [FUNCTION DOC]
                # HELPER: Builds the http URL of a guard worker from its IP and first
                # port.
                return (
                    f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"
                )

            self._guard_addrs = [_guard_addr(w) for w in guard_workers]

            client = await self._get_async_client()
            resp = await client.post(
                f"{guard_addr_0}/alloc_ports", json={"count": 1}, timeout=30.0
            )
            resp.raise_for_status()
            master_port = resp.json()["ports"][0]

            # ==============================================================
            # Step 1.5: Set NCCL env on each guard so forked workers inherit it
            # ==============================================================

            await self._async_set_guards_env(
                guard_workers,
                _guard_addr,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
            )

            # ==============================================================
            # Step 2: Fork one train worker per guard
            # ==============================================================
            async def _fork_worker(rank: int) -> str:
                # [FUNCTION DOC]
                # HELPER: Forks a training worker (admin key and log level args) on the
                # guard for rank and returns its http address.
                guard = _guard_addr(guard_workers[rank])
                worker_cmd = [
                    sys.executable,
                    "-m",
                    "areal.v2.training_service.worker",
                    "--admin-api-key",
                    cfg.admin_api_key,
                    "--log-level",
                    cfg.log_level,
                ]

                host, port = await self._async_fork_on_guard(
                    guard_addr=guard,
                    role="train-worker",
                    worker_index=rank,
                    raw_cmd=worker_cmd,
                )
                return f"http://{format_hostport(host, port)}"

            self._worker_addrs = list(
                await asyncio.gather(
                    *[_fork_worker(rank) for rank in range(world_size)]
                )
            )
            logger.info("Workers: %s", self._worker_addrs)

            # ==============================================================
            # Step 3: Create engines on all workers (coordinated NCCL init)
            # ==============================================================
            if isinstance(self.train_engine, str):
                engine_class = self.train_engine
            else:
                engine_class = (
                    f"{self.train_engine.__module__}.{self.train_engine.__name__}"
                )
            await asyncio.gather(
                *[
                    self._create_engine_on_worker(
                        worker_addr=addr,
                        engine_class=engine_class,
                        init_args=[],
                        init_kwargs={"config": self.config},
                    )
                    for addr in self._worker_addrs
                ]
            )
            logger.info("Engines created on all workers")

            pg_kwargs = {"parallel_strategy": self._parallel_strategy}
            await asyncio.gather(
                *[
                    self._call_worker_engine_endpoint(
                        addr,
                        "/create_process_group",
                        args=[],
                        kwargs=pg_kwargs,
                        timeout=self.config.setup_timeout,
                    )
                    for addr in self._worker_addrs
                ]
            )

            await asyncio.gather(
                *[
                    self._call_worker_engine_endpoint(
                        addr,
                        "/initialize",
                        args=[],
                        kwargs={
                            "addr": kwargs.get("addr"),
                            "ft_spec": ft_spec,
                        },
                        timeout=self.config.setup_timeout,
                    )
                    for addr in self._worker_addrs
                ]
            )
            logger.info("Engines initialized on all workers")

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 4: Fork Router on guard 0
            # ==============================================================
            router_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.router",
                "--admin-api-key",
                cfg.admin_api_key,
                "--log-level",
                cfg.log_level,
            ]
            router_host, router_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="router",
                worker_index=0,
                raw_cmd=router_cmd,
            )
            self._router_addr = f"http://{format_hostport(router_host, router_port)}"
            logger.info("Router: %s", self._router_addr)

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 5: Fork Data Proxy on a guard
            # ==============================================================
            data_proxy_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.data_proxy",
                "--worker-addrs",
                ",".join(self._worker_addrs),
                "--admin-api-key",
                cfg.admin_api_key,
                "--log-level",
                cfg.log_level,
            ]
            dp_host, dp_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="data-proxy",
                worker_index=0,
                raw_cmd=data_proxy_cmd,
            )
            self._model_addr = f"http://{format_hostport(dp_host, dp_port)}"
            logger.info("Model endpoint: %s", self._model_addr)

            if self._shutdown_requested.is_set():
                return

            # ==============================================================
            # Step 6: Fork Gateway on guard 0
            # ==============================================================
            gw_cmd = [
                sys.executable,
                "-m",
                "areal.v2.training_service.gateway",
                "--admin-api-key",
                cfg.admin_api_key,
                "--router-addr",
                self._router_addr,
                "--forward-timeout",
                str(cfg.request_timeout),
                "--log-level",
                cfg.log_level,
            ]
            gw_host, gw_port = await self._async_fork_on_guard(
                guard_addr=guard_addr_0,
                role="gateway",
                worker_index=0,
                raw_cmd=gw_cmd,
            )
            self._gateway_addr = f"http://{format_hostport(gw_host, gw_port)}"
            logger.info("Gateway: %s", self._gateway_addr)

            # ==============================================================
            # Step 7: Register data proxy with API key in router
            # ==============================================================
            self.api_key = f"ak-{role}-{uuid4().hex[:12]}"
            await self._register_in_router(
                self._router_addr, self._model_addr, self.api_key
            )
            logger.info("Model registered with api_key=%s", self.api_key)
        except Exception:
            logger.error(
                "GatewayTrainController initialization failed, rolling back",
                exc_info=True,
            )
            self._cleanup_runtime_state()
            raise

    # -- Engine creation ---------------------------------------------------

    async def _async_set_guards_env(
        self,
        guard_workers: list[Worker],
        guard_addr_fn: Any,
        *,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Concurrently POSTs the distributed env (RANK, LOCAL_RANK 0,
        # WORLD_SIZE, MASTER_ADDR, MASTER_PORT) to /set_env on every guard so forked
        # workers inherit it; raises on HTTP errors.
        client = await self._get_async_client()

        async def _set_env(rank: int) -> None:
            # [FUNCTION DOC]
            # HELPER: Sets the env for one guard rank.
            addr = guard_addr_fn(guard_workers[rank])
            env = {
                "RANK": str(rank),
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            }
            resp = await client.post(f"{addr}/set_env", json={"env": env}, timeout=30.0)
            resp.raise_for_status()

        await asyncio.gather(*[_set_env(rank) for rank in range(len(guard_workers))])
        logger.info("NCCL env set on %d guards", len(guard_workers))

    async def _create_engine_on_worker(
        self,
        worker_addr: str,
        engine_class: str,
        init_args: list[Any],
        init_kwargs: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: POSTs /create_engine with the engine class path and serialized init
        # args to a worker; raises RuntimeError on HTTP >= 400.
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "engine_class": engine_class,
            "init_args": serialize_value(init_args),
            "init_kwargs": serialize_value(init_kwargs),
        }
        client = await self._get_async_client()
        resp = await client.post(
            f"{worker_addr}/create_engine",
            json=payload,
            timeout=self.config.setup_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Engine creation failed on {worker_addr}: {resp.text}")

    async def _call_worker_engine_endpoint(
        self,
        worker_addr: str,
        path: str,
        *,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        timeout: float,
    ) -> Any:
        # [FUNCTION DOC]
        # HELPER: POSTs serialized args/kwargs to worker_addr+path and returns the
        # deserialized "result"; raises RuntimeError on HTTP >= 400.
        from areal.infra.rpc.serialization import deserialize_value, serialize_value

        payload = {
            "args": serialize_value(args or []),
            "kwargs": serialize_value(kwargs or {}),
        }
        client = await self._get_async_client()
        resp = await client.post(f"{worker_addr}{path}", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Worker endpoint call failed on {worker_addr}{path}: {resp.text}"
            )
        data = resp.json()
        return deserialize_value(data.get("result"))

    # -- Router registration -----------------------------------------------

    async def _register_in_router(
        self, router_addr: str, model_addr: str, api_key: str
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: POSTs {model_addr, api_key, name=role} to router /register with the
        # admin bearer; raises on HTTP errors.
        client = await self._get_async_client()
        resp = await client.post(
            f"{router_addr}/register",
            json={
                "model_addr": model_addr,
                "api_key": api_key,
                "name": self._role,
            },
            headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
            timeout=10.0,
        )
        resp.raise_for_status()

    # -- Guard fork helpers ------------------------------------------------

    def _fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        env: dict[str, str] | None = None,
        health_path: str = "/health",
    ) -> tuple[str, int]:
        # [FUNCTION DOC]
        # HELPER: Synchronous variant of guard forking using requests: allocates a port
        # on the guard, appends --host/--port to the command, POSTs /fork (role,
        # worker_index, cmd, optional env), records the service for cleanup, waits for
        # its health path and returns (host, port). Not used by the async launch path.
        import requests

        resp = requests.post(f"{guard_addr}/alloc_ports", json={"count": 1}, timeout=30)
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]

        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if env:
            fork_payload["env"] = env

        resp = requests.post(f"{guard_addr}/fork", json=fork_payload, timeout=30)
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        self._wait_for_service(f"{addr}{health_path}", role)

        return host, port

    async def _async_fork_on_guard(
        self,
        guard_addr: str,
        role: str,
        worker_index: int,
        raw_cmd: list[str],
        env: dict[str, str] | None = None,
        health_path: str = "/health",
    ) -> tuple[str, int]:
        # [FUNCTION DOC]
        # HELPER: Async version of _fork_on_guard using the shared httpx client.
        client = await self._get_async_client()
        resp = await client.post(
            f"{guard_addr}/alloc_ports", json={"count": 1}, timeout=30.0
        )
        resp.raise_for_status()
        port_data = resp.json()
        host = port_data["host"]
        port = port_data["ports"][0]

        cmd = list(raw_cmd) + ["--host", host, "--port", str(port)]
        fork_payload: dict[str, Any] = {
            "role": role,
            "worker_index": worker_index,
            "raw_cmd": cmd,
        }
        if env:
            fork_payload["env"] = env

        resp = await client.post(f"{guard_addr}/fork", json=fork_payload, timeout=30.0)
        resp.raise_for_status()

        self._forked_services.append((guard_addr, role, worker_index))

        addr = f"http://{format_hostport(host, port)}"
        await self._async_wait_for_service(f"{addr}{health_path}", role)

        return host, port

    def _kill_forked_service(
        self, guard_addr: str, role: str, worker_index: int
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: POSTs /kill_forked_worker (role, worker_index) to the guard; logs
        # success, failure or exceptions and never raises.
        import requests

        try:
            resp = requests.post(
                f"{guard_addr}/kill_forked_worker",
                json={"role": role, "worker_index": worker_index},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Killed forked service %s/%d", role, worker_index)
            else:
                logger.warning(
                    "Failed to kill %s/%d: %s", role, worker_index, resp.text
                )
        except Exception as exc:
            logger.error("Error killing %s/%d: %s", role, worker_index, exc)

    # -- Health checks -----------------------------------------------------

    def _wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Polls url every 0.1s with requests (2s timeout) until HTTP 200 or
        # until timeout (default setup_timeout) elapses, then raises TimeoutError.
        import requests as _requests

        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = _requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except _requests.RequestException:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"{name} not healthy at {url} within {timeout}s")

    async def _async_wait_for_service(
        self, url: str, name: str, timeout: float | None = None
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Async equivalent of _wait_for_service using the shared httpx client.
        timeout = timeout or self.config.setup_timeout
        deadline = time.monotonic() + timeout
        client = await self._get_async_client()
        while time.monotonic() < deadline:
            try:
                resp = await client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("%s is ready at %s", name, url)
                    return
            except Exception:
                pass
            await asyncio.sleep(0.1)
        raise TimeoutError(f"{name} not healthy at {url} within {timeout}s")

    # -- Gateway HTTP helpers (duck-type TrainController interface) ---------

    def _gateway_post(self, path: str, payload: Any = None) -> Any:
        # [FUNCTION DOC]
        # HELPER: Waits for initialization, POSTs JSON payload to gateway+path with the
        # controller's api_key as bearer (request_timeout), raises RuntimeError on HTTP
        # >= 400 and returns the decoded JSON.
        import requests

        self._ensure_initialized()
        url = f"{self._gateway_addr}{path}"
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.config.request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _gateway_get(self, path: str) -> Any:
        # [FUNCTION DOC]
        # HELPER: Same as _gateway_post but issues a GET without body.
        import requests

        self._ensure_initialized()
        url = f"{self._gateway_addr}{path}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.config.request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _gateway_post_result(self, path: str, payload: Any = None) -> Any:
        # [FUNCTION DOC]
        # HELPER: POSTs and returns the deserialized "result" field; RuntimeError if the
        # response is not a dict with "result".
        from areal.infra.rpc.serialization import deserialize_value

        data = self._gateway_post(path, payload)
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"Gateway {path} response missing 'result': {data!r}")
        return deserialize_value(data["result"])

    def _gateway_get_result(self, path: str) -> Any:
        # [FUNCTION DOC]
        # HELPER: GET counterpart of _gateway_post_result.
        from areal.infra.rpc.serialization import deserialize_value

        data = self._gateway_get(path)
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"Gateway {path} response missing 'result': {data!r}")
        return deserialize_value(data["result"])

    # -- TrainController duck-type interface --------------------------------

    @staticmethod
    def _require_list_batch(input_: Any, method_name: str) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # HELPER: Static. Returns input_ if it is a list, else raises TypeError naming
        # the method (training-service dispatch requires list-of-dict batches).
        if not isinstance(input_, list):
            raise TypeError(
                f"{method_name} expects `input_` as list[dict[str, Any]] for training-service dispatch; "
                f"got {type(input_).__name__}."
            )
        return input_

    def train_batch(
        self,
        input_: list[dict[str, Any]] | None = None,
        loss_fn: Any = None,
        loss_weight_fn: Any = None,
    ) -> Any:
        # [FUNCTION DOC]
        # Sends a list-of-dict batch with loss_fn and loss_weight_fn to /train_batch
        # (data proxy partitions it over DP) and returns the deserialized result.
        # TypeError for None or non-list input.
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError("train_batch expects non-None list[dict[str, Any]] input.")
        batch = self._require_list_batch(input_, "train_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(
                {"loss_fn": loss_fn, "loss_weight_fn": loss_weight_fn}
            ),
        }
        return self._gateway_post_result("/train_batch", payload)

    def forward_batch(
        self, input_: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> Any:
        # [FUNCTION DOC]
        # Sends a list-of-dict batch plus extra kwargs to /forward_batch and returns the
        # deserialized merged result. TypeError for None or non-list input.
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError(
                "forward_batch expects non-None list[dict[str, Any]] input."
            )
        batch = self._require_list_batch(input_, "forward_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/forward_batch", payload)

    def eval_batch(
        self,
        input_: list[dict[str, Any]] | None = None,
        loss_fn: Any = None,
        loss_weight_fn: Any = None,
    ) -> Any:
        # [FUNCTION DOC]
        # Sends a batch with loss functions to /eval_batch (padded across DP) and
        # returns the result. TypeError for None or non-list input.
        from areal.infra.rpc.serialization import serialize_value

        if input_ is None:
            raise TypeError("eval_batch expects non-None list[dict[str, Any]] input.")
        batch = self._require_list_batch(input_, "eval_batch")

        payload = {
            "args": serialize_value([batch]),
            "kwargs": serialize_value(
                {"loss_fn": loss_fn, "loss_weight_fn": loss_weight_fn}
            ),
        }
        return self._gateway_post_result("/eval_batch", payload)

    def train(self, mode: bool = True) -> GatewayTrainController:
        # [FUNCTION DOC]
        # Broadcasts train(mode) to all workers via the gateway; returns self.
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/train",
            {
                "args": serialize_value([mode]),
                "kwargs": serialize_value({}),
            },
        )
        return self

    def eval(self) -> GatewayTrainController:
        # [FUNCTION DOC]
        # Broadcasts eval mode to all workers; returns self.
        self._gateway_post("/eval")
        return self

    def set_version(self, version: int) -> None:
        # [FUNCTION DOC]
        # Records version locally (under lock) and broadcasts set_version to all
        # workers.
        from areal.infra.rpc.serialization import serialize_value

        with self._version_lock:
            self._version = version

        self._gateway_post(
            "/set_version",
            {
                "args": serialize_value([version]),
                "kwargs": serialize_value({}),
            },
        )

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # Returns the locally tracked version (no remote call).
        with self._version_lock:
            return self._version

    def save(self, meta: Any) -> None:
        # [FUNCTION DOC]
        # Broadcasts save(meta) (checkpoint/HF save) to all workers.
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/save",
            {
                "args": serialize_value([meta]),
                "kwargs": serialize_value({}),
            },
        )

    def load(self, meta: Any) -> None:
        # [FUNCTION DOC]
        # Broadcasts load(meta) to all workers.
        from areal.infra.rpc.serialization import serialize_value

        self._gateway_post(
            "/load",
            {
                "args": serialize_value([meta]),
                "kwargs": serialize_value({}),
            },
        )

    def offload(self) -> None:
        # [FUNCTION DOC]
        # Broadcasts offload (engine memory to CPU) to all workers.
        self._gateway_post("/offload")

    def onload(self) -> None:
        # [FUNCTION DOC]
        # Broadcasts onload to all workers.
        self._gateway_post("/onload")

    def step_lr_scheduler(self) -> None:
        # [FUNCTION DOC]
        # Broadcasts an LR scheduler step to all workers.
        self._gateway_post("/step_lr_scheduler")

    def optimizer_zero_grad(self) -> None:
        # [FUNCTION DOC]
        # Broadcasts optimizer zero_grad to all workers.
        self._gateway_post("/optimizer_zero_grad")

    def optimizer_step(self) -> Any:
        # [FUNCTION DOC]
        # Broadcasts optimizer step and returns the first worker's deserialized result.
        return self._gateway_post_result("/optimizer_step")

    def export_stats(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns local stats_tracker.export_all() merged with (overridden by) the first
        # worker's exported stats fetched through the gateway.
        from areal.utils import stats_tracker

        stats = stats_tracker.export_all()
        stats.update(self._gateway_get_result("/export_stats"))
        return stats

    def get_device_stats(self) -> Any:
        # [FUNCTION DOC]
        # Returns the first worker's device stats through the gateway.
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({}),
        }
        return self._gateway_post_result("/get_device_stats", payload)

    def config_perf_tracer(self, config: Any, role: str) -> None:
        # [FUNCTION DOC]
        # Waits for initialization, then calls /config_perf_tracer directly on every
        # worker (bypassing the gateway) with config, its rank and role, concurrently.
        self._ensure_initialized()

        async def _call() -> None:
            # [FUNCTION DOC]
            # HELPER: Async fan-out of the per-rank perf tracer configuration.
            tasks = [
                self._call_worker_engine_endpoint(
                    addr,
                    "/config_perf_tracer",
                    args=[],
                    kwargs={"config": config, "rank": rank, "role": role},
                    timeout=self.config.request_timeout,
                )
                for rank, addr in enumerate(self._worker_addrs)
            ]
            await asyncio.gather(*tasks)

        run_async_task(_call)

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        # [FUNCTION DOC]
        # Broadcasts save_perf_tracer(step, force) to all workers.
        from areal.infra.rpc.serialization import serialize_value

        payload = {
            "args": serialize_value([]),
            "kwargs": serialize_value({"step": step, "force": force}),
        }
        self._gateway_post("/save_perf_tracer", payload)

    def clear_batches(self, *targets: Any) -> None:
        # [FUNCTION DOC]
        # Frees remote batch storage for the given RTensor targets: concurrently sends
        # storage-clear requests to every node holding shards (errors ignored), then, if
        # any shard ids exist, broadcasts clear_batches(shard_ids) to workers through
        # the gateway.
        from areal.infra.rpc.rtensor import RTensor, flatten_shard_ids
        from areal.infra.rpc.serialization import serialize_value

        # Step 1: HTTP DELETE to storage nodes to evict _storage entries
        # (mirrors TrainController._async_clear_batches)
        shards_by_node = RTensor.collect_shards(targets)
        if shards_by_node:

            async def _clear_storage():
                # [FUNCTION DOC]
                # HELPER: Async fan-out of per-node shard deletion.
                await asyncio.gather(
                    *[
                        RTensor.clear_node(addr, sids)
                        for addr, sids in shards_by_node.items()
                    ],
                    return_exceptions=True,
                )

            run_async_task(_clear_storage)

        # Step 2: Drain _fetch_buffer on workers via engine.clear_batches(shard_ids)
        shard_ids = flatten_shard_ids(targets)
        if not shard_ids:
            return
        payload = {
            "args": serialize_value([shard_ids]),
            "kwargs": serialize_value({}),
        }
        self._gateway_post("/clear_batches", payload)

    def current_data_parallel_head(self) -> int:
        # [FUNCTION DOC]
        # Duck-type compat: always 0.
        return 0

    @property
    def context_and_model_parallel_group(self):
        # [FUNCTION DOC]
        # Property: duck-type compat, returns cpu_group (None).
        return self.cpu_group

    @property
    def parallel_strategy(self):
        # [FUNCTION DOC]
        # Property: the current parallel strategy.
        return self._parallel_strategy

    @property
    def data_parallel_world_size(self) -> int:
        # [FUNCTION DOC]
        # Property: duck-type compat, always 1 (the controller acts as a single DP
        # rank).
        return 1

    @property
    def data_parallel_rank(self) -> int:
        # [FUNCTION DOC]
        # Property: always 0.
        return 0

    # -- Properties (duck-type compat) -------------------------------------

    @property
    def cpu_group(self):
        # [FUNCTION DOC]
        # Property: always None.
        return None

    @property
    def train_worker_urls(self) -> list[str]:
        # [FUNCTION DOC]
        # Property: copy of the train worker addresses.
        return list(self._worker_addrs)

    # -- RL parity methods (connect_engine / update_weights / batch) --------

    def connect_engine(self, rollout: Any, meta: Any) -> None:
        # [FUNCTION DOC]
        # Waits for initialization and connects to a v2 rollout controller for weight
        # updates. Raises TypeError unless rollout is a RolloutControllerV2 and
        # ValueError unless meta.type is "awex" or "disk". Starts a
        # WeightUpdateController bound to this host's IP, and connects pair
        # "<role>-rollout" between train workers and inference workers: for awex,
        # allocates the NCCL master port on inference guard 0 (rank 0 lives there); for
        # disk, passes save path and LoRA settings. Stores the rollout and controller.
        self._ensure_initialized()
        import requests

        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )
        from areal.v2.weight_update.controller.config import (
            WeightUpdateControllerConfig,
        )
        from areal.v2.weight_update.controller.controller import (
            WeightUpdateController,
        )

        if not isinstance(rollout, RolloutControllerV2):
            raise TypeError(
                f"GatewayTrainController requires RolloutControllerV2, "
                f"got {type(rollout).__name__}. "
                f"Ensure _version='v2' is set on InferenceEngineConfig."
            )

        self.rollout = rollout

        if meta.type not in ("awex", "disk"):
            raise ValueError(
                f"GatewayTrainController supports 'awex' or 'disk' weight "
                f"updates, got '{meta.type}'"
            )

        ctrl = WeightUpdateController(
            WeightUpdateControllerConfig(
                # Bind gateway to this node's outbound IP so cross-host
                # train/inf workers can reach the kv_store_url the gateway
                # publishes. Default "127.0.0.1" only works single-machine.
                host=gethostip(),
                admin_api_key=self.config.admin_api_key,
                log_level=self.config.log_level,
            )
        )
        ctrl.initialize()

        inference_urls: list[str] = rollout.inference_worker_urls
        pair_name = f"{self._role}-rollout"

        if meta.type == "awex":
            # NCCL rendezvous master must live on the rank-0 process's node.
            # awex assigns rank 0 to inference[0], so allocate on the inference
            # rank-0 guard rather than a train guard.
            inf_guard_addrs = rollout.inference_guard_addrs
            resp = requests.post(
                f"{inf_guard_addrs[0]}/alloc_ports",
                json={"count": 1},
                timeout=30,
            )
            resp.raise_for_status()
            port_data = resp.json()
            ctrl.connect(
                pair_name=pair_name,
                train_worker_urls=self._worker_addrs,
                inference_worker_urls=inference_urls,
                mode="awex",
                nccl_master_addr=port_data["host"],
                nccl_master_port=port_data["ports"][0],
            )
        else:  # disk
            ctrl.connect(
                pair_name=pair_name,
                train_worker_urls=self._worker_addrs,
                inference_worker_urls=inference_urls,
                mode="disk",
                save_path=meta.path or "",
                use_lora=meta.use_lora,
                lora_name=meta.lora_name,
                lora_keep_versions=meta.lora_keep_versions,
            )
        self._weight_update_ctrl = ctrl
        logger.info(
            "WeightUpdateController connected (pair=%s, train=%d, inf=%d)",
            pair_name,
            len(self._worker_addrs),
            len(inference_urls),
        )

    def update_weights(self, meta: Any) -> None:
        # [FUNCTION DOC]
        # Requires connect_engine (RuntimeError otherwise). Pauses generation, asserts
        # meta.version is a positive int, runs the weight update for that version,
        # resumes generation and logs status/duration.
        # KNOWN ISSUE: no try/finally, so a failed assertion or update leaves rollout
        # generation paused.
        if self._weight_update_ctrl is None or self.rollout is None:
            raise RuntimeError(
                "connect_engine() must be called before update_weights()"
            )
        self.rollout.pause_generation()
        assert meta.version is not None and meta.version > 0, (
            f"meta.version must be a positive integer, got {meta.version}"
        )
        result = self._weight_update_ctrl.update_weights(version=meta.version)
        self.rollout.continue_generation()
        logger.info(
            "Weight update v%d completed (%s, %.0fms)",
            meta.version,
            result.status,
            result.duration_ms,
        )

    def prepare_batch(
        self,
        dataloader: Any,
        workflow: Any,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Delegates to rollout.prepare_batch with the same arguments (RuntimeError
        # before connect_engine).
        if self.rollout is None:
            raise RuntimeError("connect_engine() must be called before prepare_batch()")
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
        workflow: Any,
        workflow_kwargs: dict[str, Any],
        should_accept_fn: str | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Delegates to rollout.rollout_batch with the same arguments (RuntimeError
        # before connect_engine).
        if self.rollout is None:
            raise RuntimeError("connect_engine() must be called before rollout_batch()")
        return self.rollout.rollout_batch(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        # [FUNCTION DOC]
        # Stores parallel_strategy and, if torch.distributed is not initialized, creates
        # a local single-rank gloo process group on a free port and marks it owned.
        # KNOWN ISSUE: overwrites the stored parallel strategy even when called with
        # None.
        self._parallel_strategy = parallel_strategy
        import torch.distributed as dist

        from areal.utils.network import find_free_ports

        if not dist.is_initialized():
            port = find_free_ports(1)[0]
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://localhost:{port}",
                rank=0,
                world_size=1,
            )
            self._own_process_group = True

    def is_data_parallel_head(self) -> bool:
        # [FUNCTION DOC]
        # Duck-type compat: always True.
        return True

    # -- Destroy -----------------------------------------------------------

    def _graceful_shutdown_workers(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Concurrently POSTs /awex/teardown and then /destroy_engine to every
        # worker (30s total timeout), logging warnings on failures, so all ranks tear
        # down NCCL before any process exits.
        """Destroy engines on all training workers before killing processes.

        ``dist.destroy_process_group()`` is a local operation
        (``ncclCommAbort`` + HeartbeatMonitor join), but rank-0 hosts the
        TCPStore server.  All workers must stop their HeartbeatMonitor
        before any process exits, otherwise surviving ranks get a
        ``recvValue failed`` warning from the now-dead TCPStore.
        """
        if not self._worker_addrs:
            return

        async def _shutdown_all() -> None:
            # [FUNCTION DOC]
            # HELPER: Opens an aiohttp session and shuts down all workers concurrently.
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = []
                for addr in self._worker_addrs:
                    tasks.append(_shutdown_one(session, addr))
                await asyncio.gather(*tasks, return_exceptions=True)

        async def _shutdown_one(session: aiohttp.ClientSession, addr: str) -> None:
            # [FUNCTION DOC]
            # HELPER: Calls teardown then destroy_engine on one worker, logging
            # failures.
            try:
                async with session.post(f"{addr}/awex/teardown") as resp:
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Graceful shutdown: failed to call /awex/teardown on %s: %s",
                    addr,
                    e,
                )
            try:
                async with session.post(f"{addr}/destroy_engine", json={}) as resp:
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Graceful shutdown: failed to call /destroy_engine on %s: %s",
                    addr,
                    e,
                )

        run_async_task(_shutdown_all)
        logger.info("All training worker engines destroyed gracefully")

    def _cleanup_runtime_state(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Best-effort teardown: unregisters the data proxy from the router,
        # gracefully shuts down workers, kills forked services in reverse order, deletes
        # scheduler workers for each recorded role, clears addresses and api_key, closes
        # the async client, and destroys the owned gloo process group. Errors are
        # logged, not raised.
        if self._router_addr and self._model_addr:
            try:
                import requests

                requests.post(
                    f"{self._router_addr}/unregister",
                    json={"model_addr": self._model_addr},
                    headers={"Authorization": f"Bearer {self.config.admin_api_key}"},
                    timeout=10,
                )
            except Exception:
                logger.error("Failed to unregister model: %s", traceback.format_exc())

        self._graceful_shutdown_workers()

        for guard_addr, role, worker_index in reversed(self._forked_services):
            try:
                self._kill_forked_service(guard_addr, role, worker_index)
            except Exception:
                logger.error(
                    "Error killing %s/%d: %s",
                    role,
                    worker_index,
                    traceback.format_exc(),
                )
        self._forked_services.clear()

        for role in reversed(self._service_roles):
            try:
                self.scheduler.delete_workers(role=role)
                logger.info("Workers deleted for role: %s", role)
            except Exception:
                logger.error(
                    "Error deleting workers for %s: %s", role, traceback.format_exc()
                )
        self._service_roles.clear()
        self._worker_addrs.clear()
        self._router_addr = ""
        self._gateway_addr = ""
        self._model_addr = ""
        self.api_key = None

        if self._async_client is not None:
            try:
                run_async_task(self._async_client.aclose)
            except Exception:
                pass
            self._async_client = None
            self._async_client_loop = None

        import torch.distributed as dist

        if self._own_process_group:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                logger.error(
                    "Failed to destroy process group: %s", traceback.format_exc()
                )
            finally:
                self._own_process_group = False

    def destroy(self) -> None:
        # [FUNCTION DOC]
        # Sets the shutdown flag, cancels the pending init future (cannot interrupt one
        # that is running) and cleans up all runtime state.
        # KNOWN ISSUE: if background initialization is still running, cleanup races with
        # it and services forked afterwards may be left behind.
        self._shutdown_requested.set()
        future = self._init_future
        self._init_future = None
        if future is not None:
            future.cancel()

        self._cleanup_runtime_state()
