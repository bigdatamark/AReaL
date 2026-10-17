# [MODULE DOC]
# Generic HTTP client-side inference engine (RemoteInfEngine) that talks to remote LLM
# servers (SGLang, vLLM, ...) through an injected backend object implementing
# RemoteInfBackendProtocol. Handles server discovery and health waits, round-robin
# routing with per-request-id stickiness, interruptible multi-segment generation,
# logprob scoring, weight updates (disk or NCCL/XCCL) executed on a background executor,
# pause/resume/offload control requests, local server launching, and rollout submission
# via a WorkflowExecutor. Also defines GroupedRolloutWorkflow for running a workflow
# several times per prompt, and module-level helpers that run async HTTP fan-outs.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime
from logging import Logger
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp
import numpy as np
import requests
import torch
import torch.distributed as dist
import uvloop
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    RolloutWorkflow,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
)
from areal.infra import workflow_context
from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import get_executor
from areal.infra.utils.http import arequest_with_retry, get_default_connector
from areal.infra.utils.launcher import wait_llm_server_addrs
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging, name_resolve, names
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.network import (
    find_free_ports,
    format_hostport,
    gethostip,
    split_hostport,
)
from areal.utils.perf_tracer import trace_perf

from .workflow_executor import WorkflowExecutor

if TYPE_CHECKING:
    from areal.experimental.openai import InteractionWithTokenLogpReward

RID_CACHE_SIZE = 128

logger = logging.getLogger("RemoteInfEngine")


class GroupedRolloutWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow wrapper that runs the inner workflow group_size times concurrently
    # on the same input and merges the results. Holds the inner workflow, group size, a
    # logger, and flags reward_normalization (per-group reward normalization for
    # agent/OpenAI interaction results) and drop_incomplete_group (drop the whole group
    # if any sample returned None).
    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        logger: Logger,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ):
        # [FUNCTION DOC]
        # Stores arguments; raises ValueError if group_size < 1.
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.logger = logger
        self.reward_normalization = reward_normalization
        self.drop_incomplete_group = drop_incomplete_group

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Async. Runs group_size copies of the inner workflow concurrently, each under a
        # child WorkflowContext inheriting is_eval and task_id with its own sample_idx;
        # sorts results by index (RuntimeError if indices are not exactly 0..n-1).
        # Returns None if all results are None. If some are None: returns None when
        # drop_incomplete_group, otherwise warns (unless normalizing) and continues with
        # the valid ones. If results are dicts of InteractionWithTokenLogpReward,
        # optionally normalizes rewards across the group (returning None if
        # normalization fails) and returns the merged dict (None if empty). Otherwise
        # treats results as padded tensor dicts and returns their concatenation (None if
        # empty).
        from areal.experimental.openai import InteractionWithTokenLogpReward

        async def run_sample(sample_idx: int) -> tuple[int, Any]:
            # [FUNCTION DOC]
            # HELPER: Async. Sets a per-sample workflow context derived from the parent
            # context and runs the inner workflow once, returning (sample_idx, result).
            from areal.infra import workflow_context
            from areal.infra.workflow_context import WorkflowContext

            parent = workflow_context.get()
            workflow_context.set(
                WorkflowContext(
                    is_eval=parent.is_eval,
                    task_id=parent.task_id,
                    sample_idx=sample_idx,
                )
            )
            result = await self.workflow.arun_episode(engine, data)
            return sample_idx, result

        indexed_results = await asyncio.gather(
            *[run_sample(sample_idx) for sample_idx in range(self.group_size)]
        )
        indexed_results.sort(key=lambda item: item[0])
        sample_indices = [sample_idx for sample_idx, _ in indexed_results]
        if sample_indices != list(range(self.group_size)):
            raise RuntimeError(
                "Grouped rollout returned invalid sample indices: "
                f"expected {list(range(self.group_size))}, got {sample_indices}"
            )
        results = [result for _, result in indexed_results]

        valid_results = [r for r in results if r is not None]

        # All results None -> return None
        if not valid_results:
            return None

        # Some results None -> drop entire group if requested. Reward
        # normalization also requires a complete group and will log/drop below.
        if len(valid_results) < len(results):
            n_failed = len(results) - len(valid_results)
            if self.drop_incomplete_group:
                self.logger.warning(
                    f"GroupedRolloutWorkflow: {n_failed}/{len(results)} "
                    "trajectories returned None, dropping entire group "
                    "(drop_incomplete_group=True). prepare_batch will retry "
                    "with a new prompt from the dataloader."
                )
                return None
            if not self.reward_normalization:
                self.logger.warning(
                    f"GroupedRolloutWorkflow: {n_failed}/{len(results)} "
                    "trajectories returned None, using remaining results"
                )

        # Check if results are InteractionWithTokenLogpReward dicts
        first = valid_results[0]
        if (
            isinstance(first, dict)
            and first
            and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in first.values()
            )
        ):
            if self.reward_normalization and self.group_size > 1:
                if not self._normalize_group_rewards(results):
                    return None
            # Merge dicts - each result is {completion_id: InteractionWithTokenLogpReward}
            merged: dict[str, InteractionWithTokenLogpReward] = {}
            for result in valid_results:
                merged.update(result)
            return merged if merged else None

        # Otherwise, tensor dicts - concatenate
        concatenated = concat_padded_tensors(valid_results)
        return concatenated if concatenated else None

    def _normalize_group_rewards(
        self,
        results: list[dict[str, InteractionWithTokenLogpReward] | None],
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Applies normalize_group_rewards (from the OpenAI types module) to the
        # full result list, mutating rewards in place. Returns True on success;
        # otherwise counts rollouts that are missing or whose last interaction has no
        # reward, warns, and returns False so the caller drops the group.
        """Apply per-prompt reward normalization across the n_samples rollouts.

        One scalar reward per rollout is taken from the last interaction in
        each result. If any rollout failed or has no reward, the whole group is
        dropped so the normalization base always matches the configured group
        size.
        """
        from areal.experimental.openai.types import normalize_group_rewards

        if normalize_group_rewards(results):
            return True
        invalid_count = sum(
            1
            for result in results
            if not result or result[next(reversed(result))].reward is None
        )
        if invalid_count > 0:
            self.logger.warning(
                f"reward_normalization: dropping group ({invalid_count}/"
                f"{self.group_size} rollouts have None reward)"
            )
        return False


class RemoteInfBackendProtocol(Protocol):
    # [CLASS DOC]
    # Protocol of backend-specific operations: building/parsing generation and scoring
    # requests, building disk and distributed weight-update requests and the weight
    # group init request, control requests (pause, resume, health, abort-all, offload,
    # onload) and launching a server subprocess. Backends may raise NotImplementedError
    # for unsupported features. A backend may additionally provide get_pause_requests
    # returning several pause requests.
    """Protocol defining backend-specific operations for remote inference engines.

    This protocol abstracts the differences between various remote inference servers
    (SGLang, vLLM, etc.) by defining a common interface for:
    - Building HTTP requests with backend-specific formats
    - Parsing backend-specific responses
    - Handling weight updates
    - Managing control flow (pause/resume)
    - Supporting optional features (LoRA)

    Implementations can raise NotImplementedError for unsupported features.
    """

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: build the HTTP generation request for req, optionally targeting the
        # LoRA adapter for the given weight version.
        """Build HTTP request for text generation.

        Parameters
        ----------
        req : ModelRequest
            The generation request containing input and parameters
        with_lora : bool
            Whether to specify a LoRA to use
        version : int
            The current weight version for versioned LoRA names

        Returns
        -------
        HttpRequest
            The HTTP request with endpoint and payload
        """
        ...

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        # [FUNCTION DOC]
        # Protocol: parse a server JSON response into output tokens, logprobs, stop
        # reason and optional routed experts.
        """Parse generation response into standard format.

        Parameters
        ----------
        response : Dict[str, Any]
            The raw JSON response from the server

        Returns
        -------
        HttpGenerationResult
            Parsed result with tokens, logprobs, and stop reason
        """
        ...

    def build_score_request(
        self, input_ids: list[int], target_len: int, with_lora: bool, version: int
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: build a request scoring the last target_len tokens of input_ids.
        """Build HTTP request for token log-prob scoring."""
        ...

    def parse_score_response(
        self, response: dict[str, Any], target_len: int
    ) -> list[float]:
        # [FUNCTION DOC]
        # Protocol: extract target_len per-token logprobs from the scoring response.
        """Parse token log-prob scoring response."""
        ...

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        # [FUNCTION DOC]
        # Protocol: build the ordered requests that load weights from meta.path.
        """Build requests for loading weights from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing path and configuration

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests (may be multiple for LoRA workflows)
        """
        ...

    def build_distributed_weight_update_requests(
        self,
        meta: WeightUpdateMeta,
        param_specs: list[ParamSpec],
    ) -> WeightUpdateRequests:
        # [FUNCTION DOC]
        # Protocol: build the ordered requests that receive the given parameters via
        # collective broadcast.
        """Build requests for distributed weight update via NCCL/XCCL.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing communication group info
        param_specs : List[ParamSpec]
            Specifications for parameters to be updated

        Returns
        -------
        WeightUpdateRequests
            Collection of HTTP requests for distributed update
        """
        ...

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: build the request making server addr join the weight-update group
        # with rank server_idx.
        """Build request to initialize weight update xccl group.

        Parameters
        ----------
        addr : str
            Server address
        server_idx : int
            Index of this server in the server list
        meta : WeightUpdateMeta
            Metadata containing communication backend configuration

        Returns
        -------
        HttpRequest
            The HTTP request to initialize the group
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request pausing generation (may raise NotImplementedError).
        """Get request to pause generation.

        Returns
        -------
        HttpRequest
            The HTTP request to pause generation

        Raises
        ------
        NotImplementedError
            If pause is not supported by this backend
        """
        ...

    def get_resume_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request resuming generation (may raise NotImplementedError).
        """Get request to resume generation.

        Returns
        -------
        HttpRequest
            The HTTP request to resume generation

        Raises
        ------
        NotImplementedError
            If resume is not supported by this backend
        """
        ...

    def get_health_check_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request used for health checks (200 means healthy).
        """Get the health check request.

        Returns
        -------
        HttpRequest
            The HTTP request for health checks
        """
        ...

    def get_abort_all_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request aborting all in-flight generations.
        """Get request to abort all in-flight requests."""
        ...

    def get_offload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request releasing model memory (optionally only tagged components).
        """Get request to offload model memory.

        Returns
        -------
        HttpRequest
            The HTTP request to offload model memory

        Raises
        ------
        NotImplementedError
            If offload is not supported by this backend
        """
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: request restoring model memory (optionally only tagged components).
        """Get request to onload model memory.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.

        Returns
        -------
        HttpRequest
            The HTTP request to onload model memory

        Raises
        ------
        NotImplementedError
            If onload is not supported by this backend
        """
        ...

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        # [FUNCTION DOC]
        # Protocol: start a server subprocess from server_args and return its Popen.
        """Launch inference server subprocess.

        Parameters
        ----------
        server_args : dict[str, Any]
            Server configuration arguments for build_cmd_from_args

        Returns
        -------
        subprocess.Popen
            The launched server process
        """
        ...


class RemoteInfEngine(InferenceEngine):
    # [CLASS DOC]
    # InferenceEngine implementation over HTTP, parameterized by a backend (composition,
    # not subclassing). State: config, backend, server addresses and round-robin index,
    # a bounded rid-to-server map (RID_CACHE_SIZE most recent request ids) for KV-cache
    # reuse, the current weight version guarded by a lock, the WorkflowExecutor, an
    # optional proxy gateway address, and locally launched server processes.
    """
    Base implementation for HTTP-based remote inference engines.

    This class provides common functionality for communicating with remote
    inference servers via HTTP REST APIs. Backend-specific behaviors are
    delegated to an injected RemoteInfBackendProtocol implementation.

    Uses composition pattern - instantiate directly with a backend rather
    than inheriting from this class.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    backend : RemoteInfBackendProtocol
        Backend implementation providing server-specific behavior
    """

    def __init__(
        self, config: InferenceEngineConfig, backend: RemoteInfBackendProtocol
    ):
        # [FUNCTION DOC]
        # Stores config and backend and initializes empty routing state, version 0, a
        # lock, no executor, uninitialized flag and empty local server list. No network
        # I/O.
        self.config = config
        self.backend = backend

        self.rid_to_address = {}
        # Maintain the addresses for the recent 128 requests
        self.rid_queue = []
        self.addresses = []
        self.server_idx = 0

        self._version = 0

        self.lock = Lock()

        self._workflow_executor: WorkflowExecutor | None = None
        self._initialized = False
        self._proxy_gateway_addr: str | None = None
        self.local_server_processes: list[LocalInfServerInfo] = []

    def _wait_for_server(self, address: str, process: subprocess.Popen | None = None):
        # [FUNCTION DOC]
        # HELPER: Polls check_health on the (IPv6-normalized) address every second until
        # healthy or config.setup_timeout elapses (TimeoutError). If process is given
        # and has exited, raises RuntimeError with its exit code. Logs a progress
        # message every 60s.
        """Wait for a server to become healthy."""
        try:
            host, port = split_hostport(address)
            base_url = f"http://{format_hostport(host, port)}"
        except ValueError:
            base_url = f"http://{address}"
        tik = time.time()
        last_report = tik
        while time.time() - tik < self.config.setup_timeout:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"Inference server process (pid={process.pid}) exited with "
                    f"code {process.returncode} before becoming healthy at "
                    f"{address}. Search the worker log above for the server "
                    "traceback (e.g. scheduler init errors, port EADDRINUSE)."
                )
            if self.check_health(base_url):
                return
            now = time.time()
            if now - last_report >= 60:
                logger.info(
                    "Still waiting for inference server at %s to become "
                    "healthy (%.0fs elapsed, timeout %.0fs, process alive=%s)",
                    address,
                    now - tik,
                    self.config.setup_timeout,
                    process is not None and process.poll() is None,
                )
                last_report = now
            time.sleep(1)
        raise TimeoutError(
            f"Inference server at {address} failed to become healthy within "
            f"{self.config.setup_timeout}s"
        )

    def check_health(self, base_url):
        # [FUNCTION DOC]
        # Sends the backend health request to base_url with a 30s timeout; returns True
        # on HTTP 200, False on other statuses or request exceptions.
        """Check if server is healthy."""
        try:
            health_req = self.backend.get_health_check_request()
            url = f"{base_url}{health_req.endpoint}"
            response = requests.request(
                health_req.method, url, json=health_req.payload, timeout=30
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        # [FUNCTION DOC]
        # Sets engine_id (given, else the torch.distributed rank, else a random hex) and
        # a per-rank logger. Resolves server addresses in priority order: addr argument
        # (string or list), locally launched servers, name_resolve lookup for the
        # experiment/trial (1s timeout), then the AREAL_LLM_SERVER_ADDRS env var (comma-
        # separated) as fallback when name_resolve fails. Raises RuntimeError if none
        # found. Waits for each server to be healthy, picks a random starting round-
        # robin index, creates and initializes the WorkflowExecutor (with
        # train_data_parallel_size), and marks initialized.
        # KNOWN ISSUE: The env-var fallback is only consulted when experiment and trial
        # names are set and name_resolve fails, not when those names are None.
        """Initialize the engine by discovering and connecting to servers.

        Parameters
        ----------
        engine_id : Optional[str]
            Unique identifier for this engine instance
        addr : str | List[str] | None
            Server address(es) to connect to. If None, will auto-discover.
        train_data_parallel_size : int | None
            Data parallel size of the training engine
        """
        if engine_id is None:
            if dist.is_initialized():
                engine_id = str(dist.get_rank())
            else:
                engine_id = uuid.uuid4().hex
        self.engine_id = engine_id
        self.logger = logging.getLogger(f"[RemoteInfEngine Rank {engine_id}]")

        if addr:
            self.addresses = addr if isinstance(addr, list) else [addr]
            self.logger.info("Get server addresses from the `addr` argument.")
        elif len(self.local_server_processes) > 0:
            self.addresses = [
                format_hostport(s.host, s.port) for s in self.local_server_processes
            ]
            self.logger.info("Get server addresses from the local subprocess.")
        elif (
            self.config.experiment_name is not None
            and self.config.trial_name is not None
        ):
            try:
                self.addresses = wait_llm_server_addrs(
                    experiment_name=self.config.experiment_name,
                    trial_name=self.config.trial_name,
                    timeout=1,
                )
                self.logger.info("Get server addresses from name_resolve.")
            except (TimeoutError, RuntimeError):
                self.logger.info(
                    "Failed to get server addresses from name_resolve, "
                    "falling back to environment variable."
                )
                addrs_str = os.getenv("AREAL_LLM_SERVER_ADDRS")
                if addrs_str:
                    # When addr is not provided, fallback to reading addrs from env var
                    self.addresses = addrs_str.split(",")
                    self.logger.info("Get server addresses from environment variable.")

        if not self.addresses:
            raise RuntimeError(
                "No configured inference servers. "
                "Please pass in server addresses by arguments "
                "for `initialize` or environment "
                "variable `AREAL_LLM_SERVER_ADDRS`."
            )

        self.logger.info("Waiting for server ready...")
        for addr_ in self.addresses:
            self._wait_for_server(addr_)
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.logger.info("Servers are all ready!")

        self.workflow_executor = WorkflowExecutor(
            config=self.config,
            inference_engine=self,
        )
        self.workflow_executor.initialize(
            logger=self.logger, train_data_parallel_size=train_data_parallel_size
        )
        self._initialized = True

    def destroy(self):
        # [FUNCTION DOC]
        # Marks uninitialized, destroys the workflow executor if present, and tears down
        # any locally launched servers.
        """Destroy the engine and clean up resources."""
        self._initialized = False
        if self._workflow_executor is not None:
            self._workflow_executor.destroy()
        if len(self.local_server_processes) > 0:
            self.teardown_server()

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        # [FUNCTION DOC]
        # Property: returns the WorkflowExecutor; raises RuntimeError if not
        # initialized.
        """Get the workflow executor of the inference engine."""
        if self._workflow_executor is None:
            raise RuntimeError("WorkflowExecutor is not initialized")
        return self._workflow_executor

    @workflow_executor.setter
    def workflow_executor(self, workflow_executor: WorkflowExecutor):
        # [FUNCTION DOC]
        # Setter: replaces the WorkflowExecutor.
        """Set the workflow executor of the inference engine."""
        self._workflow_executor = workflow_executor

    @property
    def initialized(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether initialize completed and destroy has not been called.
        return self._initialized

    def set_version(self, version):
        # [FUNCTION DOC]
        # Sets the current weight version under the lock.
        """Set the current weight version."""
        with self.lock:
            self._version = version

    def get_version(self):
        # [FUNCTION DOC]
        # Returns the current weight version under the lock.
        """Get the current weight version."""
        with self.lock:
            return self._version

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Scores trajectories with the remote servers. For each trajectory dict (2D
        # input_ids and loss_mask; ValueError otherwise) builds a float32 output shaped
        # like loss_mask. For each row with a positive number of loss-masked tokens,
        # takes the token ids (filtered by attention_mask when present), sends a
        # blocking scoring request to the next round-robin server, checks that exactly
        # target_len logprobs came back (ValueError otherwise), and writes them at the
        # loss-mask positions. Returns one tensor per trajectory. HTTP errors raise via
        # raise_for_status.
        # KNOWN ISSUE: Assumes the loss-masked tokens are exactly the last target_len
        # scored tokens; masks with gaps would misalign. Requests are sequential and
        # synchronous.
        results: list[torch.Tensor] = []
        timeout = self.config.request_timeout
        version = self.get_version()
        for traj in data:
            input_ids = traj["input_ids"]
            loss_mask = traj["loss_mask"]
            if input_ids.dim() != 2 or loss_mask.dim() != 2:
                raise ValueError("input_ids and loss_mask must be 2D tensors")
            bs = input_ids.shape[0]
            out = torch.zeros_like(loss_mask, dtype=torch.float32)
            for i in range(bs):
                token_ids = input_ids[i].tolist()
                target_len = int(loss_mask[i].sum().item())
                if target_len <= 0:
                    continue
                if "attention_mask" in traj:
                    attn_mask = traj["attention_mask"][i]
                    active_idx = torch.nonzero(attn_mask, as_tuple=False).squeeze(-1)
                    token_ids = input_ids[i, active_idx].tolist()
                else:
                    token_ids = input_ids[i].tolist()
                server_addr = self.choose_server()
                http_req = self.backend.build_score_request(
                    input_ids=token_ids,
                    target_len=target_len,
                    with_lora=self.config.use_lora,
                    version=version,
                )
                response = requests.request(
                    http_req.method,
                    f"http://{server_addr}{http_req.endpoint}",
                    json=http_req.payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                token_logps = self.backend.parse_score_response(payload, target_len)
                if len(token_logps) != target_len:
                    raise ValueError(
                        f"Expected {target_len} token logprobs, got {len(token_logps)}"
                    )
                write_idx = torch.nonzero(loss_mask[i], as_tuple=False).squeeze(-1)
                out[i, write_idx] = torch.tensor(
                    token_logps, device=out.device, dtype=out.dtype
                )
            results.append(out)
        return results

    def set_proxy_gateway_addr(self, addr: str) -> None:
        # [FUNCTION DOC]
        # Stores the proxy gateway address later passed to OpenAIProxyWorkflow wrappers.
        """Set the proxy gateway address.

        Called by ``RolloutController.start_proxy_gateway()`` via
        collective RPC after the proxy gateway is started.
        """
        # HACK: We'd better not have this kind of setter beyond well-defined APIs,
        # but for now it's a workaround for the next release.
        self._proxy_gateway_addr = addr

    def _wrap_openai_agent(self, agent: Any, proxy_addr: str) -> RolloutWorkflow:
        # [FUNCTION DOC]
        # HELPER: Wraps an agent object (or None in online mode) in an
        # OpenAIProxyWorkflow configured from config.agent (mode, admin key, turn
        # discount, export style, subprocess workers, retry-orphan dropping) plus
        # proxy_addr and the stored proxy gateway address.
        """Wrap an agent workflow in OpenAIProxyWorkflow (HTTP mode only).

        Parameters
        ----------
        agent : Any | None
            The agent workflow to wrap (any class with async run() method).
            ``None`` is valid when ``mode='online'``.
        proxy_addr : str
            HTTP address of the proxy server (required)
        """
        from areal.experimental.openai import OpenAIProxyWorkflow

        agent_cfg = self.config.agent

        return OpenAIProxyWorkflow(
            mode=agent_cfg.mode,
            agent=agent,
            proxy_addr=proxy_addr,
            admin_api_key=agent_cfg.admin_api_key,
            discount=agent_cfg.turn_discount,
            export_style=agent_cfg.export_style,
            subproc_max_workers=agent_cfg.subproc_max_workers,
            proxy_gateway_addr=self._proxy_gateway_addr,
            drop_retry_orphans=agent_cfg.drop_retry_orphans,
        )

    def _resolve_workflow(
        self,
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> RolloutWorkflow:
        # [FUNCTION DOC]
        # HELPER: Normalizes a WorkflowLike into a RolloutWorkflow. None means online
        # agent mode (requires config.agent.mode == "online" and proxy_addr, else
        # ValueError). A RolloutWorkflow instance is used directly (kwargs ignored with
        # warning). A RolloutWorkflow subclass is instantiated with workflow_kwargs
        # (required). A string is imported: RolloutWorkflow classes are instantiated
        # (kwargs required), instances used directly, and anything else is treated as an
        # agent (instantiated with kwargs if a class) and wrapped via the OpenAI proxy
        # (proxy_addr required). Other classes and other objects are treated as agents
        # likewise. If group_size > 1 the result is wrapped in GroupedRolloutWorkflow
        # with the normalization/drop flags. Import failures raise ValueError.
        resolved: RolloutWorkflow

        # 0. None workflow = online mode (config-driven)
        if workflow is None:
            agent_cfg = self.config.agent
            if agent_cfg is None or agent_cfg.mode != "online":
                raise ValueError(
                    "workflow is None but AgentConfig.mode is not 'online'. "
                    "Provide a workflow or set mode='online' in the config."
                )
            if proxy_addr is None:
                raise ValueError("proxy_addr is required for online mode")
            resolved = self._wrap_openai_agent(None, proxy_addr=proxy_addr)
            if group_size > 1:
                resolved = GroupedRolloutWorkflow(
                    resolved,
                    group_size,
                    self.logger,
                    reward_normalization=reward_normalization,
                    drop_incomplete_group=drop_incomplete_group,
                )
            return resolved

        # 1. Already a RolloutWorkflow instance
        if isinstance(workflow, RolloutWorkflow):
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = workflow

        # 2. RolloutWorkflow class
        elif isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            if workflow_kwargs is None:
                raise ValueError(
                    f"workflow_kwargs is required when workflow is a class. "
                    f"Got workflow={workflow}, but workflow_kwargs=None."
                )
            resolved = workflow(**workflow_kwargs)

        # 3. String import path
        elif isinstance(workflow, str):
            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import workflow from string {workflow!r}: {e}"
                ) from e

            # Check if it's a RolloutWorkflow class
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                if workflow_kwargs is None:
                    raise ValueError(
                        f"workflow_kwargs is required when workflow is a class or string. "
                        f"Got workflow={workflow}, but workflow_kwargs=None."
                    )
                resolved = imported_obj(**workflow_kwargs)

            # Check if it's a RolloutWorkflow instance
            elif isinstance(imported_obj, RolloutWorkflow):
                if workflow_kwargs is not None:
                    self.logger.warning(
                        "workflow_kwargs is ignored when workflow resolves to an instance"
                    )
                resolved = imported_obj

            # Otherwise, treat it as an agent-like workflow (needs proxy)
            else:
                if proxy_addr is None:
                    raise ValueError(
                        f"proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                        f"Ensure proxy workers are initialized via RolloutController.start_proxy(). "
                        f"Got workflow={workflow!r}"
                    )

                # Instantiate if it's a class
                if isinstance(imported_obj, type):
                    agent = imported_obj(**(workflow_kwargs or {}))
                else:
                    # Already an instance
                    agent = imported_obj

                resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 4. Callable class (agent-like workflow)
        elif isinstance(workflow, type):
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            agent = workflow(**(workflow_kwargs or {}))
            resolved = self._wrap_openai_agent(agent, proxy_addr=proxy_addr)

        # 5. Instance of agent-like workflow
        else:
            if proxy_addr is None:
                raise ValueError(
                    "proxy_addr is required for agent workflows (non-RolloutWorkflow). "
                    "Ensure proxy workers are initialized via RolloutController.start_proxy()."
                )
            if workflow_kwargs is not None:
                self.logger.warning(
                    "workflow_kwargs is ignored when workflow is already an instance"
                )
            resolved = self._wrap_openai_agent(workflow, proxy_addr=proxy_addr)

        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            resolved = GroupedRolloutWorkflow(
                resolved,
                group_size,
                self.logger,
                reward_normalization=reward_normalization,
                drop_incomplete_group=drop_incomplete_group,
            )

        return resolved

    def _resolve_should_accept_fn(
        self, should_accept_fn: Callable[[dict[str, Any]], bool] | str | None
    ) -> Callable[[dict[str, Any]], bool] | None:
        # [FUNCTION DOC]
        # HELPER: Returns None or a callable unchanged; imports a dotted-path string and
        # checks it is callable. Raises ValueError on import failure and TypeError for
        # non-callables or unsupported types.
        """Resolve should_accept_fn parameter to a callable or None.

        Parameters
        ----------
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str | None
            The should_accept_fn specification

        Returns
        -------
        Callable[[Dict[str, Any]], bool] | None
            A callable for trajectory filtering, or None

        Raises
        ------
        ValueError
            If string import fails
        TypeError
            If imported object is not callable
        """
        if should_accept_fn is None or callable(should_accept_fn):
            return should_accept_fn

        if isinstance(should_accept_fn, str):
            try:
                func = import_from_string(should_accept_fn)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import should_accept_fn from string {should_accept_fn!r}: {e}"
                ) from e
            if not callable(func):
                raise TypeError(
                    f"Imported object {func} from {should_accept_fn!r} is not callable"
                )
            return func

        raise TypeError(
            f"Invalid should_accept_fn type: {type(should_accept_fn)}. "
            f"Expected callable or string module path."
        )

    def choose_server(self) -> str:
        # [FUNCTION DOC]
        # Returns the next server address in round-robin order, advancing the index.
        # Raises NotImplementedError for any routing strategy other than "round_robin".
        # KNOWN ISSUE: The index update is not protected by a lock.
        """Choose a server based on the routing strategy.

        Returns
        -------
        str
            Selected server address

        Raises
        ------
        NotImplementedError
            If routing strategy other than round-robin is used
        """
        if self.config.routing_strategy == "round_robin":
            server = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return server
        raise NotImplementedError("Only round-robin scheduling is implemented.")

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        # [FUNCTION DOC]
        # Async. Generates a completion for a single-sample request. Copies req, sets
        # return_routed_experts metadata if configured, rejects n_samples != 1
        # (ValueError), and clamps max_new_tokens to max_tokens minus prompt length
        # (RuntimeError if non-positive). Picks a server, reusing the one previously
        # used for the same rid (bounded FIFO cache). Then loops until stop reason is
        # stop/tool_calls/length or the token budget is used: waits while the workflow
        # executor is paused, pins the current weight version, sends the backend
        # generation request with retries via the shared aiohttp session, requires a
        # dict response, parses it, validates routed experts were returned when
        # requested (unless aborted), accumulates tokens, logprobs, per-token versions
        # and routed experts, and appends the new tokens to the prompt while reducing
        # the remaining budget (so aborted requests resume after weight updates). A
        # final "abort" stop reason is reported as "length". Returns a ModelResponse
        # with the original prompt tokens, outputs, versions, latency (ttft equals
        # latency) and concatenated routed experts.
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        # Create a shallow copy of the input request
        # we are going to modify it in-place
        req = req.copy()

        # Populate return_routed_experts from config to metadata
        if self.config.return_routed_experts:
            req.metadata["return_routed_experts"] = True

        # Validate n_samples
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError(
                "Inference engines do not support n_samples > 1. "
                "Please call generate multiple times with n_samples = 1."
            )

        # Validate max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids), gconfig.max_new_tokens
        )
        if max_new_tokens <= 0:
            raise RuntimeError(
                f"max_new_tokens ({max_new_tokens}) is non-positive! "
                f"max_tokens={gconfig.max_tokens}, prompt_len={len(req.input_ids)}, "
                f"max_new_tokens={gconfig.max_new_tokens}."
            )

        # Update max_new_tokens in request
        req.gconfig.max_new_tokens = max_new_tokens

        # Make request
        start_time = time.perf_counter()
        accumulated_output_tokens = []
        accumulated_output_logprobs = []
        accumulated_versions = []
        accumulated_routed_experts: list[np.ndarray] = []

        # A single "rid" shares the same server to allow KV cache reuse
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                # Remove the oldest entry if cache is full
                oldest_rid = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest_rid, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        # Get the shared session from workflow context
        session = await workflow_context.get_aiohttp_session()

        # Deal with rollout interruption
        stop_reason = None
        ori_max_new_tokens = gconfig.max_new_tokens
        while (
            stop_reason not in ["stop", "tool_calls", "length"]
            and len(accumulated_output_tokens) < ori_max_new_tokens
        ):
            # Request is interrupted, wait for some time to avoid interfering
            # with update weights requests
            while self.workflow_executor.is_paused():
                await asyncio.sleep(0.5)

            # Pin the version that serves this request. A trajectory may span
            # several weight versions, so each segment must be attributed to the
            # version that actually generated it rather than to whichever
            # version is current once the response arrives.
            request_version = self.get_version()

            # Build request using backend
            http_req = self.backend.build_generation_request(
                req,
                with_lora=self.config.use_lora,
                version=request_version,
            )

            # Loop until the generation is complete
            result = await arequest_with_retry(
                session=session,
                addr=server_addr,
                endpoint=http_req.endpoint,
                payload=http_req.payload,
                method=http_req.method,
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            # Assert response is JSON dict (not text/binary from error pages)
            if not isinstance(result, dict):
                raise ValueError(
                    f"Expected JSON dict response, got {type(result).__name__}"
                )

            # Parse response using backend
            gen_result = self.backend.parse_generation_response(result)
            stop_reason = gen_result.stop_reason

            if (
                req.metadata.get("return_routed_experts", False)
                and gen_result.routed_experts is None
            ):
                if stop_reason != "abort":  # Only validate for successful generations
                    raise RuntimeError(
                        "Requested return_routed_experts=True but received None from SGLang. "
                        "This usually means the model is not a MoE (Mixture of Experts) model. "
                        "Please use a MoE model to get routed_experts information."
                    )

            # Update accumulated outputs
            accumulated_output_tokens.extend(gen_result.output_tokens)
            accumulated_output_logprobs.extend(gen_result.output_logprobs)
            accumulated_versions.extend(
                [request_version] * len(gen_result.output_tokens)
            )
            # Accumulate routed_experts for MoE models
            if gen_result.routed_experts is not None:
                accumulated_routed_experts.append(gen_result.routed_experts)

            # Update request for next iteration
            req.input_ids += gen_result.output_tokens
            req.gconfig.max_new_tokens -= len(gen_result.output_tokens)
            assert req.gconfig.max_new_tokens >= 0, (
                req.gconfig.max_new_tokens,
                len(gen_result.output_tokens),
                len(req.input_ids),
            )

        # Final abort handling
        if stop_reason == "abort":
            # If stop_reason is "abort", the only reason we exit the loop is
            # len(accumulated_output_tokens) >= gconfig.max_new_tokens
            # so the actual reason is length
            stop_reason = "length"

        latency = time.perf_counter() - start_time

        accumulated_routed_experts = (
            np.concatenate(accumulated_routed_experts)
            if accumulated_routed_experts
            else None
        )

        response = ModelResponse(
            input_tokens=req.input_ids[
                : len(req.input_ids) - len(accumulated_output_tokens)
            ],
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,  # Simplified for non-streaming
            tokenizer=req.tokenizer,
            processor=req.processor,
            routed_experts=accumulated_routed_experts,
        )
        return response

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        # [FUNCTION DOC]
        # Asserts meta.type is "xccl" and submits _init_weights_update_group_remote to
        # the shared executor for all addresses (optional explicit per-address ranks).
        # Attaches a callback logging success or failure and returns the Future.
        """Initialize the weight update process group for distributed weight updates.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        xccl_group_ranks : list[int] | None, optional
            Explicit rank assignment for each remote inference worker, aligned with
            ``self.addresses`` (same length, same order).

            - If provided, worker at ``self.addresses[i]`` will initialize the
            communication group using rank ``xccl_group_ranks[i]``.
            - If None, ranks are assigned by address order: rank ``i`` for
            ``self.addresses[i]``.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation
        """
        assert meta.type == "xccl"

        self.logger.info(
            "Initializing weight update group: group=%s, addresses=%s",
            meta.nccl_group_name,
            self.addresses,
        )

        fut = get_executor().submit(
            _init_weights_update_group_remote,
            self.backend,
            meta,
            self.addresses,
            self.config.request_timeout,
            xccl_group_ranks,
        )

        def callback(fut):
            # [FUNCTION DOC]
            # HELPER: Done-callback logging failure (with the exception) or success;
            # ignores cancellation.
            if fut.cancelled():
                return
            if fut.exception() is not None:
                self.logger.error(
                    "Failed to initialize %s group for distributed weight update for %s: %s",
                    current_platform.communication_backend.upper(),
                    meta.nccl_group_name,
                    repr(fut.exception()),
                )
                return
            self.logger.info(
                f"Initialized {current_platform.communication_backend.upper()} group "
                f"for distributed weight update for {meta.nccl_group_name}."
            )

        fut.add_done_callback(callback)
        return fut

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        # [FUNCTION DOC]
        # Asserts meta.type is "xccl" and submits _update_weights_from_distributed for
        # all servers to the shared executor; returns the Future.
        """Update weights in the inference engine from distributed memory.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "xccl"

        fut = get_executor().submit(
            _update_weights_from_distributed,
            self.backend,
            meta,
            param_specs,
            self.addresses,
            self.config.request_timeout,
        )

        return fut

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        # [FUNCTION DOC]
        # Asserts meta.type is "disk"; raises RuntimeError if experiment/trial names are
        # missing. Submits _update_weights_from_disk (current version, all servers) to
        # the shared executor and attaches a callback that logs timing and deletes
        # meta.path when clear_checkpoint_after_load. Returns the Future (resolving to
        # load-minus-save seconds).
        # KNOWN ISSUE: Comment mentions a ProcessPool but the shared executor is used;
        # the callback calls result(), so a failed update raises inside the callback
        # (logged by the Future machinery) rather than being reported here.
        """Update weights in the inference engine from disk.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        assert meta.type == "disk"

        tik = time.perf_counter()

        # Use ProcessPool to bypass python GIL for running async coroutines
        if self.config.experiment_name is None or self.config.trial_name is None:
            raise RuntimeError(
                "Experiment and trial names must be set for disk-based weight updates."
            )

        fut = get_executor().submit(
            _update_weights_from_disk,
            self.backend,
            self.config.experiment_name,
            self.config.trial_name,
            self.get_version(),
            self.addresses,
            meta,
            self.config.request_retries,
            self.config.request_timeout,
        )

        def callback(fut):
            # [FUNCTION DOC]
            # HELPER: Logs total and respond time and optionally removes the checkpoint
            # directory.
            respond_time = fut.result()
            self.logger.info(
                f"Loading weights from disk done "
                f"in {(time.perf_counter() - tik):.2f}s. "
                f"Respond time: {respond_time:.2f}s."
            )
            if meta.clear_checkpoint_after_load:
                shutil.rmtree(meta.path, ignore_errors=True)

        fut.add_done_callback(callback)
        return fut

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        callback_addr: str | None = None,
        is_eval: bool = False,
        proxy_addr: str | None = None,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> int:
        # [FUNCTION DOC]
        # Submits one rollout task and returns its task id immediately. Raises
        # ValueError if workflow is None outside online agent mode. Registers
        # callback_addr for task_id with the dispatcher if given, resolves the workflow
        # (grouping, proxy) and should_accept_fn, and delegates to
        # WorkflowExecutor.submit with task_id and is_eval.
        """Submit a request to the inference engine and return immediately.

        Parameters
        ----------
        data : Dict[str, Any]
            The input data for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        task_id : int, optional
            The task ID to use. If None, a new task ID will be generated internally.
        is_eval : bool, optional
            Whether this is an evaluation workflow. Affects variables like trajectory dump path
            and statistics keys. By default False.
        proxy_addr : str, optional
            HTTP address of the proxy server for AgentWorkflow. If provided,
            AgentWorkflow will use this proxy instead of a local one.
        """
        if workflow is None and (
            self.config.agent is None or self.config.agent.mode != "online"
        ):
            raise ValueError(
                "workflow must be specified for submit (unless mode='online')"
            )
        if callback_addr:
            self.workflow_executor.dispatcher.register_callback(task_id, callback_addr)

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            proxy_addr=proxy_addr,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.submit(
            data,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            task_id=task_id,
            is_eval=is_eval,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # [FUNCTION DOC]
        # Delegates to WorkflowExecutor.wait: blocks until count accepted trajectories
        # are ready or timeout (raising TimeoutError if raise_timeout, else returning
        # what is available).
        """Wait for a specified number of requests to complete.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds
        raise_timeout : bool, optional
            Whether to raise a TimeoutError when the timeout is exceeded, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.
            Returns an empty list if timeout exceeded and raise_timeout is False.
        """
        return self.workflow_executor.wait(
            count, timeout=timeout, raise_timeout=raise_timeout
        )

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Delegates to WorkflowExecutor.wait_for_task for a single task id.
        """Wait for a specific submitted task to complete."""
        return self.workflow_executor.wait_for_task(task_id, timeout, raise_timeout)

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
        # Synchronously rolls out a list of inputs: resolves the workflow (no proxy
        # address, so agent workflows fail) and delegates to
        # WorkflowExecutor.rollout_batch. Asserts workflow is not None.
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        Parameters
        ----------
        data : List[Dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for rollout_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

        return self.workflow_executor.rollout_batch(
            data=data,
            workflow=resolved_workflow,
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
        # Resolves workflow and should_accept_fn and delegates to
        # WorkflowExecutor.prepare_batch, which keeps submitting from dataloader until a
        # full (optionally dynamic-size) batch of accepted trajectories is ready.
        # Asserts workflow is not None.
        """Asynchronously submit and wait until a full batch is ready.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from
        workflow : WorkflowLike
            The workflow to use for rollout generation
        workflow_kwargs : Dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor
        should_accept_fn : Callable[[Dict[str, Any]], bool] | str, optional
            A function or module path for trajectory filtering
        group_size : int
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        assert workflow is not None, "Workflow must be specified for prepare_batch."

        # Resolve workflow to a RolloutWorkflow instance
        resolved_workflow = self._resolve_workflow(
            workflow,
            workflow_kwargs,
            group_size,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )
        resolved_should_accept_fn = self._resolve_should_accept_fn(should_accept_fn)

        return self.workflow_executor.prepare_batch(
            dataloader=dataloader,
            workflow=resolved_workflow,
            should_accept_fn=resolved_should_accept_fn,
            dynamic_bs=dynamic_bs,
        )

    @trace_perf("remote_inf_engine.pause_generation", category="misc")
    def pause_generation(self):
        # [FUNCTION DOC]
        # Sends pause request(s) (backend get_pause_requests if available, else a single
        # get_pause_request) to all servers, then sleeps config.pause_grace_period so
        # in-flight requests are aborted. Traced with perf_tracer.
        """Pause request submission for async rollout."""
        get_pause_requests = getattr(self.backend, "get_pause_requests", None)
        pause_requests = (
            get_pause_requests()
            if get_pause_requests is not None
            else [self.backend.get_pause_request()]
        )
        for pause_req in pause_requests:
            self._run_request_on_all_servers(pause_req)

        # The above http request may require some time to be scheduled and executed.
        # The following line waits until all requests are indeed dropped.
        time.sleep(self.config.pause_grace_period)

    @trace_perf("remote_inf_engine.continue_generation", category="misc")
    def continue_generation(self):
        # [FUNCTION DOC]
        # Sends the backend resume request to all servers. Traced with perf_tracer.
        """Resume request submission for async rollout."""
        resume_req = self.backend.get_resume_request()
        self._run_request_on_all_servers(resume_req)

    def pause(self):
        # [FUNCTION DOC]
        # Pauses new rollout submission in the WorkflowExecutor (used during
        # evaluation).
        """Pause request submission for async rollout.
        Used during evaluation to prevent data over generation.
        """
        return self.workflow_executor.pause()

    def resume(self):
        # [FUNCTION DOC]
        # Resumes rollout submission in the WorkflowExecutor.
        """Resume request submission for async rollout."""
        return self.workflow_executor.resume()

    def offload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Sends the backend offload request (optional tags) to all servers, logging
        # before and after.
        """Offload model memory on all servers."""
        offload_req = self.backend.get_offload_request(tags=tags)
        self.logger.info(
            "RemoteInfEngine.offload(tags=%s) sending to %s: endpoint=%s",
            tags,
            self.addresses,
            offload_req.endpoint,
        )
        self._run_request_on_all_servers(offload_req)
        self.logger.info("RemoteInfEngine.offload(tags=%s) completed", tags)

    def abort_all_requests(self) -> None:
        # [FUNCTION DOC]
        # Sends the backend abort-all request to all servers.
        """Abort all in-flight requests on all servers."""
        abort_req = self.backend.get_abort_all_request()
        self._run_request_on_all_servers(abort_req)

    def onload(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Sends the backend onload request (optional tags) to all servers.
        """Onload model memory on all servers."""
        onload_req = self.backend.get_onload_request(tags=tags)
        self._run_request_on_all_servers(onload_req)

    def _run_request_on_all_servers(self, req: HttpRequest):
        # [FUNCTION DOC]
        # HELPER: Blocking. Runs a fresh uvloop event loop that opens an aiohttp session
        # (request timeout, 10MB read buffer) and concurrently sends req to every
        # address with retries; the first failure propagates.
        # KNOWN ISSUE: uvloop.run fails if called from a thread that already runs an
        # event loop.
        async def _fn():
            # [FUNCTION DOC]
            # HELPER: Async body issuing the concurrent requests.
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            ) as session:
                jobs = []
                for addr in self.addresses:
                    jobs.append(
                        arequest_with_retry(
                            session=session,
                            addr=addr,
                            endpoint=req.endpoint,
                            payload=req.payload,
                            method=req.method,
                            max_retries=self.config.request_retries,
                            timeout=self.config.request_timeout,
                        )
                    )
                await asyncio.gather(*jobs)

        uvloop.run(_fn())

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        # [FUNCTION DOC]
        # Sets server_args host to this host's IP and port to a free port, launches the
        # server via the backend, and waits for it to become healthy (failing fast if
        # the process exits). On success records and returns a LocalInfServerInfo; on
        # TimeoutError/RuntimeError logs a warning, shuts the server down and re-raises.
        # Mutates server_args.
        """Launch a local inference server."""
        server_args["host"] = gethostip()
        server_args["port"] = find_free_ports(1)[0]
        process = self.backend.launch_server(server_args)
        address = format_hostport(server_args["host"], server_args["port"])
        server_info = LocalInfServerInfo(
            host=server_args["host"],
            port=server_args["port"],
            process=process,
        )
        try:
            self._wait_for_server(address, process=process)
            self.local_server_processes.append(server_info)
            return server_info
        except (TimeoutError, RuntimeError) as e:
            logger.warning(f"Launch local server failed at {address}: {e}")
            self._shutdown_one_server(server_info)
            raise

    def _shutdown_one_server(self, server_info: LocalInfServerInfo):
        # [FUNCTION DOC]
        # HELPER: Removes the server's address from the routing list and, if its process
        # is still running, kills its process tree gracefully.
        addr = format_hostport(server_info.host, server_info.port)
        if addr in self.addresses:
            self.addresses.remove(addr)
        if server_info.process.poll() is not None:
            return
        kill_process_tree(server_info.process.pid, graceful=True)

    def teardown_server(self):
        # [FUNCTION DOC]
        # Shuts down every locally launched server and clears the list.
        """Teardown all locally launched servers."""
        for server_info in self.local_server_processes:
            self._shutdown_one_server(server_info)
        self.local_server_processes.clear()


# Helper functions that run in ProcessPoolExecutor


def _update_weights_from_disk(
    backend: RemoteInfBackendProtocol,
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: list[str],
    meta: WeightUpdateMeta,
    request_retries: int,
    request_timeout: float,
):
    # [FUNCTION DOC]
    # HELPER: Runs in an executor worker. Waits (up to 120s) on name_resolve for the
    # trainer's save timestamp for model_version, records the load start time, builds
    # the backend's disk weight update requests, and sends each request (in order) to
    # all servers concurrently with retries. Best-effort requests have failures logged
    # and ignored; others propagate. Returns load timestamp minus save timestamp
    # (seconds).
    """Helper to update weights from disk in a separate process."""

    async def _fn():
        # [FUNCTION DOC]
        # HELPER: Async body of the disk update.
        update_name = names.update_weights_from_disk(
            experiment_name, trial_name, model_version
        )
        save_timestamp = float(name_resolve.wait(update_name, timeout=120))
        load_timestamp = datetime.now().timestamp()

        # Get requests from backend with version for LoRA name
        weight_reqs = backend.build_disk_weight_update_requests(meta)

        # Execute all requests
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=request_retries,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                if http_req.best_effort:
                    # Cleanup requests (e.g. unloading a stale LoRA adapter) must
                    # not fail the weight update: the target may already be gone.
                    results = await asyncio.gather(*jobs, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            logger.warning(
                                f"Best-effort request to {http_req.endpoint} "
                                f"failed (ignored): {r}"
                            )
                else:
                    await asyncio.gather(*jobs)

        return load_timestamp - save_timestamp

    return uvloop.run(_fn())


def _init_weights_update_group_remote(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    addresses: list[str],
    request_timeout: float,
    xccl_group_ranks: list[int] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: Runs in an executor worker. Validates xccl_group_ranks length matches
    # addresses (ValueError). Concurrently sends each server its group-init request
    # (rank from xccl_group_ranks or the address index, a single attempt), logs every
    # failure, then re-raises the first exception if any.
    """Helper to initialize weight update group in a separate process.

    If xccl_group_ranks is provided, it must have the same length as addresses and will be
    used as the per-address rank passed to the backend request builder.
    Otherwise, ranks default to enumerate(addresses).
    """

    if xccl_group_ranks is not None and len(xccl_group_ranks) != len(addresses):
        raise ValueError(
            f"xccl_group_ranks must have the same length as addresses "
            f"(got {len(xccl_group_ranks)} vs {len(addresses)})"
        )

    async def _fn():
        # [FUNCTION DOC]
        # HELPER: Async body issuing the init requests.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            jobs = []
            for i, addr in enumerate(addresses):
                xccl_group_rank = (
                    xccl_group_ranks[i] if xccl_group_ranks is not None else i
                )
                http_req = backend.build_init_weights_group_request(
                    addr, xccl_group_rank, meta
                )
                jobs.append(
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                )
            results = await asyncio.gather(*jobs, return_exceptions=True)
            for _idx, _r in enumerate(results):
                if isinstance(_r, Exception):
                    logger.error(
                        "init_weights_update_group request %d to %s failed: %s",
                        _idx,
                        addresses[_idx],
                        _r,
                    )
            # Re-raise first exception if any failed
            for _r in results:
                if isinstance(_r, Exception):
                    raise _r

    return uvloop.run(_fn())


def _update_weights_from_distributed(
    backend: RemoteInfBackendProtocol,
    meta: WeightUpdateMeta,
    param_specs: list[ParamSpec],
    addresses: list[str],
    request_timeout: float,
):
    # [FUNCTION DOC]
    # HELPER: Runs in an executor worker. Builds the backend's distributed update
    # requests and sends them in order, each to all servers concurrently (single
    # attempt); any failure propagates.
    """Helper to update weights from distributed memory in a separate process."""

    async def _fn():
        # [FUNCTION DOC]
        # HELPER: Async body issuing the update requests.
        # Get requests from backend
        weight_reqs = backend.build_distributed_weight_update_requests(
            meta, param_specs
        )

        # Execute all requests sequentially (they may have dependencies)
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            for http_req in weight_reqs.requests:
                jobs = [
                    arequest_with_retry(
                        session=session,
                        addr=addr,
                        endpoint=http_req.endpoint,
                        payload=http_req.payload,
                        method=http_req.method,
                        max_retries=1,
                        timeout=request_timeout,
                    )
                    for addr in addresses
                ]
                await asyncio.gather(*jobs)

    return uvloop.run(_fn())
