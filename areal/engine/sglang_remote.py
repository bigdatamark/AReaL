# [MODULE DOC]
# SGLang remote inference engine. SGLangBackend translates AReaL engine operations
# (generation, scoring, weight updates from disk or NCCL/XCCL, LoRA loading,
# pause/resume, memory offload/onload, health checks, server launch) into SGLang HTTP
# requests and parses its responses; RemoteSGLangEngine is the public InferenceEngine
# that composes a generic RemoteInfEngine with this backend and forwards every call to
# it. Logs via "SGLangRemote".
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

import numpy as np
import pybase64
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    InferenceEngine,
    LocalInfServerInfo,
    ModelAllocation,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    Scheduler,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.cli_args import InferenceEngineConfig, PerfTracerConfig, SGLangConfig
from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    WeightUpdateRequests,
    get_versioned_lora_name,
)
from areal.infra import RemoteInfEngine, RolloutController, WorkflowExecutor
from areal.infra.platforms import current_platform
from areal.infra.utils.launcher import TRITON_CACHE_PATH
from areal.utils import perf_tracer, stats_tracker
from areal.utils.logging import getLogger
from areal.utils.network import format_host_for_url

logger = getLogger("SGLangRemote")


class SGLangBackend:
    # [CLASS DOC]
    # Stateless SGLang protocol adapter consumed by RemoteInfEngine. Each build_* /
    # get_* method returns an HttpRequest (endpoint, JSON payload, optional
    # method/best_effort) or a WeightUpdateRequests bundle; parse_* methods turn SGLang
    # JSON into AReaL result types; launch_server starts the server process.
    """SGLang-specific backend implementation for remote inference."""

    @staticmethod
    def build_server_env(env: Mapping[str, str]) -> dict[str, str]:
        # [FUNCTION DOC]
        # Return a copy of env with TRITON_CACHE_PATH set to a fresh uuid4 subdirectory
        # of the existing value (or the default launcher path) so concurrent servers do
        # not share a Triton cache.
        _env = dict(env)
        triton_cache_path = _env.get("TRITON_CACHE_PATH", TRITON_CACHE_PATH)
        _env["TRITON_CACHE_PATH"] = os.path.join(triton_cache_path, str(uuid.uuid4()))
        return _env

    def build_generation_request(
        self, req: ModelRequest, with_lora: bool, version: int
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Build a POST /generate request from a ModelRequest. Sampling params: top_p,
        # top_k, max_new_tokens, temperature (0 when greedy), stop_token_ids,
        # ignore_eos, skip_special_tokens, frequency_penalty, plus stop strings and
        # sampling_seed when set. Payload includes a copy of input_ids, image_data,
        # return_logprob True, stream False, return_routed_experts when req.metadata
        # requests it, and lora_path set to the versioned LoRA name when with_lora.
        # Raises NotImplementedError for beam search and ValueError if with_lora but
        # gconfig.lora_name is empty.
        """Build SGLang generation request."""
        gconfig = req.gconfig
        stop_token_ids = gconfig.stop_token_ids
        stop = gconfig.stop

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Currently Beam search is not supported in SGLang backend."
            )

        sample_params = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if stop:
            sample_params["stop"] = stop
        if gconfig.seed is not None:
            sample_params["sampling_seed"] = gconfig.seed

        payload = {
            "input_ids": req.input_ids.copy(),
            "image_data": req.image_data,  # ImageObject or str
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
        }

        # Add return_routed_experts to payload if set
        if req.metadata.get("return_routed_experts", False):
            payload["return_routed_experts"] = True
        # Add LoRA if initialized
        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["lora_path"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_generation_response(
        self, response: dict[str, Any]
    ) -> HttpGenerationResult:
        # [FUNCTION DOC]
        # Parse a /generate response into HttpGenerationResult. Reads
        # meta_info.finish_reason type (stop reason) and message. If routed_experts is
        # present, base64-decodes it into an int32 numpy array reshaped to
        # (prompt_tokens + completion_tokens - 1, -1). For aborts with message starting
        # "Abort before prefill" returns empty tokens/logprobs; otherwise output tokens
        # and logprobs come from output_token_logprobs entries (logprob, token_id).
        # Raises KeyError on missing fields.
        """Parse SGLang generation response."""
        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason = finish_reason["type"]
        stop_message = finish_reason.get("message", "")

        # Extract routed_experts information if available
        routed_experts = meta_info.get("routed_experts", None)
        if routed_experts is not None:
            num_sgl_token = (
                meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
            )
            # Extract expert_id and reshape to (num_sgl_token, num_layers*expert_top_k)
            routed_experts = np.frombuffer(
                pybase64.b64decode(routed_experts.encode("utf-8")), dtype=np.int32
            ).reshape(num_sgl_token, -1)

        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
            )

        output_tokens = [x[1] for x in meta_info["output_token_logprobs"]]
        output_logprobs = [x[0] for x in meta_info["output_token_logprobs"]]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    def build_score_request(
        self, input_ids: list[int], target_len: int, with_lora: bool, version: int
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Build a /generate request that scores the last target_len tokens of input_ids:
        # generate 1 token greedily with return_logprob, logprob_start_len = max(0, len
        # - target_len - 1), no top logprobs. Raises NotImplementedError when with_lora.
        # version is unused.
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0.0,
            },
            "return_logprob": True,
            "logprob_start_len": max(0, len(input_ids) - target_len - 1),
            "top_logprobs_num": 0,
            "stream": False,
        }
        if with_lora:
            raise NotImplementedError(
                "LoRA scoring request is not supported in SGLang teacher compute_logp yet."
            )
        return HttpRequest(endpoint="/generate", payload=payload)

    def parse_score_response(
        self, response: dict[str, Any], target_len: int
    ) -> list[float]:
        # [FUNCTION DOC]
        # Return the last target_len input-token logprobs (floats) from
        # meta_info.input_token_logprobs. Raises ValueError if meta_info is missing or
        # fewer than target_len logprobs are returned.
        meta_info = response.get("meta_info")
        if meta_info is None:
            raise ValueError("SGLang response missing meta_info for score request")
        # SGLang returns [logprob, token_id, ...]
        all_logprobs = [float(x[0]) for x in meta_info.get("input_token_logprobs", [])]
        if len(all_logprobs) < target_len:
            raise ValueError(
                f"SGLang returned insufficient input_token_logprobs: {len(all_logprobs)} < {target_len}"
            )
        return all_logprobs[-target_len:]

    def build_disk_weight_update_requests(
        self, meta: WeightUpdateMeta
    ) -> WeightUpdateRequests:
        # [FUNCTION DOC]
        # Build disk-based weight update requests. LoRA (meta.use_lora): requires
        # lora_name and version (ValueError otherwise); returns a best-effort
        # /load_lora_adapter for the versioned name with meta.path, plus a best-effort
        # /unload_lora_adapter for version - lora_keep_versions when keep > 0 and that
        # version is >= 0 (bounded adapter retention). Full model: a single
        # /update_weights_from_disk with model_path and abort_all_requests True.
        """Build SGLang disk weight update requests."""
        if meta.use_lora:
            if not meta.lora_name:
                raise ValueError("LoRA name is required for LoRA update.")
            if meta.version is None:
                raise ValueError("Version is required for LoRA update.")
            lora_name = get_versioned_lora_name(meta.lora_name, meta.version)
            # Load new LoRA (best_effort: if already registered, SGLang
            # returns 400 which is silently ignored).
            requests = [
                HttpRequest(
                    endpoint="/load_lora_adapter",
                    payload={"lora_name": lora_name, "lora_path": str(meta.path)},
                    best_effort=True,
                ),
            ]
            # Unload the version that has fallen outside the retention window so
            # sglang does not accumulate one adapter per train step (which leaks
            # VRAM and eventually hangs). Kept versions cover off-policy rollouts
            # (max_head_offpolicyness). Best-effort: the stale adapter may have
            # already been evicted or never loaded.
            keep = meta.lora_keep_versions
            if keep > 0 and meta.version - keep >= 0:
                stale_name = get_versioned_lora_name(
                    meta.lora_name, meta.version - keep
                )
                requests.append(
                    HttpRequest(
                        endpoint="/unload_lora_adapter",
                        payload={"lora_name": stale_name},
                        best_effort=True,
                    )
                )
            return WeightUpdateRequests(requests=requests)
        else:
            # Full model update
            return WeightUpdateRequests(
                requests=[
                    HttpRequest(
                        endpoint="/update_weights_from_disk",
                        payload={
                            "model_path": str(meta.path),
                            "abort_all_requests": True,
                        },
                    )
                ]
            )

    def build_distributed_weight_update_requests(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> WeightUpdateRequests:
        # [FUNCTION DOC]
        # Build a single /update_weights_from_distributed request listing param names,
        # dtypes and shapes from param_specs, the NCCL group name and abort_all_requests
        # True. Raises ValueError for LoRA (unsupported over NCCL with SGLang).
        """Build SGLang distributed weight update requests.

        Note: SGLang distributed weight update (NCCL-based) does not support LoRA.
        For LoRA weight updates with SGLang, use disk-based update mode instead.
        """
        if meta.use_lora:
            raise ValueError(
                "SGLang distributed (XCCL/NCCL) weight update does not support LoRA. "
                "Use weight_update_mode='disk' for LoRA weight updates with SGLang."
            )
        return WeightUpdateRequests(
            requests=[
                HttpRequest(
                    endpoint="/update_weights_from_distributed",
                    payload={
                        "names": [pspec.name for pspec in param_specs],
                        "dtypes": [pspec.dtype for pspec in param_specs],
                        "shapes": [pspec.shape for pspec in param_specs],
                        "group_name": meta.nccl_group_name,
                        "abort_all_requests": True,
                    },
                )
            ]
        )

    def build_init_weights_group_request(
        self, addr: str, server_idx: int, meta: WeightUpdateMeta
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Build the /init_weights_update_group request for server number server_idx.
        # Asserts meta.gen_allocation is set. If inference pp_size > 1 and the group
        # name ends with "_<int>" (per-PP-stage groups), pp_rank comes from that suffix,
        # rank_offset = 1 + server_idx * tp_size, world_size = n_servers * tp_size + 1
        # with n_servers = world_size / (tp*pp), and pp_rank is included. Otherwise
        # rank_offset = 1 + server_idx * tp*pp and world_size = total inference world
        # size + 1 (rank 0 is the trainer). Payload also has master address (URL-
        # formatted host), port as string, platform communication backend and group
        # name. addr is unused.
        """Build SGLang init weights group request.

        Supports two scenarios:

        1. **PP=1** (original): Single NCCL group spanning all TP workers across
           all DP instances. ``rank_offset`` is based on ``tp_size``.

        2. **PP>1, per-PP-rank groups**: The training engine creates a separate
           NCCL group per PP stage. The group name encodes the PP rank
           (e.g., ``update_weight_group_0``). Only sglang workers at that PP
           rank participate, so ``rank_offset`` is based on ``tp_size`` only,
           ``world_size = n_servers * tp_size + 1``, and ``pp_rank`` is
           included in the payload.

        All three training engines (Megatron, FSDP, Archon) use per-PP-rank
        group naming (``update_weight_group_{pp_rank}``) when PP>1, so the
        per-PP-rank path is always taken for PP>1.
        """
        assert meta.gen_allocation is not None
        gen_parallel = meta.gen_allocation.parallel
        group_name = meta.nccl_group_name

        # Determine if training side uses per-PP-rank groups.
        # Per-PP-rank groups are identified by group names ending with _{digit}
        # and pp_size > 1. All engines use this pattern when PP>1.
        per_pp_groups = False
        if gen_parallel.pp_size > 1:
            try:
                _suffix = group_name.rsplit("_", 1)[-1]
                int(_suffix)
                per_pp_groups = True
            except (ValueError, IndexError):
                per_pp_groups = False

        if per_pp_groups:
            # Scenario 2: PP>1 with per-PP-rank groups.
            # Extract pp_rank from the group name suffix.
            pp_rank = int(group_name.rsplit("_", 1)[-1])

            tp_size = gen_parallel.tp_size
            pp_size = gen_parallel.pp_size
            # gen_parallel.world_size = dp_size * tp_size * pp_size on the
            # inference side. n_servers (number of sglang server replicas)
            # therefore equals dp_size whether or not DP-attention is
            # configured: the AReaL allocation always materialises one server
            # replica per DP shard, each running ``tp_size * pp_size`` workers.
            n_servers = gen_parallel.world_size // (tp_size * pp_size)

            # Each server contributes exactly ``tp_size`` workers per PP stage.
            # Across all replicas this PP stage has ``n_servers * tp_size``
            # inference workers (= dp_size * tp_size), all of which join the
            # per-PP NCCL group together with the trainer.
            rank_offset = 1 + server_idx * tp_size

            # world_size for this group: TP workers across all DP replicas at
            # this PP rank + 1 (trainer).
            world_size = n_servers * tp_size + 1

            payload = {
                "master_address": format_host_for_url(meta.nccl_master_address),
                "master_port": str(meta.nccl_master_port),
                "rank_offset": rank_offset,
                "world_size": world_size,
                "backend": current_platform.communication_backend,
                "group_name": group_name,
                "pp_rank": pp_rank,
            }
        else:
            instance_size = gen_parallel.tp_size * gen_parallel.pp_size
            rank_offset = 1 + server_idx * instance_size
            payload = {
                "master_address": format_host_for_url(meta.nccl_master_address),
                "master_port": str(meta.nccl_master_port),
                "rank_offset": rank_offset,
                "world_size": gen_parallel.world_size + 1,
                "backend": current_platform.communication_backend,
                "group_name": group_name,
            }

        return HttpRequest(endpoint="/init_weights_update_group", payload=payload)

    def get_pause_request(self, mode: str | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Build a /pause_generation request; includes "mode" only when given.
        """Get SGLang pause request."""
        payload = {} if mode is None else {"mode": mode}
        return HttpRequest(endpoint="/pause_generation", payload=payload)

    def get_pause_requests(self) -> list[HttpRequest]:
        # [FUNCTION DOC]
        # Return the two-step pause sequence: default-mode pause (aborts in-flight
        # requests and idles the scheduler) followed by an "in_place" pause (sets the
        # scheduler's paused flag needed by colocated awex weight work).
        """Pause in two steps so memory can be released safely.

        The first request keeps SGLang's default mode, which aborts in-flight
        requests and returns their partial output so the client resumes them by
        extending the prompt. That also leaves the scheduler fully idle, which
        SGLang requires before releasing memory.

        The second request looks redundant because Scheduler.pause_generation
        sets its paused flag unconditionally, but the abort path never reaches
        the scheduler: TokenizerManager forwards the request only for non-abort
        modes, and otherwise just drains via abort_request(). Abort therefore
        raises the tokenizer's own gate while the scheduler keeps scheduling.
        Only an in-place pause raises the scheduler flag that the colocate loop
        watches before it services awex work, and by then the abort has already
        left nothing for that mode to retain.
        """
        return [
            self.get_pause_request(),
            self.get_pause_request(mode="in_place"),
        ]

    def get_resume_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Build the /continue_generation request.
        """Get SGLang resume request."""
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_abort_all_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Build an /abort_request request with abort_all True.
        """Get SGLang abort all requests."""
        return HttpRequest(endpoint="/abort_request", payload={"abort_all": True})

    def get_health_check_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Build a GET /health request.
        """Get SGLang health check request."""
        return HttpRequest(endpoint="/health", payload={}, method="GET")

    def get_offload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Build /release_memory_occupation, with optional tags (e.g. weights, kv_cache)
        # to release.
        """Get SGLang offload request."""
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/release_memory_occupation", payload=payload)

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Build /resume_memory_occupation, with optional tags to restore.
        """Get SGLang onload request.

        Parameters:
        ----------
        tags: list[str], optional
            Available tags for multi-stage resume: weights, kv_cache
        """
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def launch_server(self, server_args: dict[str, Any]) -> subprocess.Popen:
        # [FUNCTION DOC]
        # Start an SGLang server subprocess from server_args (mutated:
        # awex_meta_server_addr, awex_colocate_mode and _awex_gpus_per_server are
        # popped). In awex colocate mode with SLURM_LOCALID set, sets base_gpu_id =
        # SLURM_LOCALID * gpus_per_server and logs it. Builds the command with
        # SGLangConfig.build_cmd_from_args and an env with a unique Triton cache. If
        # colocate mode or an awex meta address (argument or AWEX_META_SERVER_ADDR env)
        # is present, swaps the sglang launch_server module entry for
        # areal.engine.awex.sglang_plugin and exports AWEX_META_SERVER_ADDR. Returns the
        # Popen with stdout/stderr redirected to this process's stdout.
        """Launch SGLang server subprocess."""
        awex_meta_addr = server_args.pop("awex_meta_server_addr", None)
        awex_colocate = server_args.pop("awex_colocate_mode", False)
        # Colocate placement: derive base_gpu_id from SLURM_LOCALID so two SGLang
        # servers sharing a node never claim the same GPU range. The controller
        # cannot do this reliably because its global rank -> node-slot mapping is
        # not guaranteed by SLURM task dispatch (a collision degrades the
        # TP group into an unsharded single-GPU load -> OOM). SLURM_LOCALID is the
        # only id guaranteed unique per node-slot, and only the worker sees it at
        # runtime. `_awex_gpus_per_server` is injected by the controller exclusively
        # for real colocation, so its presence doubles as the colocate gate; it is
        # absent for separated mode (where CVD isolation keeps base_gpu_id at 0).
        awex_gpus_per_server = server_args.pop("_awex_gpus_per_server", None)
        if awex_gpus_per_server is not None:
            slurm_localid = os.environ.get("SLURM_LOCALID")
            if slurm_localid is not None:
                base_gpu_id = int(slurm_localid) * int(awex_gpus_per_server)
                server_args["base_gpu_id"] = base_gpu_id
                logger.info(
                    "AWEX colocate base_gpu_id override: SLURM_LOCALID=%s x "
                    "gpus_per_server=%s -> base_gpu_id=%s",
                    slurm_localid,
                    awex_gpus_per_server,
                    base_gpu_id,
                )
        cmd = SGLangConfig.build_cmd_from_args(server_args)
        _env = self.build_server_env(os.environ)

        if not awex_meta_addr:
            awex_meta_addr = os.environ.get("AWEX_META_SERVER_ADDR")
        if awex_colocate or awex_meta_addr:
            sglang_entrypoints = (
                "sglang.launch_server",
                "areal.v2.inference_service.sglang.launch_server",
            )
            cmd = [
                "areal.engine.awex.sglang_plugin" if c in sglang_entrypoints else c
                for c in cmd
            ]
            if awex_meta_addr:
                _env["AWEX_META_SERVER_ADDR"] = awex_meta_addr
            logger.info("AWEX mode: using awex_sglang_plugin entry, cmd=%s", cmd[:4])

        return subprocess.Popen(
            cmd,
            env=_env,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )


class RemoteSGLangEngine(InferenceEngine):
    # [CLASS DOC]
    # Public SGLang InferenceEngine: holds config and a RemoteInfEngine(config,
    # SGLangBackend()) in _engine; all methods delegate to it (pure composition), except
    # export_stats, as_controller, RTensor fetch-buffer helpers and perf tracer hooks.
    """SGLang remote inference engine.

    This class delegates all functionality to RemoteInfEngine with
    an SGLangBackend implementation. It maintains the same public API.

    Parameters
    ----------
    config : InferenceEngineConfig
        Configuration for the inference engine
    """

    def __init__(self, config: InferenceEngineConfig):
        # [FUNCTION DOC]
        # Store config and build the internal RemoteInfEngine with an SGLangBackend. No
        # network activity.
        self.config = config
        # Pure composition - create internal engine with SGLang backend
        self._engine = RemoteInfEngine(config, SGLangBackend())

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_path: str | None = None,
        dp_size: int = 1,
        max_concurrent_rollouts: int | None = None,
        **kwargs,
    ) -> "RemoteInfEngine":
        # [FUNCTION DOC]
        # Convenience constructor: builds InferenceEngineConfig with backend
        # "sglang:d<dp_size>", max_concurrent_rollouts, tokenizer_path and extra kwargs,
        # and returns cls(config).
        # KNOWN ISSUE: annotated and documented as returning RemoteInfEngine, but it
        # returns a RemoteSGLangEngine instance.
        """Create a RemoteInfEngine without kwargs instead of InferenceEngineConfig.

        Parameters
        ----------
        tokenizer_path: str | None = None
            Path to the tokenizer
        dp_size : int
            Data parallelism size
        max_concurrent_rollouts : int | None
            Maximum concurrent rollouts
        **kwargs : dict
            Additional config parameters passed to InferenceEngineConfig

        Returns
        -------
        RemoteInfEngine
        """

        backend_str = f"sglang:d{dp_size}"

        config = InferenceEngineConfig(
            backend=backend_str,
            max_concurrent_rollouts=max_concurrent_rollouts,
            tokenizer_path=tokenizer_path,
            **kwargs,
        )

        engine = cls(config)

        return engine

    def initialize(
        self,
        engine_id: str | None = None,
        addr: str | list[str] | None = None,
        train_data_parallel_size: int | None = None,
    ):
        # [FUNCTION DOC]
        # Initialize by discovering/connecting to SGLang servers via the inner engine.
        # If train_data_parallel_size is None it is derived from the config backend
        # string's data-parallel size. Returns the inner engine's result.
        """Initialize the engine by discovering and connecting to servers."""
        if train_data_parallel_size is None:
            train_data_parallel_size = ModelAllocation.from_str(
                self.config.backend, name="rollout"
            ).parallel.data_parallel_size
        return self._engine.initialize(engine_id, addr, train_data_parallel_size)

    def destroy(self):
        # [FUNCTION DOC]
        # Destroy the inner engine (stop executors, clean up resources).
        """Destroy the engine and clean up resources."""
        return self._engine.destroy()

    @property
    def initialized(self) -> bool:
        # [FUNCTION DOC]
        # Whether the inner engine has been initialized.
        return self._engine.initialized

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        # [FUNCTION DOC]
        # The inner engine's WorkflowExecutor.
        """Get the workflow executor of the inference engine."""
        return self._engine.workflow_executor

    def set_version(self, version: int):
        # [FUNCTION DOC]
        # Set the current weight version on the inner engine.
        """Set the current weight version."""
        return self._engine.set_version(version)

    def get_version(self) -> int:
        # [FUNCTION DOC]
        # Return the inner engine's current weight version.
        """Get the current weight version."""
        return self._engine.get_version()

    def set_proxy_gateway_addr(self, addr: str) -> None:
        # [FUNCTION DOC]
        # Forward the proxy gateway address to the inner engine.
        return self._engine.set_proxy_gateway_addr(addr)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        # [FUNCTION DOC]
        # Async: generate a ModelResponse for req via the inner engine (HTTP to an
        # SGLang server).
        """Asynchronously generate a response for the given request."""
        return await self._engine.agenerate(req)

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, xccl_group_ranks: list[int] | None = None
    ) -> Future[None]:
        # [FUNCTION DOC]
        # Initialize the NCCL/XCCL weight update group on all servers; returns a Future.
        """Initialize the weight update process group."""
        return self._engine.init_weights_update_group(
            meta, xccl_group_ranks=xccl_group_ranks
        )

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        # [FUNCTION DOC]
        # Push weights over the distributed group to the servers; returns a Future.
        """Update weights from distributed memory."""
        return self._engine.update_weights_from_distributed(meta, param_specs)

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        # [FUNCTION DOC]
        # Make servers reload weights (or LoRA) from disk; returns a Future.
        """Update weights from disk."""
        return self._engine.update_weights_from_disk(meta)

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
        # Submit one data item to the workflow executor for rollout with the given
        # workflow, acceptance filter, group size and options; returns a task id. Pure
        # delegation.
        """Submit a request to the inference engine."""
        return self._engine.submit(
            data=data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            task_id=task_id,
            callback_addr=callback_addr,
            is_eval=is_eval,
            proxy_addr=proxy_addr,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        # [FUNCTION DOC]
        # Block until count rollout results are available (timeout optional;
        # raise_timeout controls raising). Returns the results list.
        """Wait for a specified number of requests to complete."""
        return self._engine.wait(count, timeout, raise_timeout)

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Block until the given task id finishes; returns its result or None.
        """Wait for a specific task to complete by task_id."""
        return self._engine.wait_for_task(task_id, timeout, raise_timeout)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Synchronously roll out a list of data items and return the batched result.
        # Intended for offline use/debugging.
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        """
        return self._engine.rollout_batch(
            data=data,
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
    ):
        # [FUNCTION DOC]
        # Keep submitting from the dataloader and return once a full training batch is
        # ready (async rollout).
        """Asynchronously submit and wait until a full batch is ready."""
        return self._engine.prepare_batch(
            dataloader=dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
            reward_normalization=reward_normalization,
            drop_incomplete_group=drop_incomplete_group,
        )

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Compute per-token logprobs for the given data through the servers (teacher
        # scoring); returns a list of tensors.
        return self._engine.compute_logp(data)

    def pause(self):
        # [FUNCTION DOC]
        # Pause rollout submission/generation via the inner engine.
        return self._engine.pause()

    def resume(self):
        # [FUNCTION DOC]
        # Resume rollout via the inner engine.
        return self._engine.resume()

    def pause_generation(self):
        # [FUNCTION DOC]
        # Send pause-generation requests to the servers.
        return self._engine.pause_generation()

    def continue_generation(self):
        # [FUNCTION DOC]
        # Send continue-generation requests to the servers.
        return self._engine.continue_generation()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        # [FUNCTION DOC]
        # Launch a local SGLang server via the inner engine; returns its
        # LocalInfServerInfo.
        return self._engine.launch_server(server_args)

    def teardown_server(self):
        # [FUNCTION DOC]
        # Tear down locally launched servers.
        return self._engine.teardown_server()

    def offload(self, tags: list[str] | None = None):
        # [FUNCTION DOC]
        # Release server memory (optionally only given tags), logging before and after;
        # returns the inner result.
        logger.info("RemoteSGLangEngine.offload(tags=%s) called", tags)
        result = self._engine.offload(tags=tags)
        logger.info("RemoteSGLangEngine.offload(tags=%s) done", tags)
        return result

    def abort_all_requests(self):
        # [FUNCTION DOC]
        # Abort all in-flight requests on the servers.
        return self._engine.abort_all_requests()

    def onload(self, tags: list[str] | None = None):
        # [FUNCTION DOC]
        # Restore released server memory (optionally only given tags).
        return self._engine.onload(tags=tags)

    def export_stats(self) -> dict[str, float]:
        # [FUNCTION DOC]
        # Export all stats_tracker scalars without cross-rank reduction.
        return stats_tracker.export_all(reduce_group=None)

    @classmethod
    def as_controller(cls, config: InferenceEngineConfig, scheduler: Scheduler):
        # [FUNCTION DOC]
        # Create the rollout controller for this engine: RolloutControllerV2 (imported
        # lazily) when config._version is "v2", else RolloutController(cls, config,
        # scheduler).
        if config._version == "v2":
            from areal.v2.inference_service.controller.controller import (
                RolloutControllerV2,
            )

            return RolloutControllerV2(config=config, scheduler=scheduler)
        return RolloutController(cls, config=config, scheduler=scheduler)

    def clear_batches(self, shard_ids: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Drop cached RTensor fetch-buffer shards for the given shard ids; no-op when
        # none are given (non-DP-head ranks).
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
        # Return local RTensor fetch-buffer statistics.
        """Expose local fetch-buffer stats for post-step drain verification."""
        from areal.infra.rpc.rtensor import fetch_buffer_stats

        return fetch_buffer_stats()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        # [FUNCTION DOC]
        # Persist perf tracer data for the given step (force to flush regardless of
        # schedule).
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        # [FUNCTION DOC]
        # Configure the global perf tracer for this rank/role unless it is already
        # configured.
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)
