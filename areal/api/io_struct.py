# [MODULE DOC]
# Plain data structures exchanged between workflows, inference engines, train engines
# and controllers: generation requests/responses, finetune/step bookkeeping, parameter
# specs, weight-update metadata and HTTP request descriptors, save/load metadata,
# rollout counters, local server info, and device memory snapshots. Uses the "IOStruct"
# logger.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL.Image import Image as ImageObject
from transformers import PreTrainedTokenizerFast

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GenerationHyperparameters
from areal.infra.platforms import current_platform
from areal.utils import logging

if TYPE_CHECKING:
    from transformers import AutoProcessor

logger = logging.getLogger("IOStruct")


@dataclass
class ModelRequest:
    # [CLASS DOC]
    # Dataclass for one generation request: rid (random UUID string by default),
    # input_ids, gconfig (GenerationHyperparameters), metadata dict, optional tokenizer
    # for encode/decode, VLM fields image_data (list of image strings, e.g. base64) and
    # processor, and vision_msg_vllm (vLLM-specific vision message list).
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    image_data: list[str] | None = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # vlm+vllm:
    vision_msg_vllm: list | None = None

    def copy(self):
        # [FUNCTION DOC]
        # Returns a new ModelRequest with the same rid, a shallow-copied input_ids list,
        # metadata dict, image_data and vision_msg_vllm (None preserved), a fresh
        # gconfig via gconfig.new(), and shared tokenizer/processor references.
        # KNOWN ISSUE: metadata is only shallow-copied, so nested mutable values are
        # shared with the original.
        return ModelRequest(
            rid=self.rid,
            input_ids=self.input_ids.copy(),
            gconfig=self.gconfig.new(),
            metadata=self.metadata.copy(),
            tokenizer=self.tokenizer,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            processor=self.processor,
            vision_msg_vllm=(
                self.vision_msg_vllm.copy()
                if self.vision_msg_vllm is not None
                else None
            ),
        )


@dataclass
class ModelResponse:
    # [CLASS DOC]
    # Dataclass for a generation result: input_tokens, output_tokens, per-token
    # output_logprobs and output_versions (policy weight version that produced each
    # token), stop_reason (length/stop/tool_calls/abort), optional tokenizer, VLM
    # input_images and processor, timing stats (latency, ttft, per-token itl; inf when
    # unset), and optional MoE routed_experts array.
    # outputs
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    output_logprobs: list[float] = field(default_factory=list)
    output_versions: list[int] = field(default_factory=list)
    stop_reason: Literal["length", "stop", "tool_calls", "abort"] = "stop"
    # tokenizer is used for encode-decode in the inference engine
    tokenizer: PreTrainedTokenizerFast | None = None

    # vlm
    input_images: list[ImageObject | str] = field(default_factory=list)
    processor: Optional["AutoProcessor"] = None

    # statistics
    latency: float = float("inf")
    ttft: float = float("inf")  # Time to first token
    itl: list[float] = field(default_factory=list)  # List of inter-token latencies

    # MoE routing (only populated when return_routed_experts=True)
    routed_experts: np.ndarray | None = None

    @property
    def input_len(self) -> int:
        # [FUNCTION DOC]
        # Property: number of input tokens.
        return len(self.input_tokens)

    @property
    def output_len(self) -> int:
        # [FUNCTION DOC]
        # Property: number of output tokens.
        return len(self.output_tokens)

    @property
    def end_with_stop(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if the last output token equals the tokenizer's eos or pad id
        # (ignoring ids that are None); False for empty output. Raises ValueError if
        # tokenizer is None.
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot check end_with_stop")
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if len(self.output_tokens) == 0:
            return False
        last_token = self.output_tokens[-1]
        return (eos_id is not None and last_token == eos_id) or (
            pad_id is not None and last_token == pad_id
        )

    @property
    def output_tokens_without_stop(self) -> list[int]:
        # [FUNCTION DOC]
        # Property: output tokens with trailing stop tokens stripped. Requires a
        # tokenizer (ValueError otherwise). For stop_reason length/abort or empty output
        # returns output_tokens unchanged. Otherwise requires the output to end with
        # eos/pad (ValueError if not), counts the trailing run of eos/pad ids and
        # returns the prefix before them; raises ValueError if every token is a stop
        # token.
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot get output_tokens_without_stop")
        if self.stop_reason not in ["length", "abort"] and self.output_tokens:
            if not self.end_with_stop:
                raise ValueError(
                    f"output_tokens does not end with eos or pad token, it ends with {self.output_tokens[-1]}, but stop_reason is {self.stop_reason}"
                )
            pad_or_eos_len = 0
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            stop_tokens = {eos_id, pad_id}
            stop_tokens.discard(None)
            for tok in reversed(self.output_tokens):
                if tok in stop_tokens:
                    pad_or_eos_len += 1
                else:
                    break
            if pad_or_eos_len == len(self.output_tokens):
                raise ValueError(
                    "All output_tokens are EOS or PAD tokens; cannot strip stop tokens without removing entire output."
                )
            return self.output_tokens[:-pad_or_eos_len]
        return self.output_tokens


@dataclass
class FinetuneSpec:
    # [CLASS DOC]
    # Dataclass of training schedule sizes: total_train_epochs, dataset_size,
    # train_batch_size.
    total_train_epochs: int
    dataset_size: int
    train_batch_size: int

    @property
    def total_train_steps(self):
        # [FUNCTION DOC]
        # Property: epochs times floor(dataset_size / train_batch_size), assuming the
        # last partial batch is dropped.
        # assuming drop_last
        return self.total_train_epochs * (self.dataset_size // self.train_batch_size)

    @property
    def steps_per_epoch(self):
        # [FUNCTION DOC]
        # Property: floor(dataset_size / train_batch_size).
        return self.dataset_size // self.train_batch_size


@dataclass
class ParamSpec:
    # [CLASS DOC]
    # Dataclass describing a model parameter for weight transfer: name, shape tuple and
    # dtype as a torch attribute name string (e.g. "bfloat16").
    name: str
    shape: tuple
    dtype: str

    @property
    def size(self) -> int:
        # [FUNCTION DOC]
        # Property: byte size computed as torch dtype itemsize times product of shape.
        # KNOWN ISSUE: Returns a numpy scalar (and a float for an empty shape) rather
        # than a Python int as annotated.
        """Param bytes"""
        return getattr(torch, self.dtype).itemsize * np.prod(self.shape)


def get_versioned_lora_name(lora_name: str, version: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "<lora_name>-v<version>".
    """Get versioned LoRA adapter name (e.g., 'lora-v1')."""
    return f"{lora_name}-v{version}"


def detect_image_mime(base64_data: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Infers MIME type from base64 magic prefixes: PNG, JPEG, GIF, WebP;
    # defaults to image/jpeg for anything unrecognized.
    """Detect image MIME type from the first bytes of base64-encoded data.

    Examines base64 magic byte prefixes to determine the actual image format.
    """
    if base64_data.startswith("iVBOR"):  # PNG: \x89PNG
        return "image/png"
    if base64_data.startswith("/9j/"):  # JPEG: \xff\xd8\xff
        return "image/jpeg"
    if base64_data.startswith("R0lGOD"):  # GIF: GIF8
        return "image/gif"
    if base64_data.startswith("UklGR"):  # WebP: RIFF
        return "image/webp"
    return "image/jpeg"


@dataclass
class WeightUpdateMeta:
    # [CLASS DOC]
    # Dataclass describing how trained weights are pushed to inference servers. type is
    # disk (save to path then reload), xccl (NCCL/HCCL broadcast using gen_allocation
    # and the nccl master address/port/group name, chunked by weight_chunked_mem_mb) or
    # awex. LoRA fields: use_lora, lora_name, lora_int_id, base_model_name, peft_config,
    # lora_keep_versions (number of adapter versions kept loaded; 0 disables unloading).
    # clear_checkpoint_after_load controls deleting the disk checkpoint after loading;
    # version is the weight version tag.
    type: Literal["disk", "xccl", "awex"]
    path: str | None = None
    gen_allocation: ModelAllocation | None = None

    nccl_master_address: str | None = None
    nccl_master_port: int | None = None
    nccl_group_name: str | None = None
    weight_chunked_mem_mb: int = 1024

    use_lora: bool = False
    lora_name: str = ""
    lora_int_id: int = 0
    base_model_name: str = ""
    peft_config: dict = field(default_factory=dict)
    # Number of recent LoRA adapter versions to keep loaded on the inference
    # server. Older versions are unloaded to bound memory; 0 disables cleanup.
    lora_keep_versions: int = 0

    clear_checkpoint_after_load: bool = True

    version: int | None = None

    def with_version(self, version: int) -> "WeightUpdateMeta":
        # [FUNCTION DOC]
        # Returns a shallow copy with version set and, if path is set, path replaced by
        # sibling directory "weight_update_v<version>" (same parent). Raises ValueError
        # for negative version. peft_config dict is shared with the original.
        """Return a copy of this meta with versioned path.

        Changes path from 'weight_update' to 'weight_update_v{version}'.
        """
        if version < 0:
            raise ValueError(f"version must be non-negative, got {version}")
        new_meta = copy.copy(self)
        new_meta.version = version
        if self.path is not None:
            base_dir = os.path.dirname(self.path)
            new_meta.path = os.path.join(base_dir, f"weight_update_v{version}")
        return new_meta

    @classmethod
    def from_disk(
        cls,
        experiment_name: str,
        trial_name: str,
        file_root: str,
        name: str = "default",
        use_lora: bool = False,
        clear_checkpoint_after_load: bool = True,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
        lora_keep_versions: int = 0,
    ) -> "WeightUpdateMeta":
        # [FUNCTION DOC]
        # Factory for disk updates: path is "<Saver model save root for
        # experiment/trial/file_root/name>/weight_update"; forwards the LoRA and cleanup
        # options. Note lora_int_id defaults to 1 here (field default is 0).
        from areal.utils.saver import Saver

        path = os.path.join(
            Saver.get_model_save_root(experiment_name, trial_name, file_root, name),
            "weight_update",
        )
        return cls(
            type="disk",
            path=path,
            use_lora=use_lora,
            clear_checkpoint_after_load=clear_checkpoint_after_load,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
            lora_keep_versions=lora_keep_versions,
        )

    @classmethod
    def from_megatron_xccl(
        cls,
        gen_allocation: ModelAllocation,
        weight_chunked_mem_mb: int = 1024,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        # [FUNCTION DOC]
        # Factory for xccl updates from a Megatron engine with the given gen_allocation,
        # chunk size and LoRA options; NCCL address fields are left None to be filled
        # later.
        return cls(
            type="xccl",
            gen_allocation=gen_allocation,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )

    @classmethod
    def from_fsdp_xccl(
        cls,
        gen_allocation: ModelAllocation,
        weight_chunked_mem_mb: int = 1024,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        # [FUNCTION DOC]
        # Factory for xccl updates from an FSDP engine; identical to from_megatron_xccl.
        return cls(
            type="xccl",
            gen_allocation=gen_allocation,
            weight_chunked_mem_mb=weight_chunked_mem_mb,
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )

    @classmethod
    def from_awex(
        cls,
        meta_server_addr: str | None = None,
        use_lora: bool = False,
        lora_name: str = "",
        lora_int_id: int = 1,
        base_model_name: str = "",
    ):
        # [FUNCTION DOC]
        # Factory for awex updates; stores meta_server_addr in nccl_master_address plus
        # LoRA options.
        return cls(
            type="awex",
            nccl_master_address=meta_server_addr,
            use_lora=use_lora,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            base_model_name=base_model_name,
        )


@dataclass
class HttpRequest:
    # [CLASS DOC]
    # Dataclass for one HTTP call to an inference server: endpoint path, JSON payload,
    # method (default POST) and best_effort (log and ignore failures).
    """Represents an HTTP request to be sent to a remote inference server."""

    endpoint: str
    payload: dict[str, Any]
    method: str = "POST"
    # When True, failures are logged and ignored instead of raised. Used for
    # cleanup requests (e.g. unloading a stale LoRA adapter that may be gone).
    best_effort: bool = False


@dataclass
class HttpGenerationResult:
    # [CLASS DOC]
    # Dataclass of parsed generation output: output_tokens, output_logprobs, stop_reason
    # string and optional routed_experts array; validated on construction.
    """Parsed result from a generation response."""

    output_tokens: list[int]
    output_logprobs: list[float]
    stop_reason: str
    routed_experts: np.ndarray | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validates that tokens and logprobs have equal length and that every logprob is
        # a finite int/float (bool rejected); raises ValueError describing the first
        # problem.
        if len(self.output_tokens) != len(self.output_logprobs):
            raise ValueError(
                "Malformed generation result: received "
                f"{len(self.output_tokens)} output tokens but "
                f"{len(self.output_logprobs)} output logprobs; every sampled "
                "output token requires exactly one sampling logprob."
            )

        for index, logprob in enumerate(self.output_logprobs):
            is_finite = (
                isinstance(logprob, (int, float))
                and not isinstance(logprob, bool)
                and math.isfinite(logprob)
            )
            if not is_finite:
                raise ValueError(
                    "Malformed generation result: "
                    f"output_logprobs[{index}] must be a real, finite number, "
                    f"got {logprob!r}."
                )


@dataclass
class WeightUpdateRequests:
    # [CLASS DOC]
    # Dataclass wrapping the ordered list of HttpRequests for one weight update.
    """Collection of HTTP requests needed for a weight update operation."""

    requests: list[HttpRequest]


@dataclass
class SaveLoadMeta:
    # [CLASS DOC]
    # Dataclass of checkpoint save/load options: path, weight_format (e.g. hf or dcp),
    # with_optim, optional tokenizer/processor to save alongside, base_model_path
    # (source of config/extra files), naive_distributed flag.
    path: str
    weight_format: str
    with_optim: bool
    tokenizer: PreTrainedTokenizerFast | None = None
    processor: Optional["AutoProcessor"] = None
    base_model_path: str | None = None
    naive_distributed: bool = False


@dataclass
class RolloutStat:
    # [CLASS DOC]
    # Dataclass of rollout counters: accepted, enqueued, rejected, running (all default
    # 0).
    accepted: int = 0
    enqueued: int = 0
    rejected: int = 0
    running: int = 0


@dataclass
class StepInfo:
    # [CLASS DOC]
    # Dataclass of training position: epoch, epoch_step (0-based within epoch),
    # global_step, steps_per_epoch.
    epoch: int
    epoch_step: int
    global_step: int
    steps_per_epoch: int

    def next(self):
        # [FUNCTION DOC]
        # Returns the following StepInfo: global_step+1, epoch_step+1, or rolls over to
        # epoch+1 with epoch_step 0 when the current step is the last of the epoch.
        return StepInfo(
            epoch=self.epoch + (self.epoch_step == self.steps_per_epoch - 1),
            epoch_step=(
                0
                if self.epoch_step == self.steps_per_epoch - 1
                else self.epoch_step + 1
            ),
            global_step=self.global_step + 1,
            steps_per_epoch=self.steps_per_epoch,
        )


@dataclass
class LocalInfServerInfo:
    # [CLASS DOC]
    # Dataclass for a locally launched inference server: host, port and the Popen handle
    # (None if not owned).
    """Information about a locally launched inference server."""

    host: str
    port: int
    process: subprocess.Popen | None


@dataclass
class DeviceRuntimeInfo:
    # [CLASS DOC]
    # Dataclass snapshot of accelerator memory: allocated, reserved, used (total minus
    # free) and total, in the given unit string.
    mem_allocated: float
    mem_reserved: float
    mem_used: float
    mem_total: float
    unit: str

    @classmethod
    def get_current(cls, unit: str = "GB"):
        # [FUNCTION DOC]
        # Classmethod: queries current_platform for allocated, reserved and (free,
        # total) memory and returns an instance scaled to unit GB, MB or KB (binary).
        # Raises ValueError for other units.
        unit_divisors = {"GB": 1024**3, "MB": 1024**2, "KB": 1024}
        if unit not in unit_divisors:
            raise ValueError(
                f"Unsupported unit '{unit}'. Must be one of {list(unit_divisors.keys())}."
            )
        divisor = unit_divisors[unit]

        mem_allocated = current_platform.memory_allocated()
        mem_reserved = current_platform.memory_reserved()
        mem_free, mem_total = current_platform.mem_get_info()
        mem_used = mem_total - mem_free
        return cls(
            mem_allocated=mem_allocated / divisor,
            mem_reserved=mem_reserved / divisor,
            mem_used=mem_used / divisor,
            mem_total=mem_total / divisor,
            unit=unit,
        )

    def log(self, head: str = "", rank: int = 0, precision: int = 2):
        # [FUNCTION DOC]
        # Logs an info line with the memory figures formatted to precision decimals,
        # prefixed by head, if torch.distributed is not initialized, rank is None, or
        # the current rank equals rank. Appends host RssAnon/RssShmem in MB parsed from
        # /proc/self/status when readable (errors silently ignored).
        mem_allocated = f"{self.mem_allocated:.{precision}f}"
        mem_reserved = f"{self.mem_reserved:.{precision}f}"
        mem_used = f"{self.mem_used:.{precision}f}"
        mem_total = f"{self.mem_total:.{precision}f}"
        if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
            # Append host RssAnon/RssShmem (from /proc/self/status) to every
            # device-stats log point so per-step host memory growth can be
            # attributed to a phase boundary. Near-zero cost; rank0-only.
            host_str = ""
            try:
                anon = shmem = 0
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("RssAnon:"):
                            anon = int(line.split()[1]) // 1024
                        elif line.startswith("RssShmem:"):
                            shmem = int(line.split()[1]) // 1024
                host_str = f" | host RssAnon: {anon}MB, RssShmem: {shmem}MB"
            except Exception:
                pass
            logger.info(
                f"Memory-Usage {head}: "
                f"memory allocated ({self.unit}): {mem_allocated}, "
                f"memory reserved ({self.unit}): {mem_reserved}, "
                f"device memory used/total ({self.unit}): {mem_used}/{mem_total}"
                f"{host_str}"
            )
