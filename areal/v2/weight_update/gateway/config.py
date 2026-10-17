# [MODULE DOC]
# Data models for the weight-update gateway: service config, update result, and per-pair
# registration info.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel  # pyright: ignore[reportMissingImports]


@dataclass
class WeightUpdateConfig:
    # [CLASS DOC]
    # Gateway service settings: bind host/port, admin API key, uvicorn log level,
    # communication backend ('nccl' or 'hccl'), debug mode flag, NCCL init and per-
    # update timeouts (seconds), and whether to use batch_isend_irecv.
    # KNOWN ISSUE: timeouts are annotated int but the CLI passes floats.
    """Configuration for the weight update service."""

    # Gateway
    host: str = "0.0.0.0"
    gateway_port: int = 7080

    # Authentication
    admin_api_key: str = "areal-admin-key"

    log_level: str = "warning"

    # NCCL
    comm_backend: str = "nccl"  # "nccl" or "hccl" (NPU)

    # Plan derivation
    enable_debug_mode: bool = False  # Dump plans to JSON, verbose logging

    # Timeouts
    init_timeout_s: int = 300  # NCCL group init timeout
    update_timeout_s: int = 120  # Per-step weight update timeout

    # Performance
    use_batch_isend_irecv: bool = True  # Use dist.batch_isend_irecv


class WeightUpdateResult(BaseModel):
    # [CLASS DOC]
    # Pydantic result of a weight update: status ('ok' or 'error'), version, duration_ms
    # and optional error message.
    """Result of a weight update operation."""

    status: str  # "ok" or "error"
    version: int
    duration_ms: float
    error: str | None = None


@dataclass
class PairInfo:
    # [CLASS DOC]
    # Registration record for one training/inference pair: pair name, worker URLs, world
    # sizes, NCCL master addr/port, last applied version, mode ('awex' or 'disk'),
    # disk/LoRA options (save_path, use_lora, lora_name, lora_keep_versions where 0
    # disables unloading old adapters) and colocate flag.
    pair_name: str
    train_worker_urls: list[str]
    inference_worker_urls: list[str]
    train_world_size: int = 0
    inference_world_size: int = 0
    master_addr: str = ""
    master_port: int = 0
    last_version: int = 0

    # Disk-mode fields (used when mode="disk")
    mode: str = "awex"  # "awex" or "disk"
    save_path: str = ""
    use_lora: bool = False
    lora_name: str = ""
    # Number of recent LoRA adapter versions to keep loaded on each inference
    # worker. Older versions are unloaded after a successful load_lora_adapter
    # to bound VRAM and avoid the adapter-accumulation hang. 0 disables cleanup.
    lora_keep_versions: int = 0

    # Colocated mode (training and inference share GPUs)
    colocate: bool = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Validates that pair_name is non-empty and mode is 'awex' or 'disk'; raises
        # ValueError otherwise.
        if not self.pair_name:
            raise ValueError("pair_name must not be empty")
        if self.mode not in ("awex", "disk"):
            raise ValueError(f"mode must be 'awex' or 'disk', got '{self.mode}'")
