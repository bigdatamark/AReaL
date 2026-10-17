# [MODULE DOC]
# Configuration dataclass for the Worker service.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerConfig:
    # [CLASS DOC]
    # Worker settings: host/port (0.0.0.0:9000), agent_cls_path (import path of the
    # agent class, default empty) and log_level.
    host: str = "0.0.0.0"
    port: int = 9000
    agent_cls_path: str = ""
    log_level: str = "warning"
