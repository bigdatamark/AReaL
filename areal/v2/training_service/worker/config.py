# [MODULE DOC]
# Configuration dataclass for a training worker.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainWorkerConfig:
    # [CLASS DOC]
    # Settings: bind host, port (0 default; set by launcher), admin_api_key and
    # log_level. The admin key is not enforced by worker routes.
    host: str = "0.0.0.0"
    port: int = 0
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
