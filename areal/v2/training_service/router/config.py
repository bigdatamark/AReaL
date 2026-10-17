# [MODULE DOC]
# Configuration dataclass for the training router.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    # [CLASS DOC]
    # Settings: bind host/port, admin_api_key, uvicorn log_level, poll_interval (seconds
    # between health sweeps) and worker_health_timeout (per probe).
    host: str = "0.0.0.0"
    port: int = 9081
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    poll_interval: float = 5.0
    worker_health_timeout: float = 2.0
