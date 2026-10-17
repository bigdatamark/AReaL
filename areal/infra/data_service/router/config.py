# [MODULE DOC]
# Configuration dataclass for the data service router.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    # [CLASS DOC]
    # Router settings: host/port to bind (0.0.0.0:8091), admin_api_key required for
    # admin calls, poll_interval (seconds between worker health polls, 5) and
    # worker_health_timeout (seconds per health check, 3).
    host: str = "0.0.0.0"
    port: int = 8091
    admin_api_key: str = "areal-data-admin"
    poll_interval: float = 5.0
    worker_health_timeout: float = 3.0
