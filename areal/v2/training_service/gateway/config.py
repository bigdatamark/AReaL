# [MODULE DOC]
# Configuration dataclass for the training gateway.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    # [CLASS DOC]
    # Settings: bind host/port, router_addr (empty default; must be set), admin_api_key,
    # uvicorn log_level, router_timeout (seconds per route lookup), forward_timeout
    # (seconds per forwarded engine call, long for training steps).
    host: str = "0.0.0.0"
    port: int = 9080
    router_addr: str = ""
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    router_timeout: float = 2.0
    forward_timeout: float = 600.0
