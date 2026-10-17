# [MODULE DOC]
# Configuration dataclass for the inference router service.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Router service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouterConfig:
    # [CLASS DOC]
    # Router settings: bind host/port, admin_api_key, poll_interval (seconds between
    # worker health polls), worker_health_timeout (per-check timeout), routing_strategy
    # name (only "round_robin" works) and uvicorn log_level.
    """Configuration for the routing service.

    The router owns worker registry, session pinning, and routing
    strategy. It is a separate FastAPI process from the gateway.
    """

    host: str = "0.0.0.0"
    port: int = 8081
    admin_api_key: str = "areal-admin-key"
    poll_interval: float = 5.0  # seconds between health polls
    worker_health_timeout: float = 2.0  # seconds per health check
    routing_strategy: str = "round_robin"  # "round_robin" (only supported strategy)
    log_level: str = "warning"
