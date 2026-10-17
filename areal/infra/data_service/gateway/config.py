# [MODULE DOC]
# Configuration dataclass for the data service gateway.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatewayConfig:
    # [CLASS DOC]
    # Gateway settings: host/port to bind (0.0.0.0:8090), router_addr (base URL of the
    # router, empty by default), admin_api_key (key granting admin endpoints),
    # forward_timeout (seconds for proxied requests to workers, 60) and router_timeout
    # (seconds for router queries, 2).
    host: str = "0.0.0.0"
    port: int = 8090
    router_addr: str = ""
    admin_api_key: str = "areal-data-admin"
    forward_timeout: float = 60.0
    router_timeout: float = 2.0
