# [MODULE DOC]
# Configuration dataclass for the DataProxy service.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataProxyConfig:
    # [CLASS DOC]
    # DataProxy server settings: bind host/port (default 0.0.0.0:9100), the backing
    # Worker address, request_timeout (seconds for Worker calls, 600), session_timeout
    # (idle seconds before a session expires, 3600) and uvicorn log_level.
    host: str = "0.0.0.0"
    port: int = 9100
    worker_addr: str = "http://localhost:9000"
    request_timeout: float = 600.0
    session_timeout: int = 3600
    log_level: str = "warning"
