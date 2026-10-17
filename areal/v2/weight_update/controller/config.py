# [MODULE DOC]
# Configuration dataclass for WeightUpdateController.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeightUpdateControllerConfig:
    # [CLASS DOC]
    # Settings for launching and talking to the gateway: host (bind/connect host), port
    # (0 = pick a free port), admin_api_key (Bearer token), log_level for the gateway,
    # setup_timeout (s to wait for /health), request_timeout (s per HTTP request), and
    # init_timeout_s / update_timeout_s forwarded to the gateway.
    host: str = "127.0.0.1"
    port: int = 0
    admin_api_key: str = "areal-admin-key"
    log_level: str = "warning"
    setup_timeout: float = 30.0
    request_timeout: float = 300.0
    init_timeout_s: float = 300.0
    update_timeout_s: float = 120.0
