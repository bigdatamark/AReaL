# [MODULE DOC]
# Synchronous CLI-side HTTP client for the Agent Router, built on the CLI's
# BaseHTTPClient.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.v2.cli.client import BaseHTTPClient


class RouterClient(BaseHTTPClient):
    # [CLASS DOC]
    # Router client used by CLI commands (distinct from the async service-side
    # RouterClient).
    def register_proxy(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # POSTs {"addr": addr} to /register with the given timeout (default 10s) and returns
        # the JSON response dict. Error handling is inherited from BaseHTTPClient._post.
        return self._post("/register", payload={"addr": addr}, timeout=timeout)
