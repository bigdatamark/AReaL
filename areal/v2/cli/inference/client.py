# [MODULE DOC]
# HTTP clients for the inference service's gateway and router, plus legacy exception
# aliases GatewayHTTPError / GatewayUnreachable for the generic service exceptions.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from areal.v2.cli.client import (
    BaseHTTPClient,
    ServiceHTTPError,
    ServiceUnreachable,
)

# Legacy aliases — preserve the old names imported by inference subcommands
# (and downstream callers) so this swap is mechanical. New code should use
# the scaffold names directly.
GatewayHTTPError = ServiceHTTPError
GatewayUnreachable = ServiceUnreachable


class GatewayClient(BaseHTTPClient):
    # [CLASS DOC]
    # Client for the inference gateway (admin API key as Bearer auth).
    def list_models(self, *, timeout: float = 5.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # GET /models; returns the JSON dict of registered models. Timeout 5s.
        return self._get("/models", timeout=timeout)

    def register_model(self, payload: dict, *, timeout: float = 30.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # POST /register_model with the given payload dict; returns the response JSON.
        # Timeout 30s.
        return self._post("/register_model", payload=payload, timeout=timeout)


class RouterClient(BaseHTTPClient):
    # [CLASS DOC]
    # Client for the inference router's worker/model registry.
    def register_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # POST /register with worker_addr = addr; returns the response JSON.
        return self._post("/register", payload={"worker_addr": addr}, timeout=timeout)

    def unregister_worker(self, addr: str, *, timeout: float = 10.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # POST /unregister with worker_addr = addr; returns the response JSON.
        return self._post("/unregister", payload={"worker_addr": addr}, timeout=timeout)

    def remove_model(self, name: str, *, timeout: float = 10.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # POST /remove_model with the model name; returns the response JSON.
        return self._post("/remove_model", payload={"name": name}, timeout=timeout)
