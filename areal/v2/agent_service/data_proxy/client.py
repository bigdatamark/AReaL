# [MODULE DOC]
# Async HTTP client for talking to a DataProxy server's session endpoints.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import httpx


class DataProxyClient:
    # [CLASS DOC]
    # Thin wrapper over a shared httpx.AsyncClient (600s timeout) targeting one
    # DataProxy base address. Every call raises httpx.HTTPStatusError on non-2xx.
    def __init__(self, data_proxy_addr: str) -> None:
        # [FUNCTION DOC]
        # Stores the DataProxy base address and creates the httpx.AsyncClient.
        self._addr = data_proxy_addr
        self._http = httpx.AsyncClient(timeout=600.0)

    async def turn(
        self,
        session_key: str,
        message: str,
        run_id: str = "",
        queue_mode: str = "collect",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Sends one conversation turn: POSTs {message, run_id, queue_mode, metadata}
        # to /session/<session_key>/turn (metadata None becomes an empty dict). queue_mode
        # defaults to "collect". Returns the parsed JSON response dict.
        resp = await self._http.post(
            f"{self._addr}/session/{session_key}/turn",
            json={
                "message": message,
                "run_id": run_id,
                "queue_mode": queue_mode,
                "metadata": metadata or {},
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def close_session(self, session_key: str) -> None:
        # [FUNCTION DOC]
        # POSTs to /session/<session_key>/close to release the session on the proxy.
        resp = await self._http.post(f"{self._addr}/session/{session_key}/close")
        resp.raise_for_status()

    async def get_history(self, session_key: str) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # GETs /session/<session_key>/history and returns the "history" list of message
        # dicts from the JSON response.
        resp = await self._http.get(f"{self._addr}/session/{session_key}/history")
        resp.raise_for_status()
        return resp.json()["history"]

    async def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying httpx client (release connections).
        await self._http.aclose()
