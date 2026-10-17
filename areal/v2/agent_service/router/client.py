# [MODULE DOC]
# Async HTTP client for the Router service's admin endpoints.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import httpx

from ..auth import DEFAULT_ADMIN_API_KEY, admin_headers


class RouterClient:
    # [CLASS DOC]
    # Wraps an httpx.AsyncClient (30s timeout) with admin bearer headers for one Router
    # address. Every method raises httpx.HTTPStatusError on non-2xx responses.
    def __init__(
        self, router_addr: str, admin_api_key: str = DEFAULT_ADMIN_API_KEY
    ) -> None:
        # [FUNCTION DOC]
        # Stores the router address, builds admin headers from admin_api_key (defaults to
        # DEFAULT_ADMIN_API_KEY) and creates the HTTP client.
        self._addr = router_addr
        self._headers = admin_headers(admin_api_key)
        self._http = httpx.AsyncClient(timeout=30.0)

    async def register(self, addr: str) -> None:
        # [FUNCTION DOC]
        # POSTs {"addr": addr} to /register to add a DataProxy.
        resp = await self._http.post(
            f"{self._addr}/register", json={"addr": addr}, headers=self._headers
        )
        resp.raise_for_status()

    async def unregister(self, addr: str) -> None:
        # [FUNCTION DOC]
        # POSTs {"addr": addr} to /unregister to remove a DataProxy and its sessions.
        resp = await self._http.post(
            f"{self._addr}/unregister", json={"addr": addr}, headers=self._headers
        )
        resp.raise_for_status()

    async def route(self, session_key: str) -> str:
        # [FUNCTION DOC]
        # POSTs {"session_key"} to /route and returns the assigned DataProxy address string.
        resp = await self._http.post(
            f"{self._addr}/route",
            json={"session_key": session_key},
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()["data_proxy_addr"]

    async def remove_session(self, session_key: str) -> None:
        # [FUNCTION DOC]
        # POSTs {"session_key"} to /remove_session to drop the session's routing entry.
        resp = await self._http.post(
            f"{self._addr}/remove_session",
            json={"session_key": session_key},
            headers=self._headers,
        )
        resp.raise_for_status()

    async def close(self) -> None:
        # [FUNCTION DOC]
        # Closes the underlying HTTP client.
        await self._http.aclose()
