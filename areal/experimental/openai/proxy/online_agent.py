# [MODULE DOC]
# Internal agent used by OpenAIProxyWorkflow in "online" mode, where trajectories are
# produced by external users rather than by a local agent function. Not for direct use.
# SPDX-License-Identifier: Apache-2.0

"""Internal agent for online training mode.

``_OnlineAgent`` is NOT intended for direct use. It is automatically
created by ``OpenAIProxyWorkflow`` when ``mode="online"``.  It registers
the assigned proxy worker as "ready" on the proxy gateway, then blocks
until an external user completes a full session lifecycle.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from areal.infra import workflow_context

from .proxy_gateway import CompletedSessionInfo
from .server import INTERNAL_WAIT_FOR_SESSION_PATHNAME


class _OnlineAgent:
    # [CLASS DOC]
    # HELPER: Agent object whose run() blocks until an external user completes a full
    # session (start, interact, set reward, end) on the assigned proxy worker, by
    # long-polling the proxy gateway. Holds gateway address, admin API key and timeout.
    """Internal agent that waits for external user sessions.

    Registers the assigned proxy worker as "ready" on the proxy
    gateway, then blocks until an external user completes a full session
    lifecycle (start_session → interact → set_reward → end_session) on
    that worker.

    Parameters
    ----------
    proxy_gateway_addr : str
        HTTP address of the proxy gateway server.
    admin_api_key : str
        Admin API key for authenticating with the proxy gateway.
    timeout : float
        Maximum seconds to wait for a session completion.
    """

    def __init__(
        self,
        proxy_gateway_addr: str,
        admin_api_key: str,
        timeout: float = 3600.0,
    ):
        # [FUNCTION DOC]
        # Stores proxy_gateway_addr, admin_api_key and timeout seconds (default 3600).
        self.proxy_gateway_addr = proxy_gateway_addr
        self.admin_api_key = admin_api_key
        self.timeout = timeout

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> CompletedSessionInfo:
        # [FUNCTION DOC]
        # POSTs {"worker_addr": base_url} with Bearer admin key to the gateway's
        # internal
        # wait-for-session endpoint using the shared workflow-context aiohttp session
        # and a
        # total timeout of self.timeout; raises for non-2xx status (and aiohttp timeout
        # errors).
        # Returns CompletedSessionInfo built from the JSON response. data is ignored;
        # extra_kwargs must contain base_url (KeyError otherwise).
        """Wait for an external user to complete a session.

        Parameters
        ----------
        data : dict
            Ignored in online mode (dataloader yields empty dicts).
        extra_kwargs : dict
            Provided by ``OpenAIProxyWorkflow``:

            - ``base_url``: proxy worker address
            - ``api_key``: admin API key
            - ``http_client``: ``httpx.AsyncClient`` (unused here)

        Returns
        -------
        CompletedSessionInfo
            Session credentials for trajectory export.
        """
        base_url = extra_kwargs["base_url"]  # proxy worker addr

        url = f"{self.proxy_gateway_addr}/{INTERNAL_WAIT_FOR_SESSION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self.admin_api_key}"}
        payload = {"worker_addr": base_url}

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        session = await workflow_context.get_aiohttp_session()
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
        return CompletedSessionInfo(**result)
