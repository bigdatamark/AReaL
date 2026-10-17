# [MODULE DOC]
# Client-side helper for the OpenAI proxy rollout server's RL session protocol.
# OpenAIProxyClient starts a session (receiving a session id and per-session API key
# that the agent uses for its OpenAI calls), sets rewards, ends the session and exports
# recorded interactions over HTTP via a caller-provided aiohttp session. Also provides
# generic JSON POST helpers with tenacity-based retry on transient HTTP errors.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING

import aiohttp
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
)

from areal.infra.utils.http import ensure_end_with_slash
from areal.utils.logging import getLogger

from .server import (
    EXPORT_TRAJECTORIES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    SetRewardRequest,
    StartSessionRequest,
    deserialize_interactions,
)

if TYPE_CHECKING:
    from ..types import InteractionWithTokenLogpReward

logger = getLogger("OpenAIProxyClient")


class OpenAIProxyClient:
    # [CLASS DOC]
    # Async context manager for one RL session on a proxy server. Entering starts the
    # session with the admin key; exiting ends it with the session key. Holds the
    # aiohttp session, base_url (normalized to end with "/"), task_id, admin key, and
    # after
    # start the session_id and session API key.
    """Client session for interacting with the OpenAI proxy server.

    This class manages RL session lifecycle (start/end session) and provides
    methods for setting rewards and exporting interactions. It uses composition
    rather than inheritance - an aiohttp.ClientSession must be passed in.

    Session isolation is achieved via unique API keys (not URL paths).
    The admin API key is used for management operations (start_session,
    grant_capacity), while a per-session API key is used for
    generation endpoints (set_reward, end_session, export_trajectories).

    Parameters
    ----------
    session : aiohttp.ClientSession
        The HTTP session to use for requests
    base_url : str
        Base URL of the proxy server (fixed, no session_id in path)
    task_id : str
        Unique identifier for this task
    admin_api_key : str
        Admin API key for management operations

    Example
    -------
    ```python
    async with aiohttp.ClientSession() as http_session:
        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url="http://localhost:8000",
            task_id="task-1",
            admin_api_key="my-admin-key",
        )
        async with proxy_client:
            # Session API key is available for agents
            api_key = proxy_client.session_api_key
            await proxy_client.set_last_reward(1.0)

        # After context exit, export interactions
        interactions = await proxy_client.export_interactions()
    ```
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        task_id: str,
        admin_api_key: str,
    ):
        # [FUNCTION DOC]
        # Stores the HTTP session, normalized base URL, task id and admin key; session
        # id
        # and session key start as None.
        self._session = session
        self.base_url = ensure_end_with_slash(base_url)
        self.task_id = task_id
        self._admin_api_key = admin_api_key
        self.session_id: str | None = None
        self._session_api_key: str | None = None

    @property
    def session_api_key(self) -> str:
        # [FUNCTION DOC]
        # Property returning the per-session API key; ValueError if the session has not
        # started.
        """Return the session API key for this session."""
        if self._session_api_key is None:
            raise ValueError("Session API key is not set")
        return self._session_api_key

    def _admin_auth_headers(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: Returns a Bearer Authorization header with the admin key.
        """Return Authorization headers with admin API key."""
        return {"Authorization": f"Bearer {self._admin_api_key}"}

    def _session_auth_headers(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # HELPER: Returns a Bearer Authorization header with the session key; ValueError
        # if
        # unset.
        """Return Authorization headers with session API key."""
        if self._session_api_key is None:
            raise ValueError("Session API key is not set")
        return {"Authorization": f"Bearer {self._session_api_key}"}

    async def set_reward(self, completion_id: str, reward: float):
        # [FUNCTION DOC]
        # Sets the reward of interaction completion_id by POSTing to rl/set_reward with
        # the
        # session key (retried on transient errors; HTTP 400 logged instead of raised).
        # ValueError if no session.
        """Set reward for a specific completion/response by its ID."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_interaction_reward(
            self._session,
            interaction_id=completion_id,
            reward=reward,
            url=f"{self.base_url}{RL_SET_REWARD_PATHNAME}",
            headers=self._session_auth_headers(),
        )

    async def set_last_reward(self, reward: float):
        # [FUNCTION DOC]
        # Same as set_reward but with interaction_id None, meaning the session's most
        # recent interaction.
        """Set reward for the most recent completion/response."""
        if self.session_id is None:
            raise ValueError("Session ID is not set")
        await set_last_interaction_reward(
            self._session,
            reward=reward,
            url=f"{self.base_url}{RL_SET_REWARD_PATHNAME}",
            headers=self._session_auth_headers(),
        )

    async def export_interactions(
        self,
        discount: float = 1.0,
        style: str = "individual",
        drop_retry_orphans: bool = False,
    ) -> dict[str, InteractionWithTokenLogpReward]:
        # [FUNCTION DOC]
        # POSTs {session_id, discount, style, drop_retry_orphans} to export_trajectories
        # with admin auth (the server waits for the session to finish), raises for HTTP
        # errors,
        # and returns deserialized interactions. ValueError if session_id is unset. No
        # retry.
        # KNOWN ISSUE: docstring lists styles "individual"/"merged", but the cache
        # accepts
        # "individual"/"concat"; "merged" would be rejected server-side.
        """Export interactions for this session via HTTP.

        This method should be called after the session context exits
        (i.e., after ``__aexit__`` has ended the RL session), since
        ``/export_trajectories`` waits for the session to finish.

        The request always includes the explicit ``session_id`` so that
        the server resolves the correct session regardless of any
        API-key-to-session remapping that may have occurred during a
        refresh cycle.  Admin auth is used because the session key is
        not guaranteed to still map to this session.

        Parameters
        ----------
        discount : float
            Discount factor for reward propagation
        style : str
            Export style ("individual" or "merged")
        drop_retry_orphans : bool
            If True, instruct the server to drop completions that look like
            orphaned outputs from agent-side retries before reward discounting
            and export. Useful when the upstream Agent SDK times out and
            retries the same request, leaving the proxy with two completions
            for the same input messages.

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping interaction IDs to their data

        Raises
        ------
        ValueError
            If ``session_id`` has not been set on this client.
        """
        if self.session_id is None:
            raise ValueError("session_id must be set before exporting interactions")

        url = f"{self.base_url}{EXPORT_TRAJECTORIES_PATHNAME}"
        payload = {
            "session_id": self.session_id,
            "discount": discount,
            "style": style,
            "drop_retry_orphans": drop_retry_orphans,
        }
        headers = self._admin_auth_headers()
        async with self._session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return deserialize_interactions(data["interactions"])

    async def __aenter__(self) -> OpenAIProxyClient:
        # [FUNCTION DOC]
        # POSTs StartSessionRequest(task_id) to rl/start_session with admin auth,
        # retrying
        # forever on transient errors; stores session_id and api_key from the reply;
        # returns
        # self.
        """Start the RL session via HTTP request."""
        data = await _start_session(
            self._session,
            url=f"{self.base_url}{RL_START_SESSION_PATHNAME}",
            payload=StartSessionRequest(task_id=self.task_id),
            headers=self._admin_auth_headers(),
        )
        self.session_id = data["session_id"]
        self._session_api_key = data["api_key"]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        # [FUNCTION DOC]
        # If a session was started, POSTs rl/end_session with the session key (up to 10
        # retries), even when exiting due to an exception; on failure logs a warning and
        # re-raises. Does not suppress exceptions.
        """End the RL session via HTTP request.

        Always attempts to end the session, even on exception, to avoid
        leaving zombie sessions on the server.
        """
        if self.session_id is None:
            return  # Session was never started

        # Always try to end the session, even on exception
        try:
            await post_json_with_retry(
                self._session,
                url=f"{self.base_url}{RL_END_SESSION_PATHNAME}",
                headers=self._session_auth_headers(),
            )
        except Exception as e:
            # Raised errors will be properly handled by OpenAIProxyWorkflow
            logger.warning(f"Failed to end session {self.session_id}: {e}")
            raise


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict | BaseModel | None = None,
    total_timeout: int = 10,
    headers: dict[str, str] | None = None,
) -> dict:
    # [FUNCTION DOC]
    # POSTs payload (None -> {}, pydantic model -> model_dump) as JSON with an aiohttp
    # total timeout and optional headers; raises ClientResponseError on non-2xx; returns
    # the
    # parsed JSON body.
    timeout = aiohttp.ClientTimeout(total=total_timeout)

    if payload is None:
        payload = {}
    elif isinstance(payload, BaseModel):
        payload = payload.model_dump()

    async with session.post(
        url, json=payload, timeout=timeout, headers=headers
    ) as response:
        response.raise_for_status()
        return await response.json()


def should_retry(exception: Exception):
    # [FUNCTION DOC]
    # HELPER: True for ClientResponseError with status 504/503/502/429/408, any
    # ClientConnectionError, or asyncio.TimeoutError; else False.
    """Check if exception is a retryable HTTP error (503, 502, 429, etc.)"""
    if isinstance(exception, aiohttp.ClientResponseError):
        return exception.status in [504, 503, 502, 429, 408]
    if isinstance(exception, aiohttp.ClientConnectionError):
        return True
    elif isinstance(exception, asyncio.TimeoutError):
        return True
    return False


def log_retry(retry_state: RetryCallState):
    # [FUNCTION DOC]
    # HELPER: tenacity before_sleep callback logging the attempt number and exception
    # ("Timeout" for timeouts) as a warning. Currently not wired in.
    exception = retry_state.outcome.exception()

    exception_message = (
        "Timeout" if isinstance(exception, asyncio.TimeoutError) else str(exception)
    )
    message = f"Retry #{retry_state.attempt_number} due to: {exception_message}"
    logger.warning(message)


def get_retry_strategy(
    allowed_attempt: int, multiplier: float = 0.5, min: float = 0.5, max: float = 5
):
    # [FUNCTION DOC]
    # HELPER: Returns a tenacity retry decorator retrying on should_retry exceptions
    # with
    # exponential backoff (multiplier, min, max seconds), stopping after allowed_attempt
    # attempts or never if allowed_attempt < 0, re-raising the last exception.
    return retry(
        retry=retry_if_exception(should_retry),
        wait=wait_exponential(multiplier=multiplier, min=min, max=max),
        stop=stop_never if allowed_attempt < 0 else stop_after_attempt(allowed_attempt),
        reraise=True,
        # before_sleep=log_retry,
    )


@get_retry_strategy(allowed_attempt=10)
async def post_json_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict | BaseModel | None = None,
    total_timeout: float = 10,
    headers: dict[str, str] | None = None,
) -> dict:
    # [FUNCTION DOC]
    # post_json wrapped with up to 10 attempts on transient errors.
    return await post_json(session, url, payload, total_timeout, headers=headers)


async def _set_reward(
    http_session: aiohttp.ClientSession,
    interaction_id: str | None,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
    headers: dict[str, str] | None = None,
):
    # [FUNCTION DOC]
    # HELPER: POSTs SetRewardRequest(interaction_id, reward) with retry; a 400 response
    # is
    # logged as an error and swallowed, other HTTP errors re-raised.
    payload = SetRewardRequest(interaction_id=interaction_id, reward=reward)
    try:
        await post_json_with_retry(
            http_session, url=url, payload=payload, headers=headers
        )
    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            logger.error(f"[error code {e.status}] Error setting reward: {e.message}")
        else:
            raise e


async def set_interaction_reward(
    http_session: aiohttp.ClientSession,
    interaction_id: str,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
    headers: dict[str, str] | None = None,
):
    # [FUNCTION DOC]
    # Sets the reward for a specific interaction id via _set_reward.
    await _set_reward(
        http_session,
        interaction_id=interaction_id,
        reward=reward,
        url=url,
        headers=headers,
    )


async def set_last_interaction_reward(
    http_session: aiohttp.ClientSession,
    reward: float,
    url: str = RL_SET_REWARD_PATHNAME,
    headers: dict[str, str] | None = None,
):
    # [FUNCTION DOC]
    # Sets the reward for the session's latest interaction (interaction_id None).
    await _set_reward(
        http_session, interaction_id=None, reward=reward, url=url, headers=headers
    )


@get_retry_strategy(allowed_attempt=-1)
async def _start_session(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: post_json retried indefinitely on transient errors (used to start sessions
    # while the server may still be booting or at capacity).
    return await post_json(*args, **kwargs)
