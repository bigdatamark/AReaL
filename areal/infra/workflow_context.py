# [MODULE DOC]
# Workflow execution context and shared HTTP clients. A ContextVar holds an immutable
# WorkflowContext (eval flag, task id, sample index in group) readable by workflows,
# with set/get helpers and a stats scope helper. HttpClientManager lazily creates one
# aiohttp session and one httpx client per thread, bound to the event loop they were
# created on, and registers loop-close cleanup to close them. Logger "WorkflowContext".
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from contextvars import ContextVar
from dataclasses import dataclass

import aiohttp
import httpx

from areal.infra.utils.concurrent import register_loop_cleanup
from areal.infra.utils.http import (
    DEFAULT_REQUEST_TIMEOUT,
    create_httpx_client,
    get_default_connector,
)
from areal.utils import logging

logger = logging.getLogger("WorkflowContext")


@dataclass(frozen=True)
class WorkflowContext:
    # [CLASS DOC]
    # Frozen dataclass: is_eval (default False), task_id (executor-assigned, or None),
    # sample_idx (index within a grouped rollout, or None). Default instance is the
    # ContextVar default.
    """Execution context available to workflows via contextvars.

    Attributes
    ----------
    is_eval : bool
        Whether the workflow is running in evaluation mode.
    task_id : int | None
        The task ID assigned by the workflow executor.
    sample_idx : int | None
        Index of this sample within its rollout group, when the workflow runs
        under a grouped workflow. Gives group members a stable identity that
        does not depend on completion order.
    """

    is_eval: bool = False
    task_id: int | None = None
    sample_idx: int | None = None


_current_context: ContextVar[WorkflowContext] = ContextVar(
    "workflow_context", default=WorkflowContext()
)


def set(ctx: WorkflowContext) -> None:
    # [FUNCTION DOC]
    # Sets the ContextVar to ctx for the current context (task/thread). Shadows the
    # builtin name set within this module.
    """Set the current workflow context."""
    _current_context.set(ctx)


def get() -> WorkflowContext:
    # [FUNCTION DOC]
    # Returns the current WorkflowContext (a default instance if never set).
    """Get the current workflow context."""
    return _current_context.get()


def stat_scope() -> str:
    # [FUNCTION DOC]
    # Returns "eval-rollout" if the current context is eval mode, else "rollout".
    """Get the appropriate stats_tracker scope based on current context.

    Returns
    -------
    str
        "eval-rollout" if in eval mode, "rollout" otherwise.
    """
    return "eval-rollout" if get().is_eval else "rollout"


class HttpClientManager:
    # [CLASS DOC]
    # Per-thread holder of a lazily created aiohttp ClientSession and httpx AsyncClient
    # plus the event loop they belong to. If accessed from a different running loop, the
    # old clients are closed and recreated.
    """Per-thread manager for shared HTTP clients.

    This class manages shared aiohttp and httpx clients for workflow execution,
    providing connection pooling and DNS caching for improved performance.
    Each thread gets its own instance via a global dict indexed by thread ID.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Initializes session, client and owning event loop to None.
        """Initialize the HttpClientManager instance."""
        self._aiohttp_session: aiohttp.ClientSession | None = None
        self._httpx_client: httpx.AsyncClient | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

    async def _check_event_loop_change(self) -> None:
        # [FUNCTION DOC]
        # HELPER: if a loop is running and differs from the recorded one, logs a
        # warning, attempts to close the old aiohttp session and httpx client (errors
        # logged as warnings), then resets both clients and the recorded loop to None.
        # No-op when no loop is running.
        """Check if event loop changed and close stale clients if so."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - nothing to check
            return

        if self._event_loop is not None and self._event_loop is not current_loop:
            # Event loop changed - need to close stale clients
            logger.warning(
                "Event loop changed. Closing stale HTTP clients and creating new ones."
            )

            old_aiohttp = self._aiohttp_session
            old_httpx = self._httpx_client

            # Close old clients asynchronously (in current loop)
            # Wrap in try-except as closing from a different loop may raise exceptions
            if old_aiohttp is not None:
                try:
                    await old_aiohttp.close()
                except Exception as e:
                    logger.warning(f"Error closing stale aiohttp session: {e}")
            if old_httpx is not None:
                try:
                    await old_httpx.aclose()
                except Exception as e:
                    logger.warning(f"Error closing stale httpx client: {e}")

            # Reset state after attempting to close to avoid resource leaks on failure
            self._aiohttp_session = None
            self._httpx_client = None
            self._event_loop = None

    async def get_aiohttp_session(self) -> aiohttp.ClientSession:
        # [FUNCTION DOC]
        # Checks for loop change, then lazily creates an aiohttp session with
        # DEFAULT_REQUEST_TIMEOUT for total/connect/sock_connect, 10MB read buffer and
        # the default no-keepalive connector; records the running loop and registers a
        # cleanup closing that specific session when the loop closes. Returns the
        # session.
        """Get the shared aiohttp.ClientSession for the current workflow execution.

        The session is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The session is
        automatically closed during AsyncTaskRunner shutdown.

        If the event loop has changed since the session was created (e.g., in tests),
        the old session is closed and a new one is created.

        Returns
        -------
        aiohttp.ClientSession
            The shared session for HTTP requests.
        """
        await self._check_event_loop_change()

        if self._aiohttp_session is None:
            timeout = aiohttp.ClientTimeout(
                total=DEFAULT_REQUEST_TIMEOUT,
                sock_connect=DEFAULT_REQUEST_TIMEOUT,
                connect=DEFAULT_REQUEST_TIMEOUT,
            )
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=timeout,
                read_bufsize=1024 * 1024 * 10,
                connector=get_default_connector(),
            )
            # Track which event loop this session belongs to
            self._event_loop = asyncio.get_running_loop()

            # Register cleanup with the event loop
            # Use closure to capture current session reference
            session_to_close = self._aiohttp_session

            async def cleanup_aiohttp_session():
                # [FUNCTION DOC]
                # HELPER: loop-close callback that closes the captured session if not
                # closed.
                if session_to_close is not None and not session_to_close.closed:
                    await session_to_close.close()

            register_loop_cleanup(cleanup_aiohttp_session, loop=self._event_loop)

        return self._aiohttp_session

    async def get_httpx_client(self) -> httpx.AsyncClient:
        # [FUNCTION DOC]
        # Checks for loop change, then lazily creates an httpx client via
        # create_httpx_client with DEFAULT_REQUEST_TIMEOUT, records the running loop and
        # registers a loop-close cleanup that acloses that client. Returns the client.
        """Get the shared httpx.AsyncClient for the current workflow execution.

        The client is lazily created on first access and reused for all subsequent
        calls within the same AsyncTaskRunner background thread. The client is
        automatically closed during AsyncTaskRunner shutdown.

        If the event loop has changed since the client was created (e.g., in tests),
        the old client is closed and a new one is created.

        Returns
        -------
        httpx.AsyncClient
            The shared client for HTTP requests.
        """
        await self._check_event_loop_change()

        if self._httpx_client is None:
            self._httpx_client = create_httpx_client(timeout=DEFAULT_REQUEST_TIMEOUT)
            # Track which event loop this client belongs to
            self._event_loop = asyncio.get_running_loop()

            # Register cleanup with the event loop
            # Use closure to capture current client reference
            client_to_close = self._httpx_client

            async def cleanup_httpx_client():
                # [FUNCTION DOC]
                # HELPER: loop-close callback that acloses the captured client if not
                # closed.
                if client_to_close is not None and not client_to_close.is_closed:
                    await client_to_close.aclose()

            register_loop_cleanup(cleanup_httpx_client, loop=self._event_loop)

        return self._httpx_client


# Global dict mapping thread ID -> HttpClientManager
# Each thread gets its own HttpClientManager instance
_managers: dict[int, HttpClientManager] = {}
_managers_lock = threading.Lock()


def _get_manager() -> HttpClientManager:
    # [FUNCTION DOC]
    # HELPER: returns (creating under a lock) the HttpClientManager keyed by the current
    # thread ident in the module-global dict.
    # KNOWN ISSUE: entries are never removed, so managers of dead threads persist and a
    # reused thread ident inherits the old manager.
    """Get or create the HttpClientManager for the current thread.

    Returns
    -------
    HttpClientManager
        The HttpClientManager instance for the current thread.
    """
    thread_id = threading.get_ident()
    with _managers_lock:
        if thread_id not in _managers:
            _managers[thread_id] = HttpClientManager()
        return _managers[thread_id]


async def get_aiohttp_session() -> aiohttp.ClientSession:
    # [FUNCTION DOC]
    # Module-level shortcut returning the current thread manager's aiohttp session.
    """Get the shared aiohttp.ClientSession for the current workflow execution.

    The session is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The session is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    aiohttp.ClientSession
        The shared session for HTTP requests.
    """
    return await _get_manager().get_aiohttp_session()


async def get_httpx_client() -> httpx.AsyncClient:
    # [FUNCTION DOC]
    # Module-level shortcut returning the current thread manager's httpx client.
    """Get the shared httpx.AsyncClient for the current workflow execution.

    The client is lazily created on first access and reused for all subsequent
    calls within the same AsyncTaskRunner background thread. The client is
    automatically closed during AsyncTaskRunner shutdown.

    Returns
    -------
    httpx.AsyncClient
        The shared client for HTTP requests.
    """
    return await _get_manager().get_httpx_client()
