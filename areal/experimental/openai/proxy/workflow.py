# [MODULE DOC]
# OpenAIProxyWorkflow: a RolloutWorkflow that trains on trajectories from arbitrary
# agent code talking to an OpenAI/Anthropic-compatible proxy server (one per rollout
# worker) instead of calling ArealOpenAI directly. Modes: "inline" (await agent.run in
# the workflow's event loop with base_url/api_key/http_client kwargs), "subproc" (run
# the
# agent in a shared lazily created ProcessPoolExecutor with OPENAI_/ANTHROPIC_ base URL
# and API key env vars) and "online" (wait for an external user session via the proxy
# gateway). Each episode grants proxy capacity, opens an RL session, runs the agent,
# assigns rewards, and exports interactions.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any

import aiohttp

from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client_session import OpenAIProxyClient
from .server import DEFAULT_ADMIN_API_KEY, GRANT_CAPACITY_PATHNAME

if TYPE_CHECKING:
    from ..client import TRolloutEngine
    from ..types import InteractionWithTokenLogpReward
    from .proxy_gateway import CompletedSessionInfo

logger = logging.getLogger("OpenAIProxyWorkflow")


# Lazy-initialized process pool for running agent tasks
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()
_executor_max_workers: int | None = None


def _get_executor(max_workers: int = 4) -> ProcessPoolExecutor:
    # [FUNCTION DOC]
    # HELPER: Returns the module-global ProcessPoolExecutor, creating it on first use
    # with
    # max_workers (double-checked locking) and registering _shutdown_executor at exit.
    # Later
    # max_workers values are ignored.
    """Get or create the shared process pool executor.

    Parameters
    ----------
    max_workers : int
        Maximum number of worker processes for the pool. Only used when
        creating a new executor. If an executor already exists, this
        parameter is ignored.
    """
    global _executor, _executor_max_workers
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ProcessPoolExecutor(max_workers=max_workers)
                _executor_max_workers = max_workers
                # Register cleanup on process exit
                atexit.register(_shutdown_executor)
    return _executor


def _shutdown_executor() -> None:
    # [FUNCTION DOC]
    # HELPER: Shuts down the global executor without waiting and clears the global.
    """Shutdown the shared process pool executor if it exists."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def _wrap_run(agent: Any, data: dict[str, Any], extra_envs: dict[str, str]):
    # [FUNCTION DOC]
    # HELPER: Runs inside a pool worker process: writes extra_envs into os.environ (they
    # persist in that worker for later tasks), then runs agent.run(data) to completion
    # with
    # asyncio.run and returns its result (must be picklable, as must agent and data).
    """Run agent in subprocess with environment variables."""
    for key, value in extra_envs.items():
        os.environ[key] = value
    return asyncio.run(agent.run(data))


class OpenAIProxyWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow running an agent against the proxy. Config: mode, agent (object
    # with
    # async run(data, **kwargs) returning a float reward for the last interaction or a
    # dict
    # {interaction_id: reward}), proxy_addr, admin API key, reward discount, export
    # style,
    # subprocess pool size, and drop_retry_orphans.
    def __init__(
        self,
        mode: str,
        agent: Any | None = None,
        proxy_addr: str = "",
        admin_api_key: str = DEFAULT_ADMIN_API_KEY,
        discount: float = 1.0,
        export_style: str = "individual",
        subproc_max_workers: int = 4,
        proxy_gateway_addr: str | None = None,
        drop_retry_orphans: bool = False,
    ):
        # [FUNCTION DOC]
        # Validates mode (ValueError if not inline/subproc/online). Online mode requires
        # proxy_gateway_addr (ValueError) and replaces agent with an _OnlineAgent; other
        # modes
        # require an agent (ValueError) whose run attribute is callable and a coroutine
        # function (TypeError otherwise). Stores settings.
        if mode not in ("inline", "subproc", "online"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'inline', 'subproc', or 'online'"
            )

        if mode == "online":
            if proxy_gateway_addr is None:
                raise ValueError("proxy_gateway_addr is required when mode='online'")
            from .online_agent import _OnlineAgent

            agent = _OnlineAgent(
                proxy_gateway_addr=proxy_gateway_addr,
                admin_api_key=admin_api_key,
            )
        else:
            if agent is None:
                raise ValueError("agent is required when mode is 'inline' or 'subproc'")
            # Validate that agent has an async 'run' method
            if not hasattr(agent, "run") or not callable(getattr(agent, "run")):
                raise TypeError(
                    f"Agent must have a callable 'run' method. "
                    f"Got agent of type {type(agent).__name__} without a callable 'run' attribute."
                )
            if not asyncio.iscoroutinefunction(agent.run):
                raise TypeError(
                    f"Agent's 'run' method must be an async function. "
                    f"Got {type(agent).__name__}.run which is not a coroutine function."
                )

        self.mode = mode
        self.agent = agent
        self.proxy_addr = proxy_addr
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.subproc_max_workers = subproc_max_workers
        self.drop_retry_orphans = drop_retry_orphans

    @trace_session("run_agent")
    async def _run_agent(self, session_api_key: str, data: dict):
        # [FUNCTION DOC]
        # HELPER: Traced as "run_agent". Inline: awaits agent.run(data,
        # base_url=proxy_addr,
        # http_client=shared httpx client, api_key=session key). Subproc: runs _wrap_run
        # in the
        # process pool with OpenAI/Anthropic base URL and key env vars. Online: awaits
        # agent.run with the admin key (returns CompletedSessionInfo). Returns the agent
        # result;
        # ValueError for unknown mode.
        if self.mode == "inline":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": session_api_key,
            }
            return await self.agent.run(data, **extra_kwargs)
        if self.mode == "subproc":
            extra_envs = {
                "OPENAI_BASE_URL": self.proxy_addr,
                "OPENAI_API_KEY": session_api_key,
                "ANTHROPIC_BASE_URL": self.proxy_addr,
                "ANTHROPIC_API_KEY": session_api_key,
            }
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _get_executor(max_workers=self.subproc_max_workers),
                _wrap_run,
                self.agent,
                data,
                extra_envs,
            )
        if self.mode == "online":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": self._admin_api_key,
            }
            return await self.agent.run(data, **extra_kwargs)
        raise ValueError(f"Unsupported mode: {self.mode}")

    async def _grant_capacity(self, session: aiohttp.ClientSession) -> None:
        # [FUNCTION DOC]
        # HELPER: POSTs to <proxy_addr>/grant_capacity with admin auth, allowing one
        # more agent
        # session to start (staleness control); raises on HTTP error.
        """Grant capacity via HTTP."""
        url = f"{self.proxy_addr}/{GRANT_CAPACITY_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        async with session.post(url, headers=headers) as resp:
            resp.raise_for_status()

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        # [FUNCTION DOC]
        # Runs one episode (within a perf-tracer session context). Derives proxy task id
        # "<task_id>:<sample_idx>" (or task_id), grants capacity, then:
        # Online: waits for a completed external session, exports its interactions by
        # session id (admin auth); returns None with a warning if empty, else records
        # the last
        # interaction's reward in stats and returns the interactions.
        # Inline/subproc: opens an OpenAIProxyClient session, runs the agent (exceptions
        # logged
        # and re-raised, the session still ended), sets rewards from the result (dict ->
        # per id,
        # float -> last; other types ValueError), ends the session, exports with
        # discount,
        # style and orphan dropping, records the last reward stat and returns the
        # interactions
        # dict (possibly empty). The engine argument is unused (generation goes through
        # the
        # proxy).
        # KNOWN ISSUE: an int reward (e.g. 1) is rejected because only float is
        # accepted.
        context = workflow_context.get()
        task_id = context.task_id
        # Qualify the proxy session with the group sample index so each group
        # member owns a distinct, run-stable session namespace.
        proxy_task_id = (
            f"{task_id}:{context.sample_idx}"
            if context.sample_idx is not None
            else str(task_id)
        )

        http_session = await workflow_context.get_aiohttp_session()

        # Grant capacity for clients, otherwise agent sessions are rejected.
        # Designed for online mode. Users' requests do not have any staleness
        # control, which may be detrimental to RL training. We use a hacky way
        # to control the staleness. The staleness is always explicitly controlled
        # by the rollout controller and staleness manager. If the code runs
        # to this point, it means that we are within the allowed staleness window,
        # so we can grant capacity to let the agent session proceed.
        await self._grant_capacity(http_session)

        if self.mode == "online":
            # Online mode: _OnlineAgent waits for external user session.
            # Returns CompletedSessionInfo with session credentials.
            session_info: CompletedSessionInfo = await self._run_agent(
                self._admin_api_key, data
            )

            # Create proxy client for export only (no start/end session).
            proxy_client = OpenAIProxyClient(
                session=http_session,
                base_url=self.proxy_addr,
                task_id=proxy_task_id,
                admin_api_key=self._admin_api_key,
            )
            proxy_client.session_id = session_info.session_id

            interactions = await proxy_client.export_interactions(
                discount=self.discount,
                style=self.export_style,
                drop_retry_orphans=self.drop_retry_orphans,
            )

            # Return None if no interactions (empty session — user never sent chat/completions)
            if not interactions:
                logger.warning(
                    f"Session {session_info.session_id} has no interactions, "
                    "trajectory will be rejected."
                )
                return None

            # Record stats
            last_id = next(reversed(interactions))
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
            return interactions

        # ---- Normal mode (inline / subproc) ----

        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url=self.proxy_addr,
            task_id=proxy_task_id,
            admin_api_key=self._admin_api_key,
        )
        async with proxy_client:
            # Run the user code.
            try:
                rewards = await self._run_agent(proxy_client.session_api_key, data)
            except Exception as exc:
                logger.warning(
                    "Agent task failed (%s: %s). This trajectory will be rejected.",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                raise

            # Assign rewards back according to user code output
            if isinstance(rewards, dict):
                for completion_id, reward in rewards.items():
                    await proxy_client.set_reward(completion_id, reward)
            elif isinstance(rewards, float):
                await proxy_client.set_last_reward(rewards)
            else:
                raise ValueError(f"Invalid reward type: {type(rewards)}")

        # Apply turn-level discount and export interactions
        interactions = await proxy_client.export_interactions(
            discount=self.discount,
            style=self.export_style,
            drop_retry_orphans=self.drop_retry_orphans,
        )

        # Record stats
        last_id = list(interactions.keys())[-1] if interactions else None
        if last_id and interactions:
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)

        return interactions
