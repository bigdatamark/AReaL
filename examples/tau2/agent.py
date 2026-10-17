# [MODULE DOC]
# Proxy-mode tau2-bench agent workflow: runs a tau2 customer-service simulation where
# the agent LLM calls go to AReaL's OpenAI-compatible proxy (so they are recorded for
# RL)
# and the user simulator calls go to a separate user LLM server. Import-time side
# effects:
# importing examples.tau2.utils (which patches tau2 cost logging and loguru), silencing
# litellm debug output, and registering a zero-cost "dummy" litellm model name used as
# the
# agent LLM id.
"""Tau2 Agent Workflow for AReaL proxy mode.

This module implements a Tau2 agent that uses the AReaL proxy server
for OpenAI-compatible API calls during RL training.
"""

import asyncio
import os
import time
from typing import Any

import litellm
from litellm import register_model
from litellm.main import ModelResponse
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent, LocalAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import BaseUser, DummyUser, UserSimulator

# Import utilities (also patches tau2.utils.llm_utils)
from examples.tau2.utils import Tau2EnvConfig, Tau2RunInfo

from areal.utils import logging

logger = logging.getLogger("Tau2Agent")

# Silence litellm verbose output (Provider List messages)
litellm.suppress_debug_info = True

# Register dummy model for litellm
register_model(
    {
        "dummy": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }
)


def _get_task(domain: str, task_id: str, split: str | None = None) -> Task:
    # [FUNCTION DOC]
    # HELPER: Loads the tasks of a domain/split from the tau2 registry and returns the
    # one
    # with the given id; raises ValueError if absent.
    """Get a task by ID from the tau2 registry."""
    tasks: list[Task] = registry.get_tasks_loader(domain)(split)
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"No task found with id {task_id} for domain {domain}")


def think(thoughts: str):
    # [FUNCTION DOC]
    # Optional "think" tool given to the agent: ignores its input and returns a fixed
    # acknowledgement string. The docstring is the tool description shown to the model.
    """Use this tool to think. The thoughts will be visible in the history.
    Only use this tool to think when necessary.
    """
    return "Your thoughts are recorded. Please continue your work."


class Tau2Runner:
    # [CLASS DOC]
    # Runs one tau2 simulation with async OpenAI clients injected as tau2 completion
    # functions, collecting timings, messages, reward and errors in a Tau2RunInfo.
    """Runner for Tau2 environment using AsyncOpenAI clients."""

    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gen_args: dict,
        agent_client: AsyncOpenAI,
        user_client: AsyncOpenAI | None = None,
    ):
        # [FUNCTION DOC]
        # Stores econfig, gen_args (agent llm args), the agent client and optional user
        # client,
        # and caches domain and solo_mode.
        self.econfig = econfig
        self.gen_args = gen_args
        self.agent_client = agent_client
        self.user_client = user_client
        self.domain = econfig.domain
        self.solo_mode = econfig.solo_mode

    def _get_environment(self) -> Environment:
        # [FUNCTION DOC]
        # HELPER: Builds the domain environment via the registry constructor with
        # solo_mode.
        environment_constructor = registry.get_env_constructor(self.domain)
        return environment_constructor(solo_mode=self.solo_mode)

    @staticmethod
    def _convert_to_model_response(completion: ChatCompletion) -> ModelResponse:
        # [FUNCTION DOC]
        # HELPER: Converts an OpenAI ChatCompletion into a LiteLLM ModelResponse by
        # dumping and
        # re-constructing it (tau2 expects LiteLLM responses).
        """Convert OpenAI ChatCompletion to LiteLLM ModelResponse format.

        tau2-bench expects ModelResponse format from litellm, so we need to convert.
        """
        return ModelResponse(**completion.model_dump())

    @staticmethod
    def _clean_messages(messages: list[dict], for_user: bool = False) -> list[dict]:
        # [FUNCTION DOC]
        # HELPER: Returns copies of message dicts with a None tool_calls key removed;
        # when
        # for_user, also drops tool-role messages and strips tool_calls from assistant
        # messages.
        # Non-dict messages pass through unchanged.
        """Clean messages for OpenAI API compatibility.

        Args:
            messages: List of message dicts
            for_user: If True, also removes tool-related content for user simulator
        """
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                msg = msg.copy()
                # Remove tool_calls if it's None
                if msg.get("tool_calls") is None:
                    msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                if for_user:
                    # Skip tool messages for user simulator
                    if msg.get("role") == "tool":
                        continue
                    # Remove tool_calls from assistant messages
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        msg = {k: v for k, v in msg.items() if k != "tool_calls"}
            cleaned.append(msg)
        return cleaned

    def _make_completion_fn(
        self,
        client: AsyncOpenAI,
        time_list: list[float],
        is_agent: bool = True,
    ):
        # [FUNCTION DOC]
        # HELPER: Returns an async completion function for tau2 bound to client. It
        # drops
        # LiteLLM's num_retries, for the agent sets extra_body chat_template_kwargs
        # enable_thinking=True, for the user defaults top_p to 1.0, cleans messages,
        # calls
        # chat.completions.create and returns a ModelResponse. Errors are logged and re-
        # raised.
        # Each call's wall time is appended to time_list.
        """Create a completion function for the given client."""

        async def _completion(*args, **kwargs):
            # [FUNCTION DOC]
            # Async completion function implementing the behavior described for the
            # factory above.
            start_time = time.perf_counter()
            # Remove litellm-specific arguments
            kwargs.pop("num_retries", None)

            # Agent-specific: add thinking template
            if is_agent:
                extra_body = kwargs.pop("extra_body", {})
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}
                kwargs["extra_body"] = extra_body

            # User-specific: set default top_p
            if not is_agent and "top_p" not in kwargs:
                kwargs["top_p"] = 1.0

            # Clean messages
            if "messages" in kwargs:
                kwargs["messages"] = self._clean_messages(
                    kwargs["messages"], for_user=not is_agent
                )

            try:
                completion = await client.chat.completions.create(**kwargs)
                return self._convert_to_model_response(completion)
            except Exception as e:
                role = "Agent" if is_agent else "User"
                logger.error(f"{role} LLM error: {type(e).__name__}: {e}")
                raise
            finally:
                time_list.append(time.perf_counter() - start_time)

        return _completion

    def _get_agent_and_user(self, task: Task, env: Environment, run_info: Tau2RunInfo):
        # [FUNCTION DOC]
        # HELPER: Gets the policy, agent tools and (if available) user tools from the
        # env,
        # optionally adds the think tool, builds agent/user completion functions
        # recording times
        # into run_info. Solo mode: LLMSoloAgent with agent plus user tools and the
        # task, and a
        # DummyUser. Otherwise: LLMAgent and a UserSimulator with user tools (or None),
        # the task's
        # user scenario as instructions and user_llm / user_llm_args. Returns (agent,
        # user).
        # KNOWN ISSUE: in non-solo mode without user_llm_base_url the user client is
        # None and the
        # user simulator's first completion fails with AttributeError.
        agent_policy_doc = env.get_policy()
        tools: list[Tool] = env.get_tools()
        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = []
        if self.econfig.add_thinking_tool:
            tools.append(Tool(think))

        agent_completion_fn = self._make_completion_fn(
            self.agent_client, run_info.agent_time, is_agent=True
        )
        user_completion_fn = self._make_completion_fn(
            self.user_client, run_info.user_time, is_agent=False
        )

        if self.solo_mode:
            agent = LLMSoloAgent(
                tools=tools + user_tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                task=task,
                completion_fn=agent_completion_fn,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="dummy",
                llm_args=self.gen_args,
                completion_fn=agent_completion_fn,
            )
            user = UserSimulator(
                tools=user_tools if len(user_tools) > 0 else None,
                instructions=str(task.user_scenario),
                llm=self.econfig.user_llm,
                llm_args=self.econfig.user_llm_args,
                completion_fn=user_completion_fn,
            )
        return agent, user

    def _get_orchestrator(
        self,
        agent: LocalAgent[LLMAgentState],
        user: BaseUser,
        env: Environment,
        task: Task,
    ) -> Orchestrator:
        # [FUNCTION DOC]
        # HELPER: Builds a tau2 Orchestrator for the domain, agent, user, env and task
        # with
        # econfig.max_steps.
        return Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.econfig.max_steps,
        )

    async def run(self, task: Task) -> Tau2RunInfo:
        # [FUNCTION DOC]
        # Runs a simulation for task: creates the env and a Tau2RunInfo (reward 0.0),
        # builds
        # agent, user and orchestrator, awaits orchestrator.arun. On a simulation
        # exception logs,
        # stores the partial trajectory and error and returns (reward 0). Then evaluates
        # with
        # EvaluationType.ALL; on error logs and returns with the error recorded.
        # Otherwise stores
        # reward_info and reward, logs and returns the run info.
        """Run a simulation for the given task."""
        domain = self.domain
        solo_mode = self.solo_mode
        logger.info(
            f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Solo Mode: {solo_mode}"
        )

        env = self._get_environment()
        run_info = Tau2RunInfo(
            reward=0.0,
            task=task,
            messages=[],
            agent_time=[],
            user_time=[],
            reward_info=None,
            error=None,
        )
        agent, user = self._get_agent_and_user(task=task, env=env, run_info=run_info)
        orchestrator = self._get_orchestrator(
            agent=agent, user=user, env=env, task=task
        )

        try:
            simulation = await orchestrator.arun()
            run_info.messages = simulation.messages
        except Exception as e:
            logger.error(
                f"ERROR RUNNING SIMULATION: Domain: {domain}, Task: {task.id}, "
                f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
                f"Error running simulation: {e}. Setting reward to 0.0"
            )
            run_info.messages = orchestrator.get_trajectory()
            run_info.error = str(e)
            return run_info

        try:
            reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=solo_mode,
            )
            run_info.reward_info = reward_info
            run_info.reward = reward_info.reward
        except Exception as e:
            logger.error(
                f"ERROR EVALUATING SIMULATION: Domain: {domain}, Task: {task.id}, "
                f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
                f"Error evaluating simulation: {e}. Setting reward to 0.0"
            )
            run_info.reward_info = None
            run_info.error = str(e)
            return run_info

        logger.info(
            f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
            f"Reward: {reward_info.reward}"
        )
        return run_info


class Tau2AgentWorkflow:
    # [CLASS DOC]
    # Proxy-mode workflow object whose run is invoked by AReaL's proxy infrastructure
    # with
    # base_url/api_key/http_client; returns the simulation reward (the proxy assigns it
    # to
    # the recorded completions).
    """Tau2 agent workflow for AReaL proxy mode.

    This workflow runs a Tau2 customer service simulation using the proxy server
    for OpenAI-compatible API calls. It supports both standard multi-turn mode
    and solo mode where the agent handles both agent and user roles.

    Args:
        econfig: Tau2 environment configuration
        gen_args: Generation arguments (temperature, max_tokens, etc.)
        timeout: Maximum time allowed for a single episode (default: 600s)
    """

    def __init__(
        self,
        econfig: Tau2EnvConfig | dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 600.0,
    ):
        # [FUNCTION DOC]
        # Normalizes econfig (None gives defaults, a dict is converted to
        # Tau2EnvConfig), stores
        # gen_args (default {}) and the per-episode timeout (600 s).
        if econfig is None:
            econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            econfig = Tau2EnvConfig(**econfig)
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        # [FUNCTION DOC]
        # Resolves base_url and api_key from extra_kwargs or OPENAI_BASE_URL /
        # OPENAI_API_KEY env
        # vars (ValueError if no base_url) and an optional httpx client. Per-record
        # "econfig"
        # replaces the config entirely and "gconfig" updates a copy of gen_args. Loads
        # the task
        # by task_id and split (default "train"), creates the agent AsyncOpenAI client
        # (no
        # retries) and, if not solo and user_llm_base_url is set, a user client (3
        # retries, 120 s
        # timeout). Runs Tau2Runner with the timeout; on timeout logs and re-raises so
        # the
        # trajectory is discarded. Returns the float reward.
        # KNOWN ISSUE: return annotation allows a dict but only a float is returned.
        """Run a Tau2 simulation episode.

        Args:
            data: Input data containing task_id, split, and optional econfig/gconfig
            **extra_kwargs: Additional kwargs including:
                - base_url: Proxy server URL for agent LLM
                - api_key: Session-wise API key for proxy server authentication
                - http_client: Optional httpx.AsyncClient for requests

        Returns:
            float: The reward from the simulation
        """
        import httpx

        # Get proxy URL from workflow context
        base_url: str | None = extra_kwargs.get("base_url", None) or os.getenv(
            "OPENAI_BASE_URL"
        )
        api_key: str | None = extra_kwargs.get("api_key", None) or os.getenv(
            "OPENAI_API_KEY"
        )
        http_client: httpx.AsyncClient | None = extra_kwargs.get("http_client", None)

        if base_url is None:
            raise ValueError("base_url is required for Tau2AgentWorkflow")

        # Override econfig from data if provided
        econfig = self.econfig
        if "econfig" in data:
            econfig = Tau2EnvConfig(**data["econfig"])

        # Override gen_args from data if provided
        gen_args = self.gen_args.copy()
        if "gconfig" in data:
            gen_args.update(data["gconfig"])

        # Get task information
        domain = econfig.domain
        split = data.get("split", "train")
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        # Create AsyncOpenAI client for agent (pointing to proxy server)
        agent_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        # Create AsyncOpenAI client for user simulator (pointing to user LLM server)
        user_client = None
        if not econfig.solo_mode and econfig.user_llm_base_url:
            user_client = AsyncOpenAI(
                base_url=econfig.user_llm_base_url,
                api_key="dummy",  # Not used by self-hosted server
                max_retries=3,
                timeout=120.0,
            )

        # Create runner and execute
        runner = Tau2Runner(
            econfig=econfig,
            gen_args=gen_args,
            agent_client=agent_client,
            user_client=user_client,
        )

        try:
            run_info = await asyncio.wait_for(runner.run(task), timeout=self.timeout)
        except TimeoutError:
            logger.error(
                f"TIMEOUT: Task {task_id} exceeded {self.timeout}s limit. "
                f"Raise and discard current trajectory."
            )
            raise

        # Return the reward
        # The proxy server handles tracking completions and assigning rewards
        return run_info.reward
