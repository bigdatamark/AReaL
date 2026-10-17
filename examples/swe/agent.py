# [MODULE DOC]
# Proxy-mode agent workflow that runs AReaL-SWEAgent agents (swe, cc, oh, opencode,
# codex) on SWE-bench records. The external AReaL-SWEAgent checkout is located via
# config, env vars or a sibling directory and added to sys.path. The proxy base_url and
# per-rollout api_key are passed explicitly (not via os.environ) to avoid cross-rollout
# races. _DEFAULT_AGENT_CONFIGS maps agent type to its default YAML config name.
"""AReaL-SWEAgent workflow for AReaL proxy mode.

This module runs AReaL-SWEAgent-backed SWE-bench agents through AReaL's proxy
server during RL training. AReaL-SWEAgent stays in a separate checkout and is
discovered through ``AWEAGENT_ROOT`` or ``SWE_AGENT_ROOT``. AReaL passes the
proxy ``base_url`` and per-rollout ``api_key`` explicitly as kwargs instead
of mutating process-wide ``os.environ`` - the latter races across concurrent
rollouts and routes requests to the wrong proxy session.
"""

import asyncio
import os
import sys
import time
from typing import Any

from areal.utils import logging

logger = logging.getLogger("AReaL-SWEAgent")

_DEFAULT_AGENT_CONFIGS = {
    "swe": "1_0_0/min-swe-agent-train-top1",
    "cc": "train_cc_time3600",
    "oh": "eval_oh",
    "opencode": "eval_opencode",
    "codex": "eval_codex",
}


def _default_aweagent_root() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the path of an "AReaL-SWEAgent" directory next to the AReaL
    # repository
    # root (three directory levels above this file).
    """Return the default sibling checkout path for AReaL-SWEAgent."""
    areal_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(os.path.dirname(areal_root), "AReaL-SWEAgent")


def _ensure_aweagent_importable(aweagent_root: str = "") -> str:
    # [FUNCTION DOC]
    # HELPER: Resolves the AReaL-SWEAgent root from the explicit argument, else the
    # first set
    # of AWEAGENT_ROOT / SWE_AGENT_ROOT / SWEAGENT_ROOT env vars, else the default
    # sibling
    # path (made absolute). If it is a directory not yet on sys.path, prepends it
    # (mutating
    # sys.path) and logs; if it does not exist, logs a warning. Returns the resolved
    # root.
    """Add the AReaL-SWEAgent checkout root to ``sys.path``.

    Resolution order:
    1. Explicit ``aweagent_root`` argument (``econfig.agent_root`` /
       ``econfig.aweagent_root`` / ``econfig.swe_agent_root``).
    2. ``AWEAGENT_ROOT`` / ``SWE_AGENT_ROOT`` / ``SWEAGENT_ROOT`` env vars.
    3. ``../AReaL-SWEAgent`` relative to the AReaL repository root.

    Returns the resolved AReaL-SWEAgent checkout root.
    """
    if aweagent_root:
        root = os.path.abspath(aweagent_root)
    elif (
        os.getenv("AWEAGENT_ROOT")
        or os.getenv("SWE_AGENT_ROOT")
        or os.getenv("SWEAGENT_ROOT")
    ):
        root = os.path.abspath(
            os.getenv("AWEAGENT_ROOT")
            or os.getenv("SWE_AGENT_ROOT")
            or os.getenv("SWEAGENT_ROOT")
        )
    else:
        root = _default_aweagent_root()

    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
        logger.info(f"Added AReaL-SWEAgent root to sys.path: {root}")
    elif not os.path.isdir(root):
        logger.warning(
            f"AReaL-SWEAgent root does not exist: {root}. "
            "Set AWEAGENT_ROOT (or SWE_AGENT_ROOT) env var or econfig.agent_root "
            "(or the legacy econfig.swe_agent_root) to the correct path."
        )
    return root


def _configured_aweagent_root(econfig: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: First truthy of econfig agent_root, aweagent_root, swe_agent_root; else
    # "".
    return (
        econfig.get("agent_root")
        or econfig.get("aweagent_root")
        or econfig.get("swe_agent_root")
        or ""
    )


def _configured_agent_type(econfig: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns econfig agent_type (default "swe") stripped and lower-cased.
    # Raises
    # ValueError if it becomes empty (e.g. whitespace only).
    agent_type = str(econfig.get("agent_type") or "swe").strip().lower()
    if not agent_type:
        raise ValueError("econfig.agent_type must not be empty")
    return agent_type


def _configured_agent_config(econfig: dict[str, Any], agent_type: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the first truthy of agent_config, <agent_type>_agent_config,
    # swe_agent_config, the built-in default for the type, or the type name itself.
    # KNOWN ISSUE: swe_agent_config is consulted before the type default, so for types
    # other
    # than swe/cc (which have their own keys) a non-empty swe_agent_config (non-empty by
    # default in SWEEnvConfig) overrides the oh/opencode/codex defaults.
    return (
        econfig.get("agent_config")
        or econfig.get(f"{agent_type}_agent_config")
        or econfig.get("swe_agent_config")
        or _DEFAULT_AGENT_CONFIGS.get(agent_type)
        or agent_type
    )


class SWEAgentWorkflow:
    # [CLASS DOC]
    # Proxy-mode agent workflow object (run is invoked by AReaL's proxy infrastructure,
    # which
    # supplies base_url and api_key). Delegates episodes to AReaL-SWEAgent's
    # run_agent_with_reward and returns its scalar reward.
    """AReaL-SWEAgent workflow for AReaL proxy mode.

    This workflow runs an AReaL-SWEAgent agent type (``swe``, ``cc``, ``oh``,
    ``opencode``, or ``codex``) on SWE-bench style records in sandboxed
    environments. The agent's LLM calls are routed through AReaL's proxy
    server for on-policy RL training.

    The workflow delegates to AReaL-SWEAgent's ``run_agent_with_reward``, passing
    the AReaL proxy URL + per-rollout API key as explicit kwargs so that the
    current rollout's endpoint/key are used without process-wide env races.

    Args:
        econfig: Environment configuration dict. Key fields include
            ``agent_type``, ``agent_config``, ``swe_agent_config``,
            ``cc_agent_config``, ``agent_root`` / ``swe_agent_root``,
            ``llm_model``, ``opencode_provider``, ``codex_provider``,
            and ``timeout``.
        gen_args: Generation arguments (unused; token limits come from
            the AReaL-SWEAgent config YAML).
        timeout: Maximum time allowed for a single episode (default: 1800s).
    """

    def __init__(
        self,
        econfig: dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 1800.0,
    ):
        # [FUNCTION DOC]
        # Stores econfig (default {}), gen_args (unused, default {}) and timeout
        # (econfig
        # "timeout" overrides the argument, default 1800 s), and eagerly ensures the
        # AReaL-SWEAgent root is on sys.path so a wrong path warns early.
        if econfig is None:
            econfig = {}
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = econfig.get("timeout", timeout)

        # Ensure AReaL-SWEAgent is importable at construction time so we get an early
        # warning if the path is wrong, rather than failing mid-training.
        _ensure_aweagent_importable(_configured_aweagent_root(econfig))

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> dict[str, float] | float:
        # [FUNCTION DOC]
        # Runs one episode. Requires extra_kwargs base_url (else ValueError); api_key
        # defaults to
        # OPENAI_API_KEY env or "dummy". Merges a per-record data["econfig"] over a copy
        # of
        # econfig, resolves agent type and config name, and optional llm_model /
        # opencode_provider / codex_provider from econfig or LLM_MODEL /
        # OPENCODE_PROVIDER /
        # CODEX_PROVIDER env vars. Awaits _run_episode with the instance timeout; on
        # timeout logs
        # an error and re-raises TimeoutError (trajectory discarded). Logs start/finish
        # with
        # elapsed time and returns float(reward).
        # KNOWN ISSUE: return annotation allows a dict but only a float is ever
        # returned.
        """Run one SWE-bench style agent episode.

        Args:
            data: Input data containing SWE-bench instance fields:
                - instance_id (str): SWE-bench instance ID
                - problem_statement (str): The GitHub issue description
                - eval_script (str): Shell script for reward evaluation
            **extra_kwargs: Additional kwargs injected by AReaL proxy infra:
                - base_url (str): Proxy server URL for the agent LLM.
                - api_key (str): Per-rollout API key for the proxy session.

        Returns:
            float: The reward from the episode (0.0 or 1.0).
        """
        base_url: str | None = extra_kwargs.get("base_url", None)
        if base_url is None:
            raise ValueError("base_url is required for SWEAgentWorkflow")

        api_key: str = extra_kwargs.get("api_key", os.getenv("OPENAI_API_KEY", "dummy"))

        econfig = self.econfig.copy()
        if "econfig" in data:
            econfig.update(data["econfig"])

        agent_type = _configured_agent_type(econfig)
        config_name = _configured_agent_config(econfig, agent_type)
        llm_model = econfig.get("llm_model") or os.getenv("LLM_MODEL") or None
        opencode_provider = (
            econfig.get("opencode_provider") or os.getenv("OPENCODE_PROVIDER") or None
        )
        codex_provider = (
            econfig.get("codex_provider") or os.getenv("CODEX_PROVIDER") or None
        )
        instance_id = data.get("instance_id", "unknown")

        logger.info(
            f"Starting {agent_type} episode: "
            f"instance_id={instance_id}, config={config_name}"
        )
        start_time = time.time()

        try:
            reward = await asyncio.wait_for(
                self._run_episode(
                    data=data,
                    agent_type=agent_type,
                    config_name=config_name,
                    base_url=base_url,
                    api_key=api_key,
                    llm_model=llm_model,
                    opencode_provider=opencode_provider,
                    codex_provider=codex_provider,
                ),
                timeout=self.timeout,
            )
        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                f"TIMEOUT: Instance {instance_id} exceeded {self.timeout}s "
                f"(elapsed: {elapsed:.1f}s). Discarding trajectory."
            )
            raise

        elapsed = time.time() - start_time
        logger.info(
            f"Finished {agent_type} episode: instance_id={instance_id}, "
            f"reward={reward}, elapsed={elapsed:.1f}s"
        )
        return float(reward)

    async def _run_episode(
        self,
        data: dict[str, Any],
        agent_type: str,
        config_name: str,
        base_url: str,
        api_key: str,
        llm_model: str | None,
        opencode_provider: str | None,
        codex_provider: str | None,
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: Lazily imports aweagent.lifecycle.run_agent_with_reward and awaits it
        # with the
        # record, agent type, config name, result_dir (LOG_DIR env or ./logs) and
        # override
        # api key/base url/model/providers. Returns float(reward); any exception is
        # logged with
        # its traceback and yields 0.0 (so agent failures become zero reward rather than
        # discards).
        """Execute one episode through AReaL-SWEAgent and return the reward.

        Delegates to ``aweagent.lifecycle.run_agent_with_reward``, which
        handles env creation, the agent loop, reward computation, and trace
        persistence. ``base_url`` / ``api_key`` are forwarded as explicit
        kwargs (NOT via ``os.environ``) so concurrent rollouts can't read a
        foreign session's key.
        """
        # Imported lazily - AReaL-SWEAgent is only added to sys.path at run time.
        from aweagent.lifecycle import (
            run_agent_with_reward,  # type: ignore[import-not-found]
        )

        result_dir = os.getenv("LOG_DIR", "./logs")
        try:
            reward, _ = await run_agent_with_reward(
                data,
                agent_type=agent_type,
                agent_config=config_name,
                result_dir=result_dir,
                override_api_key=api_key,
                override_base_url=base_url,
                override_llm_model=llm_model,
                override_opencode_provider=opencode_provider,
                override_codex_provider=codex_provider,
            )
        except Exception as e:
            import traceback

            instance_id = data.get("instance_id", "unknown")
            logger.error(
                f"[{instance_id}] Episode error: {e}\n{traceback.format_exc()}"
            )
            return 0.0

        return float(reward)
