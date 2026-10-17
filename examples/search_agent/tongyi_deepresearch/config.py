# [MODULE DOC]
# Experiment config for the Tongyi DeepResearch search-agent RL example.
from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig, InferenceEngineConfig


@dataclass
class AgentRLConfig(GRPOConfig):
    # [CLASS DOC]
    # GRPOConfig extension with agent limits: max_llm_calls_per_run (100 LLM calls per
    # trajectory), max_tokens_per_trajectory (32768), log_agent_stats (False) and
    # log_agent_stats_keys (list of stat keys to log, default empty), plus judge_engine,
    # an
    # InferenceEngineConfig for the LLM judge used to score answers.
    max_llm_calls_per_run: int = field(
        default=100,
        metadata={
            "help": "Maximum number of LLM calls per trajectory. By default max_llm_calls_per_run=100."
        },
    )
    max_tokens_per_trajectory: int = field(
        default=32768,
        metadata={
            "help": "Maximum number of tokens per trajectory. By default max_tokens_per_trajectory=32768."
        },
    )
    # Logging Agent Trajectories
    log_agent_stats: bool = field(
        default=False,
        metadata={"help": "Log stats for agent trajectories"},
    )
    log_agent_stats_keys: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Keys of log stats for agent trajectories"},
    )
    judge_engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
