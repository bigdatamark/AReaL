# [MODULE DOC]
# Config dataclass for the OpenAI Agents SDK RL example, extending GRPOConfig.
from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AgentRLConfig(GRPOConfig):
    # [CLASS DOC]
    # GRPOConfig plus reward_fn_path (dotted path to reward fn, default GSM8K),
    # agent_builder_path (dotted path to a function returning an OpenAI Agents SDK
    # Agent) and agent_builder_kwargs (kwargs for that builder).
    reward_fn_path: str = field(
        default="areal.reward.gsm8k.gsm8k_reward_fn",
        metadata={
            "help": "The path to the reward function. Should follow the API in `areal/api/reward_api.py`."
        },
    )
    agent_builder_path: str = field(
        default="areal.workflow.openai_agent.math_agent.build_math_agent",
        metadata={
            "help": "The path to the OpenAI agent builder. The function should return an `Agent` object with OpenAI SDK."
        },
    )
    agent_builder_kwargs: dict = field(
        default_factory=dict,
        metadata={
            "help": "The initialization arguments for the agent builder function."
        },
    )
