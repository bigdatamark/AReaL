# [MODULE DOC]
# Config dataclass for the multi-turn GSM8K math RL example, extending GRPOConfig.
from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class MultiTurnGRPOConfig(GRPOConfig):
    # [CLASS DOC]
    # GRPOConfig plus agent_run_args (free-form dict; the example reads "max_turns",
    # default 2) and export_style ("concat" default or "individual"), which controls how
    # multi-turn completions are exported for training.
    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for the completions. By default export_style=concat."
        },
    )
