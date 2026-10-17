# [MODULE DOC]
# Configuration dataclasses for the Terminal-Bench agentic RL example.
from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class TaskTimeouts:
    # [CLASS DOC]
    # Dataclass of per-phase timeouts in seconds for a terminal task episode:
    # environment
    # reset (default 1800), agent reset (120), one agent step (agent_astep, 300),
    # completion
    # evaluation (1200) and cleanup (None = no timeout). Underscore-prefixed field names
    # mirror
    # the names of the agent methods they bound.
    _reset_env: float = 1800.0
    _reset_agent: float = 120.0
    agent_astep: float = 300.0
    _evaluate_completion_sync: float = 1200.0
    _cleanup: float | None = None


@dataclass
class AgentRLConfig(GRPOConfig):
    # [CLASS DOC]
    # GRPOConfig extension for the terminal agent: n_trajs trajectories per task (1),
    # max_tokens_per_trajectory (32768), max_iteration agent turns (3), max_workers for
    # concurrency (25), non_think_mode (True, disables thinking in chat template),
    # async_training (False), task_timeouts (TaskTimeouts), filter_uniform_reward
    # (False, drop
    # groups whose rewards are all equal) and encourage_completion_reward (False, bonus
    # shaping
    # for finishing the task).
    n_trajs: int = field(default=1)
    max_tokens_per_trajectory: int = field(default=32768)
    max_iteration: int = field(default=3)
    max_workers: int = field(default=25)
    non_think_mode: bool = field(default=True)
    async_training: bool = field(default=False)
    task_timeouts: TaskTimeouts = field(default_factory=TaskTimeouts)
    filter_uniform_reward: bool = field(default=False)
    encourage_completion_reward: bool = field(default=False)
