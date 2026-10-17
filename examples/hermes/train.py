# [MODULE DOC]
# Entry script for online PPO training where external users interact through the proxy
# gateway: loads a PPOConfig from CLI args and runs PPOTrainer.
"""Online training: external users interact via proxy gateway URL."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args):
    # [FUNCTION DOC]
    # Loads the experiment config (PPOConfig) from the argv list, constructs PPOTrainer
    # as a context manager and calls train(). Blocks for the whole training run; side
    # effects are those of the trainer (GPU, processes, checkpoints).
    config, _ = load_expr_config(args, PPOConfig)

    with PPOTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
