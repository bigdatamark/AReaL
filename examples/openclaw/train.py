# [MODULE DOC]
# Entry script for online PPO training where external users interact through the proxy
# gateway.
"""Online training: external users interact via proxy gateway URL."""

import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config


def main(args):
    # [FUNCTION DOC]
    # Loads PPOConfig from argv and runs PPOTrainer.train inside its context manager.
    config, _ = load_expr_config(args, PPOConfig)

    with PPOTrainer(config) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
