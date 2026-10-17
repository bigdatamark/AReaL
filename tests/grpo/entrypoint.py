# [MODULE DOC]
# Training entrypoint launched by tests/grpo/test_grpo.py for the GRPO integration test.
# Parses a GRPOConfig from argv, trains with RLVRWorkflow using the GSM8K reward,
# records the average task reward per step and writes it to
# <cluster.fileroot>/rewards.json on rank 0.
import json
import os
import sys

import torch.distributed as dist

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow import RLVRWorkflow


class MinimalPPOTrainer(PPOTrainer):
    # [CLASS DOC]
    # PPOTrainer subclass that replaces stats committing with collecting the averaged
    # task reward into rewards_history (list of floats) for test assertions.
    """Trainer that collects stats to JSON for testing."""

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards all args to PPOTrainer.__init__ and initializes rewards_history as an
        # empty list.
        super().__init__(*args, **kwargs)
        self.rewards_history = []

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # [FUNCTION DOC]
        # Overrides the parent stats hook: exports actor stats and appends
        # stats['ppo_actor/task_reward/avg'] to rewards_history; epoch/step args
        # ignored.
        # KNOWN ISSUE: Stats are never committed/logged, so regular stats logging is
        # suppressed during the test.
        # Collect stats before committing
        stats = self.actor.export_stats()
        self.rewards_history.append(stats["ppo_actor/task_reward/avg"])


def main() -> None:
    # [FUNCTION DOC]
    # Main entry: loads GRPOConfig from sys.argv, loads the tokenizer, builds the train
    # dataset, then within the trainer context trains with RLVRWorkflow configured with
    # reward_fn 'areal.reward.gsm8k_reward_fn', config.gconfig, the trainer tokenizer
    # and enable_thinking=False. On rank 0 writes rewards_history as JSON to
    # <fileroot>/rewards.json.
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    with MinimalPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        workflow = RLVRWorkflow
        workflow_kwargs = dict(
            reward_fn="areal.reward.gsm8k_reward_fn",
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            enable_thinking=False,
        )

        trainer.train(workflow, workflow_kwargs=workflow_kwargs)

        # Save rewards to JSON for test assertions
        if dist.get_rank() == 0:
            with open(os.path.join(config.cluster.fileroot, "rewards.json"), "w") as f:
                json.dump(trainer.rewards_history, f)


if __name__ == "__main__":
    main()
