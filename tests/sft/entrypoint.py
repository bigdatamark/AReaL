# [MODULE DOC]
# Training entrypoint launched by tests/sft/test_sft.py for the SFT integration test.
# Parses an SFTConfig from argv, builds the training dataset through the single-
# controller data service path, trains with a trainer subclass that records per-step SFT
# loss, and writes the loss history to <cluster.fileroot>/losses.json on rank 0.
import json
import os
import sys

import torch.distributed as dist

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.data_service.rdataset import RDataset
from areal.utils.hf_utils import load_hf_tokenizer


class MinimalSFTTrainer(SFTTrainer):
    # [CLASS DOC]
    # SFTTrainer subclass that replaces stats committing with collecting the averaged
    # SFT loss into losses_history (a list of floats) for later test assertions.
    """Trainer that collects stats to JSON for testing."""

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards all args to SFTTrainer.__init__ and initializes losses_history as an
        # empty list.
        super().__init__(*args, **kwargs)
        self.losses_history = []

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        # [FUNCTION DOC]
        # Overrides the parent stats hook: exports actor stats and appends
        # stats['sft/loss/avg'] to losses_history. Arguments epoch, epoch_step,
        # global_step are ignored.
        # KNOWN ISSUE: Despite the method name, stats are not committed/logged anywhere,
        # so normal stats logging is suppressed during the test.
        # Collect stats before committing
        stats = self.actor.export_stats()
        self.losses_history.append(stats["sft/loss/avg"])


def main() -> None:
    # [FUNCTION DOC]
    # Main entry: forces AREAL_SPMD_MODE=0 in the environment, loads SFTConfig from
    # sys.argv, loads the tokenizer, builds the train dataset via get_custom_dataset and
    # asserts it is an RDataset (data-service path). Runs training inside the trainer
    # context manager; on distributed rank 0 dumps losses_history as JSON to
    # <fileroot>/losses.json. Raises AssertionError if the dataset type is wrong.
    os.environ["AREAL_SPMD_MODE"] = "0"

    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    assert isinstance(train_dataset, RDataset), (
        "SFT integration test expects get_custom_dataset() to return RDataset "
        "(single-controller data service path)."
    )

    with MinimalSFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,
    ) as trainer:
        trainer.train()

        # Save losses to JSON for test assertions
        if dist.get_rank() == 0:
            with open(os.path.join(config.cluster.fileroot, "losses.json"), "w") as f:
                json.dump(trainer.losses_history, f)


if __name__ == "__main__":
    main()
