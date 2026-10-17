# [MODULE DOC]
# Entry script for supervised fine-tuning of a vision-language model on CLEVR-Count-70k
# with AReaL's SFTTrainer. Run with CLI args forwarded to load_expr_config.
import sys

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def main(args):
    # [FUNCTION DOC]
    # Parses args into an SFTConfig, loads HF processor and tokenizer from
    # config.tokenizer_path, builds train ("train" split) and validation ("test" split)
    # datasets via get_custom_dataset with the processor, and runs SFTTrainer.train
    # inside its
    # context manager (which handles setup/teardown). Returns None; side effects are
    # distributed training, checkpointing and logging.
    config, _ = load_expr_config(args, SFTConfig)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    with SFTTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
