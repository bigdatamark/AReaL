# [MODULE DOC]
# Example GRPO training on GSM8K using the OpenAI-compatible MathAgent workflow for both
# training and evaluation.
import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    # [FUNCTION DOC]
    # Entry point: loads GRPOConfig and tokenizer, builds train/test datasets, derives
    # sampling kwargs (temperature, top_p, max_tokens, max_completion_tokens) from
    # gconfig, uses a copy with temperature 0.6 for evaluation, and runs
    # PPOTrainer.train with the MathAgent workflow.
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.openai.math_agent.MathAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.openai.math_agent.MathAgent",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
