# [MODULE DOC]
# Entry script for PPO training with framework-based agent workflows (OpenAI, Anthropic,
# LangChain agents) loaded by import path. Adds its own directory to sys.path to import
# configs.
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import AgentConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    # [FUNCTION DOC]
    # Loads AgentConfig from CLI args, tokenizer, train ('train') and valid ('test')
    # datasets; builds workflow kwargs (temperature, top_p, max_tokens,
    # max_completion_tokens from max_new_tokens, max_turns) and eval kwargs with
    # temperature 0.6; runs PPOTrainer.train with the configured workflow classes.
    config, _ = load_expr_config(args, AgentConfig)
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

    # Build workflow kwargs from config
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        # For anthropic
        max_tokens=config.gconfig.max_tokens,
        # For openai
        max_completion_tokens=config.gconfig.max_new_tokens,
        # For agent-specific kwargs
        max_turns=config.max_turns,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
