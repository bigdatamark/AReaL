# [MODULE DOC]
# Entry script for GRPO training of a vision-language model on CLEVR-Count-70k with
# PPOTrainer and VisionRLVRWorkflow; reward is exact match of the predicted count.
import re
import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

logger = logging.getLogger("clevr_count_70k_grpo")


def extract_answer(pred_str, data_name, use_last_number=True):
    # [FUNCTION DOC]
    # Returns the last number found inside square brackets in pred_str (as a string), or
    # ""
    # if none. data_name and use_last_number are unused.
    match = re.findall(r"\[([0-9\.]+)\]", pred_str)
    if match:
        return match[-1]

    return ""


def clevr_count_70k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    # [FUNCTION DOC]
    # Reward function: extracts the bracketed count from str(completions) and returns
    # 1.0 if
    # it equals str(answer) after stripping, 0.0 otherwise or if either is empty.
    # Exceptions
    # are logged as warnings and yield 0.0.
    try:
        sol = extract_answer(str(completions), data_name="")  # str number
        ans = str(answer)

        if not sol or not ans:
            return 0.0

        return float(sol.strip() == ans.strip())
    except Exception:
        logger.warning("Exception in clevr_count_70k_reward_fn", exc_info=True)
        return 0.0


def main(args):
    # [FUNCTION DOC]
    # Parses args into GRPOConfig, loads processor and tokenizer, builds train/test
    # datasets
    # with the processor and runs PPOTrainer.train with VisionRLVRWorkflow for training
    # and
    # eval (reward_fn by import path, enable_thinking=False; eval temperature 0.6).
    config, _ = load_expr_config(args, GRPOConfig)
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

    workflow_kwargs = dict(
        reward_fn="examples.vlm.clevr_count_70k_grpo.clevr_count_70k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
