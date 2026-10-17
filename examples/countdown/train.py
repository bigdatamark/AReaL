# [MODULE DOC]
# Example GRPO training on the Countdown task with a simple single-turn workflow that
# tokenizes the raw 'query' text and scores with reward_score.compute_score. Imports
# reward_score as a top-level module, so the countdown directory must be on sys.path.
import sys
import uuid

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from reward_score import compute_score
from transformers import PreTrainedTokenizerFast

from areal import PPOTrainer, workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.utils import logging, stats_tracker

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"CountDown @ {worker_id}")


class CountDownWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow generating one completion per prompt and scoring it with
    # compute_score.
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
    ):
        # [FUNCTION DOC]
        # Loads tokenizer if given as path and derives gconfig with tokenizer stop/pad
        # ids.
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer

    async def arun_episode(self, engine: InferenceEngine, data):
        # [FUNCTION DOC]
        # Encodes data['query'] without special tokens, generates one sample, builds
        # input_ids/loss_mask/logprobs/versions/attention_mask (batch dim 1, prompt
        # masked), computes the reward from decoded completion and the data row
        # (target/numbers), records it to stats_tracker, and returns the tensor dict
        # with rewards shape [1].
        input_ids = self.tokenizer.encode(data["query"], add_special_tokens=False)

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        resp = await engine.agenerate(req)

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = compute_score(
            completions_str,
            data,
        )

        # Log reward.
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return {
            # unsqueeze to add an additional batch dimension
            "input_ids": torch.tensor(seq).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask).unsqueeze(0),
            "logprobs": torch.tensor(logprobs).unsqueeze(0),
            "versions": torch.tensor(versions).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            "rewards": torch.tensor([float(reward)]),
        }


def get_countdown_dataset(dataset_path, rank, world_size):
    # [FUNCTION DOC]
    # Loads a JSON dataset's train split from dataset_path and shards it by
    # rank/world_size. Unused by main.
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def main(args):
    # [FUNCTION DOC]
    # Loads GRPOConfig, loads the JSON train dataset from config.train_dataset.path (no
    # tokenization), and runs PPOTrainer.train with CountDownWorkflow by import path.
    config, _ = load_expr_config(args, GRPOConfig)

    train_dataset = load_dataset(
        path="json",
        split="train",
        data_files=config.train_dataset.path,
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )

    with PPOTrainer(config, train_dataset=train_dataset) as trainer:
        trainer.train(
            workflow="examples.countdown.train.CountDownWorkflow",
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
