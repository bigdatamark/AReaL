# [MODULE DOC]
# RLVRWorkflow: single-turn reinforcement learning with verifiable rewards. Tokenizes a
# prompt via chat template, generates one sample from the inference engine, scores it
# with a (possibly lazily imported) reward function, and returns a batch-of-one
# trajectory tensor dict. Includes perf-tracer session instrumentation.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import apply_chat_template
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVRWorkflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Default prompt tokenizer: applies the tokenizer chat template to data with
    # generation prompt and the enable_thinking flag; returns token ids.
    return apply_chat_template(
        tokenizer,
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    # [FUNCTION DOC]
    # HELPER: Default prompt extractor: returns data['messages'].
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow generating one response per call to arun_episode. Configurable
    # reward function, tokenizer, thinking mode and prompt extraction/tokenization hooks
    # (callables or dotted import strings).
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        # [FUNCTION DOC]
        # Stores reward_fn (wrapped in AsyncRewardWrapper immediately unless it is an
        # import string, which is resolved lazily in arun_episode); loads the tokenizer
        # if given as a path; derives gconfig with tokenizer stop/pad ids; resolves
        # get_input_ids_fn and data_extract_prompt_fn from import strings if needed.
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        # Support string paths for get_input_ids_fn
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        # Support string paths for data_extract_prompt_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: Traced as session phase 'reward'. Decodes output tokens and awaits the
        # async reward with positional (prompt_str, completion_str, input_tokens,
        # output_tokens) plus task_data as kwargs; returns the reward.
        # KNOWN ISSUE: the docstring says it logs the metric to the stats tracker, but
        # logging happens in _collect_samples.
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function, and logs metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        # [FUNCTION DOC]
        # HELPER: Runs inside a new perf-tracer session. Awaits engine.agenerate(req) in
        # a 'generate' phase, computes reward, records reward to stats_tracker under the
        # current workflow scope, returns (response, reward).
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        return resp, reward

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Resolves a string reward_fn on first use, builds input ids via the hooks,
        # creates a ModelRequest (random rid, n_samples=1), decodes prompt, collects one
        # sample, and returns tensors with batch dim 1: input_ids (prompt+output,
        # int32), loss_mask (0 prompt, 1 output), logprobs (0 for prompt), versions (-1
        # for prompt), turn_ids (-1 prompt, 0 output), attention_mask (all True), and
        # scalar rewards.
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        turn_ids = [-1] * resp.input_len + [0] * resp.output_len

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "turn_ids": torch.tensor(turn_ids, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
