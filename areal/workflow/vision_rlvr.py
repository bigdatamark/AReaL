# [MODULE DOC]
# VisionRLVRWorkflow: RLVR variant for vision-language models. Uses a HF processor to
# tokenize text+images, sends base64 images with the request, passes reward args by
# keyword, and returns multi-modal inputs alongside the usual trajectory tensors.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import AsyncRewardWrapper, InferenceEngine, ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from areal.workflow.rlvr import RLVRWorkflow

logger = logging.getLogger("VisionRLVRWorkflow")


class VisionRLVRWorkflow(RLVRWorkflow):
    # [CLASS DOC]
    # Extends RLVRWorkflow with an AutoProcessor; default prompt hooks of the parent are
    # not used.
    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | str,
        enable_thinking: bool,
    ):
        # [FUNCTION DOC]
        # Initializes the parent (reward, gconfig, tokenizer, thinking) and loads the
        # processor from a path if a string is given.
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            enable_thinking,
        )
        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: Traced 'reward' phase. Decodes output tokens and awaits the reward
        # function with keyword args prompt, completions, prompt_ids, completion_ids
        # plus task_data; returns the reward.
        # KNOWN ISSUE: the docstring says it logs the metric to the stats tracker, but
        # logging happens in _collect_samples.
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function with keyword arguments, and logs
        metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """

        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
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
        # HELPER: Same as parent: generate in a 'generate' phase, compute reward, record
        # it to stats_tracker, return (response, reward).
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
        # Resolves a string reward_fn lazily; runs the processor on data['images'] and
        # data['messages'] (expected to be already-templated text) without padding;
        # takes input_ids and mm_token_type_ids of the first row; base64-encodes images;
        # builds a ModelRequest with image data, optional vLLM chat messages
        # (data['messages_chat']), n_samples=1, tokenizer and processor; collects one
        # sample. Returns batch-of-one tensors: input_ids and mm_token_type_ids (long;
        # output tokens typed 0), loss_mask, logprobs, versions, turn_ids,
        # attention_mask, rewards, plus multi_modal_input list holding pixel_values and
        # optional image_grid_thw.
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        processed_input = processor_callable(
            images=data["images"],
            text=data["messages"],
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]
        mm_token_type_ids: list[int] = processed_input["mm_token_type_ids"].tolist()[0]

        byte_images = image2base64(data["images"])
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[data["messages_chat"]]
            if "messages_chat" in data
            else None,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        mm_token_type_ids = mm_token_type_ids + [0] * resp.output_len
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        turn_ids = [-1] * resp.input_len + [0] * resp.output_len

        # Build multi-modal input
        multi_modal_input = [
            {
                "pixel_values": processed_input["pixel_values"],
            }
        ]
        if "image_grid_thw" in processed_input:
            multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
            "mm_token_type_ids": torch.tensor(
                mm_token_type_ids, dtype=torch.long
            ).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "multi_modal_input": multi_modal_input,
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "turn_ids": torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }
