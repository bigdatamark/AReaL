# [MODULE DOC]
# Data types for the experimental ArealOpenAI client: enums for interaction API kind,
# and InteractionWithTokenLogpReward, which records one chat-completion or Responses-API
# call together with token ids, logprobs, reward and parent link, and converts it to
# training tensors. Also group reward normalization and string-only export helpers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from enum import Enum

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api import ModelResponse
from areal.utils import logging

logger = logging.getLogger("TokenLogpReward")


class ApiType(str, Enum):
    # [CLASS DOC]
    # String enum of the API used for an interaction: "completion" (chat completions),
    # "response" (Responses API) or "none".
    """API type for interaction."""

    COMPLETION = "completion"
    RESPONSE = "response"
    NONE = "none"


class InputName(str, Enum):
    # [CLASS DOC]
    # String enum naming the input field for log messages: "messages", "input_data" or
    # "none".
    """Input name used for logging."""

    MESSAGES = "messages"
    INPUT_DATA = "input_data"
    NONE = "none"


@dataclass
class InteractionWithTokenLogpReward:
    # [CLASS DOC]
    # Dataclass for one recorded interaction. Fields: model_response (token-level output
    # from the inference engine: input/output tokens, logprobs, versions), reward and
    # original_reward (pre-normalization reward), parent (previous interaction in the
    # same conversation, used to build concatenated multi-turn sequences),
    # chat_template_type ("hf" or "concat"), _cache (memoized tensor dict), messages
    # (chat input messages), output_message_list (assistant output messages), completion
    # (OpenAI ChatCompletion) or response (OpenAI Response) plus input_data (Responses
    # input), and _interaction_id (explicit ID used when neither completion nor response
    # object is present, e.g. after deserialization).
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    original_reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Interaction ID cache (used for deserialization)
    _interaction_id: str | None = None

    @property
    def has_tensor_data(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if model_response or a cached tensor dict exists.
        return self.model_response is not None or self._cache is not None

    @property
    def is_completion(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if a ChatCompletion object is attached.
        return self.completion is not None

    @property
    def is_response(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if a Responses-API Response object is attached.
        return self.response is not None

    @property
    def api_type(self) -> ApiType:
        # [FUNCTION DOC]
        # Property: ApiType.COMPLETION if a completion is attached, else RESPONSE if a
        # response is attached, else NONE.
        """API type (completion/response)."""
        if self.is_completion:
            return ApiType.COMPLETION
        elif self.is_response:
            return ApiType.RESPONSE
        else:
            return ApiType.NONE

    @property
    def input_name_for_logging(self) -> InputName:
        # [FUNCTION DOC]
        # Property: the InputName matching api_type (MESSAGES, INPUT_DATA or NONE), used
        # in log text.
        """Input name used for logging."""
        if self.is_completion:
            return InputName.MESSAGES
        elif self.is_response:
            return InputName.INPUT_DATA
        else:
            return InputName.NONE

    @property
    def current_data(self) -> list[dict] | str | ResponseInputParam | None:
        # [FUNCTION DOC]
        # Property: the interaction's input: messages for completions, input_data for
        # responses, None otherwise.
        if self.is_completion:
            return self.messages
        elif self.is_response:
            return self.input_data
        else:
            return None

    @property
    def parent_data(self) -> list[dict] | str | ResponseInputParam | None:
        # [FUNCTION DOC]
        # Property: the parent's current_data, or None if there is no parent.
        if self.parent is None:
            return None
        return self.parent.current_data

    @property
    def interaction_id(self) -> str | None:
        # [FUNCTION DOC]
        # Property: the ID of the attached completion or response; otherwise the
        # explicit _interaction_id (may be None).
        if self.is_completion:
            return self.completion.id
        elif self.is_response:
            return self.response.id
        elif self._interaction_id is not None:
            return self._interaction_id
        else:
            return None

    @interaction_id.setter
    def interaction_id(self, value):
        # [FUNCTION DOC]
        # Setter: stores an explicit interaction ID. Raises ValueError if a completion
        # or response is attached (their IDs are authoritative).
        if self.is_completion or self.is_response:
            raise ValueError("Cannot set ID for completion or responses")
        self._interaction_id = value

    @property
    def created_at(self) -> float | None:
        # [FUNCTION DOC]
        # Property: creation timestamp as float from the completion's "created" or the
        # response's "created_at"; None if neither is attached.
        if self.is_completion:
            return float(self.completion.created)
        elif self.is_response:
            return float(self.response.created_at)
        else:
            return None

    @property
    def remaining_messages(self) -> list[dict]:
        # [FUNCTION DOC]
        # Property: the messages new in this turn. With no parent, all messages;
        # otherwise drops the first N messages where N is the parent's input messages
        # plus its output messages. Asserts the parent's output_message_list is set.
        if self.parent is None:
            return self.messages
        assert self.parent.output_message_list is not None, (
            "Parent output message is not set."
        )
        parent_len = len(self.parent.messages + self.parent.output_message_list)
        return self.messages[parent_len:]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Converts the interaction to a training sample of tensors, each with a leading
        # batch dimension of 1. Returns the cache if present. Requires model_response
        # (asserts). Sets self.seq_tokens to input plus output tokens (side effect
        # attribute). Default ("hf" template or no parent): logprobs are 0 for prompt
        # and engine logprobs for output, loss_mask 0/1, versions -1 for prompt and
        # engine versions for output, turn_ids -1 for prompt and 0 for output. For
        # "concat" with a parent: recursively gets the parent's tensors, reuses its
        # logprobs/mask/versions/turn_ids for the shared prefix, pads the new prompt
        # tokens beyond the parent length with 0/0/-1/-1, and gives output tokens a turn
        # id one greater than the parent's max non-negative turn id. If the child's
        # input is not longer than the parent's sequence, logs a detailed WARNING and
        # falls back to the default (parent ignored). Result keys: input_ids, loss_mask,
        # logprobs, versions, turn_ids (int32), attention_mask (all-True bool), rewards
        # and original_rewards (1-element float tensors; reward defaults to 0.0,
        # original_reward defaults to reward). Caches and returns the dict.
        # KNOWN ISSUE: the concat path assumes the parent's sequence is a token prefix
        # of the child's input without checking, and the prefix-length fallback is
        # marked FIXME in the code.
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        assert resp is not None, "Model response is not set."
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_turn_ids = parent_res["turn_ids"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert (
                parent_len
                == len(parent_loss_mask)
                == len(parent_versions)
                == len(parent_turn_ids)
            )
            valid_parent_turn_ids = [tid for tid in parent_turn_ids if tid >= 0]
            own_turn_id = max(valid_parent_turn_ids) + 1 if valid_parent_turn_ids else 0
            if resp.input_len > parent_len:
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
                turn_ids = (
                    parent_turn_ids
                    + [-1] * (resp.input_len - parent_len)
                    + [own_turn_id] * resp.output_len
                )
            else:
                # FIXME: Find out why this happens occasionally
                api_type = self.api_type
                input_name = self.input_name_for_logging
                logger.warning(
                    f"The input length of the child {api_type} ({resp.input_len}) is less than or "
                    f"equal to the length of the parent {api_type} {parent_len}. "
                    f"This should not happen if the {input_name}s are constructed properly. "
                    f"Ignoring the parent {api_type} by masking them out. \n"
                    f"Parent input token ids: {self.parent.model_response.input_tokens}\n"
                    f"Parent output token ids: {self.parent.model_response.output_tokens}\n"
                    f"Child input token ids: {resp.input_tokens}\n"
                    f"Parent input {input_name}: {self.parent_data}\n"
                    f"Child input {input_name}: {self.current_data}",
                )
                logprobs = [0.0] * resp.input_len + resp.output_logprobs
                loss_mask = [0] * resp.input_len + [1] * resp.output_len
                versions = [-1] * resp.input_len + resp.output_versions
                turn_ids = [-1] * resp.input_len + [0] * resp.output_len
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            turn_ids = [-1] * resp.input_len + [0] * resp.output_len
        reward = self.reward if self.reward is not None else 0.0
        original_reward = (
            self.original_reward if self.original_reward is not None else reward
        )
        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            turn_ids=torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
            original_rewards=torch.tensor([float(original_reward)]),
        )
        self._cache = result
        return result


def normalize_group_rewards(
    results: list[dict[str, InteractionWithTokenLogpReward] | None],
) -> bool:
    # [FUNCTION DOC]
    # Group-normalizes rewards across rollouts in place. Input: list of per-rollout
    # ordered dicts of interactions (or None). Takes each rollout's reward from its last
    # interaction; returns False (no change) if the list is empty or any rollout is
    # empty/None or has a None last reward. Otherwise computes mean and population std
    # (std fixed to 1 for a single rollout), normalizes as (r - mean)/(std + 1e-8), and
    # for every interaction with a non-None reward stores original_reward = old reward
    # and sets reward to its rollout's normalized value, also patching
    # rewards/original_rewards in any cached tensor dict. Returns True.
    # KNOWN ISSUE: original_reward is overwritten with the current reward, so calling it
    # twice loses the true raw reward.
    """Normalize one scalar reward per rollout while preserving raw rewards."""
    if not results:
        return False

    reward_per_result: list[float | None] = []
    for result in results:
        if not result:
            reward_per_result.append(None)
            continue
        last_id = next(reversed(result))
        reward_per_result.append(result[last_id].reward)

    if any(reward is None for reward in reward_per_result):
        return False

    rewards = torch.tensor(reward_per_result, dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False) if rewards.numel() > 1 else torch.tensor(1.0)
    normalized_rewards = ((rewards - mean) / (std + 1e-8)).tolist()

    for result, normalized_reward in zip(results, normalized_rewards):
        assert result is not None
        for interaction in result.values():
            if interaction.reward is None:
                continue
            interaction.original_reward = interaction.reward
            interaction.reward = normalized_reward
            if interaction._cache is not None:
                interaction._cache["rewards"] = torch.tensor([float(normalized_reward)])
                interaction._cache["original_rewards"] = torch.tensor(
                    [float(interaction.original_reward)]
                )
    return True


def concat_string_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, list[dict]]:
    # [FUNCTION DOC]
    # Exports interactions without tensor data (e.g. external API mode). Input: dict of
    # interaction ID to interaction. Returns a dict with key "interactions" holding, per
    # interaction in order, a dict of request (the messages list), response (content of
    # the first output message, or empty string) and reward. No side effects.
    """Concat interactions that lack tensor data (e.g. external API mode).

    Returns a dict with an ``"interactions"`` key containing a list of
    ``{"request": ..., "response": ..., "reward": ...}`` dicts, one per
    interaction.  This is the counterpart of
    :func:`~areal.utils.data.concat_padded_tensors` for string-only
    trajectories.
    """
    return {
        "interactions": [
            {
                "request": v.messages,
                "response": (
                    v.output_message_list[0]["content"] if v.output_message_list else ""
                ),
                "reward": v.reward,
            }
            for v in interactions.values()
        ]
    }
