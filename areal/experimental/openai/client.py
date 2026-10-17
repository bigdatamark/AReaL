# [MODULE DOC]
# ArealOpenAI: a drop-in AsyncOpenAI client whose chat.completions.create and
# responses.create run generation on an AReaL rollout/inference engine (anything with an
# async agenerate(ModelRequest) -> ModelResponse) instead of calling a remote API. Each
# call tokenizes the conversation locally (HF chat template, or "concat" mode which
# reuses the parent interaction's exact tokens and appends only new turn tokens), builds
# GenerationHyperparameters, generates, parses tool calls/reasoning, builds OpenAI
# response objects, and records an InteractionWithTokenLogpReward (token ids, logprobs,
# output messages) in an InteractionCache so rewards can be attached and trajectories
# exported for RL training. Also contains message normalization helpers (pydantic ->
# dict, Responses input items -> chat messages, image extraction for VLMs, tool schema
# alignment with SGLang rendering). Streaming is simulated from a full generation.
# Side effect at import: sets OPENAI_API_KEY and OPENAI_BASE_URL env vars to "none" if
# unset so the base AsyncOpenAI constructor does not fail.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import os
import re
import uuid
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, overload

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, Body, NotGiven
from openai.resources.chat.completions.completions import (
    AsyncCompletions as BaseAsyncCompletions,
)
from openai.resources.responses.responses import AsyncResponses as BaseAsyncResponses
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import response_create_params
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from openai.types.responses.tool_param import ToolParam
from openai.types.shared_params.metadata import Metadata
from pydantic import BaseModel

from areal.api import ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.tool_call_parser import process_tool_calls
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging
from areal.utils.hf_utils import apply_chat_template

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class _AsyncGenerateEngine(Protocol):
    # [CLASS DOC]
    # HELPER: Structural Protocol for the engine dependency: an object with async
    # agenerate(ModelRequest) -> ModelResponse.
    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        # [FUNCTION DOC]
        # Protocol method; the default body raises NotImplementedError.
        raise NotImplementedError()


TRolloutEngine = TypeVar("TRolloutEngine", bound=_AsyncGenerateEngine)

# reset OpenAI keys when using the wrapped client.
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "none")
os.environ["OPENAI_BASE_URL"] = os.environ.get("OPENAI_BASE_URL", "none")

logger = logging.getLogger("OpenAIClient")


def _align_tools_with_sglang(tools_list: list) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Normalizes tool definitions so the tools block rendered by the chat
    # template
    # matches SGLang's own /v1/chat/completions path. For each tool: pydantic models are
    # model_dump'ed, mappings copied to dicts, other types passed through with a
    # warning;
    # flat Responses-style function tools are nested under "function"; then, if SGLang's
    # protocol Tool model is importable, the dict is round-tripped through it
    # (validation
    # failures logged and the dict kept). If SGLang is missing, warns once (flag stored
    # as a
    # function attribute) and only normalizes shapes. Returns the new list.
    """Round-trip tools through sglang's pydantic Tool model so the dicts
    fed to ``apply_chat_template`` byte-match what sglang's serving_chat
    produces on its own ``/v1/chat/completions`` path.

    The goal is behavioral parity with sglang's native route, so a
    trajectory served here stays reproducible on sglang's own endpoint.
    Without it the two paths render the ``tools`` block differently (field
    order and default-field presence, e.g. sglang's ``Function`` dumps
    ``strict: false`` while LiteLLM omits it). The prompt-token delta this
    causes is a symptom of the mismatch, not the thing being optimized.

    Assumptions this function cannot verify (see FIXME at the call sites):
    - Backend is sglang. Call sites only invoke this when the engine class
      name identifies sglang, so other backends are left unaligned.
    - The worker venv's sglang version matches the serving version; a skew in
      disaggregated deployments silently aligns to a stale tool format.

    Behaviour:
    - sglang not importable → log once, then still normalize shapes (dict
      conversion and flat→nested) and skip only the pydantic round-trip, so
      ``apply_chat_template`` always receives nested dicts.
    - Responses ``FunctionToolParam`` is flat (top-level ``name``/``parameters``
      and no ``function`` key); it is normalized to the nested Chat shape that
      both sglang's ``Tool`` model and ``apply_chat_template`` expect before
      validation.
    - per-tool validation failure → log + keep that single tool unchanged
      (partial alignment is still better than no alignment for the rest).
    - BaseModel inputs are handled via ``model_dump()`` first.
    """

    def _to_chat_format(t_dict: dict) -> dict:
        # [FUNCTION DOC]
        # HELPER: If a dict is a flat function tool (type "function", has "name", no
        # "function" key) returns {"type": "function", "function": <other fields>}; else
        # returns it unchanged.
        # Flat Responses FunctionToolParam → nested Chat shape. sglang's
        # Function model drops fields it doesn't declare (e.g. defer_loading),
        # so the dump still matches the chat-completions path. Chat tools
        # already nest under ``function`` and pass through unchanged.
        if (
            t_dict.get("type") == "function"
            and "function" not in t_dict
            and "name" in t_dict
        ):
            function = {k: v for k, v in t_dict.items() if k != "type"}
            return {"type": "function", "function": function}
        return t_dict

    _SglTool = None
    try:
        from sglang.srt.entrypoints.openai.protocol import Tool as _SglTool
    except Exception as e:
        if not getattr(_align_tools_with_sglang, "_warned_no_sglang", False):
            logger.warning(
                "_align_tools_with_sglang: sglang not importable (%s); tools "
                "block will diverge from sglang chat-completions path. "
                "Install sglang in the worker venv to fix this.",
                e,
            )
            _align_tools_with_sglang._warned_no_sglang = True  # type: ignore[attr-defined]
    aligned: list[dict] = []
    for t in tools_list:
        # Accept dict (TypedDict at runtime) and BaseModel.
        if isinstance(t, BaseModel):
            t_dict = t.model_dump()
        elif isinstance(t, Mapping):
            t_dict = dict(t)
        else:
            logger.warning(
                "_align_tools_with_sglang: tool of type %s is neither dict "
                "nor BaseModel; passing through unchanged.",
                type(t).__name__,
            )
            aligned.append(t)
            continue
        t_dict = _to_chat_format(t_dict)
        if _SglTool is None:
            aligned.append(t_dict)
            continue
        try:
            aligned.append(_SglTool(**t_dict).model_dump())
        except Exception as e:
            logger.warning(
                "_align_tools_with_sglang: pydantic Tool validation failed "
                "for tool %s (%s); passing through unchanged. This will "
                "cause partial drift from sglang chat-completions path.",
                t_dict.get("function", {}).get("name", "<unknown>"),
                e,
            )
            aligned.append(t_dict)
    return aligned


_DEFAULT_MAX_TOTAL_TOKENS = 32768


def _ensure_message_dict_list(
    name: str,
    value: list[Any],
) -> list[dict[str, Any]]:
    # [FUNCTION DOC]
    # HELPER: Validates value is a list whose items are dicts or pydantic models
    # (TypeError
    # otherwise, naming the argument and index) and recursively normalizes them: models
    # via
    # model_dump(exclude_none=True), mappings with None values removed, non-string
    # iterables to lists. Returns the new list of dicts.
    """Validate that ``value`` is a list of dictionaries or BaseModel objects.

    Args:
        name: Name of the argument being validated (for error messages).
        value: The list provided by the caller.

    Returns:
        A list containing only dictionaries. BaseModel objects are
        converted into their dictionary representation with
        `model_dump(exclude_none=True)`; dictionaries are preserved.

    Raises:
        TypeError: If ``value`` is not a list or an element cannot be converted to a dict.
    """

    if not isinstance(value, list):
        raise TypeError(
            f"{name} must be provided as a list, got {type(value).__name__}"
        )

    def _normalize(item: Any):
        # [FUNCTION DOC]
        # HELPER: Recursive normalizer described in the parent (models first, since they
        # are
        # also iterable).
        # we should convert BaseModel first, because BaseModel is also Iterable
        if isinstance(item, BaseModel):
            return item.model_dump(exclude_none=True)
        elif isinstance(item, Mapping):
            return {k: _normalize(v) for k, v in item.items() if v is not None}
        elif (
            isinstance(item, Iterable)
            and not isinstance(item, str)
            and not isinstance(item, bytes)
            and not isinstance(item, bytearray)
        ):
            return [_normalize(sub_item) for sub_item in item]
        else:
            return item

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict) or isinstance(item, BaseModel):
            normalized.append(_normalize(item))
        else:
            raise TypeError(
                f"{name}[{index}] must be a dict or a BaseModel; got {type(item).__name__}"
            )
    return normalized


def _find_kth(lst: list, target, k: int) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the index of the k-th (1-based) occurrence of target in lst, or -1
    # if there are fewer than k occurrences. For k <= 0 returns -1.
    def target_indices():
        # [FUNCTION DOC]
        # HELPER: Generator yielding indices where lst equals target.
        for i, char in enumerate(lst):
            if char == target:
                yield i

    gen = target_indices()
    try:
        result = -1
        for _ in range(k):
            result = next(gen)
        return result
    except StopIteration:
        return -1


# Regex for data URI: data:image/<subtype>;base64,<data>
_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


def _extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    # [FUNCTION DOC]
    # HELPER: Scans list-typed message contents for image_url parts. Returns
    # (image_data, messages_for_tokenizer, vision_messages_for_vllm): image_data holds
    # the
    # base64 payload of data:image URIs or the raw URL otherwise, in order;
    # messages_for_tokenizer replaces each image part by {"type": "image"} for HF VLM
    # templates; vision_messages_for_vllm keeps image_url parts with url "placeholder"
    # (real data injected later by the vLLM backend). Non-image parts are deep-copied.
    # Raises ValueError for an image part with empty/missing url.
    """Extract image data from OpenAI-format messages.

    Scans message ``content`` lists for ``image_url`` content parts,
    extracts base64 data (or raw URLs), and converts messages to a
    HuggingFace-compatible format for ``apply_chat_template``.

    Args:
        messages: Normalized list of message dicts (OpenAI format).

    Returns:
        A 3-tuple of:

        - **image_data** – list of base64 image strings (no data-URI prefix)
          or raw URL strings for each image found.
        - **messages_for_tokenizer** – deep copy of *messages* where every
          ``{"type": "image_url", ...}`` part is replaced by
          ``{"type": "image"}`` so that HuggingFace VLM tokenizers insert
          the correct image-placeholder tokens.
        - **vision_messages_for_vllm** – deep copy of *messages* where
          ``image_url`` parts retain the ``image_url`` key but the ``url``
          value is replaced with a placeholder (the actual base64 data URI
          is injected later by the vLLM backend from *image_data*).
    """
    image_data: list[str] = []
    messages_for_tokenizer: list[dict[str, Any]] = []
    vision_messages_for_vllm: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            messages_for_tokenizer.append(deepcopy(msg))
            vision_messages_for_vllm.append(deepcopy(msg))
            continue

        tok_parts: list[dict[str, Any]] = []
        vllm_parts: list[dict[str, Any]] = []

        for part in content:
            if not isinstance(part, dict):
                tok_parts.append(part)
                vllm_parts.append(deepcopy(part))
                continue

            if part.get("type") == "image_url":
                image_url_obj = part.get("image_url", {})
                url = (
                    image_url_obj.get("url", "")
                    if isinstance(image_url_obj, dict)
                    else ""
                )

                if not url:
                    raise ValueError(
                        "image_url content part has an empty or missing URL. "
                        "Provide a valid data URI or HTTP(S) URL in "
                        "image_url.url."
                    )

                # Extract base64 payload from data URIs; keep raw URLs as-is.
                m = _DATA_URI_RE.match(url)
                if m:
                    image_data.append(m.group(1))
                else:
                    image_data.append(url)

                tok_parts.append({"type": "image"})

                # vLLM backend injects actual data URI from req.image_data.
                vllm_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": "placeholder"},
                    }
                )
            else:
                tok_parts.append(deepcopy(part))
                vllm_parts.append(deepcopy(part))

        tok_msg = {**msg, "content": tok_parts}
        vllm_msg = {**msg, "content": vllm_parts}
        messages_for_tokenizer.append(tok_msg)
        vision_messages_for_vllm.append(vllm_msg)

    return image_data, messages_for_tokenizer, vision_messages_for_vllm


def _convert_tool_output_format(
    item: dict,
) -> ChatCompletionToolMessageParam | dict:
    # [FUNCTION DOC]
    # HELPER: Converts a Responses FunctionCallOutput dict (type "function_call_output"
    # with "output") into a chat tool message {"role": "tool", "content": output,
    # "tool_call_id": call_id if present}; returns other inputs unchanged.
    """Convert custom tool output format to standard chat template format.

    Converts openai.types.responses.response_input_item_param.FunctionCallOutput
    to openai.types.chat.ChatCompletionToolMessageParam.

    Args:
        item: Input dict, could be FunctionCallOutput from openai-agents SDK
            with format: {'call_id': str, 'output': str, 'type': 'function_call_output'}

    Returns:
        ChatCompletionToolMessageParam (TypedDict) with format:
        {'role': 'tool', 'content': str, 'tool_call_id': str}
        or the original dict if conversion is not needed.
    """
    if (
        isinstance(item, dict)
        and "output" in item
        and item.get("type") == "function_call_output"
    ):
        converted = {
            "role": "tool",
            "content": item["output"],
        }
        # Add tool_call_id if present
        if "call_id" in item:
            converted["tool_call_id"] = item["call_id"]
        return converted
    return item


def _build_messages_list(item: dict) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Converts one normalized Responses input item into chat messages. String
    # content -> one message. List content: output_text/input_text parts become text
    # parts,
    # input_image parts become image_url parts (with optional detail; ValueError if
    # image_url missing); any other part type or non-dict part raises ValueError. With
    # any
    # image the result is a single message with list content, otherwise one message per
    # text part. Items without content (e.g. function_call_output) are converted via
    # _convert_tool_output_format and deep-copied. Other content types raise ValueError.
    # KNOWN ISSUE: Responses function_call items (no content, type "function_call") pass
    # through unconverted, so the assistant's tool-call turn is not expressed in chat
    # form.
    """Convert a Responses API input item into Chat Completions message dicts.

    Handles ``output_text``, ``input_text``, ``input_image``, and
    ``function_call_output`` content types.  When the item contains at
    least one ``input_image`` part the returned message keeps a ``list``
    content value (multimodal); otherwise each text part produces its own
    flat ``{"role": …, "content": "…"}`` message.

    Args:
        item: A single normalised Responses-API input item (dict form of
            :class:`ResponseInputItemParam`).

    Returns:
        One or more Chat-Completions-style message dicts.

    Raises:
        ValueError: On unsupported content types, non-dict content parts,
            or ``input_image`` items that lack an ``image_url`` value.
    """
    messages_list: list[dict] = []
    if "content" in item:
        if isinstance(item["content"], str):
            messages_list.append(
                {"role": item["role"], "content": item["content"]},
            )
        elif isinstance(item["content"], Iterable):
            content_parts: list[dict] = []
            has_multimodal = False
            for content in item["content"]:
                if not isinstance(content, dict):
                    raise ValueError("Unsupported content format")
                ctype = content.get("type", "")
                if ctype == "output_text" and "text" in content:
                    content_parts.append({"type": "text", "text": content["text"]})
                elif ctype == "input_text" and "text" in content:
                    content_parts.append({"type": "text", "text": content["text"]})
                elif ctype == "input_image":
                    has_multimodal = True
                    image_url = content.get("image_url", "")
                    if not image_url:
                        raise ValueError(
                            "input_image content part requires a non-empty "
                            "'image_url' field; file_id-only images are not "
                            "supported."
                        )
                    image_url_dict: dict[str, Any] = {"url": image_url}
                    if "detail" in content:
                        image_url_dict["detail"] = content["detail"]
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": image_url_dict,
                        }
                    )
                else:
                    raise ValueError(f"Unsupported content format: {ctype}")
            if has_multimodal:
                messages_list.append({"role": item["role"], "content": content_parts})
            else:
                for cp in content_parts:
                    messages_list.append(
                        {"role": item["role"], "content": cp["text"]},
                    )
        else:
            raise ValueError("Unsupported input item format")
    else:
        # Convert tool output format if needed
        converted = _convert_tool_output_format(item)
        messages_list.append(deepcopy(converted))
    return messages_list


def _resolve_max_total_tokens(
    prompt_len: int,
    max_new_tokens: int,
    engine_max_tokens: int | None,
) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns min(prompt_len + max_new_tokens, engine_max_tokens or 32768), the
    # total-length cap passed to the engine.
    # Fall back to a fixed context-length ceiling when the deployment does not
    # configure engine_max_tokens, so a large client max_tokens cannot push
    # prompt + generation past the backend model's context window.
    cap = engine_max_tokens or _DEFAULT_MAX_TOTAL_TOKENS
    return min(prompt_len + max_new_tokens, cap)


def _parse_tool_call_arguments(messages: list[dict]) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Returns a new list where any message tool_calls whose function.arguments
    # is
    # a JSON string are copied with arguments parsed to an object (invalid JSON left as
    # is).
    # Untouched messages are shared, not copied. Needed for chat templates that iterate
    # over argument items.
    """Return a new message list with tool_call arguments parsed from JSON strings
    to dicts. Some chat templates (e.g. GLM-5.1) iterate over arguments with
    .items(), which fails when arguments is a JSON string per OpenAI API convention.

    Only creates new dicts where modifications are needed; unaffected messages
    are shared with the input list to avoid expensive deep copies.
    """
    result = []
    for msg in messages:
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not tool_calls:
            result.append(msg)
            continue
        new_tool_calls = []
        modified = False
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if fn is None:
                new_tool_calls.append(tc)
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    tc = {**tc, "function": {**fn, "arguments": parsed}}
                    modified = True
                except (json.JSONDecodeError, TypeError):
                    pass
            new_tool_calls.append(tc)
        if modified:
            result.append({**msg, "tool_calls": new_tool_calls})
        else:
            result.append(msg)
    return result


def concat_prompt_token_ids_with_parent(
    message_list: list[dict],
    parent: InteractionWithTokenLogpReward | None,
    tokenizer: "PreTrainedTokenizerFast",
    tools: Iterable[ChatCompletionToolParam] | None = None,
    extra_body: Body = {},
) -> list[int]:
    # [FUNCTION DOC]
    # Builds prompt token ids for "concat" mode so a child turn extends its parent's
    # exact
    # token sequence. If parent is given (ValueError if it has no model_response),
    # parent
    # tokens = parent input tokens + output tokens without stop + one EOS token, and the
    # parent's messages and output messages are prepended to message_list. The whole
    # conversation is chat-templated (tool-call args parsed, generation prompt added,
    # chat_template_kwargs from extra_body). With E EOS tokens in the parent tokens, the
    # template output is cut after its E-th EOS and the remainder appended to the parent
    # tokens (RuntimeError if no such EOS or nothing follows it). Without a parent
    # returns
    # the full templated ids. Returns a list of ints.
    # KNOWN ISSUE: alignment counts EOS ids, so it breaks if the template or generated
    # text
    # contains EOS-equal tokens elsewhere; stop strings are unsupported (TODO in code).
    """
    Concatenate prompt token IDs with parent interaction's tokens.
    """
    parent_tokens: list[int] = []
    all_message_list: list[dict] = []
    eos_token_id = tokenizer.eos_token_id

    # To ensure compatibility across different models, we adopted the following padding scheme:
    # Apply the chat template to the full-text message_list of the new input, then count the number of eos tokens
    # in the parent tokens. Locate the final index where the same number of eos tokens appears in the child_all_tokens,
    # all tokens after this index correspond to the newly input tokens for the current round.

    if parent is not None:
        if parent.model_response is None:
            raise ValueError("Parent interaction has no model_response.")
        # TODO: (yulangz) how to handle here when stop is set?
        parent_tokens = (
            parent.model_response.input_tokens
            + parent.model_response.output_tokens_without_stop  # without stop tokens
        )
        all_message_list += parent.messages if parent.messages is not None else []
        all_message_list += (
            parent.output_message_list if parent.output_message_list is not None else []
        )

        # If the parent terminates due to output exceeding length limits or being aborted, it will not have an EOS token.
        # We will add an extra EOS token to align with the chat template. During training, this added EOS will be treated
        # as part of the child message's prompt rather than the parent message's output, and therefore will be masked out
        # by the loss_mask.
        # If the parent terminated with an EOS token, it will be removed by parent.model_response.output_tokens_without_stop, and
        # we add it here.
        # TODO: should we mask this extra eos token in loss_mask during training?
        parent_tokens += [eos_token_id]

    all_message_list += message_list
    all_message_list = _parse_tool_call_arguments(all_message_list)

    all_tokens = apply_chat_template(
        tokenizer,
        all_message_list,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        **extra_body.get("chat_template_kwargs", {}),
    )
    parent_eos_num = parent_tokens.count(eos_token_id)
    if parent_eos_num > 0:
        child_tokens_truncate_idx = _find_kth(all_tokens, eos_token_id, parent_eos_num)
        if child_tokens_truncate_idx == -1 or child_tokens_truncate_idx + 1 >= len(
            all_tokens
        ):
            raise RuntimeError(
                f"Failed to align child tokens with parent tokens in concat prompt."
                f"Find child_truncate_idx at {child_tokens_truncate_idx}, "
                f"parent_eos_num: {parent_eos_num}, "
                f"all_tokens eos count: {all_tokens.count(eos_token_id)}"
            )
    else:
        child_tokens_truncate_idx = -1

    prompt_token_ids = parent_tokens + all_tokens[child_tokens_truncate_idx + 1 :]
    return prompt_token_ids


class AsyncCompletionsWithReward(BaseAsyncCompletions):
    # [CLASS DOC]
    # Subclass of the OpenAI SDK AsyncCompletions resource that generates locally via
    # the
    # engine and caches interactions. Holds engine, tokenizer, tool-call and reasoning
    # parser names, the default cache, engine_max_tokens (context cap),
    # chat_template_type
    # ("hf" or "concat") and LoRA adapter name. A class-level _warned_parameters set
    # exists
    # for one-time warnings (currently unused).
    """Extended AsyncCompletions that adds caching and reward functionality."""

    # Class-level set to track which parameters have been warned about
    # (shared across all instances)
    _warned_parameters: set[str] = set()

    def __init__(
        self,
        client,
        engine: TRolloutEngine,
        tokenizer: "PreTrainedTokenizerFast",
        cache: InteractionCache,
        tool_call_parser: str,
        reasoning_parser: str,
        engine_max_tokens: int | None = None,
        chat_template_type: str = "hf",
        lora_name: str = "",
    ):
        # [FUNCTION DOC]
        # Calls the SDK resource init with the parent client and stores all
        # collaborators and
        # settings as attributes.
        super().__init__(client)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self._cache = cache
        self.engine_max_tokens = engine_max_tokens
        self.chat_template_type = chat_template_type
        self.lora_name = lora_name

    def _build_chat_completion(
        self,
        completion_id: str,
        current_time: int,
        output_text: str,
        tool_calls: list | None,
        response: ModelResponse,
    ) -> tuple[ChatCompletion, ChatCompletionMessage]:
        # [FUNCTION DOC]
        # HELPER: Builds the assistant ChatCompletionMessage (tool_calls None if empty)
        # and a
        # single-choice ChatCompletion with the engine stop reason as finish_reason,
        # model
        # "None", and usage counted from input/output token lengths. Returns both.
        """Build ChatCompletion and ChatCompletionMessage objects.

        Args:
            completion_id: Unique identifier for the completion.
            current_time: Unix timestamp for the completion creation time.
            output_text: The generated text output.
            tool_calls: List of tool calls, or None if no tool calls.
            response: The ModelResponse from the inference engine.

        Returns:
            A tuple of (ChatCompletion, ChatCompletionMessage).
        """
        output_message = ChatCompletionMessage(
            content=output_text,
            role="assistant",
            # For all empty tool calls, set tool_calls=None
            tool_calls=tool_calls or None,
        )
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=response.stop_reason,
                    index=0,
                    logprobs=None,  # For simplicity
                    message=output_message,
                )
            ],
            created=current_time,
            model="None",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                completion_tokens=len(response.output_tokens),
                prompt_tokens=len(response.input_tokens),
                total_tokens=len(response.input_tokens) + len(response.output_tokens),
            ),
        )
        return chat_completion, output_message

    # [FUNCTION DOC]
    # Typing-only overload of create (stream=True -> async chunk generator; otherwise
    # ChatCompletion). No runtime behavior.
    # (Placed above the definition because the body is on the signature line.)
    @overload
    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        stream: Literal[True],
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        max_total_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_cache: InteractionCache | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]: ...

    # [FUNCTION DOC]
    # Typing-only overload of create (stream=True -> async chunk generator; otherwise
    # ChatCompletion). No runtime behavior.
    # (Placed above the definition because the body is on the signature line.)
    @overload
    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        stream: Literal[False] | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        max_total_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_cache: InteractionCache | None = None,
        **kwargs: Any,
    ) -> ChatCompletion: ...

    async def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        max_total_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_cache: InteractionCache | None = None,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        # [FUNCTION DOC]
        # OpenAI-compatible chat completion backed by the engine. Steps: validate
        # messages
        # (TypeError if not iterable, ValueError if empty; n != 1 raises
        # NotImplementedError),
        # normalize to dicts. Unless store is False, creates an interaction (deep copy
        # of
        # messages, template type) in areal_cache or the default cache under a fresh
        # "chatcmpl-" id (ValueError on id collision); insertion links it to its parent.
        # Tools
        # are listified (TypeError if not iterable) and aligned to SGLang format when
        # the engine
        # class name contains "sglang". Images are extracted. Prompt ids come from the
        # HF chat
        # template ("hf") or concat_prompt_token_ids_with_parent using the interaction's
        # remaining (new) messages ("concat"); other types raise RuntimeError. Token
        # limits:
        # max_tokens is an alias of max_completion_tokens (ValueError if both given);
        # the total
        # budget is min(max_total_tokens, engine_max_tokens); max_new_tokens = budget -
        # prompt
        # length (ValueError if <= 0) further capped by max_completion_tokens, else
        # default 512
        # with a warning. Interaction is removed from cache on these ValueErrors.
        # temperature
        # defaults 1.0 (0 means greedy), top_p 1.0, stop normalized to a list (dropped
        # with a
        # warning in concat mode), frequency_penalty 0. Builds ModelRequest (random rid,
        # metadata, image data, vLLM vision messages; stop ids are EOS and PAD) and
        # awaits
        # engine.agenerate. Decodes output without stop tokens, parses tool calls unless
        # tool_choice is "none" (JSON errors logged). For streaming, records completion,
        # model_response and output message in the cache before returning the
        # _create_stream
        # generator; otherwise records the same and returns the ChatCompletion. Extra
        # kwargs
        # are ignored.
        # KNOWN ISSUE: if generation or templating raises after the interaction was
        # inserted,
        # the incomplete interaction stays in the cache (export skips it with a
        # warning).
        """Override create method to use AReaL engine and cache responses."""

        is_streaming = not is_omitted(stream) and stream is True

        # Extract and validate supported parameters
        cache, interaction = None, None
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        if not isinstance(messages, Iterable):
            raise TypeError(
                "messages must be provided as an iterable of dictionaries or BaseModel instances."
            )
        if not is_omitted(n) and n != 1:
            raise NotImplementedError("n != 1 is not supported yet")
        n = 1

        messages_list_raw = list(messages)
        if not messages_list_raw:
            raise ValueError("messages cannot be empty")
        messages_list = _ensure_message_dict_list(
            "messages",
            messages_list_raw,
        )
        if extra_body is None:
            extra_body = {}

        # Convert response to OpenAI format
        current_time = int(datetime.datetime.now().timestamp())
        # Add interaction to cache, resolve parent relationship according to input messages
        if is_omitted(store) or store:
            # Cache the completion with its input messages
            cache = areal_cache if areal_cache is not None else self._cache
            if completion_id in cache:
                raise ValueError(f"Completion {completion_id} already exists in cache")
            interaction = InteractionWithTokenLogpReward(
                messages=deepcopy(messages_list),  # Store a copy of the input messages
                chat_template_type=self.chat_template_type,
            )
            cache[completion_id] = interaction

        # Convert messages to prompt format
        tools_list = None
        if not is_omitted(tools):
            if not isinstance(tools, Iterable):
                raise TypeError("tools must be an iterable of ChatCompletionToolParam")
            tools_list = list(tools)
            # FIXME: alignment targets sglang's rendering, so it is only
            # correct for an sglang backend. Apply it only when the engine is
            # positively identified as sglang (by class name); any other
            # backend is left unaligned to avoid aligning to the wrong format
            # or a misleading "install sglang" hint. A proper fix gates on a
            # backend identifier exposed by the engine.
            # See docs/en/tutorial/online_proxy.md.
            if "sglang" in type(self.engine).__name__.lower():
                tools_list = _align_tools_with_sglang(tools_list)

        image_data, messages_for_tokenizer, vision_messages_for_vllm = (
            _extract_images_from_messages(messages_list)
        )
        has_images = len(image_data) > 0

        tokenizer_messages = messages_for_tokenizer if has_images else messages_list
        tokenizer_messages = _parse_tool_call_arguments(tokenizer_messages)
        if self.chat_template_type == "hf":
            prompt_token_ids = apply_chat_template(
                self.tokenizer,
                tokenizer_messages,
                tools=tools_list,
                add_generation_prompt=True,
                tokenize=True,
                **extra_body.get("chat_template_kwargs", {}),
            )
        elif self.chat_template_type == "concat":
            concat_messages = (
                interaction.remaining_messages
                if interaction is not None
                else messages_list
            )
            if has_images:
                _, concat_tok_messages, _ = _extract_images_from_messages(
                    concat_messages
                )
            else:
                concat_tok_messages = concat_messages
            prompt_token_ids = concat_prompt_token_ids_with_parent(
                concat_tok_messages,
                interaction.parent if interaction is not None else None,
                self.tokenizer,
                tools=tools_list,
                extra_body=extra_body,
            )
        else:
            raise RuntimeError(
                f"Unsupported chat_template_type {self.chat_template_type}"
            )

        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)
        if not is_omitted(max_tokens):
            # NOTE: support deprecated `max_tokens` usage.
            if not is_omitted(max_completion_tokens):
                if (
                    interaction is not None
                    and cache is not None
                    and completion_id in cache
                ):
                    # Remove the interaction from cache on failure
                    del cache[completion_id]
                raise ValueError(
                    "max_tokens and max_completion_tokens cannot be set at the same time. "
                    "max_tokens has been deprecated. Please use max_completion_tokens instead. "
                    "To set the total max tokens, please use max_total_tokens instead."
                )
            # NOTE (2025-12-09): the usage of max_tokens has been changed.
            max_completion_tokens = max_tokens

        max_total_tokens_final = None
        if not is_omitted(max_total_tokens):
            max_total_tokens_final = max_total_tokens
        if self.engine_max_tokens is not None:
            if max_total_tokens_final is None:
                max_total_tokens_final = self.engine_max_tokens
            else:
                max_total_tokens_final = min(
                    max_total_tokens_final, self.engine_max_tokens
                )

        max_new_tokens = None
        if max_total_tokens_final is not None:
            max_new_tokens = max_total_tokens_final - len(prompt_token_ids)
            if max_new_tokens <= 0:
                if (
                    interaction is not None
                    and cache is not None
                    and completion_id in cache
                ):
                    # Remove the interaction from cache on failure
                    del cache[completion_id]
                raise ValueError(
                    f"len of prompt tokens {len(prompt_token_ids)} exceeds max_total_tokens {max_total_tokens_final}"
                )
        if not is_omitted(max_completion_tokens):
            if max_new_tokens is None:
                max_new_tokens = max_completion_tokens
            else:
                max_new_tokens = min(max_new_tokens, max_completion_tokens)
        if max_new_tokens is None:
            max_new_tokens = 512  # Default value
            logger.warning(
                "Neither max_tokens nor max_completion_tokens is set; "
                "defaulting max_new_tokens to 512."
            )

        top_p_val = 1.0 if is_omitted(top_p) else (top_p or 1.0)
        stop_tokens = None if is_omitted(stop) else stop

        # Since the concat logic cannot properly handle stop tokens yet, so we remove stop here.
        if stop_tokens is not None and self.chat_template_type == "concat":
            logger.warning(
                "stop tokens are not supported in concat mode yet; ignoring stop tokens."
            )
            stop_tokens = None

        if stop_tokens is not None and not isinstance(stop_tokens, list):
            stop_tokens = [stop_tokens]

        if is_omitted(frequency_penalty):
            frequency_penalty = 0.0

        # Create generation config
        gconfig = GenerationHyperparameters(
            n_samples=n,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            max_tokens=_resolve_max_total_tokens(
                prompt_len=len(prompt_token_ids),
                max_new_tokens=max_new_tokens,
                engine_max_tokens=self.engine_max_tokens,
            ),
            top_p=top_p_val,
            stop=stop_tokens,
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            lora_name=self.lora_name,
            seed=None if is_omitted(seed) else seed,
            stop_token_ids=list(
                set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata if not is_omitted(metadata) else {},
            tokenizer=self.tokenizer,
            image_data=image_data if has_images else None,
            vision_msg_vllm=([vision_messages_for_vllm] if has_images else None),
        )

        # Call inference engine
        response = await self.engine.agenerate(model_request)
        output_text = self.tokenizer.decode(response.output_tokens_without_stop)

        # Parse tool calls.
        tool_calls = None
        try:
            if (is_omitted(tool_choice) or tool_choice != "none") and tools_list:
                tool_calls, output_text, response.stop_reason = process_tool_calls(
                    output_text,
                    tools_list,
                    self.tool_call_parser,
                    self.reasoning_parser,
                    response.stop_reason,
                    tokenizer=self.tokenizer,
                )
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse tool calls from output text: {e}, output_text:\n"
                f"{output_text}"
            )

        # If streaming is requested, return an async generator
        if is_streaming:
            # Update cache BEFORE returning the generator to ensure the interaction
            # is recorded even if the generator is never iterated. This is critical
            # because LiteLLM's streaming adapter generates initial chunks (e.g.,
            # message_start, content_block_start) before iterating the original
            # generator, so if the client disconnects early, the cache update
            # code inside the generator would never execute.
            if cache is not None:
                chat_completion, output_message = self._build_chat_completion(
                    completion_id=completion_id,
                    current_time=current_time,
                    output_text=output_text,
                    tool_calls=tool_calls,
                    response=response,
                )
                cache[completion_id].completion = chat_completion
                cache[completion_id].model_response = response
                cache[completion_id].output_message_list = [
                    output_message.model_dump(exclude_none=True)
                ]
            return self._create_stream(
                completion_id=completion_id,
                current_time=current_time,
                output_text=output_text,
                tool_calls=tool_calls,
                response=response,
            )

        # Create proper ChatCompletion object with all required fields
        chat_completion, output_message = self._build_chat_completion(
            completion_id=completion_id,
            current_time=current_time,
            output_text=output_text,
            tool_calls=tool_calls,
            response=response,
        )

        if cache is not None:
            cache[completion_id].completion = chat_completion
            cache[completion_id].model_response = response
            cache[completion_id].output_message_list = [
                output_message.model_dump(exclude_none=True)
            ]
        return chat_completion

    async def _create_stream(
        self,
        completion_id: str,
        current_time: int,
        output_text: str,
        tool_calls: list | None,
        response: ModelResponse,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        # [FUNCTION DOC]
        # HELPER: Async generator simulating streaming from a finished generation:
        # yields a
        # role chunk, one content chunk with the full text (if any), two chunks per tool
        # call
        # (name+id with empty arguments, then arguments; split for LiteLLM Anthropic
        # adapter
        # compatibility), and a final chunk with finish_reason and usage. No cache
        # updates.
        """Generate streaming ChatCompletionChunk objects.

        Since Inference engine doesn't support true streaming, we simulate it by
        yielding the complete response as chunks.

        Note: Cache is updated by the caller (AsyncCompletionsWithReward.create)
        before this generator is created, to ensure the interaction is recorded
        even if the generator is never iterated.
        """
        try:
            # First chunk: role
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(role="assistant", content=""),
                        index=0,
                        finish_reason=None,
                    )
                ],
                created=current_time,
                model="None",
                object="chat.completion.chunk",
            )

            # Content chunks - yield the full text as one chunk
            # (In a true streaming implementation, this would be broken into smaller pieces)
            if output_text:
                yield ChatCompletionChunk(
                    id=completion_id,
                    choices=[
                        ChunkChoice(
                            delta=ChoiceDelta(content=output_text),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                    created=current_time,
                    model="None",
                    object="chat.completion.chunk",
                )

            # Tool calls chunks (if any)
            # Split each tool call into two chunks (name then arguments) to
            # match standard OpenAI streaming behavior. LiteLLM's Anthropic
            # streaming adapter treats the first chunk with a function name as
            # a content_block_start trigger and discards the processed delta
            # from that same chunk. If name and arguments are combined in a
            # single chunk, the arguments are lost and Anthropic clients see
            # input={}.
            if tool_calls:
                for idx, tool_call in enumerate(tool_calls):
                    tool_call = cast(ChatCompletionMessageFunctionToolCall, tool_call)
                    # Chunk 1: name + id, with empty arguments.
                    yield ChatCompletionChunk(
                        id=completion_id,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=idx,
                                            id=tool_call.id,
                                            type="function",
                                            function=ChoiceDeltaToolCallFunction(
                                                name=tool_call.function.name,
                                                arguments="",
                                            ),
                                        )
                                    ]
                                ),
                                index=0,
                                finish_reason=None,
                            )
                        ],
                        created=current_time,
                        model="None",
                        object="chat.completion.chunk",
                    )
                    # Chunk 2: arguments only, emitted as input_json_delta by
                    # Anthropic adapters after the tool_use block has started.
                    yield ChatCompletionChunk(
                        id=completion_id,
                        choices=[
                            ChunkChoice(
                                delta=ChoiceDelta(
                                    tool_calls=[
                                        ChoiceDeltaToolCall(
                                            index=idx,
                                            function=ChoiceDeltaToolCallFunction(
                                                arguments=tool_call.function.arguments,
                                            ),
                                        )
                                    ]
                                ),
                                index=0,
                                finish_reason=None,
                            )
                        ],
                        created=current_time,
                        model="None",
                        object="chat.completion.chunk",
                    )

            # Final chunk with finish_reason and usage
            yield ChatCompletionChunk(
                id=completion_id,
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(),
                        index=0,
                        finish_reason=response.stop_reason,
                    )
                ],
                created=current_time,
                model="None",
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=len(response.output_tokens),
                    prompt_tokens=len(response.input_tokens),
                    total_tokens=len(response.input_tokens)
                    + len(response.output_tokens),
                ),
            )
        finally:
            # Cleanup is handled by the caller via _safe_stream_wrapper
            pass


class AsyncResponsesWithReward(BaseAsyncResponses):
    # [CLASS DOC]
    # Subclass of the OpenAI SDK AsyncResponses resource implementing Responses API
    # create() on the AReaL engine with interaction caching; same attributes as
    # AsyncCompletionsWithReward.
    """Extended AsyncResponses that adds caching and reward functionality."""

    def __init__(
        self,
        client,
        engine: TRolloutEngine,
        tokenizer: "PreTrainedTokenizerFast",
        cache: InteractionCache,
        tool_call_parser: str,
        reasoning_parser: str,
        engine_max_tokens: int | None = None,
        chat_template_type: str = "hf",
        lora_name: str = "",
    ):
        # [FUNCTION DOC]
        # Calls the SDK resource init and stores collaborators and settings.
        super().__init__(client)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self._cache = cache
        self.engine_max_tokens = engine_max_tokens
        self.chat_template_type = chat_template_type
        self.lora_name = lora_name

    async def create(
        self,
        *,
        include: list[str] | None | NotGiven = NOT_GIVEN,
        input: str | ResponseInputParam | NotGiven = NOT_GIVEN,
        instructions: str | None | NotGiven = NOT_GIVEN,
        max_output_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        tool_choice: response_create_params.ToolChoice | NotGiven = NOT_GIVEN,
        tools: Iterable[ToolParam] | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        extra_body: Body | None = None,
        areal_cache: dict[str, InteractionWithTokenLogpReward] | None = None,
        **kwargs: Any,
    ) -> Response:
        # [FUNCTION DOC]
        # Responses API create backed by the engine. Generates "resp-"/"msg-" ids
        # (ValueError
        # on cache collision). Builds chat messages: optional system message from
        # instructions,
        # then a string input becomes one user message, or a list input is normalized
        # and each
        # item converted with _build_messages_list; other input types or missing input
        # raise
        # ValueError; non-empty include raises NotImplementedError. Always caches an
        # interaction (messages copy, template type, deep copy of input as input_data).
        # Tools,
        # images and prompt tokenization as in chat completions (hf or concat).
        # max_new_tokens =
        # engine_max_tokens - prompt length (ValueError if <= 0, interaction removed),
        # capped by
        # max_output_tokens, default 512 with warning. stop comes from kwargs (ignored
        # in concat
        # mode). Generates, decodes, parses tool calls in Responses format, counts
        # reasoning
        # tokens between <think> tags, and builds a Response whose output is the tool-
        # call items
        # if any, else a single output_text message, with usage. Stores a deep copy of
        # the
        # Response, model_response and dumped output items in the cache; returns the
        # Response.
        # KNOWN ISSUE: when tool calls are present the assistant text (if any) is
        # dropped from
        # the output; store is not honored; tool_choice is reported as "none" when
        # omitted even
        # if tools are given.
        """Override create method to use AReaL engine"""
        # Initialize IDs and timestamps
        resp_id = f"resp-{uuid.uuid4().hex[:29]}"
        msg_id = f"msg-{uuid.uuid4().hex[:29]}"
        current_time = float(int(datetime.datetime.now().timestamp()))
        interaction, cache = None, None
        # Add interaction to cache, resolve parent relationship according to input messages

        # Cache the completion with its input messages
        cache = areal_cache if areal_cache is not None else self._cache
        if resp_id in cache:
            raise ValueError(f"Response {resp_id} already exists in cache")
        if extra_body is None:
            extra_body = {}

        # Build a simple messages list compatible with tokenizer chat template
        messages_list: list[dict] = []
        if not is_omitted(instructions):
            messages_list = [
                {"role": "system", "content": instructions},
            ]
        if not is_omitted(include) and len(include) > 0:
            raise NotImplementedError("include is not supported yet")

        if is_omitted(input):
            raise ValueError("input is required for Responses.create")

        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        if isinstance(input, list):
            normalized_input = _ensure_message_dict_list(
                "input",
                input,
            )
            for item in normalized_input:
                messages_list += _build_messages_list(item)
        else:
            raise ValueError(
                "Unsupported Responses input format: "
                "expected str or list of message items with input_text."
            )
        interaction = InteractionWithTokenLogpReward(
            messages=deepcopy(messages_list),  # Store a copy of the input messages
            chat_template_type=self.chat_template_type,
            input_data=(
                deepcopy(input) if not is_omitted(input) else ""
            ),  # Store a copy of the input data
        )
        cache[resp_id] = interaction

        # Apply chat template
        tools_list = None
        if not is_omitted(tools):
            if not isinstance(tools, Iterable):
                raise TypeError("tools must be an iterable of ChatCompletionToolParam")
            tools_list = list(tools)
            # FIXME: alignment targets sglang's rendering, so it is only
            # correct for an sglang backend. Apply it only when the engine is
            # positively identified as sglang (by class name); any other
            # backend is left unaligned to avoid aligning to the wrong format
            # or a misleading "install sglang" hint. A proper fix gates on a
            # backend identifier exposed by the engine.
            # See docs/en/tutorial/online_proxy.md.
            if "sglang" in type(self.engine).__name__.lower():
                tools_list = _align_tools_with_sglang(tools_list)

        image_data, messages_for_tokenizer, vision_messages_for_vllm = (
            _extract_images_from_messages(messages_list)
        )
        has_images = len(image_data) > 0

        tokenizer_messages = messages_for_tokenizer if has_images else messages_list
        tokenizer_messages = _parse_tool_call_arguments(tokenizer_messages)
        if self.chat_template_type == "hf":
            prompt_token_ids = apply_chat_template(
                self.tokenizer,
                tokenizer_messages,
                tools=tools_list,
                add_generation_prompt=True,
                tokenize=True,
                **extra_body.get("chat_template_kwargs", {}),
            )
        elif self.chat_template_type == "concat":
            remaining = interaction.remaining_messages
            if has_images:
                _, remaining_tok, _ = _extract_images_from_messages(remaining)
            else:
                remaining_tok = remaining
            prompt_token_ids = concat_prompt_token_ids_with_parent(
                remaining_tok,
                interaction.parent if interaction is not None else None,
                self.tokenizer,
                tools=tools_list,
                extra_body=extra_body,
            )
        else:
            raise RuntimeError(
                f"Unsupported chat_template_type {self.chat_template_type}"
            )

        # Map sampling params
        temp = 1.0 if is_omitted(temperature) else (temperature or 0.0)
        top_p_val = 1.0 if is_omitted(top_p) else (top_p or 1.0)
        max_new_tokens = None
        if self.engine_max_tokens is not None:
            max_new_tokens = self.engine_max_tokens - len(prompt_token_ids)
            if max_new_tokens <= 0:
                if interaction is not None and cache is not None and resp_id in cache:
                    # Remove the interaction from cache on failure
                    del cache[resp_id]
                raise ValueError(
                    f"len of prompt tokens {len(prompt_token_ids)} exceeds engine_max_tokens {self.engine_max_tokens}"
                )
        if not is_omitted(max_output_tokens):
            if max_new_tokens is None:
                max_new_tokens = max_output_tokens
            else:
                max_new_tokens = min(max_new_tokens, max_output_tokens)
        if max_new_tokens is None:
            max_new_tokens = 512  # Default value
            logger.warning("max_output_tokens not specified, defaulting to 512.")

        stop = kwargs.get("stop", None)
        if stop is not None and self.chat_template_type == "concat":
            logger.warning(
                "stop tokens are not supported in concat mode yet; ignoring stop tokens."
            )
            stop = None
        if is_omitted(frequency_penalty):
            frequency_penalty = 0.0

        # Create generation config and request
        gconfig = GenerationHyperparameters(
            n_samples=1,
            temperature=temp,
            max_new_tokens=max_new_tokens,
            max_tokens=_resolve_max_total_tokens(
                prompt_len=len(prompt_token_ids),
                max_new_tokens=max_new_tokens,
                engine_max_tokens=self.engine_max_tokens,
            ),
            top_p=top_p_val,
            stop=stop,
            greedy=temp == 0,
            frequency_penalty=frequency_penalty,
            lora_name=self.lora_name,
            seed=None if is_omitted(seed) else seed,
            stop_token_ids=list(
                set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
            ),
        )

        model_request = ModelRequest(
            input_ids=prompt_token_ids,
            gconfig=gconfig,
            rid=str(uuid.uuid4()),
            metadata=metadata if not is_omitted(metadata) else {},
            tokenizer=self.tokenizer,
            image_data=image_data if has_images else None,
            vision_msg_vllm=([vision_messages_for_vllm] if has_images else None),
        )

        # Call inference engine
        engine_resp = await self.engine.agenerate(model_request)
        output_text = self.tokenizer.decode(engine_resp.output_tokens_without_stop)

        # Parse tool calls.
        tool_calls = None
        try:
            if (is_omitted(tool_choice) or tool_choice != "none") and tools_list:
                tool_calls, output_text, engine_resp.stop_reason = process_tool_calls(
                    output_text,
                    tools_list,
                    self.tool_call_parser,
                    self.reasoning_parser,
                    engine_resp.stop_reason,
                    use_responses=True,
                    tokenizer=self.tokenizer,
                )
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse tool calls from output text: {e}, output_text:\n"
                f"{output_text}"
            )

        # Extract reasoning tokens from output
        reasoning_token_count = self._count_reasoning_tokens(output_text)

        # Build Responses API objects
        output_message = ResponseOutputMessage(
            id=msg_id,
            role="assistant",
            status="completed",
            type="message",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text=output_text,
                    type="output_text",
                )
            ],
        )

        if tool_calls:
            resp_output = tool_calls
        else:
            resp_output = [output_message]

        usage = ResponseUsage(
            input_tokens=len(engine_resp.input_tokens),
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=len(engine_resp.output_tokens),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=reasoning_token_count
            ),
            total_tokens=len(engine_resp.input_tokens) + len(engine_resp.output_tokens),
        )

        response = Response(
            id=resp_id,
            created_at=current_time,
            error=None,
            incomplete_details=None,
            instructions=None if is_omitted(instructions) else instructions,
            metadata=None if is_omitted(metadata) else metadata,
            model="None",
            object="response",
            output=resp_output,
            parallel_tool_calls=False,
            temperature=temp,
            tool_choice=tool_choice if not is_omitted(tool_choice) else "none",
            tools=tools_list,
            top_p=top_p_val,
            background=None,
            conversation=None,
            max_output_tokens=max_new_tokens,
            max_tool_calls=None,
            previous_response_id=None,
            prompt=None,
            prompt_cache_key=None,
            reasoning=None,
            safety_identifier=None,
            service_tier=None,
            status="completed",
            text=None,
            top_logprobs=None,
            truncation=None,
            usage=usage,
            user=None,
        )

        cache[resp_id].response = deepcopy(response)
        cache[resp_id].model_response = engine_resp
        cache[resp_id].output_message_list = [
            o.model_dump(exclude_none=True) for o in resp_output
        ]
        return response

    def _count_reasoning_tokens(
        self,
        output_text: str,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
    ) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns the number of tokenizer tokens (no special tokens) between the
        # first
        # thinking_start_token and the following thinking_end_token (or end of text if
        # unclosed); 0 if no start token.
        """
        Count reasoning tokens from output text by extracting content within thinking start and end tokens.
        """

        if thinking_start_token not in output_text:
            return 0
        processed_text = output_text.split(thinking_start_token, maxsplit=1)[1]
        if thinking_end_token in processed_text:
            processed_text = processed_text.split(thinking_end_token, maxsplit=1)[0]
        return len(self.tokenizer.encode(processed_text, add_special_tokens=False))


class ArealOpenAI(AsyncOpenAI):
    # [CLASS DOC]
    # AsyncOpenAI subclass wiring a shared InteractionCache into custom
    # chat.completions and responses resources, plus reward/export helpers. Typical use:
    # an
    # agent workflow calls the client like OpenAI, then sets rewards and exports
    # interactions for training.
    """
    Extended AsyncOpenAI client that uses AReaL's inference engine
    and supports reward setting.
    """

    def __init__(
        self,
        engine: TRolloutEngine,
        tokenizer: "PreTrainedTokenizerFast",
        tool_call_parser: str = "qwen",
        reasoning_parser: str = "qwen3",
        engine_max_tokens: int | None = None,
        chat_template_type: str = "hf",
        lora_name: str = "",
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Passes kwargs to AsyncOpenAI, stores engine, tokenizer, parser names and LoRA
        # name,
        # creates an empty InteractionCache and replaces self.responses and
        # self.chat.completions with the reward-aware resources sharing that cache and
        # the
        # engine_max_tokens / chat_template_type settings.
        super().__init__(**kwargs)
        self.engine = engine
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.lora_name = lora_name

        # Use an ordered dict to maintain insertion order of completions/responses
        self._cache: InteractionCache = InteractionCache()

        # Override responses with our extended implementation
        self.responses = AsyncResponsesWithReward(
            self,
            engine,
            tokenizer,
            self._cache,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            engine_max_tokens=engine_max_tokens,
            chat_template_type=chat_template_type,
            lora_name=lora_name,
        )

        # Override chat.completions with our extended implementation
        self.chat.completions = AsyncCompletionsWithReward(
            self,
            engine,
            tokenizer,
            self._cache,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            engine_max_tokens=engine_max_tokens,
            chat_template_type=chat_template_type,
            lora_name=lora_name,
        )

    def get_interaction(self, id: str) -> InteractionWithTokenLogpReward | None:
        # [FUNCTION DOC]
        # Returns the cached interaction for id, or None.
        """Get completion/response with its reward from cache."""
        return self._cache.get(id)

    def set_reward(self, id: str, reward: float) -> None:
        # [FUNCTION DOC]
        # Sets the reward for interaction id; raises KeyError if not cached.
        """Set reward for a specific completion/response by its ID."""
        if id not in self._cache:
            raise KeyError(f"Interaction with ID {id} not found in cache")
        return self._cache.set_reward(id, reward)

    def set_last_reward(self, reward: float) -> None:
        # [FUNCTION DOC]
        # Sets the reward of the most recent interaction; RuntimeError if the cache is
        # empty.
        """Set reward for the most recent completion/response."""
        if not self._cache:
            raise RuntimeError("No interaction in cache to set reward for")
        return self._cache.set_last_reward(reward)

    def apply_reward_discount(self, turn_discount: float = 1.0) -> None:
        # [FUNCTION DOC]
        # Delegates to InteractionCache.apply_reward_discount (one-shot backward
        # geometric
        # discounting) and returns its result.
        # KNOWN ISSUE: annotated as returning None but returns the dict from the cache.
        """Apply backward discounted rewards across cached completions/responses.

        This method iterates over the cached completions/responses in reverse creation
        (insertion) order and applies a geometric discount to propagate reward
        signal backward in time. The most recent completion/response is treated as the
        starting point. If it does not have an explicit reward, a warning is
        logged and a default reward of ``0.0`` is used. For each earlier
        completion/response, its reward is initialized to ``0.0`` if unset, then the
        discounted reward from the next later completion/response is added:

        ``reward[i] += reward[i+1] * turn_discount``.

        Typically called before exporting completions/responses in 'individual' style
        to each completion/response is assigned with a valid reward value.

        Parameters
        ----------
        turn_discount : float, optional
            The per-turn discount factor applied when propagating reward
            backward from a later completion/response to an earlier one, by default 1.0.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A shallow copy of the completion/response cache after rewards have been
            updated in-place.
        """
        return self._cache.apply_reward_discount(turn_discount)

    def export_interactions(
        self, style: str
    ) -> dict[str, InteractionWithTokenLogpReward]:
        # [FUNCTION DOC]
        # Delegates to InteractionCache.export_interactions(style) ("concat" leaves or
        # "individual" all) without discount or orphan dropping.
        # KNOWN ISSUE: docstring says style defaults to 'concat', but it is required.
        """Export cached completions/responses in different formats.

        When ``style='concat'``, this method constructs a conversation tree by
        linking completions/responses whose input message lists form a strict-prefix
        relationship. The longest-prefix rule is used to determine each node's
        parent. It then returns only leaf-node completions/responses (those without
        children). No reward propagation is performed here.

        When ``style='individual'``, all cached completions/responses are returned as-is
        without constructing the tree.

        Parameters
        ----------
        style : str, optional
            The export style, either ``'concat'`` (build tree and return leaves)
            or ``'individual'`` (return all), by default 'concat'.

        Returns
        -------
        Dict[str, InteractionWithTokenLogpReward]
            A mapping from completion/response ID to completion/response objects. For
            ``'concat'``, this contains only leaf nodes. For ``'individual'``,
            this contains all cached completions/responses.

        Raises
        ------
        ValueError
            If an unsupported ``style`` is provided.
        """
        return self._cache.export_interactions(style)


def is_omitted(value) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if value is NOT_GIVEN, None, an openai Omit instance, or any object
    # whose
    # class name is "NotGiven" or "Omit"; else False.
    """Check if a value is NOT_GIVEN or Omit type or None."""
    if value is NOT_GIVEN or value is None:
        return True
    # Use isinstance for type safety and robustness
    # Check for common omitted types from OpenAI SDK
    try:
        from openai import Omit

        if isinstance(value, Omit):
            return True
    except ImportError:
        pass

    # Fallback for other omit types
    if hasattr(value, "__class__"):
        return value.__class__.__name__ in ("NotGiven", "Omit")
    return False
