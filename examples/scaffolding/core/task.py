# [MODULE DOC]
# Vendored scaffolding task and message types. A Task is a mutable dataclass that a
# controller yields and a worker fills in place (inputs, sampling params and result
# fields live on the same object). Also defines chat role messages and tool descriptions
# used by ChatTask.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.task

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .result import ScaffoldingOutput


@dataclass
class Task:
    # [CLASS DOC]
    # Base task dataclass: worker_tag (routes the task to a worker in ScaffoldingLlm),
    # streaming_output_flag / streaming_output_list (buffered streaming outputs
    # forwarded by the scheduler), custom_input_params / custom_output_params (free-form
    # extension dicts).
    # Scaffolding delivers the task to the Worker by worker_tag.
    worker_tag: str = field(default=None)

    # For streaming output.
    streaming_output_flag: bool = field(default=False)
    streaming_output_list: list[Any] = field(default_factory=list)

    # Reserve for custom input params.
    custom_input_params: dict | None = None

    # Reserve for custom output params.
    custom_output_params: dict | None = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "Task":
        # [FUNCTION DOC]
        # Factory hook to build a task from a prompt; base implementation is a stub
        # returning None.
        pass

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        # [FUNCTION DOC]
        # Hook returning the task's ScaffoldingOutput; base implementation returns None.
        pass

    def create_scaffolding_output_stream(self) -> list[ScaffoldingOutput]:
        # [FUNCTION DOC]
        # Hook returning streaming ScaffoldingOutputs; base implementation returns None.
        pass


class TaskStatus(Enum):
    # [CLASS DOC]
    # Enum of worker results for run_task: SUCCESS, WORKER_NOT_SUPPORTED,
    # WORKER_EXECEPTION (sic).
    SUCCESS = "success"
    WORKER_NOT_SUPPORTED = "worker_not_supported"
    WORKER_EXECEPTION = "worker_exception"


@dataclass
class GenerationTask(Task):
    # [CLASS DOC]
    # Text generation task. Inputs: input_tokens / input_str, skip_tokenizer /
    # skip_detokenizer flags; OpenAI-style sampling params (best_of, echo,
    # frequency_penalty, logit_bias, num_logprobs, max_tokens, n, presence_penalty,
    # seed, stop, suffix, temperature, top_p, user, ignore_eos) plus top_k and
    # return_context_logits. Results: output_str, output_tokens, finish_reason,
    # context_logits, logprobs, customized_result_fields, perf_metrics.
    # input field
    input_tokens: list[int] | None = None
    input_str: str | None = None
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False

    # sampling params for openai
    best_of: int | None = None
    echo: bool | None = False
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    num_logprobs: int | None = None
    max_tokens: int | None = None
    n: int = 1
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stop: str | list[str] | None = field(default_factory=list)
    suffix: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None
    ignore_eos: bool = False

    # sampling params
    top_k: int | None = None
    return_context_logits: bool | None = False

    # suggest to use Controller.WorkerTag
    worker_tag: str | None = None

    # result field
    output_str: str | None = None
    output_tokens: list[int] | None = None
    finish_reason: str | None = None
    context_logits: Any = None
    logprobs: Any = None
    customized_result_fields: dict[str, Any] = field(default_factory=dict)

    perf_metrics: dict[str, float] | None = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        # [FUNCTION DOC]
        # Returns a new GenerationTask with input_str = prompt and tokenizer/detokenizer
        # not skipped.
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        # [FUNCTION DOC]
        # Returns ScaffoldingOutput(output_str, output_tokens).
        return ScaffoldingOutput(self.output_str, self.output_tokens)


@dataclass
class StreamGenerationTask(GenerationTask):
    # [CLASS DOC]
    # GenerationTask variant for incremental streaming: cancel_flag, streaming_step
    # (tokens per step) inputs and request_handle / end_flag result fields.
    # input field
    cancel_flag: bool | None = field(default=False)
    streaming_step: int | None = field(default=1)

    # result field
    request_handle: Any = field(default=None)
    end_flag: bool = field(default=False)

    @staticmethod
    def create_from_generation_task(
        task: GenerationTask, streaming_step
    ) -> "StreamGenerationTask":
        # [FUNCTION DOC]
        # Builds a StreamGenerationTask by copying every instance attribute of task
        # (shallow) and setting streaming_step.
        stream_task = StreamGenerationTask()
        for k, v in task.__dict__.items():
            stream_task.__dict__[k] = v
        stream_task.streaming_step = streaming_step
        return stream_task


@dataclass
class RewardTask(Task):
    # [CLASS DOC]
    # Reward-scoring task with input_tokens / input_str inputs.
    # input field
    input_tokens: list[int] | None = field(default=None)
    input_str: str | None = field(default=None)


@dataclass
class RoleMessage:
    # [CLASS DOC]
    # Chat message with role, content and an optional prefix.
    role: str | None = field(default=None)
    content: str | None = field(default=None)
    prefix: str | None = field(default=None)

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns a JSON string with role and content.
        return json.dumps(
            {
                "role": self.role,
                "content": self.content,
            }
        )

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Returns "role: content" followed by a newline.
        return f"{self.role}: {self.content}\n"

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {"role", "content"} dict (prefix omitted).
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        # [FUNCTION DOC]
        # Builds an instance from a dict's "role" and "content" keys (KeyError if
        # missing); prefix is not restored.
        return cls(role=data["role"], content=data["content"])


@dataclass
class UserMessage(RoleMessage):
    # [CLASS DOC]
    # RoleMessage with role fixed to "user".
    def __init__(self, content: str, prefix: str | None = None):
        # [FUNCTION DOC]
        # Initializes with role "user", content and optional prefix.
        super().__init__(role="user", content=content, prefix=prefix)


@dataclass
class AssistantMessage(RoleMessage):
    # [CLASS DOC]
    # Assistant message with optional reasoning, reasoning_content and tool_calls.
    reasoning: str | None = field(default=None)
    reasoning_content: str | None = field(default=None)
    tool_calls: list[Any] | None = field(default=None)

    def __init__(
        self,
        content: str,
        reasoning: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes role "assistant" and the given content, reasoning fields and
        # tool_calls; prefix is left None.
        super().__init__(role="assistant", content=content)
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls

    def __str__(self) -> str:
        # [FUNCTION DOC]
        # Returns a JSON string with role, content, reasoning, reasoning_content and
        # tool_calls (each stringified) or null.
        # KNOWN ISSUE: Inherited to_dict omits reasoning and tool_calls, so they are
        # lost when messages are converted for an API request.
        return json.dumps(
            {
                "role": "assistant",
                "content": self.content,
                "reasoning": self.reasoning,
                "reasoning_content": self.reasoning_content,
                "tool_calls": [str(tool) for tool in self.tool_calls]
                if self.tool_calls is not None
                else None,
            }
        )


@dataclass
class SystemMessage(RoleMessage):
    # [CLASS DOC]
    # RoleMessage with role fixed to "system".
    def __init__(self, content: str, prefix: str | None = None):
        # [FUNCTION DOC]
        # Initializes with role "system", content and optional prefix.
        super().__init__(role="system", content=content, prefix=prefix)


class ToolDescription:
    # [CLASS DOC]
    # Base description of a callable tool: name, description, parameters (JSON-schema
    # properties dict).
    def __init__(self, name: str, description: str, parameters: dict[str, Any]):
        # [FUNCTION DOC]
        # Stores name, description and parameters.
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract hook to serialize the tool; base returns None.
        pass


class OpenAIToolDescription(ToolDescription):
    # [CLASS DOC]
    # Tool description serialized in OpenAI function-calling format.
    def to_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {"type": "function", "function": {name, description, parameters:
        # {"type": "object", "properties": parameters}}} (no "required" list).
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                },
            },
        }


@dataclass
class ChatTask(StreamGenerationTask):
    # [CLASS DOC]
    # Chat completion task: messages (list of RoleMessage), tools, token-counting flag
    # and counters (prompt/completion/reasoning tokens), sub_request_markers and
    # unique_id for tracing.
    messages: list[RoleMessage] = field(default_factory=list)
    tools: Any = field(default=None)

    # for token counting
    enable_token_counting: bool = field(default=False)
    prompt_tokens_num: int = field(default=0)
    completion_tokens_num: int = field(default=0)
    reasoning_tokens_num: int = field(default=0)

    # for sub request marker
    sub_request_markers: list[tuple[str, int]] = field(default_factory=list)
    unique_id: int | None = field(default=None)

    def messages_to_dict_content(self, start_index: int = 0) -> list[Mapping[str, str]]:
        # [FUNCTION DOC]
        # Returns to_dict() of messages from start_index on, skipping messages whose
        # content is None.
        ret = []
        for message in self.messages[start_index:]:
            if message.content is not None:
                ret.append(message.to_dict())
        return ret

    def add_message(self, message: RoleMessage):
        # [FUNCTION DOC]
        # Appends one message.
        self.messages.append(message)

    def add_messages(self, messages: list[RoleMessage]):
        # [FUNCTION DOC]
        # Appends several messages.
        self.messages.extend(messages)

    @staticmethod
    def create_from_prompt(
        user_prompt: str | None,
        system_prompts: list[SystemMessage] | None = None,
        tools: Any | None = None,
    ) -> "ChatTask":
        # [FUNCTION DOC]
        # Builds a ChatTask: extends with system_prompts if given, appends a UserMessage
        # if user_prompt is not None, sets tools.
        task = ChatTask()
        if system_prompts is not None:
            task.messages.extend(system_prompts)
        if user_prompt is not None:
            task.add_message(UserMessage(user_prompt))
        task.tools = tools
        return task

    @staticmethod
    def create_from_messages(
        messages: list[RoleMessage], tools: Any | None = None
    ) -> "ChatTask":
        # [FUNCTION DOC]
        # Builds a ChatTask that uses the given messages list (by reference, not copied)
        # and tools.
        task = ChatTask()
        task.messages = messages
        task.tools = tools
        return task
