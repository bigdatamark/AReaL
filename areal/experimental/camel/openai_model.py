# [MODULE DOC]
# Experimental CAMEL-AI integration: a CAMEL model backend that sends chat completions
# through an injected AsyncOpenAI-compatible client (typically AReaL's OpenAI-compatible
# proxy client that records completions for RL), and a token counter backed by a
# HuggingFace tokenizer. At import, picks the tracing decorator "observe" from langfuse
# (LANGFUSE_ENABLED=true), traceroot (TRACEROOT_ENABLED=true), falling back to
# camel.utils.observe. Uses CAMEL's get_logger.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any

from camel.logger import get_logger
from camel.messages import OpenAIMessage
from camel.models.base_model import BaseModelBackend
from camel.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ModelType,
)
from camel.utils import BaseTokenCounter, OpenAITokenCounter
from openai import AsyncOpenAI, AsyncStream, Stream
from openai.lib.streaming.chat import (
    AsyncChatCompletionStreamManager,
    ChatCompletionStreamManager,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


if os.environ.get("LANGFUSE_ENABLED", "False").lower() == "true":
    try:
        from langfuse.decorators import observe
    except ImportError:
        from camel.utils import observe
elif os.environ.get("TRACEROOT_ENABLED", "False").lower() == "true":
    try:
        from traceroot import trace as observe  # type: ignore[import]
    except ImportError:
        from camel.utils import observe
else:
    from camel.utils import observe


logger = get_logger(__name__)


class AReaLTokenCounter(BaseTokenCounter):
    # [CLASS DOC]
    # CAMEL BaseTokenCounter using a HuggingFace tokenizer. State: tokenizer,
    # tokens_per_message (fixed per-message overhead).
    def __init__(self, tokenizer: "PreTrainedTokenizerFast", tokens_per_message: int):
        # [FUNCTION DOC]
        # Store the tokenizer and per-message token overhead.
        self.tokenizer = tokenizer
        self.tokens_per_message = tokens_per_message

    def count_tokens_from_messages(self, messages: list[OpenAIMessage]) -> int:
        # [FUNCTION DOC]
        # Approximate prompt token count for OpenAI-format messages: for each message
        # add tokens_per_message plus the encoded length of every field value
        # (stringified); list values must contain only {"type": "text"} items whose text
        # is encoded, other item types raise ValueError. Adds 3 for the assistant reply
        # priming. Returns int.
        num_tokens = 0
        for message in messages:
            num_tokens += self.tokens_per_message
            for key, value in message.items():
                if not isinstance(value, list):
                    num_tokens += len(self.tokenizer.encode(str(value)))
                else:
                    for item in value:
                        if item["type"] == "text":
                            num_tokens += len(self.tokenizer.encode(str(item["text"])))
                        else:
                            raise ValueError(f"Unsupported item type: {item['type']}")
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def encode(self, text: str) -> list[int]:
        # [FUNCTION DOC]
        # Encode text to token ids with the tokenizer.
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        # [FUNCTION DOC]
        # Decode token ids to text with the tokenizer.
        return self.tokenizer.decode(token_ids)


class AReaLOpenAICompatibleModel(BaseModelBackend):
    # [CLASS DOC]
    # CAMEL BaseModelBackend whose async run delegates to a caller-supplied AsyncOpenAI
    # client; synchronous run is unsupported and streaming is always off. State:
    # tokenizer, _client, plus BaseModelBackend fields (model_type, model_config_dict,
    # api key, url, timeout, retries, token counter).
    # KNOWN ISSUE: the docstring omits the required openai_client and tokenizer
    # arguments, and extra **kwargs are accepted but ignored.
    r"""Constructor for model backend supporting AReaL OpenAI compatibility.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into:obj:`openai.ChatCompletion.create()`. If
            :obj:`None`, :obj:`{}` will be used. (default: :obj:`None`)
        api_key (str): The API key for authenticating with the model service.
        url (str): The url to the model service.
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used.
            (default: :obj:`None`)
        timeout (Optional[float], optional): The timeout value in seconds for
            API calls. If not provided, will fall back to the MODEL_TIMEOUT
            environment variable or default to 180 seconds.
            (default: :obj:`None`)
        max_retries (int, optional): Maximum number of retries for API calls.
            (default: :obj:`3`)
        **kwargs (Any): Additional arguments to pass to the
            OpenAI client initialization. These can include parameters like
            'organization', 'default_headers', 'http_client', etc.
    """

    def __init__(
        self,
        model_type: ModelType | str,
        openai_client: AsyncOpenAI,
        tokenizer: "PreTrainedTokenizerFast",
        model_config_dict: dict[str, Any] | None = None,
        api_key: str | None = None,
        url: str | None = None,
        token_counter: BaseTokenCounter | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        # [FUNCTION DOC]
        # Resolve api_key from OPENAI_COMPATIBILITY_API_KEY and url from
        # OPENAI_COMPATIBILITY_API_BASE_URL when not given, timeout from MODEL_TIMEOUT
        # (default 180 s), initialize BaseModelBackend with these, then store the
        # tokenizer and the injected client.
        api_key = api_key or os.environ.get("OPENAI_COMPATIBILITY_API_KEY")
        url = url or os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL")
        if timeout is None:
            timeout = float(os.environ.get("MODEL_TIMEOUT", 180))
        super().__init__(
            model_type,
            model_config_dict,
            api_key,
            url,
            token_counter,
            timeout,
            max_retries,
        )
        self.tokenizer = tokenizer
        self._client = openai_client

    @observe()
    def _run(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | Stream[ChatCompletionChunk]
        | ChatCompletionStreamManager[BaseModel]
    ):
        # [FUNCTION DOC]
        # Synchronous inference is not supported; always raises NotImplementedError.
        # Wrapped by the tracing decorator.
        r"""Runs inference of OpenAI chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, Stream[ChatCompletionChunk]]:
                `ChatCompletion` in the non-stream mode, or
                `Stream[ChatCompletionChunk]` in the stream mode.
                `ChatCompletionStreamManager[BaseModel]` for
                structured output streaming.
        """

        raise NotImplementedError("Not implemented")

    @observe()
    async def _arun(
        self,
        messages: list[OpenAIMessage],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> (
        ChatCompletion
        | AsyncStream[ChatCompletionChunk]
        | AsyncChatCompletionStreamManager[BaseModel]
    ):
        # [FUNCTION DOC]
        # Async: call self._client.chat.completions.create with the messages and a copy
        # of model_config_dict (plus tools when given) and return the result. Wrapped by
        # the tracing decorator.
        # KNOWN ISSUE: response_format is accepted but ignored, and the model name is
        # not passed unless present in model_config_dict.
        r"""Runs inference of OpenAI chat completion in async mode.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.
            response_format (Optional[Type[BaseModel]]): The format of the
                response.
            tools (Optional[List[Dict[str, Any]]]): The schema of the tools to
                use for the request.

        Returns:
            Union[ChatCompletion, AsyncStream[ChatCompletionChunk],
                AsyncChatCompletionStreamManager[BaseModel]]:
                `ChatCompletion` in the non-stream mode,
                `AsyncStream[ChatCompletionChunk]` in the stream mode,
                or `AsyncChatCompletionStreamManager[BaseModel]` for
                structured output streaming.
        """
        request_config = self.model_config_dict.copy()
        if tools:
            request_config["tools"] = tools
        result = await self._client.chat.completions.create(
            messages=messages,
            **request_config,
        )

        return result

    @property
    def token_counter(self) -> BaseTokenCounter:
        # [FUNCTION DOC]
        # Return the token counter, lazily defaulting to CAMEL's OpenAITokenCounter for
        # GPT-4o-mini (the tokenizer-based AReaLTokenCounter is not used automatically).
        r"""Initialize the token counter for the model backend.

        Returns:
            OpenAITokenCounter: The token counter following the model's
                tokenization style.
        """

        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    @property
    def stream(self) -> bool:
        # [FUNCTION DOC]
        # Always False: streaming is not supported.
        r"""Returns whether the model is in stream mode, which sends partial
        results each time.

        Returns:
            bool: Whether the model is in stream mode.
        """
        return False
