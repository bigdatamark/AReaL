# [MODULE DOC]
# Configuration dataclass for the data proxy service.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class DataProxyConfig:
    # [CLASS DOC]
    # Settings: bind host/port, backend_addr and backend_type, tokenizer_path,
    # log_level, request_timeout per backend call, set_reward_finish_timeout,
    # max_resubmit_retries and resubmit_wait for abort/resubmit cycles, admin_api_key,
    # callback_server_addr, deterministic_sampling, serving_addr (resolved host:port
    # used as RTensor node address), and OpenAI-client options (tool_call_parser,
    # reasoning_parser, engine_max_tokens, chat_template_type, message_preprocessors,
    # prefix_matcher).
    host: str = "0.0.0.0"
    port: int = 8082
    backend_addr: str = "http://localhost:30000"  # co-located SGLang/vLLM
    backend_type: str = "sglang"
    tokenizer_path: str = ""
    log_level: str = "warning"
    request_timeout: float = 120.0  # seconds per SGLang call
    set_reward_finish_timeout: float = 0.0
    max_resubmit_retries: int = 20  # max abort/resubmit cycles before giving up
    resubmit_wait: float = 0.5  # seconds between is_paused polls
    admin_api_key: str = "areal-admin-key"  # admin key for authentication
    callback_server_addr: str = ""
    deterministic_sampling: bool = False
    # Resolved serving address (host:port) used as node_addr for RTensor shards.
    # Set at startup by __main__.py after the host is resolved.
    serving_addr: str = ""

    # ArealOpenAI client parameters (forwarded from AgentConfig)
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    engine_max_tokens: int | None = None
    chat_template_type: str = "hf"
    message_preprocessors: tuple[str, ...] = ()
    prefix_matcher: str | None = None
