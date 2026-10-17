# [PACKAGE DOC]
# OpenAI-compatible client layer for AReaL agentic RL. Exposes ArealOpenAI (an
# OpenAI SDK compatible client that generates through an AReaL inference engine and
# records token ids/logprobs/rewards per interaction), InteractionWithTokenLogpReward
# (the recorded interaction type), and the proxy-based OpenAIProxyClient and
# OpenAIProxyWorkflow for running unmodified agents against an HTTP proxy.
# SPDX-License-Identifier: Apache-2.0

from .client import ArealOpenAI  # noqa
from .types import InteractionWithTokenLogpReward  # noqa
from .proxy import (
    OpenAIProxyClient,
    OpenAIProxyWorkflow,
)  # noqa

__all__ = [
    "ArealOpenAI",
    "InteractionWithTokenLogpReward",
    "OpenAIProxyClient",
    "OpenAIProxyWorkflow",
]
