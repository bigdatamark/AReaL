# [PACKAGE DOC]
# Agent subpackage of the Terminal-Bench example. Re-exports CamelTerminalAgent (agent
# driving a terminal task), ChatAgentTrace (CAMEL ChatAgent variant recording traces)
# and
# get_developer_agent_prompt (system prompt builder).
from .camel_terminal_agent import CamelTerminalAgent
from .chat_agent_trace import ChatAgentTrace
from .prompts import get_developer_agent_prompt

__all__ = [
    "CamelTerminalAgent",
    "ChatAgentTrace",
    "get_developer_agent_prompt",
]
