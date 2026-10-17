# [MODULE DOC]
# Core abstractions for TIR tools: status/type enums, tool call/description/marker data
# structures and the BaseTool abstract base class.
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from areal.utils import logging

logger = logging.getLogger("Tool Base")


class ToolCallStatus(Enum):
    # [CLASS DOC]
    # Enum of tool execution outcomes: SUCCESS, ERROR, NOT_FOUND (string values).
    """Tool call status enumeration"""

    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ToolType(Enum):
    # [CLASS DOC]
    # Enum of supported tools: PYTHON (local subprocess), CALCULATOR, DAYTONA_PYTHON
    # (remote Daytona sandbox).
    """Tool type enumeration"""

    PYTHON = "python"
    CALCULATOR = "calculator"
    DAYTONA_PYTHON = "daytona_python"


@dataclass
class ToolCall:
    # [CLASS DOC]
    # Dataclass of a parsed tool invocation: tool_type, parameters dict and raw_text
    # that
    # contained the call.
    """Tool call data structure"""

    tool_type: ToolType
    parameters: dict[str, Any]
    raw_text: str


@dataclass
class ToolDescription:
    # [CLASS DOC]
    # Dataclass describing a tool for prompts: name, description, parameters (name to
    # description), parameter_prompt and an example invocation.
    """Tool description data structure"""

    name: str
    description: str
    parameters: dict[str, str]  # Parameter name -> parameter description
    parameter_prompt: str  # Prompt for parameter parsing
    example: str


@dataclass
class ToolMarkers:
    # [CLASS DOC]
    # Dataclass of the start and end marker strings delimiting a tool call in generated
    # text.
    """Tool markers data structure"""

    start_markers: list[str]  # Start markers for the tool
    end_markers: list[str]  # End markers for the tool


class BaseTool(ABC):
    # [CLASS DOC]
    # Abstract tool. Subclasses define tool_type, description, markers, parse_parameters
    # and
    # execute; aexecute/aclose/close have default implementations. Holds timeout
    # (seconds,
    # default 30) and debug_mode (if True tools return fake output).
    """Base tool abstract class"""

    def __init__(self, timeout: int = 30, debug_mode: bool = False):
        # [FUNCTION DOC]
        # Stores timeout and debug_mode attributes.
        self.timeout = timeout
        self.debug_mode = debug_mode

    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        # [FUNCTION DOC]
        # Abstract property: the ToolType of this tool.
        """Tool type"""

    @property
    @abstractmethod
    def description(self) -> ToolDescription:
        # [FUNCTION DOC]
        # Abstract property: the ToolDescription used in prompts.
        """Tool description"""

    @property
    @abstractmethod
    def markers(self) -> ToolMarkers:
        # [FUNCTION DOC]
        # Abstract property: the ToolMarkers used to detect calls in generated text.
        """Tool markers for parsing"""

    @abstractmethod
    def parse_parameters(self, text: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract: parse the raw tool-call text into a parameters dict.
        """Parse parameters"""

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Abstract: run the tool synchronously; returns (result text, ToolCallStatus).
        """Execute tool.

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Default async execution: runs the synchronous execute in a worker thread via
        # asyncio.to_thread and returns its result. Native async tools override this.
        """Async version of execute.

        Override directly for tools that are natively async. The default keeps
        existing synchronous tools working unchanged.
        """

        return await asyncio.to_thread(self.execute, parameters)

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Default async cleanup hook; does nothing.
        """Async cleanup hook."""

        return None

    def close(self) -> None:
        # [FUNCTION DOC]
        # Default synchronous cleanup hook; does nothing.
        """Release any tool resources.

        Most tools are stateless and do not need cleanup.
        """

        return None
