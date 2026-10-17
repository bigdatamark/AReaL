# [PACKAGE DOC]
# Tools package for the TIR example. Re-exports the base tool types (BaseTool, ToolCall,
# ToolCallStatus, ToolDescription, ToolMarkers, ToolType), CalculatorTool, PythonTool
# and
# extract_python_code. If the optional daytona package is installed, also lazily imports
# and exports DaytonaPythonTool (appending it to __all__); on ImportError or if daytona
# is
# absent, any stale DaytonaPythonTool global is removed so that module reloads reflect
# the current environment.
# Export base types and classes
from __future__ import annotations

import importlib
import importlib.util

from .base import (
    BaseTool,
    ToolCall,
    ToolCallStatus,
    ToolDescription,
    ToolMarkers,
    ToolType,
)
from .calculator_tool import CalculatorTool
from .python_tool import PythonTool, extract_python_code

__all__ = [
    "ToolCallStatus",
    "ToolType",
    "ToolCall",
    "ToolDescription",
    "ToolMarkers",
    "BaseTool",
    "PythonTool",
    "extract_python_code",
    "CalculatorTool",
]

if importlib.util.find_spec("daytona") is not None:
    try:
        DaytonaPythonTool = importlib.import_module(
            ".daytona_python_tool", __name__
        ).DaytonaPythonTool
    except ImportError:
        globals().pop("DaytonaPythonTool", None)
    else:
        __all__.append("DaytonaPythonTool")
else:
    globals().pop("DaytonaPythonTool", None)
