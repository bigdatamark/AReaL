# [MODULE DOC]
# Calculator tool for the TIR example: evaluates arithmetic expressions written between
# calculator tags in model output.
import re
from typing import Any

from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType

logger = logging.getLogger("Calculator Tool")


class CalculatorTool(BaseTool):
    # [CLASS DOC]
    # BaseTool implementation of type CALCULATOR with calculator start/end markers.
    # Supports
    # only digits, + - * / parentheses, dots and whitespace. In debug_mode (from
    # BaseTool)
    # returns a fake output without evaluating.
    """Basic calculator tool"""

    @property
    def tool_type(self) -> ToolType:
        # [FUNCTION DOC]
        # Returns ToolType.CALCULATOR.
        return ToolType.CALCULATOR

    @property
    def description(self) -> ToolDescription:
        # [FUNCTION DOC]
        # Returns a ToolDescription named "python calculator" with an "expression"
        # parameter and
        # an example using calculator tags.
        return ToolDescription(
            name="python calculator",
            description="Perform basic mathematical calculations, supporting addition, subtraction, multiplication, division, and parentheses.",
            parameters={"expression": "Mathematical expression string"},
            parameter_prompt="Please provide a mathematical expression. Supports addition, subtraction, multiplication, division, and parentheses.",
            example="<calculator>1 + 2 * 3</calculator>",
        )

    @property
    def markers(self) -> ToolMarkers:
        # [FUNCTION DOC]
        # Returns ToolMarkers with start "<calculator>" and end "</calculator>".
        return ToolMarkers(
            start_markers=["<calculator>"], end_markers=["</calculator>"]
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Regex-searches (case-insensitive, dot matches newline) the first calculator-
        # tagged
        # span and returns its stripped content as {"expression": ...}; logs a debug
        # message.
        # If no tag is found, logs a warning and returns an empty expression.
        """Extract mathematical expression from <calculator> tags"""
        pattern = r"<calculator>(.*?)</calculator>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            expression = match.group(1).strip()
            logger.debug(f"Extracted expression: {expression}")
            return {"expression": expression}
        else:
            logger.warning("No <calculator> tag found")
            return {"expression": ""}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Evaluates parameters["expression"]. Returns (error text, ERROR) if empty; in
        # debug mode
        # returns ("dummy calculator output", SUCCESS). Otherwise rejects expressions
        # with
        # characters outside the safe arithmetic set, evaluates with Python eval and
        # returns
        # (str(result), SUCCESS); any exception (e.g. division by zero, syntax error) is
        # returned
        # as ("Error: <msg>", ERROR).
        # KNOWN ISSUE: eval on the whitelisted alphabet can still be expensive (e.g.
        # huge powers
        # via repeated "**"), so it is not fully safe against resource exhaustion.
        """Execute mathematical calculation"""
        expression = parameters.get("expression", "")
        if not expression:
            return "Error: No expression provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Executing calculator: {expression}")
            return "dummy calculator output", ToolCallStatus.SUCCESS

        try:
            # Simple mathematical expression calculation
            safe_pattern = r"^[0-9+\-*/().\s]+$"
            if not re.match(safe_pattern, expression):
                return "Error: Invalid expression", ToolCallStatus.ERROR

            # Use eval for calculation (in controlled environment)
            result = eval(expression)
            return str(result), ToolCallStatus.SUCCESS

        except Exception as e:
            return f"Error: {str(e)}", ToolCallStatus.ERROR
