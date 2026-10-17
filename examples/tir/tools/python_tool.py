# [MODULE DOC]
# Local (unsandboxed) Python execution tool for TIR, built on a trimmed copy of
# qwen-agent's PythonExecutor, plus a helper to extract code blocks from model output.
import io
import os
import pickle
import re
import traceback
from contextlib import redirect_stdout
from functools import partial
from typing import Any

import json5
from qwen_agent.tools.base import BaseTool as QwenAgentBaseTool
from qwen_agent.tools.python_executor import (
    GenericRuntime,
    _check_deps_for_python_executor,
)
from qwen_agent.utils.utils import extract_code
from tqdm import tqdm

from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType

logger = logging.getLogger("Python Tool")


def extract_python_code(text: str) -> str:
    # [FUNCTION DOC]
    # Returns the stripped body of the last python-fenced block (fence, newline, code,
    # newline, closing fence; case-insensitive), else of the last python-tagged block,
    # else
    # "" with a warning. Logs the first 100 chars at debug.
    """Extract Python code from text, supporting two formats:
    1. ```python\n...\n```
    2. <python>...</python>

    Args:
        text: Text containing Python code

    Returns:
        Extracted Python code, returns empty string if not found
    """
    # Try to match ```python``` format, match only the last occurrence from back to front
    pattern1 = r"```python\n(.*?)\n```"
    matches1 = list(re.finditer(pattern1, text, re.DOTALL | re.IGNORECASE))
    if matches1:
        last_match = matches1[-1]
        code = last_match.group(1).strip()
        logger.debug(
            f"Extracted Python code from ```python``` format (last occurrence): {code[:100]}..."
        )
        return code

    # Try to match <python></python> format, match only the last occurrence from back to front
    pattern2 = r"<python>(.*?)</python>"
    matches2 = list(re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE))
    if matches2:
        last_match = matches2[-1]
        code = last_match.group(1).strip()
        logger.debug(
            f"Extracted Python code from <python> format (last occurrence): {code[:100]}..."
        )
        return code

    logger.warning("No Python code block found in either format")
    return ""


# Copied from https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/tools/python_executor.py
# Removed unused `self.pool=Pool(multiprocess.cpu_count())` to avoid hang when exiting experiments.
class PythonExecutor(QwenAgentBaseTool):
    # [CLASS DOC]
    # Copy of qwen-agent's python_executor tool without its process pool created at init
    # (to
    # avoid hangs at exit). Executes code via a GenericRuntime in a pebble ProcessPool
    # with a
    # per-snippet timeout. Not sandboxed. cfg keys: runtime, get_answer_symbol,
    # get_answer_expr, get_answer_from_stdout (default True), timeout_length (20 s).
    name = "python_executor"
    description = "For executing python code. Not sandboxed. Do not use it for production purposes."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "description": "The python code.",
                "type": "string",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: dict | None = None):
        # [FUNCTION DOC]
        # Checks optional executor dependencies (raises if missing), initializes the
        # qwen-agent
        # base with cfg, and stores runtime (new GenericRuntime if not provided) and
        # answer
        # extraction settings and timeout.
        _check_deps_for_python_executor()
        super().__init__(cfg)

        runtime: Any | None = self.cfg.get("runtime", None)
        get_answer_symbol: str | None = self.cfg.get("get_answer_symbol", None)
        get_answer_expr: str | None = self.cfg.get("get_answer_expr", None)
        get_answer_from_stdout: bool = self.cfg.get("get_answer_from_stdout", True)
        timeout_length: int = self.cfg.get("timeout_length", 20)

        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length

    def call(self, params: str | dict, **kwargs) -> list:
        # [FUNCTION DOC]
        # qwen-agent tool entry: takes code from a dict, or parses a string as JSON5
        # (falling
        # back to extracting a code block from the raw string); other types give empty
        # code.
        # Returns ["", ""] for blank code, else the (result, report) pair from apply.
        # KNOWN ISSUE: returns a list for blank code but a tuple otherwise.
        if isinstance(params, dict):
            code = params.get("code", "")
        elif isinstance(params, str):
            try:
                params = json5.loads(params)
                code = params["code"]
            except Exception:
                code = extract_code(params)
        else:
            code = ""

        if not code.strip():
            return ["", ""]

        predictions = self.apply(code)
        return predictions

    def apply(self, code: str) -> list:
        # [FUNCTION DOC]
        # Runs a single code string through batch_apply and returns its (result,
        # report).
        return self.batch_apply([code])[0]

    def process_generation_to_code(self, gens: str):
        # [FUNCTION DOC]
        # HELPER: Splits each code string into a list of lines.
        return [g.split("\n") for g in gens]

    @staticmethod
    def execute(
        code,
        get_answer_from_stdout=None,
        runtime=None,
        answer_symbol=None,
        answer_expr=None,
        timeout_length=20,
    ):
        # [FUNCTION DOC]
        # HELPER: Static worker function executed in a pool process. Given code as a
        # list of
        # lines and the runtime, either captures stdout of executing the code (default),
        # returns
        # the runtime global named answer_symbol, evaluates answer_expr after execution,
        # or
        # executes all but the last line and evaluates the last line. Each step is
        # wrapped in a
        # timeout_decorator timeout. Verifies the result is stringifiable and picklable.
        # Returns
        # (result, "Done") or ("", last traceback line) on any exception.
        from timeout_decorator import timeout

        try:
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                timeout(timeout_length)(runtime.exec_code)("\n".join(code))
                result = timeout(timeout_length)(runtime.eval_code)(answer_expr)
            else:
                timeout(timeout_length)(runtime.exec_code)("\n".join(code[:-1]))
                result = timeout(timeout_length)(runtime.eval_code)(code[-1])
            report = "Done"
            str(result)
            pickle.dumps(result)  # serialization check
        except Exception:
            result = ""
            report = traceback.format_exc().split("\n")[-2]
        return result, report

    @staticmethod
    def truncate(s, max_length=256):
        # [FUNCTION DOC]
        # HELPER: If s is longer than max_length (256), keeps the first and last half
        # lengths
        # joined by "..."; returns s.
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    def batch_apply(self, batch_code: list[str]) -> list:
        # [FUNCTION DOC]
        # Executes several code strings in a pebble ProcessPool (workers = min(count,
        # CPU count))
        # with the timeout per task, iterating results in order; timeouts produce
        # ("", "Timeout Error") and other errors ("", "Internal Error"), logged at info.
        # Shows a
        # tqdm bar for more than 100 snippets. Returns a list of (result, report) string
        # pairs,
        # stripped and truncated to 256 chars. Spawns processes.
        from pebble import ProcessPool

        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []
        with ProcessPool(
            max_workers=min(len(all_code_snippets), os.cpu_count())
        ) as pool:
            executor = partial(
                self.execute,
                get_answer_from_stdout=self.get_answer_from_stdout,
                runtime=self.runtime,
                answer_symbol=self.answer_symbol,
                answer_expr=self.answer_expr,
                timeout_length=self.timeout_length,  # this timeout not work
            )
            future = pool.map(executor, all_code_snippets, timeout=self.timeout_length)
            iterator = future.result()

            if len(all_code_snippets) > 100:
                progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
            else:
                progress_bar = None

            while True:
                try:
                    result = next(iterator)
                    all_exec_results.append(result)
                except StopIteration:
                    break
                except TimeoutError as e:
                    logger.info(f"In PythonExecutor: {e}")
                    all_exec_results.append(("", "Timeout Error"))
                    timeout_cnt += 1
                except Exception as e:
                    logger.info(f"In PythonExecutor: {e}\n{traceback.format_exc()}")
                    all_exec_results.append(("", "Internal Error"))
                if progress_bar is not None:
                    progress_bar.update(1)

            if progress_bar is not None:
                progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results


class PythonTool(BaseTool):
    # [CLASS DOC]
    # TIR BaseTool of type PYTHON using python fence / python tag markers, executing
    # code
    # locally via PythonExecutor. In debug_mode returns a fake output.
    """Qwen Python code execution tool"""

    def __init__(self, timeout: int = 30, debug_mode: bool = False):
        # [FUNCTION DOC]
        # Initializes BaseTool fields and creates a PythonExecutor with default config.
        # KNOWN ISSUE: the timeout argument is stored but not passed to the executor,
        # which uses
        # its own 20 s default.
        super().__init__(timeout, debug_mode)
        self.python_executor = PythonExecutor()

    @property
    def tool_type(self) -> ToolType:
        # [FUNCTION DOC]
        # Returns ToolType.PYTHON.
        return ToolType.PYTHON

    @property
    def description(self) -> ToolDescription:
        # [FUNCTION DOC]
        # Returns a ToolDescription "python_executor" with a "code" parameter and
        # examples.
        return ToolDescription(
            name="python_executor",
            description="Execute Python code. Supports variable calculation, data processing, algorithm implementation, etc.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute. Supports variable calculation, data processing, algorithm implementation, etc.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>",
        )

    @property
    def markers(self) -> ToolMarkers:
        # [FUNCTION DOC]
        # Returns start markers python fence / python tag and matching end markers.
        return ToolMarkers(
            start_markers=["```python", "<python>"], end_markers=["```", "</python>"]
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {"code": extract_python_code(text)}.
        """Extract Python code from text, supporting two formats: ```python``` and <python>"""
        code = extract_python_code(text)
        return {"code": code}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Runs parameters["code"]: empty gives ("Error: No code provided", ERROR); debug
        # mode gives
        # ("dummy python output", SUCCESS). Otherwise calls the executor's apply; a
        # report other
        # than "Done" is logged and returned as ("Error: <report>", ERROR); success
        # returns
        # (str(result), SUCCESS). Exceptions are logged and returned as ERROR.
        # KNOWN ISSUE: the inline comment says this avoids the ProcessPool, but apply
        # delegates to
        # batch_apply, which always uses a ProcessPool.
        """Execute Python code"""
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Executing Python code: {code[:100]}...")
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            # Directly call apply to avoid using ProcessPool in async environment
            res, report = self.python_executor.apply(code)

            if report != "Done":
                logger.error(f"Error in Python execution: {report}")
                return f"Error: {report}", ToolCallStatus.ERROR

            logger.debug(f"Python execution completed: {str(res)[:100]}...")
            return str(res), ToolCallStatus.SUCCESS
        except Exception as e:
            logger.error(f"Python execution error: {e}")
            return f"Error: {str(e)}", ToolCallStatus.ERROR
