# [MODULE DOC]
# Async-only TIR Python tool executing code in a stateful, ephemeral Daytona cloud
# sandbox. The daytona SDK is optional and loaded lazily; module-level globals cache the
# SDK classes once loaded.
"""Daytona-backed Python tool for stateful TIR code execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from areal.infra.sandbox import DaytonaClientManager
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

if TYPE_CHECKING:
    from daytona import Image, Resources

logger = logging.getLogger("DaytonaPythonTool")

CreateSandboxFromImageParams = None
CreateSandboxFromSnapshotParams = None
DaytonaError = None
DaytonaNotFoundError = None


def _load_daytona_sdk() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Returns a dict of the Daytona SDK symbols (sandbox create params from
    # image and
    # from snapshot, DaytonaError, DaytonaNotFoundError), importing them on first call
    # and
    # caching them in module globals. Raises ImportError with an install hint if daytona
    # is
    # missing.
    global CreateSandboxFromImageParams
    global CreateSandboxFromSnapshotParams
    global DaytonaError
    global DaytonaNotFoundError

    if all(
        value is not None
        for value in (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            DaytonaError,
            DaytonaNotFoundError,
        )
    ):
        return {
            "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
            "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
            "DaytonaError": DaytonaError,
            "DaytonaNotFoundError": DaytonaNotFoundError,
        }

    try:
        from daytona import (
            CreateSandboxFromImageParams as ImportedCreateSandboxFromImageParams,
        )
        from daytona import (
            CreateSandboxFromSnapshotParams as ImportedCreateSandboxFromSnapshotParams,
        )
        from daytona import DaytonaError as ImportedDaytonaError
        from daytona import DaytonaNotFoundError as ImportedDaytonaNotFoundError
    except ImportError as exc:
        raise ImportError(
            "DaytonaPythonTool requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    CreateSandboxFromImageParams = ImportedCreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = ImportedCreateSandboxFromSnapshotParams
    DaytonaError = ImportedDaytonaError
    DaytonaNotFoundError = ImportedDaytonaNotFoundError
    return {
        "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
        "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
        "DaytonaError": DaytonaError,
        "DaytonaNotFoundError": DaytonaNotFoundError,
    }


class DaytonaPythonTool(BaseTool):
    # [CLASS DOC]
    # BaseTool of type DAYTONA_PYTHON using python fence or python tag markers. Lazily
    # creates one sandbox per tool instance (from a snapshot or an image, not both) and
    # reuses
    # it across calls so interpreter state persists. Sync execute is unsupported.
    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        *,
        snapshot: str | None = None,
        image: str | Image | None = None,
        resources: Resources | None = None,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        labels: dict[str, str] | None = None,
        create_timeout: float = 60.0,
    ):
        # [FUNCTION DOC]
        # Stores timeout/debug_mode via BaseTool plus sandbox options: snapshot, image,
        # resources, env_vars, network_block_all, labels and create_timeout (60 s).
        # Raises
        # ValueError if both image and snapshot are given, or if resources are given
        # without an
        # image. Initializes sandbox id None, an asyncio lock and SDK cache None.
        super().__init__(timeout, debug_mode)
        if image is not None and snapshot is not None:
            raise ValueError("Pass either snapshot or image, not both")
        if resources is not None and image is None:
            raise ValueError(
                "Daytona resources require image-based sandbox creation in SDK 0.167.0"
            )

        self.snapshot = snapshot
        self.image = image
        self.resources = resources
        self.env_vars = dict(env_vars or {})
        self.network_block_all = network_block_all
        self.labels = dict(labels or {})
        self.create_timeout = create_timeout
        self._sandbox_id: str | None = None
        self._sandbox_lock = asyncio.Lock()
        self._sdk: dict[str, Any] | None = None

    @property
    def tool_type(self) -> ToolType:
        # [FUNCTION DOC]
        # Returns ToolType.DAYTONA_PYTHON.
        return ToolType.DAYTONA_PYTHON

    @property
    def description(self) -> ToolDescription:
        # [FUNCTION DOC]
        # Returns a ToolDescription "daytona_python_executor" with a "code" parameter
        # and
        # fenced/tagged examples.
        return ToolDescription(
            name="daytona_python_executor",
            description="Execute Python code in a Daytona cloud sandbox with state preserved across calls.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute inside the Daytona sandbox.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>",
        )

    @property
    def markers(self) -> ToolMarkers:
        # [FUNCTION DOC]
        # Returns start markers python fence / python tag and the matching end markers.
        return ToolMarkers(
            start_markers=["```python", "<python>"],
            end_markers=["```", "</python>"],
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {"code": extract_python_code(text)}.
        return {"code": extract_python_code(text)}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Always raises NotImplementedError directing callers to aexecute.
        raise NotImplementedError("DaytonaPythonTool is async-only; use aexecute().")

    def _build_create_params(self):
        # [FUNCTION DOC]
        # HELPER: Builds the SDK create-params object: image-based (with resources) if
        # image is
        # set, else snapshot-based; language python, env vars and labels (None if
        # empty),
        # network_block_all and ephemeral=True. Asserts the SDK has been loaded.
        assert self._sdk is not None

        if self.image is not None:
            return self._sdk["CreateSandboxFromImageParams"](
                image=self.image,
                resources=self.resources,
                language="python",
                env_vars=self.env_vars or None,
                network_block_all=self.network_block_all,
                labels=self.labels or None,
                ephemeral=True,
            )

        return self._sdk["CreateSandboxFromSnapshotParams"](
            snapshot=self.snapshot,
            language="python",
            env_vars=self.env_vars or None,
            network_block_all=self.network_block_all,
            labels=self.labels or None,
            ephemeral=True,
        )

    async def _aget_sandbox(self):
        # [FUNCTION DOC]
        # HELPER: Loads the SDK, gets the shared async Daytona client and returns the
        # existing
        # sandbox by id, or, under the lock with a double check, creates one with
        # create_timeout and stores its id. Network I/O.
        self._sdk = _load_daytona_sdk()
        client = await DaytonaClientManager.get_client()

        if self._sandbox_id is not None:
            return await client.get(self._sandbox_id)

        async with self._sandbox_lock:
            if self._sandbox_id is not None:
                return await client.get(self._sandbox_id)

            sandbox = await client.create(
                self._build_create_params(),
                timeout=self.create_timeout,
            )
            self._sandbox_id = sandbox.id
            logger.debug("Created Daytona sandbox %s", self._sandbox_id)
            return sandbox

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Runs parameters["code"]. Empty code returns an error with ERROR; debug mode
        # returns a
        # dummy output with SUCCESS. Otherwise gets the sandbox and runs the code
        # through its code
        # interpreter with the tool timeout. DaytonaError is returned as (message,
        # ERROR); other
        # exceptions propagate. If the result has an error, returns ERROR with stdout,
        # stderr and
        # the traceback (prefixed with a "Traceback (most recent call last):" header if
        # missing,
        # or synthesized from the error name and value when the traceback is empty).
        # Otherwise
        # returns the right-stripped non-empty stdout and stderr joined by newlines with
        # SUCCESS.
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug("[FAKE] Executing Daytona Python code: %s", code[:100])
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            sandbox = await self._aget_sandbox()
            result = await sandbox.code_interpreter.run_code(code, timeout=self.timeout)
        except Exception as exc:
            daytona_error = _load_daytona_sdk()["DaytonaError"]
            if isinstance(exc, daytona_error):
                return str(exc), ToolCallStatus.ERROR
            raise

        if result.error is not None:
            error_lines = []
            if result.stdout.strip():
                error_lines.append(result.stdout.strip())
            if result.stderr.strip():
                error_lines.append(result.stderr.strip())
            if result.error.traceback.strip():
                traceback_text = result.error.traceback.strip()
                if not traceback_text.startswith("Traceback"):
                    error_lines.append("Traceback (most recent call last):")
                error_lines.append(traceback_text)
            else:
                error_lines.append("Traceback (most recent call last):")
                error_lines.append(f"{result.error.name}: {result.error.value}")

            return "\n".join(error_lines), ToolCallStatus.ERROR

        output = "\n".join(
            part.rstrip()
            for part in (result.stdout, result.stderr)
            if part and part.rstrip()
        )
        return output, ToolCallStatus.SUCCESS

    async def _aclose(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If a sandbox exists, clears the id and deletes the sandbox via the
        # client;
        # not-found errors are logged at debug, other errors as warnings.
        # KNOWN ISSUE: if loading the SDK fails, self._sdk may still be None and
        # evaluating the
        # except clause raises TypeError instead of logging.
        if self._sandbox_id is None:
            return

        sandbox_id = self._sandbox_id
        self._sandbox_id = None

        try:
            self._sdk = _load_daytona_sdk()
            client = await DaytonaClientManager.get_client()
            sandbox = await client.get(sandbox_id)
            await sandbox.delete(timeout=self.create_timeout)
        except self._sdk["DaytonaNotFoundError"]:
            logger.debug("Daytona sandbox already deleted")
        except Exception as exc:
            logger.warning("DaytonaPythonTool cleanup error: %s", exc)

    async def aclose(self) -> None:
        # [FUNCTION DOC]
        # Async cleanup: awaits _aclose.
        await self._aclose()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Synchronous cleanup. No-op without a sandbox. If called with no running event
        # loop:
        # schedules _aclose on the Daytona client manager's active loop and waits, or,
        # if none,
        # runs _aclose with asyncio.run and then closes the client manager. If called
        # inside a
        # running loop it only logs a warning (callers should use a thread) and does
        # nothing.
        if self._sandbox_id is None:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = DaytonaClientManager.active_loop()
            if loop is not None:
                asyncio.run_coroutine_threadsafe(self._aclose(), loop).result()
                return

            try:
                asyncio.run(self._aclose())
            finally:
                asyncio.run(DaytonaClientManager.close())
            return

        logger.warning(
            "DaytonaPythonTool.close() was called from an active event loop; use asyncio.to_thread(tool.close)."
        )
