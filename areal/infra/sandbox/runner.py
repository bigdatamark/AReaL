# [MODULE DOC]
# Synchronous wrapper around the async Daytona SDK for running code in a single remote
# sandbox from non-async code. Runs a private event loop on a daemon thread and submits
# coroutines to it; lazily imports SDK types into module globals.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous Daytona sandbox runner for non-async user code."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from areal.utils import logging

from ._client import DaytonaClientManager
from ._client import _load_daytona_sdk as _load_daytona_client_sdk

if TYPE_CHECKING:
    from daytona import Chart, Image, Resources

logger = logging.getLogger("DaytonaRunner")

CreateSandboxFromImageParams = None
CreateSandboxFromSnapshotParams = None
DaytonaError = None
DaytonaNotFoundError = None
DaytonaTimeoutError = None


def _load_daytona_sdk() -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Lazily imports the Daytona sandbox-creation param classes and error
    # classes, caching them in module globals. Returns a dict mapping their names to the
    # classes. Raises ImportError with install instructions if daytona is missing.
    global CreateSandboxFromImageParams
    global CreateSandboxFromSnapshotParams
    global DaytonaError
    global DaytonaNotFoundError
    global DaytonaTimeoutError

    if all(
        value is not None
        for value in (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            DaytonaError,
            DaytonaNotFoundError,
            DaytonaTimeoutError,
        )
    ):
        return {
            "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
            "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
            "DaytonaError": DaytonaError,
            "DaytonaNotFoundError": DaytonaNotFoundError,
            "DaytonaTimeoutError": DaytonaTimeoutError,
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
        from daytona import DaytonaTimeoutError as ImportedDaytonaTimeoutError
    except ImportError as exc:
        raise ImportError(
            "DaytonaRunner requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    CreateSandboxFromImageParams = ImportedCreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = ImportedCreateSandboxFromSnapshotParams
    DaytonaError = ImportedDaytonaError
    DaytonaNotFoundError = ImportedDaytonaNotFoundError
    DaytonaTimeoutError = ImportedDaytonaTimeoutError

    return {
        "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
        "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
        "DaytonaError": DaytonaError,
        "DaytonaNotFoundError": DaytonaNotFoundError,
        "DaytonaTimeoutError": DaytonaTimeoutError,
    }


def _last_non_empty_line(text: str) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the last non-blank line of text (stripped), or None if all blank.
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@dataclass
class DaytonaRunResult:
    # [CLASS DOC]
    # Dataclass result of a code run: stdout, stderr, exit_code (0 ok, 124 timeout, 1
    # SDK error, else the process code), charts produced by the run, and error (short
    # message, None on success).
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    charts: list[Chart] = field(default_factory=list)
    error: str | None = None


class DaytonaRunner:
    # [CLASS DOC]
    # Owns one Daytona sandbox (created from either a snapshot or an image) plus its own
    # AsyncDaytona client and background event loop thread. Usable as a context manager;
    # start creates the sandbox, run executes code, close deletes it and stops the loop.
    def __init__(
        self,
        *,
        snapshot: str | None = None,
        image: str | Image | None = None,
        resources: Resources | None = None,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        labels: dict[str, str] | None = None,
        create_timeout: float = 60.0,
        default_timeout: int = 30,
        language: str = "python",
        ephemeral: bool = True,
    ):
        # [FUNCTION DOC]
        # Loads SDK types, validates arguments (ValueError if both snapshot and image
        # are given, or if resources are given without image), and stores sandbox
        # settings: env_vars and labels (copied), network_block_all, create_timeout
        # (seconds for create/delete), default_timeout (seconds per code run), language,
        # ephemeral. Initializes loop/thread, client, sandbox id and locks to empty.
        # Does no network I/O.
        self._sdk = _load_daytona_sdk()
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
        self.default_timeout = default_timeout
        self.language = language
        self.ephemeral = ephemeral
        self._sandbox_id: str | None = None
        self._client = None
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._sandbox_lock: asyncio.Lock | None = None

    def _build_create_params(self):
        # [FUNCTION DOC]
        # HELPER: Builds image-based creation params (with resources) if image is set,
        # else snapshot-based params; empty env_vars/labels are passed as None.
        if self.image is not None:
            return self._sdk["CreateSandboxFromImageParams"](
                image=self.image,
                resources=self.resources,
                language=self.language,
                env_vars=self.env_vars or None,
                network_block_all=self.network_block_all,
                labels=self.labels or None,
                ephemeral=self.ephemeral,
            )

        return self._sdk["CreateSandboxFromSnapshotParams"](
            snapshot=self.snapshot,
            language=self.language,
            env_vars=self.env_vars or None,
            network_block_all=self.network_block_all,
            labels=self.labels or None,
            ephemeral=self.ephemeral,
        )

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        # [FUNCTION DOC]
        # HELPER: Raises RuntimeError if closed. Under a thread lock, returns the
        # existing loop or creates a new event loop running forever on a daemon thread
        # named DaytonaRunnerLoop (which on stop shuts down async generators and closes
        # the loop), waiting until it started.
        if self._closed:
            raise RuntimeError("DaytonaRunner is closed")

        with self._loop_lock:
            if self._loop is not None:
                return self._loop

            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_loop() -> None:
                # [FUNCTION DOC]
                # HELPER: Thread target: sets the loop as current, signals readiness,
                # runs it until stopped, then shuts down async generators and closes it.
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

            self._loop = loop
            self._thread = threading.Thread(
                target=run_loop,
                name="DaytonaRunnerLoop",
                daemon=True,
            )
            self._thread.start()
            ready.wait()
            return loop

    def _run(self, coro):
        # [FUNCTION DOC]
        # HELPER: Submits coro to the background loop and blocks for its result. If the
        # loop cannot be obtained the coroutine is closed and the error re-raised; if
        # waiting raises (including KeyboardInterrupt), cancels the future and re-
        # raises.
        try:
            loop = self._ensure_loop()
        except BaseException:
            coro.close()
            raise

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    async def _aget_client(self):
        # [FUNCTION DOC]
        # HELPER: Async. Lazily creates this runner's own AsyncDaytona client using
        # DaytonaClientManager's config overrides and connection_pool_maxsize=None;
        # returns it.
        if self._client is None:
            async_daytona_cls, daytona_config_cls = _load_daytona_client_sdk()
            config = daytona_config_cls(
                connection_pool_maxsize=None,
                **DaytonaClientManager.config_overrides(),
            )
            self._client = async_daytona_cls(config)
            logger.debug("Initialized DaytonaRunner AsyncDaytona client")
        return self._client

    async def _acreate(self):
        # [FUNCTION DOC]
        # HELPER: Async. Creates a sandbox with the built params and create_timeout,
        # records its id, returns the sandbox.
        client = await self._aget_client()
        sandbox = await client.create(
            self._build_create_params(),
            timeout=self.create_timeout,
        )
        self._sandbox_id = sandbox.id
        logger.debug("Created Daytona sandbox %s", self._sandbox_id)
        return sandbox

    async def _aget_sandbox(self):
        # [FUNCTION DOC]
        # HELPER: Async. If no sandbox exists, creates one under a lazily created
        # asyncio lock (double-checked) and returns it; otherwise fetches the sandbox by
        # id from the client.
        if self._sandbox_id is None:
            if self._sandbox_lock is None:
                self._sandbox_lock = asyncio.Lock()
            async with self._sandbox_lock:
                if self._sandbox_id is None:
                    return await self._acreate()

        client = await self._aget_client()
        return await client.get(self._sandbox_id)

    def start(self) -> DaytonaRunner:
        # [FUNCTION DOC]
        # Synchronously ensures the sandbox exists and returns self. On any failure
        # closes the runner and re-raises.
        try:
            self._run(self._aget_sandbox())
            return self
        except BaseException:
            self.close()
            raise

    def run(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> DaytonaRunResult:
        # [FUNCTION DOC]
        # Synchronously executes code in the sandbox via _arun and returns a
        # DaytonaRunResult. timeout defaults to default_timeout; on_stdout/on_stderr are
        # optional callbacks invoked with the whole output text (from the loop thread).
        # Creates the sandbox if needed.
        return self._run(
            self._arun(code, timeout=timeout, on_stdout=on_stdout, on_stderr=on_stderr)
        )

    async def _arun(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> DaytonaRunResult:
        # [FUNCTION DOC]
        # HELPER: Async implementation of run. Gets the sandbox and runs code. Daytona
        # timeout errors yield exit_code 124; other Daytona errors yield exit_code 1;
        # other exceptions with HTTP status 408 or "timeout" in the message yield 124
        # (error = last line), and anything else is re-raised. For error cases the
        # message is sent to on_stderr. On exit_code 0, output goes to stdout (and
        # on_stdout); otherwise output goes to stderr (and on_stderr) with error set to
        # its last non-empty line or a generic exit-code message. Charts come from the
        # response artifacts.
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            sandbox = await self._aget_sandbox()
            response = await sandbox.process.code_run(code, timeout=timeout)
        except self._sdk["DaytonaTimeoutError"] as exc:
            error = str(exc)
            if on_stderr is not None:
                on_stderr(error)
            return DaytonaRunResult(stderr=error, exit_code=124, charts=[], error=error)
        except self._sdk["DaytonaError"] as exc:
            error = str(exc)
            if on_stderr is not None:
                on_stderr(error)
            return DaytonaRunResult(stderr=error, exit_code=1, charts=[], error=error)
        except Exception as exc:
            status_code = getattr(exc, "status", None)
            error_text = str(exc)
            if status_code == 408 or "timeout" in error_text.lower():
                if on_stderr is not None:
                    on_stderr(error_text)
                error = _last_non_empty_line(error_text) or error_text
                return DaytonaRunResult(
                    stderr=error_text,
                    exit_code=124,
                    charts=[],
                    error=error,
                )
            raise

        output = response.result or ""
        charts = (
            response.artifacts.charts
            if response.artifacts and response.artifacts.charts
            else []
        )

        if response.exit_code == 0:
            if on_stdout is not None and output:
                on_stdout(output)
            return DaytonaRunResult(
                stdout=output,
                stderr="",
                exit_code=response.exit_code,
                charts=charts,
                error=None,
            )

        error = (
            _last_non_empty_line(output)
            or f"Process exited with code {response.exit_code}"
        )
        if on_stderr is not None and output:
            on_stderr(output)
        return DaytonaRunResult(
            stdout="",
            stderr=output,
            exit_code=response.exit_code,
            charts=charts,
            error=error,
        )

    def close(self) -> None:
        # [FUNCTION DOC]
        # If no loop was ever started just marks closed. Otherwise synchronously deletes
        # the sandbox and closes the client, then always marks closed and stops the loop
        # thread.
        if self._loop is None:
            self._closed = True
            return

        try:
            self._run(self._aclose())
        finally:
            self._closed = True
            self._stop_loop()

    async def _aclose(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Async. Clears the sandbox id; if one existed, fetches and deletes the
        # sandbox (create_timeout), ignoring not-found errors. Always closes and drops
        # the client.
        sandbox_id = self._sandbox_id
        self._sandbox_id = None

        try:
            if sandbox_id is not None:
                client = await self._aget_client()
                sandbox = await client.get(sandbox_id)
                await sandbox.delete(timeout=self.create_timeout)
        except self._sdk["DaytonaNotFoundError"]:
            logger.debug("Daytona sandbox already deleted")
        finally:
            if self._client is not None:
                client = self._client
                self._client = None
                await client.close()

    def _stop_loop(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Clears loop/thread references, stops the loop thread-safely if
        # running, and joins the thread if alive.
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join()

    def __enter__(self) -> DaytonaRunner:
        # [FUNCTION DOC]
        # Starts the runner (creating the sandbox) and returns it.
        return self.start()

    def __exit__(self, *exc) -> None:
        # [FUNCTION DOC]
        # Closes the runner; does not suppress exceptions.
        self.close()
