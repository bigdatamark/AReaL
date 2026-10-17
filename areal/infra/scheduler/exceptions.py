# [MODULE DOC]
# Exception hierarchy for schedulers. All errors derive from SchedulerError and store
# their constructor arguments as attributes while building a readable message.
# SPDX-License-Identifier: Apache-2.0

"""Custom exceptions for the scheduler module."""


class SchedulerError(Exception):
    # [CLASS DOC]
    # Base exception for scheduler-related failures; no extra fields.
    """Base exception for all scheduler-related errors."""


class WorkerCreationError(SchedulerError):
    # [CLASS DOC]
    # Raised when spawning or initializing a worker fails. Fields: worker_key, reason,
    # details.
    """Raised when worker creation fails during subprocess spawn or initialization."""

    def __init__(self, worker_key: str, reason: str, details: str = ""):
        # [FUNCTION DOC]
        # Stores fields; message is "Failed to create worker '<key>': <reason>" plus a
        # Details line when details is non-empty.
        self.worker_key = worker_key
        self.reason = reason
        self.details = details
        message = f"Failed to create worker '{worker_key}': {reason}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class WorkerConfigurationError(SchedulerError):
    # [CLASS DOC]
    # Raised when configuring a worker fails. Fields: worker_key, reason, details.
    def __init__(self, worker_key: str, reason: str, details: str = ""):
        # [FUNCTION DOC]
        # Stores fields; message is "Failed to configure worker '<key>': <reason>" plus
        # a Details line when details is non-empty.
        self.worker_key = worker_key
        self.reason = reason
        self.details = details
        message = f"Failed to configure worker '{worker_key}': {reason}"
        if details:
            message += f"\nDetails: {details}"
        super().__init__(message)


class WorkerFailedError(SchedulerError):
    # [CLASS DOC]
    # Raised when a worker process exits unexpectedly. Fields: worker_id, exit_code,
    # stderr.
    """Raised when a worker process fails or exits unexpectedly."""

    def __init__(self, worker_id: str, exit_code: int, stderr: str = ""):
        # [FUNCTION DOC]
        # Stores fields; message reports the exit code and appends captured stderr if
        # non-empty.
        self.worker_id = worker_id
        self.exit_code = exit_code
        self.stderr = stderr
        message = f"Worker '{worker_id}' failed with exit code {exit_code}"
        if stderr:
            message += f"\nStderr output:\n{stderr}"
        super().__init__(message)


class WorkerNotFoundError(SchedulerError):
    # [CLASS DOC]
    # Raised when a requested worker id is unknown. Field: worker_id.
    """Raised when attempting to access a worker that doesn't exist."""

    def __init__(self, worker_id: str):
        # [FUNCTION DOC]
        # Stores worker_id; message "Worker '<id>' not found".
        self.worker_id = worker_id
        super().__init__(f"Worker '{worker_id}' not found")


class EngineCreationError(SchedulerError):
    # [CLASS DOC]
    # Raised when creating an engine on a worker fails. Fields: worker_id, reason,
    # status_code.
    """Raised when engine creation fails on a worker."""

    def __init__(self, worker_id: str, reason: str, status_code: int = None):
        # [FUNCTION DOC]
        # Stores fields; appends " (HTTP <code>)" when status_code is truthy.
        # KNOWN ISSUE: status_code is annotated int but defaults to None (should be
        # Optional).
        self.worker_id = worker_id
        self.reason = reason
        self.status_code = status_code
        message = f"Failed to create engine on worker '{worker_id}': {reason}"
        if status_code:
            message += f" (HTTP {status_code})"
        super().__init__(message)


class EngineCallError(SchedulerError):
    # [CLASS DOC]
    # Raised when invoking an engine method on a worker fails. Fields: worker_id,
    # method, reason, attempt.
    """Raised when calling an engine method fails."""

    def __init__(self, worker_id: str, method: str, reason: str, attempt: int = 1):
        # [FUNCTION DOC]
        # Stores fields; appends "(after N attempts)" when attempt is greater than 1.
        self.worker_id = worker_id
        self.method = method
        self.reason = reason
        self.attempt = attempt
        message = f"Failed to call method '{method}' on worker '{worker_id}': {reason}"
        if attempt > 1:
            message += f" (after {attempt} attempts)"
        super().__init__(message)


class WorkerTimeoutError(SchedulerError):
    # [CLASS DOC]
    # Raised when waiting for a worker to become ready exceeds a timeout. Fields:
    # worker_key, timeout (seconds).
    """Raised when waiting for a worker exceeds the timeout."""

    def __init__(self, worker_key: str, timeout: float):
        # [FUNCTION DOC]
        # Stores fields; message reports the key and seconds waited.
        self.worker_key = worker_key
        self.timeout = timeout
        super().__init__(
            f"Timeout waiting for worker '{worker_key}' (waited {timeout}s)"
        )


class PortAllocationError(SchedulerError):
    # [CLASS DOC]
    # Raised when free ports cannot be allocated. Field: reason.
    """Raised when port allocation fails."""

    def __init__(self, reason: str):
        # [FUNCTION DOC]
        # Stores reason; message "Failed to allocate ports: <reason>".
        self.reason = reason
        super().__init__(f"Failed to allocate ports: {reason}")


class GPUAllocationError(SchedulerError):
    # [CLASS DOC]
    # Raised when GPU resources cannot be allocated. Field: reason.
    """Raised when GPU allocation fails."""

    def __init__(self, reason: str):
        # [FUNCTION DOC]
        # Stores reason; message "Failed to allocate GPU resources: <reason>".
        self.reason = reason
        super().__init__(f"Failed to allocate GPU resources: {reason}")


class RPCConnectionError(SchedulerError):
    # [CLASS DOC]
    # Raised when connecting to a worker's RPC server fails. Fields: worker_id, host,
    # port, reason.
    """Raised when RPC connection to a worker fails."""

    def __init__(self, worker_id: str, host: str, port: int, reason: str):
        # [FUNCTION DOC]
        # Stores fields; message includes host:port and reason.
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.reason = reason
        super().__init__(
            f"Failed to connect to worker '{worker_id}' at {host}:{port}: {reason}"
        )


class EngineImportError(SchedulerError):
    # [CLASS DOC]
    # Raised when a worker cannot import an engine class by path. Fields: import_path,
    # reason.
    """Raised when importing an engine class fails on the worker."""

    def __init__(self, import_path: str, reason: str):
        # [FUNCTION DOC]
        # Stores fields; message "Failed to import engine '<path>': <reason>".
        self.import_path = import_path
        self.reason = reason
        super().__init__(f"Failed to import engine '{import_path}': {reason}")
