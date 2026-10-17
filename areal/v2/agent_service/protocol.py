# [MODULE DOC]
# Wire protocol for the Agent Service gateway WebSocket (OpenClaw-like): JSON text
# frames discriminated by "type" (req / res / event), enums for frame type, request
# method, queue mode and run status, frame dataclasses, JSON (de)serialization and
# factory helpers. Also defines PASSTHROUGH_HEADER, the HTTP header the Worker sets
# to mark a raw passthrough turn. All functions are pure (no I/O).
# SPDX-License-Identifier: Apache-2.0

"""Gateway protocol frame parsing for the Agent Service.

Implements the OpenClaw-like WebSocket protocol with three frame types:
``req`` (client→gateway), ``res`` (gateway→client), and ``event``
(gateway→client streaming). All frames are JSON text with a ``type``
discriminator.

Reference: https://docs.openclaw.ai/gateway/protocol
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# HTTP response header the Worker sets on a raw-passthrough turn (an agent that
# returns a :class:`~areal.v2.agent_service.types.StreamResponse`).
# The DataProxy keys its relay-vs-parse decision on this marker rather than on
# ``Content-Type`` — a *non-streaming* passthrough body is itself
# ``application/json`` and would otherwise be mistaken for a structured turn.
PASSTHROUGH_HEADER = "x-areal-passthrough"


class FrameType(str, Enum):
    # [CLASS DOC]
    # String enum of frame discriminators: "req", "res", "event".
    """WebSocket frame type discriminator."""

    REQ = "req"
    RES = "res"
    EVENT = "event"


class RequestMethod(str, Enum):
    # [CLASS DOC]
    # String enum of supported req methods; only "agent".
    """Supported request methods in ``req`` frames."""

    AGENT = "agent"


class QueueMode(str, Enum):
    # [CLASS DOC]
    # String enum for how new messages are handled while a session run is active:
    # "collect" (coalesce, default) or "followup" (enqueue for the next turn).
    """Queue mode for inbound messages when a session is already running.

    Controls how the gateway handles new ``req`` frames that arrive while
    a prior run for the same session is still active.

    Reference: https://docs.openclaw.ai/concepts/queue
    """

    COLLECT = "collect"
    """Coalesce queued messages into a single followup turn (default)."""

    FOLLOWUP = "followup"
    """Enqueue for the next agent turn after the current run ends."""


class RunStatus(str, Enum):
    # [CLASS DOC]
    # String enum of statuses carried in res payloads: accepted, complete, failed.
    """Status values for ``res`` frame payloads."""

    ACCEPTED = "accepted"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Frame dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestFrame:
    # [CLASS DOC]
    # Client->gateway request: id, method and a params dict (camelCase keys message,
    # sessionKey, idempotencyKey, queueMode) exposed through convenience properties.
    """Client-to-gateway request frame.

    Attributes
    ----------
    id : str
        Unique request identifier, client-generated.
    method : RequestMethod
        The method to invoke (currently only ``agent``).
    params : dict[str, Any]
        Parameters including ``message``, ``sessionKey``, and
        ``idempotencyKey``.
    """

    id: str
    method: RequestMethod
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        # [FUNCTION DOC]
        # Returns params["message"] or "".
        return self.params.get("message", "")

    @property
    def session_key(self) -> str:
        # [FUNCTION DOC]
        # Returns params["sessionKey"] or "".
        return self.params.get("sessionKey", "")

    @property
    def idempotency_key(self) -> str:
        # [FUNCTION DOC]
        # Returns params["idempotencyKey"] or "".
        return self.params.get("idempotencyKey", "")

    @property
    def queue_mode(self) -> QueueMode:
        # [FUNCTION DOC]
        # Returns params["queueMode"] coerced to QueueMode; missing or invalid values fall
        # back to COLLECT.
        raw = self.params.get("queueMode", QueueMode.COLLECT)
        try:
            return QueueMode(raw)
        except ValueError:
            return QueueMode.COLLECT


@dataclass
class ResponseFrame:
    # [CLASS DOC]
    # Gateway->client response: id of the originating request, ok flag and a payload
    # dict (runId, status, optional summary/error).
    """Gateway-to-client response frame.

    Attributes
    ----------
    id : str
        Matching request ID from the originating ``req`` frame.
    ok : bool
        Whether the request succeeded.
    payload : dict[str, Any]
        Response payload containing ``runId`` and ``status``.
    """

    id: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        # [FUNCTION DOC]
        # Returns payload["runId"] or "".
        return self.payload.get("runId", "")

    @property
    def status(self) -> str:
        # [FUNCTION DOC]
        # Returns payload["status"] or "".
        return self.payload.get("status", "")


@dataclass
class EventFrame:
    # [CLASS DOC]
    # Gateway->client streaming event: event category (e.g. "agent") and payload with
    # runId and optional delta / toolCall.
    """Gateway-to-client streaming event frame.

    Attributes
    ----------
    event : str
        Event category (e.g. ``"agent"``).
    payload : dict[str, Any]
        Event payload containing ``runId`` and optional ``delta`` /
        ``toolCall`` fields.
    """

    event: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        # [FUNCTION DOC]
        # Returns payload["runId"] or "".
        return self.payload.get("runId", "")

    @property
    def delta(self) -> str | None:
        # [FUNCTION DOC]
        # Returns payload["delta"] or None.
        return self.payload.get("delta")

    @property
    def tool_call(self) -> dict[str, Any] | None:
        # [FUNCTION DOC]
        # Returns payload["toolCall"] dict or None.
        return self.payload.get("toolCall")


# Type alias for any frame.
Frame = RequestFrame | ResponseFrame | EventFrame


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def parse_frame(raw: str | bytes) -> Frame:
    # [FUNCTION DOC]
    # Decodes JSON text/bytes and builds the frame dataclass for its "type": req needs
    # "id" (method defaults to "agent", params {}), res needs "id" (ok defaults True,
    # payload {}), event uses event (default "agent") and payload. Raises ValueError for
    # a missing/unknown type or invalid method, KeyError for missing "id", and JSON
    # decode errors for malformed input.
    # KNOWN ISSUE: the non-str/bytes branch (passing an already-decoded dict) is
    # unreachable per the type hint but silently accepted; a missing "id" raises
    # KeyError rather than the documented ValueError.
    """Parse a raw WebSocket text frame into a typed dataclass.

    Parameters
    ----------
    raw : str | bytes
        JSON-encoded WebSocket text frame.

    Returns
    -------
    Frame
        One of :class:`RequestFrame`, :class:`ResponseFrame`, or
        :class:`EventFrame`.

    Raises
    ------
    ValueError
        If the frame is missing required fields or has an unknown type.
    """
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

    frame_type = data.get("type")
    if frame_type is None:
        raise ValueError("Frame missing 'type' field")

    if frame_type == FrameType.REQ:
        return RequestFrame(
            id=data["id"],
            method=RequestMethod(data.get("method", "agent")),
            params=data.get("params", {}),
        )
    if frame_type == FrameType.RES:
        return ResponseFrame(
            id=data["id"],
            ok=data.get("ok", True),
            payload=data.get("payload", {}),
        )
    if frame_type == FrameType.EVENT:
        return EventFrame(
            event=data.get("event", "agent"),
            payload=data.get("payload", {}),
        )
    raise ValueError(f"Unknown frame type: {frame_type!r}")


def serialize_frame(frame: Frame) -> str:
    # [FUNCTION DOC]
    # Encodes a frame dataclass to a JSON string with its "type" field plus its own
    # fields (method emitted as its string value). Raises TypeError for other objects.
    """Serialize a frame dataclass to a JSON string.

    Parameters
    ----------
    frame : Frame
        A :class:`RequestFrame`, :class:`ResponseFrame`, or
        :class:`EventFrame`.

    Returns
    -------
    str
        JSON-encoded string ready to send over WebSocket.
    """
    if isinstance(frame, RequestFrame):
        return json.dumps(
            {
                "type": FrameType.REQ,
                "id": frame.id,
                "method": frame.method.value,
                "params": frame.params,
            }
        )
    if isinstance(frame, ResponseFrame):
        return json.dumps(
            {
                "type": FrameType.RES,
                "id": frame.id,
                "ok": frame.ok,
                "payload": frame.payload,
            }
        )
    if isinstance(frame, EventFrame):
        return json.dumps(
            {
                "type": FrameType.EVENT,
                "event": frame.event,
                "payload": frame.payload,
            }
        )
    raise TypeError(f"Cannot serialize {type(frame).__name__}")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_accepted_response(request_id: str, run_id: str) -> ResponseFrame:
    # [FUNCTION DOC]
    # Builds an ok ResponseFrame with payload {runId, status: accepted}.
    """Create an ``accepted`` response frame for a request."""
    return ResponseFrame(
        id=request_id,
        ok=True,
        payload={"runId": run_id, "status": RunStatus.ACCEPTED},
    )


def make_complete_response(
    request_id: str, run_id: str, summary: str = ""
) -> ResponseFrame:
    # [FUNCTION DOC]
    # Builds an ok ResponseFrame with {runId, status: complete}, adding "summary" only
    # if non-empty.
    """Create a ``complete`` response frame for a finished run."""
    payload: dict[str, Any] = {"runId": run_id, "status": RunStatus.COMPLETE}
    if summary:
        payload["summary"] = summary
    return ResponseFrame(id=request_id, ok=True, payload=payload)


def make_failed_response(
    request_id: str, run_id: str, error: str = ""
) -> ResponseFrame:
    # [FUNCTION DOC]
    # Builds a ResponseFrame with ok False and {runId, status: failed}, adding "error"
    # only if non-empty.
    """Create a ``failed`` response frame for an errored run."""
    payload: dict[str, Any] = {"runId": run_id, "status": RunStatus.FAILED}
    if error:
        payload["error"] = error
    return ResponseFrame(id=request_id, ok=False, payload=payload)


def make_delta_event(run_id: str, delta: str) -> EventFrame:
    # [FUNCTION DOC]
    # Builds an "agent" EventFrame with payload {runId, delta}.
    """Create a text delta streaming event."""
    return EventFrame(
        event="agent",
        payload={"runId": run_id, "delta": delta},
    )


def make_tool_call_event(run_id: str, tool_name: str, tool_args: str) -> EventFrame:
    # [FUNCTION DOC]
    # Builds an "agent" EventFrame with payload {runId, toolCall: {name, args}}.
    """Create a tool call streaming event."""
    return EventFrame(
        event="agent",
        payload={
            "runId": run_id,
            "toolCall": {"name": tool_name, "args": tool_args},
        },
    )


def generate_run_id() -> str:
    # [FUNCTION DOC]
    # Returns "run-" followed by the first 12 hex chars of a random UUID4.
    """Generate a unique run ID."""
    return f"run-{uuid.uuid4().hex[:12]}"
