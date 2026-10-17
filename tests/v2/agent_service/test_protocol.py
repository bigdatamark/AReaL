# [MODULE DOC]
# Unit tests for areal.v2.agent_service.protocol: the JSON frame protocol (req/res/event
# frames with camelCase payload accessors), parse/serialize round-trips, factory helpers
# and run-id generation. Pure, no I/O.
"""Unit tests for the Agent Service protocol module."""

from __future__ import annotations

import json

import pytest

from areal.v2.agent_service.protocol import (
    EventFrame,
    FrameType,
    RequestFrame,
    RequestMethod,
    ResponseFrame,
    RunStatus,
    generate_run_id,
    make_accepted_response,
    make_complete_response,
    make_delta_event,
    make_failed_response,
    make_tool_call_event,
    parse_frame,
    serialize_frame,
)


class TestFrameTypes:
    # [CLASS DOC]
    # String values of the FrameType, RequestMethod and RunStatus enums.
    def test_frame_type_values(self):
        # [FUNCTION DOC]
        # Asserts FrameType values req/res/event.
        assert FrameType.REQ == "req"
        assert FrameType.RES == "res"
        assert FrameType.EVENT == "event"

    def test_request_method_values(self):
        # [FUNCTION DOC]
        # Asserts RequestMethod.AGENT == "agent".
        assert RequestMethod.AGENT == "agent"

    def test_run_status_values(self):
        # [FUNCTION DOC]
        # Asserts RunStatus values accepted/complete/failed.
        assert RunStatus.ACCEPTED == "accepted"
        assert RunStatus.COMPLETE == "complete"
        assert RunStatus.FAILED == "failed"


class TestRequestFrame:
    # [CLASS DOC]
    # RequestFrame construction and param accessors.
    def test_basic_construction(self):
        # [FUNCTION DOC]
        # Asserts message, session_key and idempotency_key are read from camelCase params.
        frame = RequestFrame(
            id="req-1",
            method=RequestMethod.AGENT,
            params={"message": "hello", "sessionKey": "s1", "idempotencyKey": "idk-1"},
        )
        assert frame.id == "req-1"
        assert frame.method == RequestMethod.AGENT
        assert frame.message == "hello"
        assert frame.session_key == "s1"
        assert frame.idempotency_key == "idk-1"

    def test_empty_params_defaults(self):
        # [FUNCTION DOC]
        # Asserts accessors default to empty strings without params.
        frame = RequestFrame(id="r", method=RequestMethod.AGENT)
        assert frame.message == ""
        assert frame.session_key == ""
        assert frame.idempotency_key == ""


class TestResponseFrame:
    # [CLASS DOC]
    # ResponseFrame construction and payload accessors.
    def test_basic_construction(self):
        # [FUNCTION DOC]
        # Asserts run_id/status are read from payload runId/status.
        frame = ResponseFrame(
            id="req-1",
            ok=True,
            payload={"runId": "run-abc", "status": "accepted"},
        )
        assert frame.id == "req-1"
        assert frame.ok is True
        assert frame.run_id == "run-abc"
        assert frame.status == "accepted"

    def test_empty_payload_defaults(self):
        # [FUNCTION DOC]
        # Asserts run_id/status default to empty strings.
        frame = ResponseFrame(id="r", ok=False)
        assert frame.run_id == ""
        assert frame.status == ""


class TestEventFrame:
    # [CLASS DOC]
    # EventFrame payload accessors.
    def test_delta_event(self):
        # [FUNCTION DOC]
        # Asserts run_id and delta are read and tool_call is None for a delta event.
        frame = EventFrame(
            event="agent",
            payload={"runId": "run-1", "delta": "Hello"},
        )
        assert frame.event == "agent"
        assert frame.run_id == "run-1"
        assert frame.delta == "Hello"
        assert frame.tool_call is None

    def test_tool_call_event(self):
        # [FUNCTION DOC]
        # Asserts tool_call dict is read from toolCall and delta is None.
        frame = EventFrame(
            event="agent",
            payload={
                "runId": "run-1",
                "toolCall": {"name": "read_file", "args": "/tmp/x"},
            },
        )
        assert frame.delta is None
        assert frame.tool_call == {"name": "read_file", "args": "/tmp/x"}

    def test_empty_payload(self):
        # [FUNCTION DOC]
        # Asserts empty-payload defaults (run_id "", delta/tool_call None).
        frame = EventFrame(event="agent")
        assert frame.run_id == ""
        assert frame.delta is None
        assert frame.tool_call is None


class TestParseFrame:
    # [CLASS DOC]
    # parse_frame dispatch on the "type" field.
    def test_parse_req_frame(self):
        # [FUNCTION DOC]
        # Asserts a req JSON string parses to a RequestFrame with fields set.
        raw = json.dumps(
            {
                "type": "req",
                "id": "req-1",
                "method": "agent",
                "params": {"message": "hi", "sessionKey": "s1"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, RequestFrame)
        assert frame.id == "req-1"
        assert frame.method == RequestMethod.AGENT
        assert frame.message == "hi"

    def test_parse_res_frame(self):
        # [FUNCTION DOC]
        # Asserts a res JSON string parses to a ResponseFrame.
        raw = json.dumps(
            {
                "type": "res",
                "id": "req-1",
                "ok": True,
                "payload": {"runId": "run-1", "status": "complete"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, ResponseFrame)
        assert frame.ok is True
        assert frame.run_id == "run-1"

    def test_parse_event_frame(self):
        # [FUNCTION DOC]
        # Asserts an event JSON string parses to an EventFrame.
        raw = json.dumps(
            {
                "type": "event",
                "event": "agent",
                "payload": {"runId": "run-1", "delta": "chunk"},
            }
        )
        frame = parse_frame(raw)
        assert isinstance(frame, EventFrame)
        assert frame.delta == "chunk"

    def test_parse_bytes_input(self):
        # [FUNCTION DOC]
        # Asserts bytes input is accepted.
        raw = json.dumps({"type": "req", "id": "r1", "method": "agent"}).encode()
        frame = parse_frame(raw)
        assert isinstance(frame, RequestFrame)

    def test_parse_missing_type_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError "missing 'type'" when type is absent.
        with pytest.raises(ValueError, match="missing 'type'"):
            parse_frame(json.dumps({"id": "r1"}))

    def test_parse_unknown_type_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError "Unknown frame type" for an unknown type.
        with pytest.raises(ValueError, match="Unknown frame type"):
            parse_frame(json.dumps({"type": "unknown"}))


class TestSerializeFrame:
    # [CLASS DOC]
    # serialize_frame output and parse/serialize round-trips.
    def test_serialize_request(self):
        # [FUNCTION DOC]
        # Asserts serialized request has type req, id, method and params.
        frame = RequestFrame(
            id="r1",
            method=RequestMethod.AGENT,
            params={"message": "hello"},
        )
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "req"
        assert data["id"] == "r1"
        assert data["method"] == "agent"
        assert data["params"]["message"] == "hello"

    def test_serialize_response(self):
        # [FUNCTION DOC]
        # Asserts serialized response has type res and ok.
        frame = ResponseFrame(
            id="r1", ok=True, payload={"runId": "run-1", "status": "complete"}
        )
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "res"
        assert data["ok"] is True

    def test_serialize_event(self):
        # [FUNCTION DOC]
        # Asserts serialized event has type event and payload delta.
        frame = EventFrame(event="agent", payload={"runId": "r", "delta": "x"})
        data = json.loads(serialize_frame(frame))
        assert data["type"] == "event"
        assert data["payload"]["delta"] == "x"

    def test_roundtrip_request(self):
        # [FUNCTION DOC]
        # Asserts a request survives a round-trip.
        frame = RequestFrame(
            id="r1", method=RequestMethod.AGENT, params={"sessionKey": "s1"}
        )
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, RequestFrame)
        assert parsed.id == frame.id
        assert parsed.session_key == "s1"

    def test_roundtrip_response(self):
        # [FUNCTION DOC]
        # Asserts a failed response survives a round-trip.
        frame = ResponseFrame(
            id="r1", ok=False, payload={"runId": "run-1", "status": "failed"}
        )
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, ResponseFrame)
        assert parsed.ok is False

    def test_roundtrip_event(self):
        # [FUNCTION DOC]
        # Asserts an event survives a round-trip.
        frame = EventFrame(event="agent", payload={"runId": "r", "delta": "hi"})
        parsed = parse_frame(serialize_frame(frame))
        assert isinstance(parsed, EventFrame)
        assert parsed.delta == "hi"


class TestFactoryHelpers:
    # [CLASS DOC]
    # Factory helper tests.
    def test_make_accepted_response(self):
        # [FUNCTION DOC]
        # Asserts make_accepted_response builds an ok frame with status accepted.
        frame = make_accepted_response("req-1", "run-1")
        assert frame.id == "req-1"
        assert frame.ok is True
        assert frame.run_id == "run-1"
        assert frame.status == RunStatus.ACCEPTED

    def test_make_complete_response_no_summary(self):
        # [FUNCTION DOC]
        # Asserts make_complete_response omits summary when not given.
        frame = make_complete_response("req-1", "run-1")
        assert frame.status == RunStatus.COMPLETE
        assert "summary" not in frame.payload

    def test_make_complete_response_with_summary(self):
        # [FUNCTION DOC]
        # Asserts summary is included when given.
        frame = make_complete_response("req-1", "run-1", summary="done")
        assert frame.payload["summary"] == "done"

    def test_make_failed_response(self):
        # [FUNCTION DOC]
        # Asserts make_failed_response is not ok, status failed, with the error.
        frame = make_failed_response("req-1", "run-1", error="timeout")
        assert frame.ok is False
        assert frame.status == RunStatus.FAILED
        assert frame.payload["error"] == "timeout"

    def test_make_delta_event(self):
        # [FUNCTION DOC]
        # Asserts make_delta_event builds an "agent" event with the delta.
        frame = make_delta_event("run-1", "Hello")
        assert frame.event == "agent"
        assert frame.delta == "Hello"

    def test_make_tool_call_event(self):
        # [FUNCTION DOC]
        # Asserts make_tool_call_event builds a tool_call dict of name and args.
        frame = make_tool_call_event("run-1", "read_file", "/tmp/x")
        assert frame.tool_call == {"name": "read_file", "args": "/tmp/x"}


class TestGenerateRunId:
    # [CLASS DOC]
    # generate_run_id tests.
    def test_format(self):
        # [FUNCTION DOC]
        # Asserts ids look like "run-" plus 12 characters (length 16).
        run_id = generate_run_id()
        assert run_id.startswith("run-")
        assert len(run_id) == 16  # "run-" + 12 hex chars

    def test_uniqueness(self):
        # [FUNCTION DOC]
        # Asserts 100 generated ids are unique.
        ids = {generate_run_id() for _ in range(100)}
        assert len(ids) == 100
