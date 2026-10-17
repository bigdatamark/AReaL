# [MODULE DOC]
# Tests request validation of the RPC guard engine blueprint
# (areal.infra.rpc.guard.engine_blueprint). Builds a bare Flask app with a GuardState
# stored under app.config['guard_state'], registers engine_bp, and checks that malformed
# payloads to /create_engine, /call and /set_env are rejected with HTTP 400. No GPU or
# network needed.
import pytest
from flask import Flask

from areal.infra.rpc.guard.app import GuardState
from areal.infra.rpc.guard.engine_blueprint import engine_bp


@pytest.fixture
def client():
    # [FUNCTION DOC]
    # Pytest fixture yielding a Flask test client. Creates a Flask app, stores a fresh
    # GuardState in app.config['guard_state'] (the key the blueprint's state lookup
    # uses), registers engine_bp and yields the test client inside its context manager.
    app = Flask(__name__)
    # The blueprint calls get_state(), which looks for this config key
    state = GuardState()
    app.config["guard_state"] = state
    app.register_blueprint(engine_bp)

    with app.test_client() as client:
        yield client


def test_create_engine_empty_string(client):
    # [FUNCTION DOC]
    # POSTs /create_engine with engine set to an empty string; expects HTTP 400 and an
    # 'error' key in the JSON response.
    """Ensure empty strings are rejected (Functional parity with old manual check)."""
    resp = client.post("/create_engine", json={"engine": "", "engine_name": "test"})
    assert resp.status_code == 400
    # Pydantic errors are returned in the 'error' key per your route logic
    assert "error" in resp.get_json()


def test_create_engine_missing_fields(client):
    # [FUNCTION DOC]
    # POSTs /create_engine without the required 'engine' field; expects HTTP 400 from
    # schema validation.
    """Ensure missing required fields are caught by Pydantic."""
    resp = client.post(
        "/create_engine", json={"engine_name": "test"}
    )  # missing 'engine'
    assert resp.status_code == 400


def test_call_engine_missing_method(client):
    # [FUNCTION DOC]
    # POSTs /call with only engine_name (no 'method'); expects HTTP 400.
    """Ensure missing method is rejected."""
    resp = client.post("/call", json={"engine_name": "actor/0"})
    assert resp.status_code == 400


def test_set_env_invalid_json(client):
    # [FUNCTION DOC]
    # POSTs /set_env with 'env' as a string instead of a mapping; expects HTTP 400.
    """Ensure malformed JSON or invalid types are rejected."""
    # Sending a string where an object is expected for 'env'
    resp = client.post("/set_env", json={"env": "not-a-dict"})
    assert resp.status_code == 400
