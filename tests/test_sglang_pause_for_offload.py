# [MODULE DOC]
# Tests for RemoteInfEngine.pause_generation request sequencing with SGLang and with a
# backend that only provides a single pause request. The engine is built without
# __init__ and _run_request_on_all_servers is replaced with a list append to capture
# requests.
from types import SimpleNamespace

from areal.engine.sglang_remote import SGLangBackend
from areal.infra.remote_inf_engine import RemoteInfEngine


def test_pause_generation_drains_then_pauses_scheduler():
    # [FUNCTION DOC]
    # Calls the unwrapped pause_generation on an SGLang-backed engine with zero grace
    # period; asserts two requests were sent: a default pause (empty payload, draining)
    # followed by an in-place scheduler pause.
    requests = []
    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.backend = SGLangBackend()
    engine.config = SimpleNamespace(pause_grace_period=0)
    engine._run_request_on_all_servers = requests.append

    pause = getattr(
        RemoteInfEngine.pause_generation,
        "__wrapped__",
        RemoteInfEngine.pause_generation,
    )
    pause(engine)

    assert [request.payload for request in requests] == [
        {},
        {"mode": "in_place"},
    ]


def test_default_pause_request_matches_upstream_payload():
    # [FUNCTION DOC]
    # Asserts SGLangBackend's default pause request payload is empty (matches upstream).
    assert SGLangBackend().get_pause_request().payload == {}


def test_pause_generation_keeps_single_request_backends_compatible():
    # [FUNCTION DOC]
    # With a backend that only has get_pause_request, asserts pause_generation sends
    # exactly that one request.
    requests = []

    class _SingleRequestBackend:
        # [CLASS DOC]
        # HELPER: Minimal backend exposing only get_pause_request.
        def get_pause_request(self):
            # [FUNCTION DOC]
            # HELPER: Returns a namespace with payload {"backend": "single"}.
            return SimpleNamespace(payload={"backend": "single"})

    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.backend = _SingleRequestBackend()
    engine.config = SimpleNamespace(pause_grace_period=0)
    engine._run_request_on_all_servers = requests.append

    pause = getattr(
        RemoteInfEngine.pause_generation,
        "__wrapped__",
        RemoteInfEngine.pause_generation,
    )
    pause(engine)

    assert [request.payload for request in requests] == [{"backend": "single"}]
