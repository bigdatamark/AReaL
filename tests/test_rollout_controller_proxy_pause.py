# [MODULE DOC]
# Unit tests for RolloutController.pause/resume ordering. Uses a fake controller that
# borrows the real unbound pause/resume methods and records the sequence of dispatcher,
# proxy and worker RPC calls, verifying proxy workers are paused after the dispatcher
# and before inference workers (and resumed in reverse), and skipped when no proxy
# started.
# SPDX-License-Identifier: Apache-2.0

from areal.infra.controller.rollout_controller import RolloutController


class _RecordingController:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for RolloutController. Records call names into self.calls
    # and exposes the attributes the real pause/resume use (_proxy_started, dispatcher,
    # _collective_rpc, _proxy_collective_rpc). pause/resume are bound from
    # RolloutController.
    def __init__(self, proxy_started: bool):
        # [FUNCTION DOC]
        # HELPER: Initializes empty call log, stores proxy_started flag (bool) as
        # _proxy_started, and builds a nested fake dispatcher sharing the same call
        # list.
        self.calls: list[str] = []
        self._proxy_started = proxy_started

        class _Dispatcher:
            # [CLASS DOC]
            # HELPER: Fake dispatcher that appends
            # "dispatcher.pause"/"dispatcher.resume" to a shared call list.
            def __init__(self, calls: list[str]):
                # [FUNCTION DOC]
                # HELPER: Stores the shared call list (list[str]).
                self.calls = calls

            def pause(self):
                # [FUNCTION DOC]
                # HELPER: Records "dispatcher.pause".
                self.calls.append("dispatcher.pause")

            def resume(self):
                # [FUNCTION DOC]
                # HELPER: Records "dispatcher.resume".
                self.calls.append("dispatcher.resume")

        self.dispatcher = _Dispatcher(self.calls)

    def _collective_rpc(self, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake worker RPC; records "workers.<method>" and ignores args. Returns
        # None.
        self.calls.append(f"workers.{method}")

    def _proxy_collective_rpc(self, method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake proxy RPC; records "proxy.<method>" and ignores args. Returns
        # None.
        self.calls.append(f"proxy.{method}")

    pause = RolloutController.pause
    resume = RolloutController.resume


def test_pause_and_resume_include_proxy_workers_in_safe_order():
    # [FUNCTION DOC]
    # With a started proxy, asserts pause order is dispatcher -> proxy -> workers and
    # resume order is workers -> proxy -> dispatcher.
    controller = _RecordingController(proxy_started=True)

    controller.pause()
    controller.resume()

    assert controller.calls == [
        "dispatcher.pause",
        "proxy.pause",
        "workers.pause",
        "workers.resume",
        "proxy.resume",
        "dispatcher.resume",
    ]


def test_pause_and_resume_skip_proxy_rpc_when_not_started():
    # [FUNCTION DOC]
    # With no proxy started, asserts proxy RPCs are skipped and only dispatcher and
    # worker pause/resume calls happen in order.
    controller = _RecordingController(proxy_started=False)

    controller.pause()
    controller.resume()

    assert controller.calls == [
        "dispatcher.pause",
        "workers.pause",
        "workers.resume",
        "dispatcher.resume",
    ]
