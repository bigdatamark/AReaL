# [MODULE DOC]
# ZMQ proxy used by the SGLang HTTP process to issue collective RPCs to scheduler
# subprocesses and receive data results.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight ZMQ proxy for dispatching RPC to scheduler subprocesses."""

from __future__ import annotations

from typing import Any

import zmq
from sglang.srt.managers.io_struct import RpcReqInput, RpcReqOutput
from sglang.srt.server_args import PortArgs


class RpcProxy:
    # [CLASS DOC]
    # Owns two ZMQ channels: a DEALER socket on SGLang's rpc_ipc_name (send RpcReqInput,
    # receive RpcReqOutput, same protocol as Engine.collective_rpc) and a bound PULL
    # socket on result_ipc that receives pickled results pushed by the rank-0 scheduler
    # bridge.
    """ZMQ proxy bridging the HTTP process to scheduler subprocesses.

    Two independent channels:

    * **RPC channel** (DEALER on ``rpc_ipc_name``): sends :class:`RpcReqInput`,
      receives :class:`RpcReqOutput` — same protocol as ``Engine.collective_rpc``.
    * **Result channel** (PULL on ``result_ipc``): receives pyobj results
      pushed by :class:`AwexSchedulerBridge` rank 0 via its PUSH socket.
    """

    def __init__(self, port_args: PortArgs, result_ipc: str) -> None:
        # [FUNCTION DOC]
        # Creates a context and DEALER socket bound to port_args.rpc_ipc_name, and a
        # second context with a PULL socket bound to result_ipc.
        from sglang.srt.utils.network import get_zmq_socket

        self._rpc_ctx = zmq.Context(1)
        self._rpc_socket = get_zmq_socket(
            self._rpc_ctx, zmq.DEALER, port_args.rpc_ipc_name, True
        )

        self._result_ctx = zmq.Context(1)
        self._result_pull = self._result_ctx.socket(zmq.PULL)
        self._result_pull.bind(result_ipc)

    def collective_rpc(self, method: str, **kwargs: Any) -> None:
        # [FUNCTION DOC]
        # Sends RpcReqInput(method, kwargs or None) and blocks for the RpcReqOutput
        # reply; raises RuntimeError with the message if success is false
        # (AssertionError if the reply type is wrong). Not thread-safe.
        req = RpcReqInput(method=method, parameters=kwargs if kwargs else None)
        self._rpc_socket.send_pyobj(req)
        resp: RpcReqOutput = self._rpc_socket.recv_pyobj()
        assert isinstance(resp, RpcReqOutput)
        if not resp.success:
            raise RuntimeError(f"RPC {method} failed: {resp.message}")

    def collective_rpc_with_result(self, method: str, **kwargs: Any) -> Any:
        # [FUNCTION DOC]
        # Runs collective_rpc, then blocks on the PULL socket for one pushed result and
        # returns it.
        # KNOWN ISSUE: no timeout; if no scheduler pushes (e.g. result env unset) it
        # blocks forever, and a stale extra result would desynchronize later calls.
        self.collective_rpc(method, **kwargs)
        return self._result_pull.recv_pyobj()

    def close(self) -> None:
        # [FUNCTION DOC]
        # Closes both sockets with linger 0 and terminates both contexts.
        for sock in (self._rpc_socket, self._result_pull):
            if sock is not None:
                sock.close(linger=0)
        for ctx in (self._rpc_ctx, self._result_ctx):
            if ctx is not None:
                ctx.term()
