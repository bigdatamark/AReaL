# [MODULE DOC]
# Copy of SGLang's http_server launch_server flow with AReaL additions (marked blocks):
# a custom scheduler process function that binds awex/PP bridges, a result IPC channel,
# an RpcProxy and /awex endpoints. Run as a script, it parses SGLang server args from
# argv, launches the server and finally kills all child processes.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Adapted from sglang.srt.entrypoints.http_server.launch_server
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers. Everything else mirrors the upstream launch_server flow.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys


def areal_launch_server(server_args) -> None:
    # [FUNCTION DOC]
    # Launches a full SGLang server: creates the awex result IPC path (exported via env
    # so scheduler subprocesses inherit it), spawns tokenizer/scheduler/detokenizer
    # subprocesses using areal_run_scheduler_process, returns early on non-head nodes
    # (no tokenizer manager), creates an RpcProxy and registers /awex endpoints on
    # SGLang's app, then runs the HTTP server (with warmup) blocking; closes the proxy
    # on exit.
    from sglang.srt.entrypoints.engine import Engine, init_tokenizer_manager
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
        app,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    # ---- BEGIN AREAL ----
    from areal.v2.inference_service.sglang.awex import (
        register_awex_endpoints,
    )
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.v2.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    result_ipc = create_result_ipc()
    # ---- END AREAL ----

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        # ---- BEGIN AREAL ----
        run_scheduler_process_func=areal_run_scheduler_process,
        # ---- END AREAL ----
        run_detokenizer_process_func=run_detokenizer_process,
    )

    # ---- BEGIN AREAL ----
    if tokenizer_manager is None:
        return
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    rpc_proxy = RpcProxy(port_args, result_ipc)
    register_awex_endpoints(app, rpc_proxy)
    # ---- END AREAL ----

    try:
        _setup_and_run_http_server(
            server_args,
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result.scheduler_infos,
            subprocess_watchdog,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        # ---- BEGIN AREAL ----
        rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
