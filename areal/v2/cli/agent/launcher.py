# [MODULE DOC]
# Spawns the local agent service stack as detached subprocesses (router, gateway, and
# worker/data-proxy pairs from areal.v2.agent_service), waits for each to become
# healthy, and returns the resulting ServiceState. All services bind to 127.0.0.1; logs
# go to the service's log directory in the agent state store.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from areal.utils.network import find_free_ports
from areal.v2.cli.agent.client import RouterClient
from areal.v2.cli.agent.state import (
    PairState,
    ProcessState,
    ServiceState,
    store,
)
from areal.v2.cli.process import spawn_process
from areal.v2.cli.utils import wait_http_health


def launch_agent_stack(
    *,
    service: str,
    agent: str,
    admin_api_key: str,
    num_pairs: int,
    setup_timeout: float,
    session_timeout: float,
    health_poll_interval: float,
    drain_timeout: float,
    log_level: str,
) -> ServiceState:
    # [FUNCTION DOC]
    # Launches the full stack. Reserves 2 + 2*num_pairs free ports at once (router,
    # gateway, then worker/proxy port per pair) to avoid port races. Spawns the router
    # and waits for its /health (setup_timeout); creates a RouterClient with
    # admin_api_key; for each pair index spawns a worker plus data proxy via _spawn_pair
    # and registers the proxy URL with the router; then spawns the gateway pointed at
    # the router and waits for health. Returns an unsaved ServiceState (launch_mode
    # "detached") holding pids, URLs, log files and the timeouts. Raises whatever
    # wait_http_health / RouterClient raise (e.g. RuntimeError, ServiceHTTPError) on
    # failure; already-spawned processes are not cleaned up here.
    log_dir = store.logs_dir(service)

    # Reserve every port we need up front from the non-ephemeral pool to
    # avoid TOCTOU collisions between successive bind(0) calls.
    ports = find_free_ports(2 + 2 * num_pairs)
    router_port = ports[0]
    gateway_port = ports[1]
    pair_ports = ports[2:]

    router_log = log_dir / "router.log"
    router_pid = _spawn_router(
        host="127.0.0.1",
        port=router_port,
        admin_api_key=admin_api_key,
        log_level=log_level,
        log_file=router_log,
    )
    router_url = f"http://127.0.0.1:{router_port}"
    wait_http_health(router_url, pid=router_pid, timeout=setup_timeout, label="router")

    pairs: list[PairState] = []
    router = RouterClient(router_url, admin_api_key)
    for idx in range(num_pairs):
        pair = _spawn_pair(
            index=idx,
            agent=agent,
            worker_port=pair_ports[2 * idx],
            proxy_port=pair_ports[2 * idx + 1],
            session_timeout=session_timeout,
            log_level=log_level,
            log_dir=log_dir,
            setup_timeout=setup_timeout,
        )
        router.register_proxy(pair.data_proxy.url)
        pairs.append(pair)

    gateway_log = log_dir / "gateway.log"
    gateway_pid = _spawn_gateway(
        host="127.0.0.1",
        port=gateway_port,
        router_url=router_url,
        admin_api_key=admin_api_key,
        log_level=log_level,
        log_file=gateway_log,
    )
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    wait_http_health(
        gateway_url, pid=gateway_pid, timeout=setup_timeout, label="gateway"
    )

    return ServiceState(
        service=service,
        launch_mode="detached",
        agent=agent,
        admin_api_key=admin_api_key,
        gateway=ProcessState(
            component="gateway",
            pid=gateway_pid,
            url=gateway_url,
            log_file=str(gateway_log),
        ),
        router=ProcessState(
            component="router",
            pid=router_pid,
            url=router_url,
            log_file=str(router_log),
        ),
        pairs=pairs,
        session_timeout=session_timeout,
        health_poll_interval=health_poll_interval,
        drain_timeout=drain_timeout,
    )


def _spawn_router(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    log_level: str,
    log_file: Path,
) -> int:
    # [FUNCTION DOC]
    # HELPER: Starts "python -m areal.v2.agent_service.router" with host, port, admin
    # API key and log level, stdout/stderr to log_file. Returns the child pid.
    return spawn_process(
        [
            sys.executable,
            "-m",
            "areal.v2.agent_service.router",
            "--host",
            host,
            "--port",
            str(port),
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            log_level,
        ],
        log_file,
    )


def _spawn_gateway(
    *,
    host: str,
    port: int,
    router_url: str,
    admin_api_key: str,
    log_level: str,
    log_file: Path,
) -> int:
    # [FUNCTION DOC]
    # HELPER: Starts "python -m areal.v2.agent_service.gateway" with host, port, router
    # address, admin API key and log level, logging to log_file. Returns the child pid.
    return spawn_process(
        [
            sys.executable,
            "-m",
            "areal.v2.agent_service.gateway",
            "--host",
            host,
            "--port",
            str(port),
            "--router-addr",
            router_url,
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            log_level,
        ],
        log_file,
    )


def _spawn_pair(
    *,
    index: int,
    agent: str,
    worker_port: int,
    proxy_port: int,
    session_timeout: float,
    log_level: str,
    log_dir: Path,
    setup_timeout: float,
) -> PairState:
    # [FUNCTION DOC]
    # HELPER: Starts one worker ("areal.v2.agent_service.worker" with the agent import
    # path on worker_port) and waits for its health, then starts its data proxy
    # ("areal.v2.agent_service.data_proxy" pointed at the worker, on proxy_port, with
    # session_timeout truncated to int seconds) and waits for health. Logs to
    # worker-<index>.log / proxy-<index>.log in log_dir. Returns a PairState with
    # component names worker-<index> and proxy-<index>.
    worker_log = log_dir / f"worker-{index}.log"
    worker_pid = spawn_process(
        [
            sys.executable,
            "-m",
            "areal.v2.agent_service.worker",
            "--agent",
            agent,
            "--host",
            "127.0.0.1",
            "--port",
            str(worker_port),
            "--log-level",
            log_level,
        ],
        worker_log,
    )
    worker_url = f"http://127.0.0.1:{worker_port}"
    wait_http_health(
        worker_url, pid=worker_pid, timeout=setup_timeout, label=f"worker-{index}"
    )

    proxy_log = log_dir / f"proxy-{index}.log"
    proxy_pid = spawn_process(
        [
            sys.executable,
            "-m",
            "areal.v2.agent_service.data_proxy",
            "--worker-addr",
            worker_url,
            "--host",
            "127.0.0.1",
            "--port",
            str(proxy_port),
            "--session-timeout",
            str(int(session_timeout)),
            "--log-level",
            log_level,
        ],
        proxy_log,
    )
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    wait_http_health(
        proxy_url, pid=proxy_pid, timeout=setup_timeout, label=f"proxy-{index}"
    )

    return PairState(
        index=index,
        worker=ProcessState(
            component=f"worker-{index}",
            pid=worker_pid,
            url=worker_url,
            log_file=str(worker_log),
        ),
        data_proxy=ProcessState(
            component=f"proxy-{index}",
            pid=proxy_pid,
            url=proxy_url,
            log_file=str(proxy_log),
        ),
    )
