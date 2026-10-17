# [MODULE DOC]
# Process launch helpers for the inference service: spawns the router and gateway
# directly as subprocesses and builds scheduler TaskSpecs for sglang/vllm workers and
# data proxies. Engine commands always use GPU base id 0 because the scheduler restricts
# visible devices.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from areal.utils.network import find_free_ports
from areal.v2.cli.inference.scheduler import (
    TaskAllocation,
    TaskResources,
    TaskSpec,
)
from areal.v2.cli.process import spawn_process

# Gateway and router are service-level singletons spawned directly via
# spawn_process — not submitted to a Scheduler — so their lifecycle is
# anchored to the CLI's state files regardless of scheduler backend.


def spawn_router(
    *,
    host: str,
    admin_api_key: str,
    routing_strategy: str,
    log_level: str,
    log_file: Path,
) -> tuple[int, int]:
    # [FUNCTION DOC]
    # Picks a free port and spawns "python -m areal.v2.inference_service.router" with
    # host, port, admin key, routing strategy and log level, logging to log_file.
    # Returns (pid, port).
    port = find_free_ports(1)[0]
    cmd = [
        sys.executable,
        "-m",
        "areal.v2.inference_service.router",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--routing-strategy",
        routing_strategy,
        "--log-level",
        log_level,
    ]
    return spawn_process(cmd, log_file), port


def spawn_gateway(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    router_url: str,
    log_level: str,
    log_file: Path,
) -> int:
    # [FUNCTION DOC]
    # Spawns "python -m areal.v2.inference_service.gateway" on host:port pointed at
    # router_url with admin key and log level, logging to log_file. Returns the pid.
    cmd = [
        sys.executable,
        "-m",
        "areal.v2.inference_service.gateway",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--router-addr",
        router_url,
        "--log-level",
        log_level,
    ]
    return spawn_process(cmd, log_file)


# sglang / vllm builders always pass base_gpu_id=0; the scheduler masks
# physical GPUs via CUDA_VISIBLE_DEVICES so the launched process sees
# its devices as 0..N-1.


def build_sglang_task_spec(
    *,
    name: str,
    model_path: str,
    tp: int,
    extra_args: list[str],
    log_file: Path,
) -> TaskSpec:
    # [FUNCTION DOC]
    # Returns a TaskSpec named name requesting tp GPUs and 1 port whose command builder
    # produces an SGLang server command (SGLangConfig with model_path and request
    # logging, tp_size tp, single node, pp 1) on the allocated host/port, followed by
    # extra_args.
    def cmd_builder(alloc: TaskAllocation) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Builds the SGLang launch argv for the given allocation (host, first
        # port).
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(model_path=model_path, log_requests=True)
        cmd = list(
            SGLangConfig.build_cmd(
                sglang_config=cfg,
                tp_size=tp,
                base_gpu_id=0,
                host=alloc.host,
                port=alloc.ports[0],
                n_nodes=1,
                node_rank=0,
                pp_size=1,
            )
        )
        cmd.extend(extra_args)
        return cmd

    return TaskSpec(
        name=name,
        cmd_builder=cmd_builder,
        log_file=log_file,
        resources=TaskResources(gpu=tp, ports=1),
    )


def build_vllm_task_spec(
    *,
    name: str,
    model_path: str,
    tp: int,
    pp: int,
    extra_args: list[str],
    log_file: Path,
) -> TaskSpec:
    # [FUNCTION DOC]
    # Returns a TaskSpec requesting tp*pp GPUs and 1 port whose command builder produces
    # a vLLM server command (vLLMConfig with model path, tp and pp sizes) on the
    # allocated host/port, followed by extra_args.
    def cmd_builder(alloc: TaskAllocation) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Builds the vLLM launch argv for the given allocation.
        from areal.api.cli_args import vLLMConfig

        cfg = vLLMConfig(model=model_path)
        cmd = list(
            vLLMConfig.build_cmd(
                vllm_config=cfg,
                tp_size=tp,
                pp_size=pp,
                host=alloc.host,
                port=alloc.ports[0],
            )
        )
        cmd.extend(extra_args)
        return cmd

    return TaskSpec(
        name=name,
        cmd_builder=cmd_builder,
        log_file=log_file,
        resources=TaskResources(gpu=tp * pp, ports=1),
    )


def build_data_proxy_task_spec(
    *,
    name: str,
    backend_addr: str,
    backend_type: str,
    tokenizer_path: str,
    admin_api_key: str,
    log_level: str,
    extra_args: list[str],
    log_file: Path,
) -> TaskSpec:
    # [FUNCTION DOC]
    # Returns a CPU-only TaskSpec (1 port) running "python -m
    # areal.v2.inference_service.data_proxy" with host/port from the allocation, backend
    # address and type, tokenizer path, admin key, log level and extra_args.
    def cmd_builder(alloc: TaskAllocation) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Builds the data proxy argv for the given allocation.
        cmd = [
            sys.executable,
            "-m",
            "areal.v2.inference_service.data_proxy",
            "--host",
            alloc.host,
            "--port",
            str(alloc.ports[0]),
            "--backend-addr",
            backend_addr,
            "--backend-type",
            backend_type,
            "--tokenizer-path",
            tokenizer_path,
            "--admin-api-key",
            admin_api_key,
            "--log-level",
            log_level,
        ]
        cmd.extend(extra_args)
        return cmd

    return TaskSpec(
        name=name,
        cmd_builder=cmd_builder,
        log_file=log_file,
        resources=TaskResources(gpu=0, ports=1),
    )
