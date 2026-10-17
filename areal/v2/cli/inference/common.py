# [MODULE DOC]
# Inference-CLI domain helpers: backend spec parsing, help strings for pass-through
# args, ordered teardown of a service, TaskHandle formatting for status tables, and
# spawning plus registering a model's worker/data-proxy replicas via a Scheduler.
# KNOWN ISSUE: the comment above __all__ talks about "imports kept at bottom" / stdlib
# re-exports, but the block only declares __all__.
# SPDX-License-Identifier: Apache-2.0

"""Domain-specific helpers for the inference CLI.

Lifecycle predicates, HTTP polling, output dispatch, and logger setup
live in the shared scaffold (`areal.v2.cli.lifecycle`,
`.utils`, `.client`, `.status`). This module keeps only the
inference-specific pieces: backend spec parsing, model registration,
and TaskHandle column formatters.
"""

from __future__ import annotations

from pathlib import Path

import click

from areal.api.alloc_mode import ModelAllocation
from areal.v2.cli.client import ServiceHTTPError, ServiceUnreachable
from areal.v2.cli.inference.client import GatewayClient, RouterClient
from areal.v2.cli.inference.launcher import (
    build_data_proxy_task_spec,
    build_sglang_task_spec,
    build_vllm_task_spec,
)
from areal.v2.cli.inference.scheduler import (
    Scheduler,
    TaskHandle,
    build_scheduler,
)
from areal.v2.cli.inference.state import (
    ModelEntry,
    ModelReplica,
    RuntimeState,
)
from areal.v2.cli.process import kill_pids
from areal.v2.cli.utils import register_cli_logger, wait_http_health

logger = register_cli_logger("InfCli")


ENGINE_ARGS_HELP = (
    "Shell-style string forwarded verbatim to the sglang / vllm process. "
    "Common sglang knobs: --mem-fraction-static 0.85, "
    "--max-running-requests 256, --chunked-prefill-size 4096, "
    "--disable-radix-cache, --enable-torch-compile. "
    "See the sglang / vllm CLI docs for the full surface."
)

PROXY_ARGS_HELP = (
    "Shell-style string forwarded verbatim to the data-proxy process. "
    "Available flags: --request-timeout, --set-reward-finish-timeout, "
    "--tool-call-parser, --reasoning-parser, --engine-max-tokens, "
    "--chat-template-type {hf|concat}."
)


def parse_backend_spec(spec: str) -> tuple[str, int, int, int]:
    # [FUNCTION DOC]
    # Parses a spec like "sglang:d4" or "vllm:d2t4" with ModelAllocation.from_str and
    # returns (engine, tp, dp, pp). Raises click.BadParameter if parsing fails or the
    # engine is not sglang or vllm.
    """Parse a backend spec into ``(engine, tp, dp, pp)``.

    Delegates to ``ModelAllocation.from_str`` so the CLI accepts the same
    grammar as ``InferenceEngineConfig.backend`` in YAML configs —
    ``"sglang:d4"``, ``"vllm:d2t4"``, etc. The CLI restricts the engine
    set to the two inference backends it can spawn locally.
    """

    try:
        alloc = ModelAllocation.from_str(spec)
    except Exception as exc:
        raise click.BadParameter(f"invalid --backend spec {spec!r}: {exc}") from exc
    if alloc.backend not in ("sglang", "vllm"):
        raise click.BadParameter(
            f"backend must be one of: sglang, vllm; got {alloc.backend!r}"
        )
    return (
        alloc.backend,
        alloc.parallel.tensor_parallel_size,
        alloc.parallel.data_parallel_size,
        alloc.parallel.pipeline_parallel_size,
    )


def split_args(value: str) -> list[str]:
    # [FUNCTION DOC]
    # Shell-splits a pass-through argument string into a list; empty string gives [].
    import shlex

    return shlex.split(value) if value else []


def terminate_runtime_state(
    state: RuntimeState, *, grace_s: float, force: bool = False
) -> None:
    # [FUNCTION DOC]
    # Kills the whole service in data-flow order: all data proxies, then all workers,
    # then the gateway, then the router, each phase via kill_pids on positive pids with
    # grace_s (0 when force). Empty phases are skipped. No state files are touched.
    """Tear down the whole inference service in data-flow order: kill the
    data-proxies first (so in-flight requests fail at the proxy boundary
    rather than mid-worker), then workers, then gateway, then router.

    ``force=True`` skips the grace period entirely — equivalent to
    ``kill_pids(pids, grace_s=0)`` for every phase.
    """

    def _pids(handles: list[TaskHandle]) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Returns the positive pids of the given handles.
        return [h.pid for h in handles if h.pid > 0]

    phases = (
        _pids(state.model_state.all_data_proxies()),
        _pids(state.model_state.all_workers()),
        _pids([state.gateway_handle]),
        _pids([state.router_handle]),
    )
    effective_grace = 0.0 if force else grace_s
    for pids in phases:
        if not pids:
            continue
        kill_pids(pids, grace_s=effective_grace)


def format_placement(backend: str, handle: TaskHandle) -> str:
    # [FUNCTION DOC]
    # Returns a placement string for a handle: local host name for "local", "node/pod"
    # for "k8s", "node/job=<id>" for "slurm" ("?" for missing ref keys), "-" otherwise.
    if backend == "local":
        import socket

        return socket.gethostname()
    if backend == "k8s":
        node = handle.ref.get("node", "?")
        pod = handle.ref.get("pod_name", "?")
        return f"{node}/{pod}"
    if backend == "slurm":
        node = handle.ref.get("node", "?")
        job = handle.ref.get("job_id", "?")
        return f"{node}/job={job}"
    return "-"


def format_ref(backend: str, handle: TaskHandle) -> str:
    # [FUNCTION DOC]
    # Returns "pid=<pid>" (local, if pid > 0), "pod=<name>" (k8s), "job=<id>" (slurm),
    # or "-" when unknown/missing.
    if backend == "local":
        return f"pid={handle.pid}" if handle.pid > 0 else "-"
    if backend == "k8s":
        pod = handle.ref.get("pod_name", "")
        return f"pod={pod}" if pod else "-"
    if backend == "slurm":
        job = handle.ref.get("job_id", "")
        return f"job={job}" if job else "-"
    return "-"


def format_gpu_count(handle: TaskHandle) -> str:
    # [FUNCTION DOC]
    # Returns a multiplication-sign prefixed GPU count (e.g. "x2") or "-" when no GPUs.
    n = len(handle.gpu_devices)
    return f"×{n}" if n > 0 else "-"


def register_internal(
    *,
    model: str,
    backend: str,
    model_path: str,
    tokenizer_path: str,
    engine_extra: list[str],
    proxy_extra: list[str],
    model_health_timeout: float,
    log_level: str,
    admin_api_key: str,
    gateway: GatewayClient,
    router: RouterClient,
    log_dir: Path,
    scheduler: Scheduler,
) -> list[ModelReplica]:
    # [FUNCTION DOC]
    # Spawns dp replicas for a model and registers them. Parses backend into
    # engine/tp/dp/pp (ClickException if pp > 1). For each rank: builds an sglang or
    # vllm worker TaskSpec (log <model>-worker-<rank>.log), submits it to the scheduler,
    # waits up to model_health_timeout for its /health; then builds and submits a data-
    # proxy TaskSpec pointing at the worker (with tokenizer, admin key, log level, extra
    # args; log <model>-data-proxy-<rank>.log) and waits 30s for health. After all
    # ranks, registers each proxy address with the router and the model (with all proxy
    # addresses) with the gateway, converting HTTP/network errors to ClickException. On
    # any exception kills all spawned pids (grace 10s), logs an error, and re-raises.
    # Returns the list of ModelReplica.
    engine, tp, dp, pp = parse_backend_spec(backend)
    if pp > 1:
        raise click.ClickException(
            "pp > 1 is not supported by `areal inf` (single-node only)."
        )

    replicas: list[ModelReplica] = []
    spawned_handles: list[TaskHandle] = []

    try:
        for rank in range(dp):
            worker_log = log_dir / f"{model}-worker-{rank}.log"
            if engine == "sglang":
                worker_spec = build_sglang_task_spec(
                    name=f"worker/{model}/{rank}",
                    model_path=model_path,
                    tp=tp,
                    extra_args=engine_extra,
                    log_file=worker_log,
                )
            else:
                worker_spec = build_vllm_task_spec(
                    name=f"worker/{model}/{rank}",
                    model_path=model_path,
                    tp=tp,
                    pp=pp,
                    extra_args=engine_extra,
                    log_file=worker_log,
                )
            worker_handle = scheduler.submit(worker_spec)
            spawned_handles.append(worker_handle)
            logger.info(
                "spawned %s worker %d/%d pid=%d port=%d gpus=%s",
                engine,
                rank,
                dp,
                worker_handle.pid,
                worker_handle.ports[0],
                worker_handle.gpu_devices,
            )
            wait_http_health(
                worker_handle.addr,
                pid=worker_handle.pid,
                timeout=model_health_timeout,
                label=f"{engine} worker {rank}",
                poll_interval=1.0,
            )

            proxy_log = log_dir / f"{model}-data-proxy-{rank}.log"
            proxy_spec = build_data_proxy_task_spec(
                name=f"data_proxy/{model}/{rank}",
                backend_addr=worker_handle.addr,
                backend_type=engine,
                tokenizer_path=tokenizer_path,
                admin_api_key=admin_api_key,
                log_level=log_level,
                extra_args=proxy_extra,
                log_file=proxy_log,
            )
            proxy_handle = scheduler.submit(proxy_spec)
            spawned_handles.append(proxy_handle)
            logger.info(
                "spawned data-proxy %d/%d pid=%d port=%d",
                rank,
                dp,
                proxy_handle.pid,
                proxy_handle.ports[0],
            )
            wait_http_health(
                proxy_handle.addr,
                pid=proxy_handle.pid,
                timeout=30.0,
                label=f"data-proxy {rank}",
                poll_interval=1.0,
            )

            replicas.append(ModelReplica(data_proxy=proxy_handle, worker=worker_handle))

        proxy_addrs = [r.data_proxy.addr for r in replicas]
        for addr in proxy_addrs:
            try:
                router.register_worker(addr)
            except (ServiceUnreachable, ServiceHTTPError) as exc:
                raise click.ClickException(
                    f"router register_worker {addr} failed: {exc}"
                ) from exc

        try:
            gateway.register_model(
                {
                    "model": model,
                    "url": "",
                    "api_key": "",
                    "data_proxy_addrs": proxy_addrs,
                }
            )
        except (ServiceUnreachable, ServiceHTTPError) as exc:
            raise click.ClickException(f"gateway register_model failed: {exc}") from exc

    except BaseException:
        if spawned_handles:
            logger.error(
                "internal register failed; killing %d spawned worker(s)",
                len(spawned_handles),
            )
            pids = [h.pid for h in spawned_handles if h.pid > 0]
            kill_pids(pids, grace_s=10.0)
        raise

    return replicas


def validate_register_opts(opts: dict) -> None:
    # [FUNCTION DOC]
    # Raises click.UsageError if opts lacks backend or model_path.
    if not opts.get("backend"):
        raise click.UsageError("--backend <spec> is required.")
    if not opts.get("model_path"):
        raise click.UsageError("--backend requires --model-path <path>.")


def register_model(
    *,
    model_name: str,
    opts: dict,
    gateway: GatewayClient,
    router: RouterClient,
    log_dir: Path,
    admin_api_key: str,
    scheduler_backend: str,
    occupied_gpus: set[int],
) -> ModelEntry:
    # [FUNCTION DOC]
    # Validates opts, builds a scheduler for scheduler_backend aware of occupied_gpus,
    # and calls register_internal with values from opts (tokenizer defaults to the model
    # path, extra args shell-split, health timeout default 600s, log level default
    # info). Returns a ModelEntry with the backend spec and replicas; does not modify
    # model state.
    """Validate registration opts and spawn the model's worker + data-proxy
    fleet. The returned ModelEntry is meant to be slotted into model_state
    by the caller — this helper does not touch model_state itself so the
    caller can keep the file lock scope explicit."""

    validate_register_opts(opts)
    scheduler = build_scheduler(scheduler_backend, occupied_gpus=occupied_gpus)
    replicas = register_internal(
        model=model_name,
        backend=opts["backend"],
        model_path=opts["model_path"],
        tokenizer_path=opts.get("tokenizer_path") or opts["model_path"],
        engine_extra=split_args(opts.get("engine_args", "")),
        proxy_extra=split_args(opts.get("proxy_args", "")),
        model_health_timeout=opts.get("model_health_timeout", 600.0),
        log_level=opts.get("log_level", "info"),
        admin_api_key=admin_api_key,
        gateway=gateway,
        router=router,
        log_dir=log_dir,
        scheduler=scheduler,
    )
    return ModelEntry(backend=opts["backend"], replicas=replicas)


# Imports kept at bottom: these are stdlib re-exports preserved for
# downstream callers that still import them via this module.
__all__ = [
    "ENGINE_ARGS_HELP",
    "PROXY_ARGS_HELP",
    "format_gpu_count",
    "format_placement",
    "format_ref",
    "logger",
    "parse_backend_spec",
    "register_internal",
    "register_model",
    "split_args",
    "terminate_runtime_state",
    "validate_register_opts",
]
