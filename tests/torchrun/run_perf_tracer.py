# [MODULE DOC]
# Torchrun script that generates perf_tracer traces for tests: each rank runs traced
# warmup and 1000-iteration matmul scopes with rank-dependent sizes and saves the
# trace under AREAL_PERF_TRACE_BASE.
import os
from pathlib import Path

import torch
import torch.distributed as dist

from areal.api.cli_args import PerfTracerConfig
from areal.utils import perf_tracer
from areal.utils.perf_tracer import Category


def main() -> None:
    # [FUNCTION DOC]
    # Reads AREAL_PERF_TRACE_BASE (required; created), RANK, WORLD_SIZE; inits gloo if
    # multi-rank. Configures perf_tracer (experiment "torchrun", trial "world-<N>"),
    # resets it, and inside a "torchrun-step" INSTR scope runs a warmup matmul and 1000
    # matmuls (size 512 on even ranks, 4096 on odd; CUDA if available) in COMPUTE scopes,
    # then emits an instant event with the result norm. Barriers if multi-rank; finally
    # force-saves the trace and destroys the process group.
    base_dir = Path(os.environ["AREAL_PERF_TRACE_BASE"])
    base_dir.mkdir(parents=True, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")

    try:
        config = PerfTracerConfig(
            experiment_name="torchrun",
            trial_name=f"world-{world_size}",
            fileroot=str(base_dir),
            enabled=True,
        )
        tracer = perf_tracer.configure(
            config,
            rank=rank,
        )
        tracer.reset()

        with perf_tracer.trace_scope(
            "torchrun-step",
            category=Category.INSTR,
            args={"rank": rank},
        ):
            device = (
                torch.device("cuda", rank % torch.cuda.device_count())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            size = 512 if rank % 2 == 0 else 4096
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)

            with perf_tracer.trace_scope(
                "warmup",
                category=Category.COMPUTE,
                args={"rank": rank, "size": size},
            ):
                _c = torch.matmul(a, b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            with perf_tracer.trace_scope(
                "matmul",
                category=Category.COMPUTE,
                args={"rank": rank, "size": size, "iters": 1000},
            ):
                c = None
                for _ in range(1000):
                    c = torch.matmul(a, b)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

            perf_tracer.instant(
                "torchrun-mark",
                args={
                    "rank": rank,
                    "size": size,
                    "result_norm": float(
                        c.norm().item() if c is not None else _c.norm().item()
                    ),
                },
            )
        if world_size > 1:
            dist.barrier()
    finally:
        perf_tracer.save(force=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
