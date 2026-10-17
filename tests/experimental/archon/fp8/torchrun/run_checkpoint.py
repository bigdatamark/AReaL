#!/usr/bin/env python3
# [MODULE DOC]
# torchrun worker script for the FP8 checkpoint end-to-end test. Each rank initializes
# NCCL, builds an ArchonLMEngine with blockwise FP8 enabled on a pure data-parallel
# mesh, loads an FP8 HF checkpoint, runs a deterministic forward, sanity-checks logits,
# optionally compares against a BF16 checkpoint, and rank 0 writes the result via
# write_result to --output.
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.experimental.archon.torchrun.dist_utils import print_rank0, write_result

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import (
    ArchonEngineConfig,
    MicroBatchSpec,
    TrainEngineConfig,
)
from areal.api.cli_args import (
    ArchonFP8Config as FP8Config,
)
from areal.experimental.engine.archon_engine import ArchonLMEngine


def _create_engine_config(
    model_path: str,
    world_size: int,
    enable_fp8: bool,
) -> TrainEngineConfig:
    # [FUNCTION DOC]
    # HELPER: Builds a TrainEngineConfig for inference-only use. Inputs: model_path,
    # world_size (encoded into backend "archon:d{world_size}"), enable_fp8 (when True
    # sets ArchonEngineConfig fp8_config mode "blockwise", else default
    # ArchonEngineConfig). Uses one micro-batch, no optimizer, fixed experiment/trial
    # names. Returns the config.
    archon_config = (
        ArchonEngineConfig(fp8_config=FP8Config(mode="blockwise"))
        if enable_fp8
        else ArchonEngineConfig()
    )
    return TrainEngineConfig(
        backend=f"archon:d{world_size}",
        experiment_name="test_fp8_checkpoint",
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=None,
        archon=archon_config,
    )


def _run_forward(engine: ArchonLMEngine) -> tuple[torch.Tensor, int, int, int]:
    # [FUNCTION DOC]
    # HELPER: Runs a no-grad eval forward on engine.model with a seeded (42) packed
    # batch of 2 sequences of length 32: token ids uniform in [100, vocab-100) shaped
    # [1, 64], positions 0..63 (not reset per sequence), int32 cu_seqlens [0,32,64],
    # max_seqlen 32. Returns (logits, batch_size, seq_len, vocab_size). Sets model to
    # eval mode as a side effect.
    torch.manual_seed(42)
    batch_size, seq_len = 2, 32
    vocab_size = engine.model_config.vocab_size
    input_ids = torch.randint(
        100,
        vocab_size - 100,
        (1, batch_size * seq_len),
        device=engine.device,
    )
    positions = torch.arange(batch_size * seq_len, device=engine.device).unsqueeze(0)
    cu_seqlens = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)],
        dtype=torch.int32,
        device=engine.device,
    )

    engine.model.eval()
    with torch.no_grad():
        logits = engine.model(input_ids, positions, cu_seqlens, max_seqlen=seq_len)

    return logits, batch_size, seq_len, vocab_size


def test_fp8_load_and_forward(
    fp8_model_path: str,
    bf16_model_path: str,
    output: str | None = None,
) -> bool:
    # [FUNCTION DOC]
    # Main check run on every rank. Creates and initializes the FP8 engine
    # (DP=world_size, dummy FinetuneSpec), runs the forward and asserts logits are
    # finite, max logit above 15 and self cross-entropy against greedy argmax labels
    # below 5. Destroys the FP8 engine; if bf16_model_path is given, builds a BF16
    # engine, runs the same forward and asserts mean relative difference below 0.5. Any
    # exception is caught, printed and marks failure. Always destroys remaining engines
    # and barriers. Rank 0 writes the result to output if given. Returns the local
    # success flag.
    # KNOWN ISSUE: Success is evaluated per rank but only rank 0 writes the result, so a
    # failure on another rank is not reflected in the output file (the process exit code
    # still reports it only if an exception propagates, which it does not here).
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: fp8_load_and_forward ===")
    print_rank0(f"FP8 model path: {fp8_model_path}")
    print_rank0(f"BF16 model path: {bf16_model_path}")
    print_rank0(f"World size: {world_size}")

    success = True
    error_msg = ""
    fp8_engine = None
    bf16_engine = None

    try:
        fp8_config = _create_engine_config(
            model_path=fp8_model_path,
            world_size=world_size,
            enable_fp8=True,
        )
        fp8_engine = ArchonLMEngine(fp8_config)

        parallel_strategy = ParallelStrategy(data_parallel_size=world_size)
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=128, train_batch_size=8
        )

        fp8_engine.create_process_group(parallel_strategy=parallel_strategy)
        fp8_engine.initialize(addr=None, ft_spec=ft_spec)

        fp8_logits, batch_size, seq_len, vocab_size = _run_forward(fp8_engine)

        assert not torch.isnan(fp8_logits).any()
        assert not torch.isinf(fp8_logits).any()

        max_logit = fp8_logits.max().item()
        print_rank0(f"FP8 max logit: {max_logit:.6f}")
        assert max_logit > 15.0

        greedy_labels = fp8_logits.reshape(-1, vocab_size).argmax(dim=-1)
        self_loss = F.cross_entropy(fp8_logits.reshape(-1, vocab_size), greedy_labels)
        print_rank0(f"FP8 self-CE loss (should be ~0): {self_loss.item():.6f}")
        assert self_loss.item() < 5.0

        fp8_logits = fp8_logits.detach().clone()
        fp8_engine.destroy()
        fp8_engine = None

        if bf16_model_path:
            bf16_config = _create_engine_config(
                model_path=bf16_model_path,
                world_size=world_size,
                enable_fp8=False,
            )
            bf16_engine = ArchonLMEngine(bf16_config)
            bf16_engine.create_process_group(parallel_strategy=parallel_strategy)
            bf16_engine.initialize(addr=None, ft_spec=ft_spec)

            bf16_logits, _, _, _ = _run_forward(bf16_engine)
            rel_diff = (fp8_logits - bf16_logits).abs() / (bf16_logits.abs() + 1e-6)
            rel_diff_mean = rel_diff.mean().item()
            print_rank0(f"FP8 vs BF16 rel_diff mean: {rel_diff_mean:.6f}")
            assert rel_diff_mean < 0.5

            bf16_engine.destroy()
            bf16_engine = None

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        error_msg = traceback.format_exc()
        traceback.print_exc()
        success = False

    finally:
        if fp8_engine is not None:
            fp8_engine.destroy()
        if bf16_engine is not None:
            bf16_engine.destroy()
        dist.barrier()

    if success:
        print_rank0("fp8_load_and_forward: PASSED")
    else:
        print_rank0("fp8_load_and_forward: FAILED")

    if rank == 0 and output:
        write_result(output, success, error=error_msg if not success else "")

    return success


def main():
    # [FUNCTION DOC]
    # Entry point: parses --fp8_model_path (required), --bf16_model_path, --output;
    # initializes the NCCL process group, sets the CUDA device to the global rank
    # (assumes single node), runs test_fp8_load_and_forward, barriers and prints the
    # overall status. On unexpected exceptions prints the traceback, rank 0 writes a
    # failure result, and re-raises. Always destroys the process group.
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8_model_path", type=str, required=True)
    parser.add_argument("--bf16_model_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    try:
        success = test_fp8_load_and_forward(
            args.fp8_model_path,
            args.bf16_model_path,
            args.output,
        )

        dist.barrier()

        if success:
            print_rank0("\n=== FP8 checkpoint test: PASSED ===")
        else:
            print_rank0("\n=== FP8 checkpoint test: FAILED ===")

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False, error=traceback.format_exc())
        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
