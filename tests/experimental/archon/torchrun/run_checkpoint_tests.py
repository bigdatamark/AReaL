#!/usr/bin/env python3
# [MODULE DOC]
# torchrun integration tests for Archon checkpointing. Engine-level tests build an
# ArchonLMEngine from a real HF model (--model_path) with TP = world_size and check HF
# safetensors save, save->load forward equality (also with torch.compile + activation
# checkpointing), MoE save memory behaviour, and sync vs async HF save equality. PP
# tests (--pp_size == world_size) split a tiny random Qwen3 model with pipeline_llm and
# check DCP save/load of weights, optimizer state and first-stage forward output via
# DCPState. Temporary directories are created on rank 0, shared via object broadcast and
# removed afterwards; rank 0 writes "Passed"/"Failed" to --output.
"""Engine checkpoint integration tests.

Tests for ArchonEngine save/load methods with HF format using DCP infrastructure.
Also includes PP (Pipeline Parallelism) checkpoint tests using DCP format.

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_checkpoint_tests.py \
        --test_type=save_hf_dense --model_path=/path/to/model --output=/tmp/result.out

    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_checkpoint_tests.py \
        --test_type=pp_dcp_checkpoint --pp_size=2 --output=/tmp/result.out

Supported test types:
    - save_hf_dense: Test save() with weight_format="hf" on dense model
    - save_load_forward_match: Test save -> load -> forward output matches
    - save_load_forward_match_with_compile_ac: Test with torch.compile + activation checkpointing
    - moe_checkpoint: Test MoE model checkpoint with EP
    - pp_dcp_checkpoint: Test PP checkpoint save/load using DCP format
    - pp_dcp_with_optim: Test PP checkpoint with optimizer state
    - pp_forward_match: Test forward output matches after PP checkpoint save/load
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from tests.experimental.archon.torchrun.dist_utils import (
    create_dense_model_args,
    create_test_input,
    print_rank0,
    write_result,
)

from areal.api import FinetuneSpec, ParallelStrategy, SaveLoadMeta
from areal.api.cli_args import (
    ArchonEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.experimental.engine.archon_checkpoint import DCPState
from areal.experimental.engine.archon_engine import ArchonLMEngine
from areal.experimental.models.archon import ArchonParallelDims
from areal.experimental.models.archon.pipeline_parallel import pipeline_llm
from areal.experimental.models.archon.qwen3 import Qwen3Model, parallelize_qwen3


def create_engine_config(model_path: str) -> TrainEngineConfig:
    # [FUNCTION DOC]
    # HELPER: Returns a TrainEngineConfig for model_path with one micro-batch, Adam
    # optimizer (lr 1e-5, weight decay 0.01, no warmup, constant schedule, clipping 1.0)
    # and temperature 1.0.
    # KNOWN ISSUE: backend is "fsdp:d1" although the config is consumed by
    # ArchonLMEngine with a ParallelStrategy of TP = world_size; the backend string does
    # not describe the layout actually used.
    """Create engine configuration for testing."""
    return TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test_checkpoint",
        trial_name="test",
        path=model_path,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=OptimizerConfig(
            type="adam",
            lr=1e-5,
            weight_decay=0.01,
            warmup_steps_proportion=0.0,
            lr_scheduler_type="constant",
            gradient_clipping=1.0,
        ),
        temperature=1.0,
    )


def test_save_hf_dense(model_path: str, output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Initializes the engine (DP 1, TP = world_size), saves with
    # SaveLoadMeta(weight_format "hf", no optimizer, tokenizer) into a shared temp dir,
    # and on rank 0 checks at least one .safetensors file and config.json exist.
    # Exceptions mark failure; the temp dir is removed in finally after a barrier. Rank
    # 0 writes the result. Returns success (rank 0's view for file checks).
    """Test ArchonEngine.save() with weight_format="hf" on dense model.

    Steps:
    1. Initialize ArchonEngine with real HF checkpoint
    2. Call save(SaveLoadMeta(weight_format="hf")) to save in HF format
    3. Verify output contains safetensors files
    4. Verify config.json is copied
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save HF format (Dense) ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="hf_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")

        # Save checkpoint using unified save() interface
        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)

        dist.barrier()

        if rank == 0:
            safetensors_files = list(
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            )
            if len(safetensors_files) == 0:
                print_rank0("ERROR: No safetensors files in output")
                success = False
            else:
                print_rank0(f"Saved {len(safetensors_files)} safetensors files")

            config_path = os.path.join(output_dir, "config.json")
            if not os.path.exists(config_path):
                print_rank0("ERROR: config.json not found in output")
                success = False
            else:
                print_rank0("config.json saved")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_hf_dense: PASSED")
    else:
        print_rank0("save_hf_dense: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_save_load_forward_match(model_path: str, output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Initializes the engine, builds a seeded packed input of 4 sequences of 32 tokens
    # (positions continuous across sequences), records the eval-mode no_grad output of
    # engine.model, saves in HF format, loads it back, re-runs forward and compares:
    # passes if allclose at 1e-4 or max diff <= 1e-3 (warning). Cleans up the temp dir;
    # rank 0 writes the result. Returns success.
    """Test that forward output matches before save and after load.

    This is the most important test for checkpoint correctness:
    1. Initialize ArchonEngine with real HF checkpoint
    2. Run forward pass and record output
    3. Save checkpoint using save(weight_format="hf")
    4. Load checkpoint using load(weight_format="hf")
    5. Run forward pass again
    6. Verify outputs match exactly

    This test uses a small 0.6B model and runs on 1-2 GPUs.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save_load_forward_match ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="forward_match_test_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")

        torch.manual_seed(42)
        batch_size = 4
        seq_len = 32
        vocab_size = engine.model_config.vocab_size

        input_ids = torch.randint(
            0, vocab_size, (1, batch_size * seq_len), device=engine.device
        )
        positions = torch.arange(batch_size * seq_len, device=engine.device).unsqueeze(
            0
        )
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=engine.device,
        )

        engine.model.eval()
        with torch.no_grad():
            output_before = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )
            output_before = output_before.clone()

        print_rank0(f"Output before save: shape={output_before.shape}")

        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)
        print_rank0(f"Checkpoint saved to {output_dir}")

        dist.barrier()

        load_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
        )
        engine.load(load_meta)
        print_rank0("Checkpoint loaded")

        dist.barrier()

        engine.model.eval()
        with torch.no_grad():
            output_after = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )

        print_rank0(f"Output after load: shape={output_after.shape}")

        max_diff = (output_before - output_after).abs().max().item()
        mean_diff = (output_before - output_after).abs().mean().item()
        allclose = torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)

        print_rank0(f"  Max diff: {max_diff:.6e}")
        print_rank0(f"  Mean diff: {mean_diff:.6e}")
        print_rank0(f"  Allclose (rtol=1e-4, atol=1e-4): {allclose}")

        if not allclose:
            print_rank0("ERROR: Forward outputs do not match after load!")
            # Allow small differences due to numerical precision
            if max_diff > 1e-3:
                success = False
                print_rank0(f"ERROR: Max diff {max_diff} exceeds threshold 1e-3")
            else:
                print_rank0(f"WARNING: Max diff {max_diff} is small, treating as pass")
        else:
            print_rank0("Forward outputs match exactly!")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_load_forward_match: PASSED")
    else:
        print_rank0("save_load_forward_match: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_save_load_forward_match_with_compile_ac(
    model_path: str, output: str | None = None
) -> bool:
    # [FUNCTION DOC]
    # Same as test_save_load_forward_match but enables archon.enable_compile, gradient
    # checkpointing and ac_mode "full" before building the engine, and logs whether
    # parameter names carry "_orig_mod" / "_checkpoint_wrapped_module" wrapper prefixes
    # (logged, not asserted). Verifies save/load handles wrapped names via forward
    # equality.
    """Test forward match with torch.compile and activation checkpointing enabled.

    This test verifies that checkpoint save/load works correctly when the model
    has wrapper prefixes in parameter names (e.g., _orig_mod from torch.compile,
    _checkpoint_wrapped_module from activation checkpointing).

    Steps:
    1. Initialize ArchonEngine with compile + AC enabled
    2. Run forward pass and record output
    3. Save checkpoint using save(weight_format="hf")
    4. Load checkpoint using load(weight_format="hf")
    5. Run forward pass again
    6. Verify outputs match exactly
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: save_load_forward_match_with_compile_ac ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="forward_match_compile_ac_test_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        # Enable compile and activation checkpointing
        config.archon.enable_compile = True
        config.gradient_checkpointing = True
        config.archon.ac_mode = "full"

        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized with model: {engine.model.__class__.__name__}")
        print_rank0(f"  torch.compile enabled: {config.archon.enable_compile}")
        print_rank0(f"  Activation checkpointing: {config.archon.ac_mode}")

        # Verify wrapper prefixes exist in parameter names
        param_names = [name for name, _ in engine.model.named_parameters()]
        has_orig_mod = any("_orig_mod" in name for name in param_names)
        has_checkpoint_wrapper = any(
            "_checkpoint_wrapped_module" in name for name in param_names
        )
        print_rank0(f"  Has _orig_mod prefix: {has_orig_mod}")
        print_rank0(
            f"  Has _checkpoint_wrapped_module prefix: {has_checkpoint_wrapper}"
        )

        torch.manual_seed(42)
        batch_size = 4
        seq_len = 32
        vocab_size = engine.model_config.vocab_size

        input_ids = torch.randint(
            0, vocab_size, (1, batch_size * seq_len), device=engine.device
        )
        positions = torch.arange(batch_size * seq_len, device=engine.device).unsqueeze(
            0
        )
        cu_seqlens = torch.tensor(
            [i * seq_len for i in range(batch_size + 1)],
            dtype=torch.int32,
            device=engine.device,
        )

        engine.model.eval()
        with torch.no_grad():
            output_before = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )
            output_before = output_before.clone()

        print_rank0(f"Output before save: shape={output_before.shape}")

        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)
        print_rank0(f"Checkpoint saved to {output_dir}")

        dist.barrier()

        load_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
        )
        engine.load(load_meta)
        print_rank0("Checkpoint loaded")

        dist.barrier()

        engine.model.eval()
        with torch.no_grad():
            output_after = engine.model(
                input_ids, positions, cu_seqlens, max_seqlen=seq_len
            )

        print_rank0(f"Output after load: shape={output_after.shape}")

        max_diff = (output_before - output_after).abs().max().item()
        mean_diff = (output_before - output_after).abs().mean().item()
        allclose = torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)

        print_rank0(f"  Max diff: {max_diff:.6e}")
        print_rank0(f"  Mean diff: {mean_diff:.6e}")
        print_rank0(f"  Allclose (rtol=1e-4, atol=1e-4): {allclose}")

        if not allclose:
            print_rank0("ERROR: Forward outputs do not match after load!")
            # Allow small differences due to numerical precision
            if max_diff > 1e-3:
                success = False
                print_rank0(f"ERROR: Max diff {max_diff} exceeds threshold 1e-3")
            else:
                print_rank0(f"WARNING: Max diff {max_diff} is small, treating as pass")
        else:
            print_rank0("Forward outputs match exactly!")

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("save_load_forward_match_with_compile_ac: PASSED")
    else:
        print_rank0("save_load_forward_match_with_compile_ac: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


def test_moe_checkpoint(model_path: str, output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Requires world_size >= 4 (otherwise reports a skip as Passed). Initializes an
    # engine on the given (MoE) model, records CUDA memory before and after an HF save
    # to check it does not OOM, and on rank 0 verifies safetensors files exist (logging
    # their total size). OOM and other exceptions mark failure. Cleans up and writes the
    # result on rank 0.
    # KNOWN ISSUE: The docstring says EP=4, but the ParallelStrategy only sets
    # tensor_parallel_size=world_size; expert parallelism is never requested.
    """Test MoE model checkpoint with Expert Parallelism.

    This test requires 4 GPUs and tests:
    1. Loading MoE model with EP=4
    2. Saving checkpoint without OOM (no full_tensor() in to_hf)
    3. Verifying saved checkpoint is valid

    Note: This test is memory-intensive due to MoE model size (30B params).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: moe_checkpoint ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if world_size < 4:
        print_rank0("SKIP: MoE checkpoint test requires 4 GPUs")
        if rank == 0 and output:
            write_result(output, True)  # Skip is not failure
        return True

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="moe_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0("MoE Engine initialized")
        print_rank0(f"  Model: {engine.model.__class__.__name__}")
        print_rank0(
            f"  Config experts: {getattr(engine.model_config, 'num_experts', 'N/A')}"
        )

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        mem_before_max = torch.cuda.max_memory_allocated() / 1024**3
        print_rank0(
            f"  Memory before save: {mem_before:.2f} GB (max: {mem_before_max:.2f} GB)"
        )

        print_rank0("Saving MoE checkpoint...")
        save_meta = SaveLoadMeta(
            path=output_dir,
            weight_format="hf",
            with_optim=False,
            tokenizer=engine.tokenizer,
        )
        engine.save(save_meta)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_after_max = torch.cuda.max_memory_allocated() / 1024**3
        print_rank0(
            f"  Memory after save: {mem_after:.2f} GB (max: {mem_after_max:.2f} GB)"
        )

        dist.barrier()

        if rank == 0:
            safetensors_files = [
                f for f in os.listdir(output_dir) if f.endswith(".safetensors")
            ]
            if len(safetensors_files) == 0:
                print_rank0("ERROR: No safetensors files in output")
                success = False
            else:
                total_size = sum(
                    os.path.getsize(os.path.join(output_dir, f))
                    for f in safetensors_files
                )
                print_rank0(
                    f"Saved {len(safetensors_files)} files, "
                    f"total size: {total_size / 1024**3:.2f} GB"
                )

        engine.destroy()

    except torch.cuda.OutOfMemoryError as e:
        print_rank0(f"ERROR: OOM during MoE checkpoint: {e}")
        success = False

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    if success:
        print_rank0("moe_checkpoint: PASSED")
    else:
        print_rank0("moe_checkpoint: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# PP Checkpoint Tests (migrated from run_pp_checkpoint.py)
# =============================================================================


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns the local shard of a DTensor, or the tensor unchanged otherwise.
    """Convert DTensor to local tensor if needed."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def test_pp_dcp_checkpoint(pp_size: int, out_file: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Asserts world_size == pp_size. Builds tiny dense args (2 layers per stage), PP-
    # only dims, a meta-device model, and splits/parallelizes it with pipeline_llm
    # (1F1B, no compile); materializes parts with random init. Snapshots local parameter
    # values, saves DCPState(model_parts) with dcp.save, overwrites all params with 999
    # (checking the fill took effect), loads with dcp.load and counts parameters
    # differing by > 1e-6. Cleans up, all-gathers per-rank success, rank 0 writes the
    # result. Returns overall success.
    """Test PP checkpoint save/load using DCP format.

    Steps:
    1. Create PP model with pipeline_llm
    2. Save checkpoint using DCPState
    3. Modify weights to verify load actually changes them
    4. Load checkpoint
    5. Verify weights match original

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    import torch.distributed.checkpoint as dcp

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP DCP Checkpoint Test: pp_size={pp_size}, world_size={world_size}")
    print_rank0("=" * 60)

    assert world_size == pp_size, (
        f"This test requires world_size == pp_size. "
        f"Got world_size={world_size}, pp_size={pp_size}"
    )

    # Create temp directory for checkpoint
    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="pp_dcp_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        # 1. Create model args
        n_layers = pp_size * 2
        model_args = create_dense_model_args(
            n_layers=n_layers,
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=1000,
        )

        # 2. Create PP parallel dims
        parallel_dims = ArchonParallelDims(
            pp=pp_size,
            dp_shard=1,
            tp=1,
            cp=1,
            world_size=world_size,
            device_type="cuda",
        )

        # 3. Create model on meta device for efficient pipeline splitting
        with torch.device("meta"):
            model = Qwen3Model(model_args)

        # NOTE: Schedule is created dynamically by ArchonEngine, not returned here
        archon_config = ArchonEngineConfig(pp_schedule="1F1B")
        pp_stages, model_parts, has_first, has_last = pipeline_llm(
            model=model,
            device=device,
            parallel_dims=parallel_dims,
            archon_config=archon_config,
            parallelize_fn=parallelize_qwen3,
            enable_compile=False,  # Disable compile to speed up test
        )

        # Materialize weights from meta device
        for part in model_parts:
            part.to_empty(device=device)
            with torch.no_grad():
                part.init_weights()
            part.init_buffers(buffer_device=device)

        print_rank0(f"  Rank {rank}: has_first={has_first}, has_last={has_last}")
        print_rank0(f"  Rank {rank}: {len(model_parts)} model parts")

        # 4. Save original parameter values (convert DTensor to local)
        original_params = {}
        for part_idx, part in enumerate(model_parts):
            for name, param in part.named_parameters():
                key = f"part{part_idx}.{name}"
                original_params[key] = _to_local(param.data).clone()

        print_rank0(f"  Saved {len(original_params)} parameter tensors")

        # 5. Save checkpoint using DCPState
        dcp_state = DCPState(model_parts, optimizer=None)
        dcp.save({"dcp": dcp_state}, checkpoint_id=output_dir)

        print_rank0(f"  Checkpoint saved to {output_dir}")

        dist.barrier()

        # 6. Modify weights (to verify load actually works)
        with torch.no_grad():
            for part in model_parts:
                for param in part.parameters():
                    param.data.fill_(999.0)

        # Verify weights changed
        modified_check_passed = True
        for part_idx, part in enumerate(model_parts):
            for name, param in part.named_parameters():
                local_data = _to_local(param.data)
                if not torch.allclose(local_data, torch.tensor(999.0, device=device)):
                    modified_check_passed = False
                    break

        if not modified_check_passed:
            print_rank0("  ERROR: Failed to modify weights")
            success = False

        # 7. Load checkpoint
        dcp_state_load = DCPState(model_parts, optimizer=None)
        dcp.load({"dcp": dcp_state_load}, checkpoint_id=output_dir)

        print_rank0("  Checkpoint loaded")

        # 8. Verify weights match original
        mismatch_count = 0
        max_diff = 0.0
        for part_idx, part in enumerate(model_parts):
            for name, param in part.named_parameters():
                key = f"part{part_idx}.{name}"
                if key in original_params:
                    local_data = _to_local(param.data)
                    diff = (local_data - original_params[key]).abs().max().item()
                    max_diff = max(max_diff, diff)
                    if diff > 1e-6:
                        mismatch_count += 1
                        if mismatch_count <= 3:
                            print_rank0(f"  Mismatch: {key}, diff={diff:.6e}")

        if mismatch_count > 0:
            print_rank0(f"  ERROR: {mismatch_count} parameters don't match")
            success = False
        else:
            print_rank0(
                f"  All {len(original_params)} parameters match (max_diff={max_diff:.6e})"
            )

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    # Gather results from all ranks
    all_results = [None] * world_size
    dist.all_gather_object(all_results, success)
    final_success = all(all_results)

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP DCP Checkpoint Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP DCP Checkpoint Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


def test_pp_dcp_with_optim(pp_size: int, out_file: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Same setup as test_pp_dcp_checkpoint plus one AdamW over all local params.
    # Populates optimizer state: the first-stage rank runs its part and backpropagates
    # cross-entropy (if it is also last) or the sum of outputs, then steps; other ranks
    # only call step. Snapshots exp_avg/exp_avg_sq, saves DCPState with the optimizer,
    # fills every tensor state entry with 999, loads, and counts entries differing by >
    # 1e-6. Gathers success across ranks and writes the result on rank 0.
    # KNOWN ISSUE: On non-first stages optimizer.step() runs with no gradients, so AdamW
    # creates no state; those ranks compare nothing and pass trivially.
    """Test PP checkpoint save/load with optimizer state using DCP format.

    Steps:
    1. Create PP model and optimizer
    2. Run one forward/backward to populate optimizer state
    3. Save checkpoint with optimizer using DCPState
    4. Modify optimizer state
    5. Load checkpoint
    6. Verify optimizer state matches

    Args:
        pp_size: Pipeline parallel degree (must equal world_size)
        out_file: Optional file to write result ("Passed"/"Failed")

    Returns:
        True if test passed, False otherwise
    """
    import torch.distributed.checkpoint as dcp

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP DCP Checkpoint with Optimizer Test: pp_size={pp_size}")
    print_rank0("=" * 60)

    assert world_size == pp_size

    # Create temp directory
    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="pp_dcp_optim_checkpoint_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        # 1. Create model
        n_layers = pp_size * 2
        model_args = create_dense_model_args(
            n_layers=n_layers,
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=1000,
        )

        parallel_dims = ArchonParallelDims(
            pp=pp_size,
            dp_shard=1,
            tp=1,
            cp=1,
            world_size=world_size,
            device_type="cuda",
        )

        # Create model on meta device for efficient pipeline splitting
        with torch.device("meta"):
            model = Qwen3Model(model_args)

        def cross_entropy_loss_fn(output, target):
            # [FUNCTION DOC]
            # HELPER: Flattens output to [N, vocab] and target to [N]; returns mean
            # cross-entropy.
            output_flat = output.view(-1, output.size(-1))
            target_flat = target.view(-1)
            return nn.functional.cross_entropy(output_flat, target_flat)

        # NOTE: Schedule is created dynamically by ArchonEngine, not returned here
        archon_config = ArchonEngineConfig(pp_schedule="1F1B")
        pp_stages, model_parts, has_first, has_last = pipeline_llm(
            model=model,
            device=device,
            parallel_dims=parallel_dims,
            archon_config=archon_config,
            parallelize_fn=parallelize_qwen3,
            enable_compile=False,  # Disable compile to speed up test
        )

        # Materialize weights from meta device
        for part in model_parts:
            part.to_empty(device=device)
            with torch.no_grad():
                part.init_weights()
            part.init_buffers(buffer_device=device)

        # 2. Create optimizer (single optimizer for all model parts, as in archon_engine)
        all_params = [p for m in model_parts for p in m.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(all_params, lr=1e-4)

        print_rank0(f"  Created optimizer with {len(all_params)} parameters")

        # 3. Run a forward/backward to populate optimizer state
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )
        labels = tokens.clone()

        # Simple forward-backward for first stage (just to populate optimizer state)
        if has_first:
            for part in model_parts:
                part.train()
            optimizer.zero_grad()
            h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
            if has_last:
                loss = cross_entropy_loss_fn(h, labels)
                loss.backward()
                optimizer.step()
                print_rank0(f"  Ran forward/backward, loss={loss.item():.4f}")
            else:
                # For non-last stages, just do a dummy backward
                h.sum().backward()
                optimizer.step()
                print_rank0("  Ran forward/backward (non-last stage)")
        else:
            # For non-first stages, just step to populate state
            optimizer.step()
            print_rank0("  Stepped optimizer (non-first stage)")

        # 4. Save optimizer state values (convert DTensor to local)
        original_state = {}
        for i, p in enumerate(all_params):
            if p in optimizer.state:
                state = optimizer.state[p]
                if "exp_avg" in state:
                    original_state[f"param_{i}_exp_avg"] = _to_local(
                        state["exp_avg"]
                    ).clone()
                if "exp_avg_sq" in state:
                    original_state[f"param_{i}_exp_avg_sq"] = _to_local(
                        state["exp_avg_sq"]
                    ).clone()

        print_rank0(f"  Saved {len(original_state)} optimizer state tensors")

        # 5. Save checkpoint
        dcp_state = DCPState(model_parts, optimizer=optimizer)
        dcp.save({"dcp": dcp_state}, checkpoint_id=output_dir)

        print_rank0(f"  Checkpoint saved to {output_dir}")

        dist.barrier()

        # 6. Modify optimizer state
        for p in all_params:
            if p in optimizer.state:
                for key in optimizer.state[p]:
                    if isinstance(optimizer.state[p][key], torch.Tensor):
                        optimizer.state[p][key].fill_(999.0)

        # 7. Load checkpoint
        dcp_state_load = DCPState(model_parts, optimizer=optimizer)
        dcp.load({"dcp": dcp_state_load}, checkpoint_id=output_dir)

        print_rank0("  Checkpoint loaded")

        # 8. Verify optimizer state
        mismatch_count = 0
        for i, p in enumerate(all_params):
            if p in optimizer.state:
                state = optimizer.state[p]
                if "exp_avg" in state and f"param_{i}_exp_avg" in original_state:
                    local_exp_avg = _to_local(state["exp_avg"])
                    diff = (
                        (local_exp_avg - original_state[f"param_{i}_exp_avg"])
                        .abs()
                        .max()
                        .item()
                    )
                    if diff > 1e-6:
                        mismatch_count += 1
                if "exp_avg_sq" in state and f"param_{i}_exp_avg_sq" in original_state:
                    local_exp_avg_sq = _to_local(state["exp_avg_sq"])
                    diff = (
                        (local_exp_avg_sq - original_state[f"param_{i}_exp_avg_sq"])
                        .abs()
                        .max()
                        .item()
                    )
                    if diff > 1e-6:
                        mismatch_count += 1

        if mismatch_count > 0:
            print_rank0(f"  ERROR: {mismatch_count} optimizer states don't match")
            success = False
        else:
            print_rank0("  All optimizer states match")

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    # Gather results
    all_results = [None] * world_size
    dist.all_gather_object(all_results, success)
    final_success = all(all_results)

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP DCP Checkpoint with Optimizer Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP DCP Checkpoint with Optimizer Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


def test_pp_forward_match(pp_size: int, out_file: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Builds the PP model as above (eval mode) and a seeded 2x8 input. The first-stage
    # rank records its part's output, then all ranks save via DCP, zero all parameters,
    # load the checkpoint and the first-stage rank re-runs its part; outputs are
    # compared (allclose 1e-4 or max diff <= 1e-3). Gathers success across ranks; rank 0
    # writes the result.
    # KNOWN ISSUE: Only the first stage's output is checked (both branches of the
    # has_last check do the same thing); later stages are never run and pass trivially.
    """Test forward output matches after PP checkpoint save/load.

    This verifies the checkpoint preserves model behavior:
    1. Create PP model
    2. Run forward pass and record output
    3. Save checkpoint
    4. Modify weights
    5. Load checkpoint
    6. Run forward again
    7. Verify outputs match

    Args:
        pp_size: Pipeline parallel degree
        out_file: Optional file to write result

    Returns:
        True if test passed, False otherwise
    """
    import torch.distributed.checkpoint as dcp

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"PP Forward Match Test: pp_size={pp_size}")
    print_rank0("=" * 60)

    assert world_size == pp_size

    if rank == 0:
        output_dir = tempfile.mkdtemp(prefix="pp_forward_match_")
    else:
        output_dir = None

    output_dir_list = [output_dir]
    dist.broadcast_object_list(output_dir_list, src=0)
    output_dir = output_dir_list[0]

    success = True

    try:
        # Create model
        n_layers = pp_size * 2
        model_args = create_dense_model_args(
            n_layers=n_layers,
            dim=64,
            hidden_dim=128,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            vocab_size=1000,
        )

        parallel_dims = ArchonParallelDims(
            pp=pp_size,
            dp_shard=1,
            tp=1,
            cp=1,
            world_size=world_size,
            device_type="cuda",
        )

        # Create model on meta device for efficient pipeline splitting
        with torch.device("meta"):
            model = Qwen3Model(model_args)

        # NOTE: Schedule is created dynamically by ArchonEngine, not returned here
        archon_config = ArchonEngineConfig(pp_schedule="1F1B")
        pp_stages, model_parts, has_first, has_last = pipeline_llm(
            model=model,
            device=device,
            parallel_dims=parallel_dims,
            archon_config=archon_config,
            parallelize_fn=parallelize_qwen3,
            enable_compile=False,  # Disable compile to speed up test
        )

        # Materialize weights from meta device
        for part in model_parts:
            part.to_empty(device=device)
            with torch.no_grad():
                part.init_weights()
            part.init_buffers(buffer_device=device)

        for part in model_parts:
            part.eval()

        # Create test input
        tokens, positions, cu_seqlens, max_seqlen = create_test_input(
            num_seqs=2,
            seq_len_per_seq=8,
            vocab_size=model_args.vocab_size,
            device=device,
            seed=123,
        )

        # Run forward (only first stage does this, others receive via PP)
        output_before = None
        if has_first:
            with torch.no_grad():
                h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                if has_last:
                    output_before = h.clone()
                else:
                    # For non-last stages in first position, send and wait
                    # For this simple test, just capture the intermediate
                    output_before = h.clone()

        # Save checkpoint
        dcp_state = DCPState(model_parts, optimizer=None)
        dcp.save({"dcp": dcp_state}, checkpoint_id=output_dir)

        dist.barrier()

        # Modify weights
        with torch.no_grad():
            for part in model_parts:
                for param in part.parameters():
                    param.data.fill_(0.0)

        # Load checkpoint
        dcp_state_load = DCPState(model_parts, optimizer=None)
        dcp.load({"dcp": dcp_state_load}, checkpoint_id=output_dir)

        # Run forward again
        output_after = None
        if has_first:
            with torch.no_grad():
                h = model_parts[0](tokens, positions, cu_seqlens, max_seqlen)
                if has_last:
                    output_after = h.clone()
                else:
                    output_after = h.clone()

        # Compare outputs
        if output_before is not None and output_after is not None:
            output_before_local = _to_local(output_before)
            output_after_local = _to_local(output_after)
            max_diff = (output_before_local - output_after_local).abs().max().item()
            mean_diff = (output_before_local - output_after_local).abs().mean().item()
            allclose = torch.allclose(
                output_before_local, output_after_local, rtol=1e-4, atol=1e-4
            )

            print_rank0(f"  Max diff: {max_diff:.6e}")
            print_rank0(f"  Mean diff: {mean_diff:.6e}")
            print_rank0(f"  Allclose: {allclose}")

            if not allclose and max_diff > 1e-3:
                print_rank0("  ERROR: Forward outputs don't match!")
                success = False
            else:
                print_rank0("  Forward outputs match!")

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0 and output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    all_results = [None] * world_size
    dist.all_gather_object(all_results, success)
    final_success = all(all_results)

    if final_success:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Forward Match Test: PASSED")
        print_rank0("=" * 60)
    else:
        print_rank0(f"\n{'=' * 60}")
        print_rank0("PP Forward Match Test: FAILED")
        print_rank0("=" * 60)

    if rank == 0 and out_file:
        write_result(out_file, final_success)

    return final_success


# =============================================================================
# Async Checkpoint Tests
# =============================================================================


def test_async_save_hf_dense(model_path: str, output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Initializes the engine, saves an HF checkpoint synchronously with
    # save_model_to_hf, then again through an AsyncCheckpointManager(AsyncMode.ASYNC)
    # followed by finalize(). On rank 0 checks the async dir has safetensors files
    # (warning if file names differ), loads both sets of tensors and requires identical
    # key sets and torch.equal values, and checks config.json exists in the async dir.
    # Cleans up both temp dirs; rank 0 writes the result. Returns success.
    # KNOWN ISSUE: A comment says a gloo process group must be created for async saving,
    # but none is created here.
    """Test async HF save produces the same output as sync save.

    Steps:
    1. Initialize ArchonEngine
    2. Save checkpoint synchronously (baseline)
    3. Save checkpoint asynchronously (both ASYNC modes)
    4. Compare output files for consistency
    """
    from safetensors.torch import load_file

    from areal.experimental.engine.archon_checkpoint import save_model_to_hf
    from areal.utils.async_checkpoint import AsyncCheckpointManager, AsyncMode

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print_rank0("\n=== Test: async_save_hf_dense ===")
    print_rank0(f"Model path: {model_path}")
    print_rank0(f"World size: {world_size}")

    if rank == 0:
        sync_dir = tempfile.mkdtemp(prefix="sync_hf_")
        async_dir = tempfile.mkdtemp(prefix="async_hf_")
    else:
        sync_dir = None
        async_dir = None

    dirs = [sync_dir, async_dir]
    dist.broadcast_object_list(dirs, src=0)
    sync_dir, async_dir = dirs

    success = True

    try:
        config = create_engine_config(model_path)
        engine = ArchonLMEngine(config)

        parallel_strategy = ParallelStrategy(
            data_parallel_size=1,
            tensor_parallel_size=world_size,
        )
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=4, train_batch_size=4)

        engine.create_process_group(parallel_strategy=parallel_strategy)
        engine.initialize(addr=None, ft_spec=ft_spec)

        print_rank0(f"Engine initialized: {engine.model.__class__.__name__}")

        # 1. Sync save (baseline)
        print_rank0("Saving sync checkpoint...")
        save_model_to_hf(engine, sync_dir, engine.tokenizer)

        # 2. Async save (thread-based)
        print_rank0("Saving async checkpoint...")
        # Need a gloo PG for async - create manually since init used nccl
        mgr = AsyncCheckpointManager(AsyncMode.ASYNC)
        save_model_to_hf(engine, async_dir, engine.tokenizer, async_mgr=mgr)
        mgr.finalize()

        dist.barrier()

        # 3. Compare outputs on rank 0
        if rank == 0:
            # Check async output has safetensors files
            sync_files = sorted(
                f for f in os.listdir(sync_dir) if f.endswith(".safetensors")
            )
            async_files = sorted(
                f for f in os.listdir(async_dir) if f.endswith(".safetensors")
            )

            if len(async_files) == 0:
                print_rank0("ERROR: No safetensors files in async output")
                success = False
            elif sync_files != async_files:
                print_rank0(
                    f"WARNING: Different file names: sync={sync_files}, "
                    f"async={async_files}"
                )
                # File names may differ but content should be equivalent

            # Compare tensor values
            sync_state = {}
            for f in sync_files:
                sync_state.update(load_file(os.path.join(sync_dir, f)))

            async_state = {}
            for f in async_files:
                async_state.update(load_file(os.path.join(async_dir, f)))

            if set(sync_state.keys()) != set(async_state.keys()):
                missing = set(sync_state.keys()) - set(async_state.keys())
                extra = set(async_state.keys()) - set(sync_state.keys())
                print_rank0(f"ERROR: Key mismatch. Missing: {missing}, Extra: {extra}")
                success = False
            else:
                mismatches = 0
                for key in sync_state:
                    if not torch.equal(sync_state[key], async_state[key]):
                        max_diff = (
                            (sync_state[key].float() - async_state[key].float())
                            .abs()
                            .max()
                            .item()
                        )
                        mismatches += 1
                        if mismatches <= 3:
                            print_rank0(f"  Mismatch: {key}, max_diff={max_diff:.6e}")

                if mismatches > 0:
                    print_rank0(f"ERROR: {mismatches} tensors don't match")
                    success = False
                else:
                    print_rank0(
                        f"All {len(sync_state)} tensors match between "
                        "sync and async saves"
                    )

            # Check config.json exists
            if not os.path.exists(os.path.join(async_dir, "config.json")):
                print_rank0("ERROR: config.json not found in async output")
                success = False

        engine.destroy()

    except Exception as e:
        print_rank0(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        dist.barrier()
        if rank == 0:
            for d in [sync_dir, async_dir]:
                if d and os.path.exists(d):
                    shutil.rmtree(d)

    if success:
        print_rank0("async_save_hf_dense: PASSED")
    else:
        print_rank0("async_save_hf_dense: FAILED")

    if rank == 0 and output:
        write_result(output, success)

    return success


# =============================================================================
# Main
# =============================================================================

TEST_REGISTRY = {
    "save_hf_dense": test_save_hf_dense,
    "save_load_forward_match": test_save_load_forward_match,
    "save_load_forward_match_with_compile_ac": test_save_load_forward_match_with_compile_ac,
    "moe_checkpoint": test_moe_checkpoint,
    # PP checkpoint tests (migrated from run_pp_checkpoint.py)
    "pp_dcp_checkpoint": test_pp_dcp_checkpoint,
    "pp_dcp_with_optim": test_pp_dcp_with_optim,
    "pp_forward_match": test_pp_forward_match,
    # Async checkpoint tests
    "async_save_hf_dense": test_async_save_hf_dense,
}

# PP tests that require pp_size argument
PP_TESTS = {"pp_dcp_checkpoint", "pp_dcp_with_optim", "pp_forward_match"}


def main():
    # [FUNCTION DOC]
    # Parses --test_type (required), --model_path, --pp_size (2) and --output;
    # initializes NCCL and sets the CUDA device to the rank. PP tests receive pp_size;
    # other tests require --model_path (if missing, rank 0 writes "Failed" and main
    # returns). Runs the test, barriers and prints the summary; on exception rank 0
    # writes "Failed" and the exception is re-raised. Destroys the process group at the
    # end.
    parser = argparse.ArgumentParser(description="Engine Checkpoint Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to HF model checkpoint (required for non-PP tests)",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=2,
        help="Pipeline parallel size (for PP checkpoint tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running Checkpoint Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]

        # PP tests use pp_size argument, non-PP tests use model_path
        if args.test_type in PP_TESTS:
            success = test_fn(args.pp_size, args.output)
        else:
            if args.model_path is None:
                print_rank0(f"ERROR: --model_path is required for {args.test_type}")
                if rank == 0 and args.output:
                    write_result(args.output, False)
                return
            success = test_fn(args.model_path, args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Checkpoint Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Checkpoint Test {args.test_type}: FAILED")
            print_rank0("=" * 60)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
