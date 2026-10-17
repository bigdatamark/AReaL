# [MODULE DOC]
# Torchrun script testing FSDPEngine save/load with the DCP (distributed checkpoint)
# weight format on Qwen3-0.6B: a plain weights round-trip and a train/save/train/
# load/retrain round-trip with optimizer state. Rank 0 optionally writes the result.
import argparse
import copy
import os
import tempfile
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from tests.utils import get_model_path

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform
from areal.utils import seeding

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def write_result(out: str, succ: bool):
    # [FUNCTION DOC]
    # HELPER: Writes "Passed" if succ else "Failed" to the file at path out.
    with open(out, "w") as f:
        if succ:
            f.write("Passed")
        else:
            f.write("Failed")


def mock_input(
    batch_size=128,
    min_seqlen=1,
    max_seqlen=1024,
    device=current_platform.device_type,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds a random right-padded batch: per-sample lengths in
    # [min_seqlen, max_seqlen), token ids in [10000, 50000), padding id 0. Returns dict with
    # input_ids (long) and attention_mask (bool) of shape [batch, longest].
    # KNOWN ISSUE: docstring claims position_ids are returned, but they are not.
    """Create mock padded input data for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        10000, 50000, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Loss = mean of logprobs; other args ignored.
    """Mock loss function for testing."""
    return torch.mean(logprobs)


def make_fsdp_engine(
    backend: str, mb_spec: MicroBatchSpec, init_optimizer: bool = False
):
    # [FUNCTION DOC]
    # HELPER: Builds a TrainEngineConfig for the model (optimizer default config only if
    # init_optimizer), creates FSDPEngine, creates its process group from the parsed
    # allocation string, initializes with a small FinetuneSpec and returns the engine.
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_fsdp_dcp_distributed",
        trial_name="test",
        mb_spec=mb_spec,
        path=MODEL_PATH,
        optimizer=OptimizerConfig() if init_optimizer else None,
    )
    alloc_mode = ModelAllocation.from_str(backend)
    engine = FSDPEngine(config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(None, ft_spec)
    return engine


def test_simple_dcp_save_load(alloc_mode: str, output: str | None = None):
    # [FUNCTION DOC]
    # Saves weights in DCP format to a temp dir (rank 0 creates it, barrier), zeroes all
    # parameters, loads, and asserts every local shard equals a deep copy taken before
    # save (printing diffs on mismatch). Then syncs, barriers, destroys the engine, and
    # rank 0 writes the result to output if given.
    """Test simple DCP save and load in distributed setting."""
    rank = int(os.environ["RANK"])

    print(f"Running simple DCP save/load test on rank {rank}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_fsdp_engine(alloc_mode, mb_spec, init_optimizer=False)

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "fsdp_engine_simple_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    # Wait for rank 0 to create directory
    dist.barrier()

    seeding.set_random_seed(0, key=f"trainer{rank}")

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))

    engine.save(save_load_meta)

    for p in engine.model.parameters():
        p.data.zero_()

    engine.load(save_load_meta)

    succ = True
    for name, param in engine.model.named_parameters():
        if isinstance(param, DTensor):
            param_after_load = param.to_local()
        else:
            param_after_load = param
        if isinstance(params[name], DTensor):
            param_train = params[name].to_local()
        else:
            param_train = params[name]
        if not torch.allclose(param_after_load, param_train):
            diff = torch.abs(param_train - param_after_load)
            print(
                f"Rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
            )
            succ = False

    assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()

    if rank == 0 and output:
        write_result(output, succ)

    print(f"Simple DCP save/load test completed on rank {rank}")


def test_train_dcp_save_load(alloc_mode: str, output: str | None = None):
    # [FUNCTION DOC]
    # With optimizer: train step 1 on mock data, save (weights+optim, DCP), train step 2
    # and snapshot params, zero params, load, re-run step 2, and assert params match the
    # snapshot (i.e. weights and optimizer state were restored). Barrier/destroy and rank 0
    # writes the result as in the simple test.
    rank = int(os.environ["RANK"])
    print(f"Running train DCP save/load test on rank {rank}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    mb_spec = MicroBatchSpec(max_tokens_per_mb=256)
    engine = make_fsdp_engine(alloc_mode, mb_spec, init_optimizer=True)

    base_dir = tempfile.gettempdir()
    path = os.path.join(base_dir, "fsdp_engine_train_dcp_test")
    if rank == 0:
        os.makedirs(path, exist_ok=True)

    # Wait for rank 0 to create directory
    dist.barrier()

    seeding.set_random_seed(0, key=f"trainer{rank}")

    input_data = mock_input(batch_size=16, max_seqlen=128, device=engine.device)

    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    # Train step 1
    engine.train()
    train_result = engine.train_batch(
        input_data, loss_fn=mock_loss_fn, loss_weight_fn=lambda x: x["cu_seqlens"][-1]
    )
    print(f"Rank {rank} train step 1 result: {train_result}")

    # Save checkpoint
    engine.save(save_load_meta)
    print(f"Rank {rank} checkpoint saved")

    # Train step 2
    train_result = engine.train_batch(
        input_data, loss_fn=mock_loss_fn, loss_weight_fn=lambda x: x["cu_seqlens"][-1]
    )
    print(f"Rank {rank} train step 2 result: {train_result}")

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))

    for p in engine.model.parameters():
        p.data.zero_()

    # Load checkpoint
    engine.load(save_load_meta)
    print(f"Rank {rank} checkpoint loaded")

    # Train step 2 again after load
    engine.train()
    train_result = engine.train_batch(
        input_data, loss_fn=mock_loss_fn, loss_weight_fn=lambda x: x["cu_seqlens"][-1]
    )
    print(f"Rank {rank} train step 2 after load result: {train_result}")

    succ = True
    with torch.no_grad():
        engine.eval()
        for name, param in engine.model.named_parameters():
            if isinstance(param, DTensor):
                param_train_after_load = param.to_local()
            else:
                param_train_after_load = param
            if isinstance(params[name], DTensor):
                param_train = params[name].to_local()
            else:
                param_train = params[name]
            if not torch.allclose(param_train_after_load, param_train):
                diff = torch.abs(param_train - param_train_after_load)
                print(
                    f"Rank {rank} diff of {name}: {diff}, max(diff)={torch.max(diff)} avg(diff)={torch.mean(diff)}, count(diff)={torch.count_nonzero(diff)}"
                )
                succ = False

    assert succ, "Weights should be same after recover"

    current_platform.synchronize()
    dist.barrier()

    engine.destroy()

    if rank == 0 and output:
        write_result(output, succ)

    print(f"Train DCP save/load test completed on rank {rank}")


def main():
    # [FUNCTION DOC]
    # Parses --test_type (simple_dcp_save_load | train_dcp_save_load), --output and
    # --backend (allocation string, default fsdp:d2t1c1) and runs the selected test;
    # raises NotImplementedError for unknown types (unreachable due to choices).
    parser = argparse.ArgumentParser(description="Run FSDP DCP Distributed Test")
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["simple_dcp_save_load", "train_dcp_save_load"],
        default="simple_dcp_save_load",
        help="Type of test to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output result",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp:d2t1c1",
        help="Backend allocation string for the model (e.g., 'fsdp:d2t1c1')",
    )
    args = parser.parse_args()

    print(f"Running {args.test_type} test")

    if args.test_type == "simple_dcp_save_load":
        test_simple_dcp_save_load(alloc_mode=args.backend, output=args.output)
    elif args.test_type == "train_dcp_save_load":
        test_train_dcp_save_load(alloc_mode=args.backend, output=args.output)
    else:
        raise NotImplementedError(f"Test type {args.test_type} not implemented")


if __name__ == "__main__":
    # run with `torchrun` to test with multiple GPUs
    main()
