# [MODULE DOC]
# torchrun script checking ArchonParallelDims device-mesh construction: EP mesh without
# expert TP, 2D ep_tp mesh when etp == tp, and inclusion of the PP dimension. Selected
# with --test_type from TEST_REGISTRY; rank 0 writes "Passed"/"Failed" to --output.
# Checks print diagnostics on rank 0 and accumulate a success flag instead of asserting
# (except world-size preconditions).
"""Distributed tests for ArchonParallelDims mesh building.

These tests require a distributed environment (torchrun) because
init_device_mesh() requires torch.distributed to be initialized.

Run with:
    torchrun --nproc_per_node=2 tests/experimental/archon/torchrun/run_parallel_dims.py \
        --test_type=ep_mesh --output=/tmp/result.out

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_parallel_dims.py \
        --test_type=etp_mesh --output=/tmp/result.out

    torchrun --nproc_per_node=4 tests/experimental/archon/torchrun/run_parallel_dims.py \
        --test_type=pp_mesh --output=/tmp/result.out

Supported test types:
    - ep_mesh: Test EP mesh is 1D when etp=1 (2 GPU)
    - etp_mesh: Test ep_tp mesh is 2D when etp=tp (4 GPU)
    - pp_mesh: Test PP mesh dimension is included (4 GPU)
"""

import argparse

import torch.distributed as dist

from tests.experimental.archon.torchrun.dist_utils import (
    print_rank0,
    write_result,
)

from areal.experimental.models.archon import ArchonParallelDims


def test_ep_mesh_when_etp_disabled(output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Requires world_size 2 (assert). Builds ArchonParallelDims(dp_shard 1, tp 2, cp 1,
    # ep 2, etp 1) and checks ep_enabled True, etp_enabled False, the "ep" mesh exists
    # and is 1D, and the "ep_tp" mesh is None. Barriers, rank 0 writes the result to
    # output if given; returns success.
    """Test that ep mesh is 1D and ep_tp mesh is None when etp=1.

    Configuration: dp_shard=1, tp=2, cp=1, ep=2, etp=1 (2 GPU)
    - EP borrows from dp_shard * cp * tp = 1 * 1 * 2 = 2
    - ep_tp mesh should NOT exist (etp=1)
    - ep mesh should be 1D with size=2
    """
    print_rank0("\n=== EP Mesh Test (etp=1) ===")

    world_size = dist.get_world_size()
    assert world_size == 2, f"This test requires 2 GPUs, got {world_size}"

    dims = ArchonParallelDims(
        dp_shard=1,
        tp=2,
        cp=1,
        ep=2,
        etp=1,
        world_size=world_size,
        device_type="cuda",
    )

    success = True

    # Verify ep_enabled and etp_enabled flags
    if not dims.ep_enabled:
        print_rank0("  FAILED: ep_enabled should be True")
        success = False
    else:
        print_rank0("  ep_enabled: True (correct)")

    if dims.etp_enabled:
        print_rank0("  FAILED: etp_enabled should be False")
        success = False
    else:
        print_rank0("  etp_enabled: False (correct)")

    # Build mesh and verify
    ep_mesh = dims.get_mesh("ep")
    ep_tp_mesh = dims.get_mesh("ep_tp")

    # ep mesh should exist and be 1D
    if ep_mesh is None:
        print_rank0("  FAILED: ep mesh should exist")
        success = False
    else:
        if ep_mesh.ndim != 1:
            print_rank0(f"  FAILED: ep mesh should be 1D, got ndim={ep_mesh.ndim}")
            success = False
        else:
            print_rank0(
                f"  ep mesh: ndim={ep_mesh.ndim}, size={ep_mesh.size()} (correct)"
            )

    # ep_tp mesh should NOT exist when etp=1
    if ep_tp_mesh is not None:
        print_rank0(
            f"  FAILED: ep_tp mesh should be None when etp=1, got ndim={ep_tp_mesh.ndim}"
        )
        success = False
    else:
        print_rank0("  ep_tp mesh: None (correct)")

    if success:
        print_rank0("  ep_mesh_test: PASSED")
    else:
        print_rank0("  ep_mesh_test: FAILED")

    dist.barrier()

    if dist.get_rank() == 0 and output:
        write_result(output, success)

    return success


def test_etp_mesh_is_2d(output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Requires world_size 4. Builds dims (dp_shard 2, tp 2, cp 1, ep 2, etp 2) and
    # checks ep_enabled and etp_enabled, "ep" and "tp" meshes exist and are 1D, and
    # "ep_tp" exists, is 2D, has dim names ("ep", "tp") and supports indexing sub-meshes
    # by name (regression for a placements/ndim mismatch). Barriers, writes result on
    # rank 0, returns success.
    """Test that ep_tp mesh is 2D when etp=tp.

    Configuration: dp_shard=2, tp=2, cp=1, ep=2, etp=2 (4 GPU)
    - EP borrows from dp_shard * cp = 2 * 1 = 2
    - ep_tp mesh should be 2D with dimensions ["ep", "tp"]
    - ep mesh should be 1D with size=2
    - tp mesh should be 1D with size=2

    This tests the fix for the ValueError:
    `placements` must have the same length as `device_mesh.ndim`!
    Found placements length: 2, and device_mesh.ndim: 3.
    """
    print_rank0("\n=== ETP Mesh Test (etp=tp) ===")

    world_size = dist.get_world_size()
    assert world_size == 4, f"This test requires 4 GPUs, got {world_size}"

    dims = ArchonParallelDims(
        dp_shard=2,
        tp=2,
        cp=1,
        ep=2,
        etp=2,
        world_size=world_size,
        device_type="cuda",
    )

    success = True

    # Verify ep_enabled and etp_enabled flags
    if not dims.ep_enabled:
        print_rank0("  FAILED: ep_enabled should be True")
        success = False
    else:
        print_rank0("  ep_enabled: True (correct)")

    if not dims.etp_enabled:
        print_rank0("  FAILED: etp_enabled should be True")
        success = False
    else:
        print_rank0("  etp_enabled: True (correct)")

    # Build mesh and verify
    ep_mesh = dims.get_mesh("ep")
    tp_mesh = dims.get_mesh("tp")
    ep_tp_mesh = dims.get_mesh("ep_tp")

    # ep mesh should exist and be 1D
    if ep_mesh is None:
        print_rank0("  FAILED: ep mesh should exist")
        success = False
    else:
        if ep_mesh.ndim != 1:
            print_rank0(f"  FAILED: ep mesh should be 1D, got ndim={ep_mesh.ndim}")
            success = False
        else:
            print_rank0(
                f"  ep mesh: ndim={ep_mesh.ndim}, size={ep_mesh.size()} (correct)"
            )

    # tp mesh should exist and be 1D
    if tp_mesh is None:
        print_rank0("  FAILED: tp mesh should exist")
        success = False
    else:
        if tp_mesh.ndim != 1:
            print_rank0(f"  FAILED: tp mesh should be 1D, got ndim={tp_mesh.ndim}")
            success = False
        else:
            print_rank0(
                f"  tp mesh: ndim={tp_mesh.ndim}, size={tp_mesh.size()} (correct)"
            )

    # ep_tp mesh should exist and be 2D
    if ep_tp_mesh is None:
        print_rank0("  FAILED: ep_tp mesh should exist when etp=tp")
        success = False
    else:
        if ep_tp_mesh.ndim != 2:
            print_rank0(
                f"  FAILED: ep_tp mesh should be 2D, got ndim={ep_tp_mesh.ndim}"
            )
            success = False
        else:
            print_rank0(
                f"  ep_tp mesh: ndim={ep_tp_mesh.ndim}, "
                f"mesh_dim_names={ep_tp_mesh.mesh_dim_names} (correct)"
            )

            # Verify dimension names are ["ep", "tp"]
            expected_names = ("ep", "tp")
            if ep_tp_mesh.mesh_dim_names != expected_names:
                print_rank0(
                    f"  FAILED: ep_tp mesh dim names should be {expected_names}, "
                    f"got {ep_tp_mesh.mesh_dim_names}"
                )
                success = False
            else:
                print_rank0(f"  ep_tp mesh dim names: {expected_names} (correct)")

            # Verify we can extract submeshes by name
            try:
                ep_submesh = ep_tp_mesh["ep"]
                tp_submesh = ep_tp_mesh["tp"]
                print_rank0(
                    f"  ep_tp['ep']: ndim={ep_submesh.ndim}, size={ep_submesh.size()}"
                )
                print_rank0(
                    f"  ep_tp['tp']: ndim={tp_submesh.ndim}, size={tp_submesh.size()}"
                )
            except Exception as e:
                print_rank0(f"  FAILED: Cannot extract submesh from ep_tp: {e}")
                success = False

    if success:
        print_rank0("  etp_mesh_test: PASSED")
    else:
        print_rank0("  etp_mesh_test: FAILED")

    dist.barrier()

    if dist.get_rank() == 0 and output:
        write_result(output, success)

    return success


def test_pp_mesh(output: str | None = None) -> bool:
    # [FUNCTION DOC]
    # Requires world_size 4. Builds dims (pp 2, dp_shard 1, tp 2, cp 1) and checks
    # pp_enabled, context_and_model_parallel_size == cp*tp*pp, world mesh dim names
    # ("pp", "dp_shard", "cp", "tp"), a 1D pp mesh of size 2, and existence of dp_shard
    # and tp meshes. Barriers, writes result on rank 0, returns success.
    """Test that PP mesh dimension is correctly included.

    Configuration: pp=2, dp_shard=1, tp=2, cp=1 (4 GPU)
    - Mesh dims should be: [pp, dp_shard, cp, tp]
    - pp mesh should exist and be 1D with size=2
    """
    print_rank0("\n=== PP Mesh Test ===")

    world_size = dist.get_world_size()
    assert world_size == 4, f"This test requires 4 GPUs, got {world_size}"

    dims = ArchonParallelDims(
        pp=2,
        dp_shard=1,
        tp=2,
        cp=1,
        world_size=world_size,
        device_type="cuda",
    )

    success = True

    # Verify pp_enabled flag
    if not dims.pp_enabled:
        print_rank0("  FAILED: pp_enabled should be True")
        success = False
    else:
        print_rank0("  pp_enabled: True (correct)")

    # Verify context_and_model_parallel_size
    expected_non_dp_size = dims.cp * dims.tp * dims.pp  # 1 * 2 * 2 = 4
    if dims.context_and_model_parallel_size != expected_non_dp_size:
        print_rank0(
            f"  FAILED: context_and_model_parallel_size should be {expected_non_dp_size}, "
            f"got {dims.context_and_model_parallel_size}"
        )
        success = False
    else:
        print_rank0(
            f"  context_and_model_parallel_size: {dims.context_and_model_parallel_size} (correct)"
        )

    # Build mesh and verify
    world_mesh = dims.world_mesh
    pp_mesh = dims.get_mesh("pp")
    dp_shard_mesh = dims.get_mesh("dp_shard")
    tp_mesh = dims.get_mesh("tp")

    # Verify world mesh has correct dimensions
    expected_dim_names = ("pp", "dp_shard", "cp", "tp")
    if world_mesh.mesh_dim_names != expected_dim_names:
        print_rank0(
            f"  FAILED: world mesh dim names should be {expected_dim_names}, "
            f"got {world_mesh.mesh_dim_names}"
        )
        success = False
    else:
        print_rank0(f"  world mesh dim names: {world_mesh.mesh_dim_names} (correct)")

    # pp mesh should exist and be 1D with size=2
    if pp_mesh is None:
        print_rank0("  FAILED: pp mesh should exist")
        success = False
    else:
        if pp_mesh.ndim != 1:
            print_rank0(f"  FAILED: pp mesh should be 1D, got ndim={pp_mesh.ndim}")
            success = False
        elif pp_mesh.size() != 2:
            print_rank0(f"  FAILED: pp mesh size should be 2, got {pp_mesh.size()}")
            success = False
        else:
            print_rank0(
                f"  pp mesh: ndim={pp_mesh.ndim}, size={pp_mesh.size()} (correct)"
            )

    # dp_shard mesh should exist
    if dp_shard_mesh is None:
        print_rank0("  FAILED: dp_shard mesh should exist")
        success = False
    else:
        print_rank0(
            f"  dp_shard mesh: ndim={dp_shard_mesh.ndim}, size={dp_shard_mesh.size()}"
        )

    # tp mesh should exist
    if tp_mesh is None:
        print_rank0("  FAILED: tp mesh should exist")
        success = False
    else:
        print_rank0(f"  tp mesh: ndim={tp_mesh.ndim}, size={tp_mesh.size()}")

    if success:
        print_rank0("  pp_mesh_test: PASSED")
    else:
        print_rank0("  pp_mesh_test: FAILED")

    dist.barrier()

    if dist.get_rank() == 0 and output:
        write_result(output, success)

    return success


TEST_REGISTRY = {
    "ep_mesh": test_ep_mesh_when_etp_disabled,
    "etp_mesh": test_etp_mesh_is_2d,
    "pp_mesh": test_pp_mesh,
}


def main():
    # [FUNCTION DOC]
    # Parses --test_type (required) and --output, initializes NCCL, sets the CUDA device
    # to the global rank, runs the selected test, barriers and prints a summary; on
    # failure or exception rank 0 writes "Failed" (exceptions are re-raised after
    # printing a traceback). Always destroys the process group.
    parser = argparse.ArgumentParser(description="Parallel Dims Mesh Tests")
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=list(TEST_REGISTRY.keys()),
        help="Type of test to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for test result (Passed/Failed)",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    import torch

    torch.cuda.set_device(rank)

    print_rank0("=" * 60)
    print_rank0(f"Running Parallel Dims Test: {args.test_type}")
    print_rank0("=" * 60)

    try:
        test_fn = TEST_REGISTRY[args.test_type]
        success = test_fn(args.output)

        dist.barrier()

        if success:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Parallel Dims Test {args.test_type}: PASSED")
            print_rank0("=" * 60)
        else:
            print_rank0(f"\n{'=' * 60}")
            print_rank0(f"Parallel Dims Test {args.test_type}: FAILED")
            print_rank0("=" * 60)
            if rank == 0 and args.output:
                write_result(args.output, False)

    except Exception as e:
        print(f"Rank {rank} failed with: {e}")
        import traceback

        traceback.print_exc()
        if rank == 0 and args.output:
            write_result(args.output, False)
        raise

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
