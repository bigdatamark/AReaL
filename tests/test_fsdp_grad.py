# [MODULE DOC]
# Unit tests for FSDP2 gradient utilities in areal.engine.fsdp_utils.grad:
# to_local_if_dtensor, get_main_grads_for_grad_norm, get_grad_norm_fp32 (with
# torch.distributed.all_reduce patched out, so results equal local norms) and
# clip_grad_by_total_norm_fp32 (scales grads by max_norm/(total_norm+1e-6) when the norm
# exceeds max_norm), on CPU (offload path) and CUDA (when available).
"""Tests for fsdp2_clip_grad_norm and related gradient utilities."""

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from areal.engine.fsdp_utils.grad import (
    clip_grad_by_total_norm_fp32,
    get_grad_norm_fp32,
    get_main_grads_for_grad_norm,
    to_local_if_dtensor,
)

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()


class SimpleModel(nn.Module):
    # [CLASS DOC]
    # HELPER: Two-layer linear model (hidden_size x hidden_size) used to produce
    # parameters with gradients.
    """A simple model for testing gradient clipping."""

    def __init__(self, hidden_size: int = 64):
        # [FUNCTION DOC]
        # HELPER: Creates fc1 and fc2 Linear layers of the given width.
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # [FUNCTION DOC]
        # HELPER: Applies fc1 then fc2.
        return self.fc2(self.fc1(x))


def create_model_with_grads(
    hidden_size: int = 64,
    device: str = "cpu",
    grad_scale: float = 1.0,
) -> nn.Module:
    # [FUNCTION DOC]
    # HELPER: Builds a SimpleModel on device, runs a backward pass of the output sum on
    # a random 4xhidden input, multiplies all grads in place by grad_scale, and returns
    # the model.
    """Create a model and populate gradients for testing."""
    model = SimpleModel(hidden_size).to(device)

    # Create dummy input and compute gradients
    x = torch.randn(4, hidden_size, device=device)
    loss = model(x).sum()
    loss.backward()

    # Scale gradients for controlled testing
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(grad_scale)

    return model


def compute_expected_grad_norm(
    parameters: list[nn.Parameter],
    norm_type: float,
) -> float:
    # [FUNCTION DOC]
    # HELPER: Reference grad norm over params with grads: max absolute value for inf
    # norm, otherwise the p-th root of summed per-tensor norms to the p-th power.
    # Returns 0.0 if no grads. Causes GPU sync via item().
    """Compute expected gradient norm using PyTorch's clip_grad_norm_ logic."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    if norm_type == float("inf"):
        return max(g.abs().max().item() for g in grads)
    else:
        total_norm = sum(g.norm(norm_type).item() ** norm_type for g in grads)
        return total_norm ** (1.0 / norm_type)


class TestToLocalIfDtensor:
    # [CLASS DOC]
    # Groups to_local_if_dtensor tests.
    """Tests for to_local_if_dtensor utility."""

    def test_regular_tensor_unchanged(self):
        # [FUNCTION DOC]
        # Asserts a regular tensor is returned as the same object.
        tensor = torch.randn(4, 4)
        result = to_local_if_dtensor(tensor)
        assert result is tensor

    def test_cpu_tensor(self):
        # [FUNCTION DOC]
        # Asserts an explicit CPU tensor is returned as the same object.
        tensor = torch.randn(4, 4, device="cpu")
        result = to_local_if_dtensor(tensor)
        assert result is tensor


class TestGetMainGradsForGradNorm:
    # [CLASS DOC]
    # Groups get_main_grads_for_grad_norm tests.
    """Tests for get_main_grads_for_grad_norm."""

    def test_returns_grads_for_rank_0(self):
        # [FUNCTION DOC]
        # Asserts on TP rank 0 one grad is returned per parameter that has a grad.
        model = create_model_with_grads()
        grads = get_main_grads_for_grad_norm(
            list(model.parameters()), tensor_parallel_rank=0
        )
        expected_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert len(grads) == expected_count

    def test_filters_none_grads(self):
        # [FUNCTION DOC]
        # Asserts parameters with no grads yield an empty list.
        model = SimpleModel()
        # No backward called, grads are None
        grads = get_main_grads_for_grad_norm(
            list(model.parameters()), tensor_parallel_rank=0
        )
        assert len(grads) == 0


class TestGetGradNormFp32:
    # [CLASS DOC]
    # Groups get_grad_norm_fp32 tests.
    """Tests for get_grad_norm_fp32."""

    @pytest.fixture
    def mock_process_groups(self):
        # [FUNCTION DOC]
        # HELPER: Fixture returning two MagicMock process groups (dp, mp).
        """Create mock process groups for testing."""
        dp_group = MagicMock()
        mp_group = MagicMock()
        return dp_group, mp_group

    def test_empty_grads_returns_zero(self, mock_process_groups):
        # [FUNCTION DOC]
        # Asserts empty grads return 0.0 yet still call all_reduce twice (dp and mp) so
        # peers do not hang.
        # Empty grads must still participate in all_reduce (e.g. LoRA frozen ranks)
        # so that ranks with real grads don't hang.
        dp_group, mp_group = mock_process_groups
        with patch("torch.distributed.all_reduce") as mock_allreduce:
            result = get_grad_norm_fp32([], dp_group, mp_group)
        assert result == 0.0
        assert mock_allreduce.call_count == 2  # called for dp_group and mp_group

    def test_empty_grads_participates_in_allreduce_no_groups(self):
        # [FUNCTION DOC]
        # Asserts empty grads with no groups return 0.0.
        # With no process groups, empty grads should still return 0.0 without hanging.
        result = get_grad_norm_fp32([], None, None)
        assert result == 0.0

    @pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
    def test_norm_types_cpu_offload(self, mock_process_groups, norm_type):
        # [FUNCTION DOC]
        # Parametrized over norm types 1, 2, 3, inf with CPU grads and offload: asserts
        # result matches the reference norm within 1e-4.
        """Test different norm types with CPU tensors (offload_params=True)."""
        dp_group, mp_group = mock_process_groups
        grads = [torch.randn(4, 4, device="cpu") for _ in range(3)]

        with patch("torch.distributed.all_reduce"):
            result = get_grad_norm_fp32(
                grads, dp_group, mp_group, norm_type=norm_type, offload_params=True
            )

        # Compute expected
        if norm_type == float("inf"):
            expected = max(g.abs().max().item() for g in grads)
        else:
            total = sum(g.norm(norm_type).item() ** norm_type for g in grads)
            expected = total ** (1.0 / norm_type)

        assert abs(result - expected) < 1e-4, (
            f"norm_type={norm_type}: {result} != {expected}"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    @pytest.mark.parametrize("norm_type", [1.0, 2.0, 3.0, float("inf")])
    def test_norm_types_gpu_no_offload(self, mock_process_groups, norm_type):
        # [FUNCTION DOC]
        # CUDA only: same norm-type check with GPU grads and no offload.
        """Test different norm types with GPU tensors (offload_params=False)."""
        dp_group, mp_group = mock_process_groups
        grads = [torch.randn(4, 4, device="cuda") for _ in range(3)]

        with patch("torch.distributed.all_reduce"):
            result = get_grad_norm_fp32(
                grads, dp_group, mp_group, norm_type=norm_type, offload_params=False
            )

        # Compute expected
        if norm_type == float("inf"):
            expected = max(g.abs().max().item() for g in grads)
        else:
            total = sum(g.norm(norm_type).item() ** norm_type for g in grads)
            expected = total ** (1.0 / norm_type)

        assert abs(result - expected) < 1e-4, (
            f"norm_type={norm_type}: {result} != {expected}"
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_offload_vs_non_offload_consistency_l2(self, mock_process_groups):
        # [FUNCTION DOC]
        # CUDA only: asserts offload (CPU) and GPU L2 norms of identical grads agree
        # within 1e-4.
        """Test that offload and non-offload paths produce consistent L2 norm results."""
        dp_group, mp_group = mock_process_groups

        # Use same random seed to create identical gradients
        torch.manual_seed(42)
        grads_cpu = [torch.randn(4, 4, device="cpu") for _ in range(3)]
        grads_gpu = [g.clone().cuda() for g in grads_cpu]

        with patch("torch.distributed.all_reduce"):
            result_offload = get_grad_norm_fp32(
                grads_cpu, dp_group, mp_group, norm_type=2.0, offload_params=True
            )
            result_no_offload = get_grad_norm_fp32(
                grads_gpu, dp_group, mp_group, norm_type=2.0, offload_params=False
            )

        assert abs(result_offload - result_no_offload) < 1e-4

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_offload_vs_non_offload_consistency_inf(self, mock_process_groups):
        # [FUNCTION DOC]
        # CUDA only: asserts offload and GPU inf norms agree within 1e-6.
        """Test that offload and non-offload paths produce consistent inf norm results."""
        dp_group, mp_group = mock_process_groups

        torch.manual_seed(42)
        grads_cpu = [torch.randn(4, 4, device="cpu") for _ in range(3)]
        grads_gpu = [g.clone().cuda() for g in grads_cpu]

        with patch("torch.distributed.all_reduce"):
            result_offload = get_grad_norm_fp32(
                grads_cpu,
                dp_group,
                mp_group,
                norm_type=float("inf"),
                offload_params=True,
            )
            result_no_offload = get_grad_norm_fp32(
                grads_gpu,
                dp_group,
                mp_group,
                norm_type=float("inf"),
                offload_params=False,
            )

        assert abs(result_offload - result_no_offload) < 1e-6

    def test_single_tensor_input(self, mock_process_groups):
        # [FUNCTION DOC]
        # Asserts a single tensor (not a list) is accepted and yields its L2 norm.
        """Test that a single tensor input works correctly."""
        dp_group, mp_group = mock_process_groups
        grad = torch.randn(4, 4, device="cpu")

        with patch("torch.distributed.all_reduce"):
            result = get_grad_norm_fp32(
                grad, dp_group, mp_group, norm_type=2.0, offload_params=True
            )

        expected = grad.norm(2.0).item()
        assert abs(result - expected) < 1e-5

    def test_no_data_parallel_group(self, mock_process_groups):
        # [FUNCTION DOC]
        # Asserts a None data-parallel group is tolerated and yields the local L2 norm.
        """Test when data_parallel_group is None."""
        _, mp_group = mock_process_groups
        grads = [torch.randn(4, 4, device="cpu")]

        with patch("torch.distributed.all_reduce"):
            result = get_grad_norm_fp32(
                grads,
                None,  # type: ignore[arg-type]
                mp_group,
                norm_type=2.0,
                offload_params=True,
            )

        expected = grads[0].norm(2.0).item()
        assert abs(result - expected) < 1e-5


class TestClipGradByTotalNormFp32:
    # [CLASS DOC]
    # Groups clip_grad_by_total_norm_fp32 tests.
    """Tests for clip_grad_by_total_norm_fp32."""

    def test_no_clipping_when_norm_below_max(self):
        # [FUNCTION DOC]
        # Asserts grads are unchanged when the given total norm is below max_norm.
        """Test that gradients are not modified when norm is below max_norm."""
        model = create_model_with_grads(device="cpu", grad_scale=0.1)
        original_grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        # Set max_norm much higher than actual norm
        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=1000.0,
            total_norm=0.5,
            offload_params=True,
        )

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, original_grads[name])

    def test_clipping_reduces_grad_magnitude_cpu(self):
        # [FUNCTION DOC]
        # CPU: asserts clipping with total norm above max_norm reduces the grad norm.
        """Test that clipping reduces gradient magnitude when norm exceeds max (CPU)."""
        model = create_model_with_grads(device="cpu", grad_scale=10.0)

        original_norm = compute_expected_grad_norm(
            list(model.parameters()), norm_type=2.0
        )
        max_norm = 1.0

        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=max_norm,
            total_norm=original_norm,
            offload_params=True,
        )

        # Compute new norm
        new_norm = compute_expected_grad_norm(list(model.parameters()), norm_type=2.0)
        assert new_norm < original_norm

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_clipping_reduces_grad_magnitude_gpu(self):
        # [FUNCTION DOC]
        # CUDA only: same reduction check on GPU.
        """Test that clipping reduces gradient magnitude when norm exceeds max (GPU)."""
        model = create_model_with_grads(device="cuda", grad_scale=10.0)

        original_norm = compute_expected_grad_norm(
            list(model.parameters()), norm_type=2.0
        )
        max_norm = 1.0

        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=max_norm,
            total_norm=original_norm,
            offload_params=False,
        )

        # Compute new norm
        new_norm = compute_expected_grad_norm(list(model.parameters()), norm_type=2.0)
        assert new_norm < original_norm

    def test_clip_coefficient_applied_correctly_cpu(self):
        # [FUNCTION DOC]
        # CPU: asserts every grad is scaled by exactly max_norm/(total_norm+1e-6).
        """Test that the clip coefficient is applied correctly (CPU)."""
        model = create_model_with_grads(device="cpu", grad_scale=1.0)
        original_grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        total_norm = 10.0
        max_norm = 1.0
        expected_scale = max_norm / (total_norm + 1e-6)

        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=max_norm,
            total_norm=total_norm,
            offload_params=True,
        )

        for name, param in model.named_parameters():
            if param.grad is not None:
                expected = original_grads[name] * expected_scale
                assert torch.allclose(param.grad, expected, atol=1e-6)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_clip_coefficient_applied_correctly_gpu(self):
        # [FUNCTION DOC]
        # CUDA only: same scaling check on GPU.
        """Test that the clip coefficient is applied correctly (GPU)."""
        model = create_model_with_grads(device="cuda", grad_scale=1.0)
        original_grads = {
            name: param.grad.clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

        total_norm = 10.0
        max_norm = 1.0
        expected_scale = max_norm / (total_norm + 1e-6)

        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=max_norm,
            total_norm=total_norm,
            offload_params=False,
        )

        for name, param in model.named_parameters():
            if param.grad is not None:
                expected = original_grads[name] * expected_scale
                assert torch.allclose(param.grad, expected, atol=1e-6)

    def test_empty_parameters(self):
        # [FUNCTION DOC]
        # Asserts an empty parameter list does not raise.
        """Test with empty parameter list."""
        # Should not raise
        clip_grad_by_total_norm_fp32(
            [], max_norm=1.0, total_norm=10.0, offload_params=True
        )

    def test_parameters_without_grads(self):
        # [FUNCTION DOC]
        # Asserts parameters without grads do not raise.
        """Test with parameters that have no gradients."""
        model = SimpleModel()  # No backward called
        # Should not raise
        clip_grad_by_total_norm_fp32(
            list(model.parameters()),
            max_norm=1.0,
            total_norm=10.0,
            offload_params=True,
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_offload_vs_non_offload_clipping_consistency(self):
        # [FUNCTION DOC]
        # CUDA only: clips identical CPU and GPU grad copies and asserts the results
        # match within 1e-6.
        """Test that CPU and GPU paths produce same clipping results."""
        torch.manual_seed(42)

        # Create model on CPU first, then copy to GPU to ensure identical grads
        model_cpu = create_model_with_grads(device="cpu", grad_scale=5.0)
        model_gpu = copy.deepcopy(model_cpu).cuda()
        for p_cpu, p_gpu in zip(model_cpu.parameters(), model_gpu.parameters()):
            if p_cpu.grad is not None:
                p_gpu.grad = p_cpu.grad.clone().cuda()

        total_norm = 10.0
        max_norm = 1.0

        clip_grad_by_total_norm_fp32(
            list(model_cpu.parameters()),
            max_norm=max_norm,
            total_norm=total_norm,
            offload_params=True,
        )

        clip_grad_by_total_norm_fp32(
            list(model_gpu.parameters()),
            max_norm=max_norm,
            total_norm=total_norm,
            offload_params=False,
        )

        # Compare results
        for (name_cpu, param_cpu), (name_gpu, param_gpu) in zip(
            model_cpu.named_parameters(), model_gpu.named_parameters()
        ):
            if param_cpu.grad is not None:
                assert torch.allclose(
                    param_cpu.grad, param_gpu.grad.cpu(), atol=1e-6
                ), f"Mismatch in {name_cpu}"


class TestIntegration:
    # [CLASS DOC]
    # Groups combined norm-then-clip workflow tests.
    """Integration tests combining norm computation and clipping."""

    def test_compute_and_clip_workflow_cpu(self):
        # [FUNCTION DOC]
        # CPU: computes the norm then clips to 1.0; if clipping applied, asserts the new
        # norm equals total_norm times the clip ratio.
        """Test the complete workflow of computing norm and clipping (CPU)."""
        model = create_model_with_grads(device="cpu", grad_scale=5.0)
        params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]

        dp_group = MagicMock()
        mp_group = MagicMock()

        with patch("torch.distributed.all_reduce"):
            # Compute norm
            total_norm = get_grad_norm_fp32(
                grads, dp_group, mp_group, norm_type=2.0, offload_params=True
            )

            # Clip gradients
            max_norm = 1.0
            clip_grad_by_total_norm_fp32(
                params,
                max_norm=max_norm,
                total_norm=total_norm,
                offload_params=True,
            )

        # Verify clipping happened
        new_norm = compute_expected_grad_norm(params, norm_type=2.0)
        if total_norm > max_norm:
            expected_ratio = max_norm / (total_norm + 1e-6)
            assert abs(new_norm - total_norm * expected_ratio) < 1e-4

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_compute_and_clip_workflow_gpu(self):
        # [FUNCTION DOC]
        # CUDA only: same workflow on GPU without offload.
        """Test the complete workflow of computing norm and clipping (GPU)."""
        model = create_model_with_grads(device="cuda", grad_scale=5.0)
        params = list(model.parameters())
        grads = [p.grad for p in params if p.grad is not None]

        dp_group = MagicMock()
        mp_group = MagicMock()

        with patch("torch.distributed.all_reduce"):
            # Compute norm
            total_norm = get_grad_norm_fp32(
                grads, dp_group, mp_group, norm_type=2.0, offload_params=False
            )

            # Clip gradients
            max_norm = 1.0
            clip_grad_by_total_norm_fp32(
                params,
                max_norm=max_norm,
                total_norm=total_norm,
                offload_params=False,
            )

        # Verify clipping happened
        new_norm = compute_expected_grad_norm(params, norm_type=2.0)
        if total_norm > max_norm:
            expected_ratio = max_norm / (total_norm + 1e-6)
            assert abs(new_norm - total_norm * expected_ratio) < 1e-4

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cpu_gpu_consistency_end_to_end(self):
        # [FUNCTION DOC]
        # CUDA only: asserts CPU and GPU norms of identical grads agree within 1e-3 and
        # match the reference norm within 1e-4.
        """Test that CPU and GPU paths produce consistent results end-to-end."""
        torch.manual_seed(42)
        # Create model on CPU first, then copy to GPU to ensure identical grads
        model_cpu = create_model_with_grads(device="cpu", grad_scale=5.0)
        model_gpu = copy.deepcopy(model_cpu).cuda()
        for p_cpu, p_gpu in zip(model_cpu.parameters(), model_gpu.parameters()):
            if p_cpu.grad is not None:
                p_gpu.grad = p_cpu.grad.clone().cuda()

        dp_group = MagicMock()
        mp_group = MagicMock()

        with patch("torch.distributed.all_reduce"):
            grads_cpu = [p.grad for p in model_cpu.parameters() if p.grad is not None]
            grads_gpu = [p.grad for p in model_gpu.parameters() if p.grad is not None]

            norm_cpu = get_grad_norm_fp32(
                grads_cpu, dp_group, mp_group, norm_type=2.0, offload_params=True
            )
            norm_gpu = get_grad_norm_fp32(
                grads_gpu, dp_group, mp_group, norm_type=2.0, offload_params=False
            )

        # Allow slightly larger tolerance for CPU vs GPU floating point differences
        assert abs(norm_cpu - norm_gpu) < 1e-3, f"{norm_cpu} != {norm_gpu}"

        # Also verify against independent computation
        expected_norm = compute_expected_grad_norm(
            list(model_cpu.parameters()), norm_type=2.0
        )
        assert abs(norm_cpu - expected_norm) < 1e-4
