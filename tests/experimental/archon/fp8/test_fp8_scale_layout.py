# [MODULE DOC]
# Tests that torchao blockwise FP8 kernels used by enable_fp8_linear produce scale
# tensors with the shapes and column-major strides torch._scaled_mm expects. Skipped
# unless CUDA SM90+ and torchao FP8 are available.
import pytest
import torch
from torch import nn

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import torchao.prototype.blockwise_fp8_training.linear  # noqa: F401

    TORCHAO_FP8_AVAILABLE = True
except ImportError:
    TORCHAO_FP8_AVAILABLE = False

SM90_AVAILABLE = False
if CUDA_AVAILABLE:
    major, _ = torch.cuda.get_device_capability()
    SM90_AVAILABLE = major >= 9

pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and SM90_AVAILABLE and TORCHAO_FP8_AVAILABLE),
    reason="FP8 scale layout tests require CUDA with SM90+ and torchao",
)


def _make_simple_model(
    in_features: int = 256, hidden: int = 512, out_features: int = 128
):
    # [FUNCTION DOC]
    # HELPER: Returns a SimpleMLP class (fc1 in->hidden, ReLU, fc2 hidden->out, no
    # bias); defaults 256/512/128.
    class SimpleMLP(nn.Module):
        # [CLASS DOC]
        # HELPER: Two-layer bias-free MLP closed over the factory dims.
        def __init__(self):
            # [FUNCTION DOC]
            # Creates fc1 and fc2.
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden, bias=False)
            self.fc2 = nn.Linear(hidden, out_features, bias=False)

        def forward(self, x):
            # [FUNCTION DOC]
            # Returns fc2(relu(fc1(x))).
            return self.fc2(torch.relu(self.fc1(x)))

    return SimpleMLP


def _make_fp8_model(model_cls, device="cuda"):
    # [FUNCTION DOC]
    # HELPER: Builds the model on meta in bfloat16, applies enable_fp8_linear with no
    # exclusions, materializes uninitialized on device (default cuda) and casts to
    # bfloat16. Weights are random garbage, which is fine for layout tests.
    with torch.device("meta"):
        model = model_cls().to(dtype=torch.bfloat16)

    from areal.experimental.models.archon.fp8 import enable_fp8_linear

    enable_fp8_linear(model, exclude_fqns=set())

    model = model.to_empty(device=device)
    model.to(dtype=torch.bfloat16)
    return model


class TestTritonScaleLayoutCorrectness:
    # [CLASS DOC]
    # Checks FP8 forward runs and quantization scale layouts.
    def test_fp8_forward_no_error(self):
        # [FUNCTION DOC]
        # Runs a 128x256 forward; converts a RuntimeError containing "Invalid scaling
        # configuration" into a test failure explaining a torchao stride
        # incompatibility, re-raises other errors; asserts an output exists.
        model_cls = _make_simple_model(in_features=256, hidden=512, out_features=128)
        fp8_model = _make_fp8_model(model_cls)
        x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

        try:
            out = fp8_model(x)
        except RuntimeError as e:
            if "Invalid scaling configuration" in str(e):
                pytest.fail(
                    f"torch._scaled_mm rejected scale tensor strides. "
                    f"This is a torchao compatibility issue.\n"
                    f"Error: {e}"
                )
            raise

        assert out is not None

    @pytest.mark.parametrize("m", [1, 64, 128, 256])
    def test_fp8_various_m_dims(self, m):
        # [FUNCTION DOC]
        # Parametrized M in 1, 64, 128, 256: same error handling as above and asserts
        # output shape (M,128).
        model_cls = _make_simple_model(in_features=256, hidden=512, out_features=128)
        fp8_model = _make_fp8_model(model_cls)
        x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)

        try:
            out = fp8_model(x)
        except RuntimeError as e:
            if "Invalid scaling configuration" in str(e):
                pytest.fail(f"FP8 forward failed for M={m}: {e}")
            raise

        assert out.shape == (m, 128)

    def test_fp8_blockwise_mm_scale_layout(self):
        # [FUNCTION DOC]
        # Imports the triton activation (lhs) and transposed weight (rhs) quant kernels
        # (skips if missing). Quantizes x [128,256] and w [512,256] with block 128 and
        # asserts activation scale shape (m, k_blocks), weight scale shape (k_blocks,
        # n_blocks), and that both scales are column-major (stride 0 smaller than stride
        # 1).
        try:
            from torchao.prototype.blockwise_fp8_training.kernels import (
                triton_fp8_blockwise_act_quant_lhs,
                triton_fp8_blockwise_weight_quant_transposed_rhs,
            )
        except ImportError:
            pytest.skip("torchao FP8 kernels not available")

        block_size = 128
        m, k, n = 128, 256, 512
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        x_fp8, x_scale = triton_fp8_blockwise_act_quant_lhs(x, block_size)
        w_t_fp8, w_t_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
            w, block_size=block_size
        )

        k_blocks = (k + block_size - 1) // block_size
        n_blocks = (n + block_size - 1) // block_size

        assert x_scale.shape == (m, k_blocks), (
            f"scale_a shape: got {tuple(x_scale.shape)}, expected ({m}, {k_blocks})"
        )
        assert w_t_scale.shape == (k_blocks, n_blocks), (
            f"scale_b shape: got {tuple(w_t_scale.shape)}, expected ({k_blocks}, {n_blocks})"
        )

        assert x_scale.stride(0) < x_scale.stride(1), (
            f"scale_a not column-major: stride={x_scale.stride()}"
        )
        assert w_t_scale.stride(0) < w_t_scale.stride(1), (
            f"scale_b not column-major: stride={w_t_scale.stride()}"
        )
