# [MODULE DOC]
# GPU correctness tests for enable_fp8_linear-patched MLPs versus identical BF16
# baselines: forward similarity, backward gradient presence/direction/magnitude, and
# loss decrease over optimizer steps. Skipped unless CUDA with SM90+ and the torchao
# blockwise FP8 prototype are available. Allocates on cuda and prints diagnostic
# metrics.
"""FP8 blockwise linear correctness tests.

Verifies that ``enable_fp8_linear``-patched modules produce correct results
for forward pass, backward pass, and optimizer steps by comparing against
BF16 baselines.

Requires GPU with SM90+ (H100/H800) and torchao.
"""

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import torchao.prototype.blockwise_fp8_training.linear  # noqa: F401

    TORCHAO_FP8_AVAILABLE = True
except ImportError:
    TORCHAO_FP8_AVAILABLE = False

# SM90+ (Hopper) check — FP8 requires H100/H800
SM90_AVAILABLE = False
if CUDA_AVAILABLE:
    major, _ = torch.cuda.get_device_capability()
    SM90_AVAILABLE = major >= 9

_SKIP_REASON = "FP8 correctness tests require CUDA with SM90+ (H100/H800) and torchao"
pytestmark = pytest.mark.skipif(
    not (CUDA_AVAILABLE and SM90_AVAILABLE and TORCHAO_FP8_AVAILABLE),
    reason=_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_model(
    in_features: int = 256, hidden: int = 512, out_features: int = 128
):
    # [FUNCTION DOC]
    # HELPER: Returns (not instantiates) a SimpleMLP class: fc1 in_features->hidden and
    # fc2 hidden->out_features, both bias-free, with ReLU between. Defaults 256/512/128
    # (all 128-aligned).
    """Create a simple 2-layer MLP on meta device for FP8 conversion testing."""

    class SimpleMLP(nn.Module):
        # [CLASS DOC]
        # HELPER: Two-layer bias-free MLP closed over the factory dimensions.
        def __init__(self):
            # [FUNCTION DOC]
            # Creates fc1 and fc2 linears.
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden, bias=False)
            self.fc2 = nn.Linear(hidden, out_features, bias=False)

        def forward(self, x):
            # [FUNCTION DOC]
            # Returns fc2 applied to ReLU of fc1(x).
            return self.fc2(torch.relu(self.fc1(x)))

    return SimpleMLP


def _make_bf16_model(model_cls, device="cuda"):
    # [FUNCTION DOC]
    # HELPER: Instantiates model_cls and moves it to device (default cuda) in bfloat16.
    # Returns the model.
    """Instantiate model in BF16 on device."""
    model = model_cls().to(dtype=torch.bfloat16, device=device)
    return model


def _make_fp8_model(model_cls, device="cuda"):
    # [FUNCTION DOC]
    # HELPER: Instantiates model_cls on meta in bfloat16, applies enable_fp8_linear with
    # no exclusions, materializes with to_empty on device (uninitialized memory) and
    # casts to bfloat16. Returns the model; weights must be filled afterwards.
    """Create model on meta, enable FP8, then materialize on device."""
    with torch.device("meta"):
        model = model_cls().to(dtype=torch.bfloat16)

    from areal.experimental.models.archon.fp8 import enable_fp8_linear

    enable_fp8_linear(model, exclude_fqns=set())

    model = model.to_empty(device=device)
    model.to(dtype=torch.bfloat16)
    return model


def _sync_weights(src_model: nn.Module, dst_model: nn.Module):
    # [FUNCTION DOC]
    # HELPER: Copies every parameter of dst_model whose name exists in src_model from
    # the source tensor, in place under no_grad. Names missing in src are silently
    # skipped.
    """Copy weights from src to dst by name matching."""
    src_sd = dict(src_model.named_parameters())
    for name, param in dst_model.named_parameters():
        if name in src_sd:
            with torch.no_grad():
                param.copy_(src_sd[name])


def _make_paired_models(model_cls, device="cuda"):
    # [FUNCTION DOC]
    # HELPER: Builds a BF16 model and an FP8-patched model of the same class and copies
    # BF16 weights into the FP8 one. Returns (bf16_model, fp8_model).
    """Create BF16 and FP8 models with identical weights."""
    bf16_model = _make_bf16_model(model_cls, device)
    fp8_model = _make_fp8_model(model_cls, device)
    _sync_weights(bf16_model, fp8_model)
    return bf16_model, fp8_model


# =============================================================================
# Test 1: Forward pass — FP8 vs BF16 output comparison
# =============================================================================


class TestFP8ForwardCorrectness:
    # [CLASS DOC]
    # Forward-pass comparisons between FP8 and BF16 models.
    def test_fp8_forward_matches_bf16(self):
        # [FUNCTION DOC]
        # Inputs uniform in [-1,1] of shape 128x256. Prints fc1-level cosine similarity
        # and mean abs diff, then compares full outputs: same shape, no NaN/Inf, cosine
        # similarity above 0.9 and mean abs diff below 5.0.
        bf16_model, fp8_model = _make_paired_models(_make_simple_model())
        x = torch.empty(128, 256, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)

        with torch.no_grad():
            h_bf16 = torch.relu(bf16_model.fc1(x))
            h_fp8 = torch.relu(fp8_model.fc1(x))
            layer1_cos = nn.functional.cosine_similarity(
                h_bf16.flatten().unsqueeze(0).float(),
                h_fp8.flatten().unsqueeze(0).float(),
            ).item()
            layer1_abs = (h_fp8 - h_bf16).abs().mean().item()
            print(f"Layer fc1: cosine_sim={layer1_cos:.4f}, abs_diff={layer1_abs:.4f}")

        out_bf16 = bf16_model(x)
        out_fp8 = fp8_model(x)

        assert out_bf16.shape == out_fp8.shape
        assert not torch.isnan(out_fp8).any(), "FP8 output contains NaN"
        assert not torch.isinf(out_fp8).any(), "FP8 output contains Inf"

        cos_sim = nn.functional.cosine_similarity(
            out_bf16.flatten().unsqueeze(0).float(),
            out_fp8.flatten().unsqueeze(0).float(),
        ).item()
        abs_diff = (out_fp8 - out_bf16).abs().mean().item()
        print(f"Output: cosine_sim={cos_sim:.4f}, abs_diff={abs_diff:.4f}")

        assert cos_sim > 0.9, f"FP8 output direction diverged: cosine_sim={cos_sim:.4f}"
        assert abs_diff < 5.0, f"FP8 absolute diff too large: {abs_diff:.4f}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128])
    def test_fp8_forward_various_batch_sizes(self, batch_size):
        # [FUNCTION DOC]
        # Parametrized batch sizes 1, 4, 16, 128: asserts output shape (batch,128) and
        # finite values from the FP8 model.
        _, fp8_model = _make_paired_models(_make_simple_model())
        x = torch.empty(batch_size, 256, device="cuda", dtype=torch.bfloat16).uniform_(
            -1, 1
        )

        out = fp8_model(x)

        assert out.shape == (batch_size, 128)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize(
        "shape",
        [(128, 256), (1, 128, 256), (2, 64, 256)],
        ids=["2D", "3D-bs1", "3D-bs2"],
    )
    def test_fp8_forward_various_input_shapes(self, shape):
        # [FUNCTION DOC]
        # Parametrized 2D and 3D inputs: asserts matching shapes with BF16, finite
        # output and cosine similarity above 0.9.
        bf16_model, fp8_model = _make_paired_models(_make_simple_model())
        x = torch.empty(*shape, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)

        out_bf16 = bf16_model(x)
        out_fp8 = fp8_model(x)

        assert out_bf16.shape == out_fp8.shape, (
            f"Shape mismatch: bf16={out_bf16.shape}, fp8={out_fp8.shape}"
        )
        assert not torch.isnan(out_fp8).any()
        assert not torch.isinf(out_fp8).any()

        cos_sim = nn.functional.cosine_similarity(
            out_bf16.flatten().unsqueeze(0).float(),
            out_fp8.flatten().unsqueeze(0).float(),
        ).item()
        assert cos_sim > 0.9, f"FP8 cosine_sim too low for shape {shape}: {cos_sim:.4f}"


# =============================================================================
# Test 2: Backward pass — gradient correctness
# =============================================================================


class TestFP8BackwardCorrectness:
    # [CLASS DOC]
    # Backward-pass checks for FP8 layers.
    """Verify FP8 backward pass produces meaningful gradients."""

    def test_fp8_backward_produces_gradients(self):
        # [FUNCTION DOC]
        # Backprops the output sum and asserts every parameter has a non-None, NaN-free,
        # non-zero gradient.
        """FP8 model should produce non-zero gradients."""
        _, fp8_model = _make_paired_models(_make_simple_model())
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        out = fp8_model(x)
        loss = out.sum()
        loss.backward()

        for name, param in fp8_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_fp8_backward_gradient_direction_matches_bf16(self):
        # [FUNCTION DOC]
        # Computes MSE-loss gradients for both models on the same random data; for each
        # shared parameter with non-negligible norms asserts cosine similarity between
        # flattened gradients exceeds 0.5.
        """FP8 gradients should point in similar direction as BF16 gradients.

        If forward is wrong but backward mechanism works, gradients will
        point in a different direction. Cosine similarity should be > 0.5.
        """
        bf16_model, fp8_model = _make_paired_models(_make_simple_model())

        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)

        # BF16 backward
        out_bf16 = bf16_model(x)
        loss_bf16 = ((out_bf16 - target) ** 2).mean()
        loss_bf16.backward()
        bf16_grads = {n: p.grad.clone() for n, p in bf16_model.named_parameters()}

        # FP8 backward
        out_fp8 = fp8_model(x)
        loss_fp8 = ((out_fp8 - target) ** 2).mean()
        loss_fp8.backward()
        fp8_grads = {n: p.grad.clone() for n, p in fp8_model.named_parameters()}

        for name in bf16_grads:
            if name not in fp8_grads:
                continue
            g_bf16 = bf16_grads[name].flatten().float()
            g_fp8 = fp8_grads[name].flatten().float()

            if g_bf16.norm() < 1e-8 or g_fp8.norm() < 1e-8:
                continue

            cosine_sim = torch.nn.functional.cosine_similarity(
                g_bf16.unsqueeze(0), g_fp8.unsqueeze(0)
            ).item()

            assert cosine_sim > 0.5, (
                f"Gradient direction mismatch for {name}: "
                f"cosine_similarity={cosine_sim:.4f} (expect > 0.5). "
                f"BF16 grad norm={g_bf16.norm():.4f}, FP8 grad norm={g_fp8.norm():.4f}"
            )

    def test_fp8_backward_gradient_magnitude_reasonable(self):
        # [FUNCTION DOC]
        # Computes global gradient L2 norms of both models from MSE loss and asserts
        # their max/min ratio is below 10.
        """FP8 gradient magnitude should be within 10x of BF16."""
        bf16_model, fp8_model = _make_paired_models(_make_simple_model())

        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)

        # BF16
        out_bf16 = bf16_model(x)
        ((out_bf16 - target) ** 2).mean().backward()
        bf16_grad_norm = (
            sum(
                p.grad.float().norm() ** 2
                for p in bf16_model.parameters()
                if p.grad is not None
            )
            .sqrt()
            .item()
        )

        # FP8
        out_fp8 = fp8_model(x)
        ((out_fp8 - target) ** 2).mean().backward()
        fp8_grad_norm = (
            sum(
                p.grad.float().norm() ** 2
                for p in fp8_model.parameters()
                if p.grad is not None
            )
            .sqrt()
            .item()
        )

        ratio = max(bf16_grad_norm, fp8_grad_norm) / (
            min(bf16_grad_norm, fp8_grad_norm) + 1e-8
        )
        assert ratio < 10.0, (
            f"Gradient magnitude mismatch: BF16 norm={bf16_grad_norm:.4f}, "
            f"FP8 norm={fp8_grad_norm:.4f}, ratio={ratio:.2f} (expect < 10x)"
        )


# =============================================================================
# Test 3: Training step — loss should decrease
# =============================================================================


class TestFP8TrainingStep:
    # [CLASS DOC]
    # Training-step sanity checks with Adam and cross-entropy over 128 classes.
    def _training_step(self, model, x, labels, lr=1e-3):
        # [FUNCTION DOC]
        # HELPER: Creates a fresh Adam optimizer (lr default 1e-3), sets train mode,
        # computes cross-entropy of model(x) against labels, backprops and steps once.
        # Returns the pre-step loss as a Python float (GPU sync). Mutates model
        # parameters.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        loss_val = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss_val

    def test_bf16_loss_decreases(self):
        # [FUNCTION DOC]
        # Sanity baseline: one step on the BF16 model must lower the loss on the same
        # batch.
        """Sanity check: BF16 model loss should decrease after one step."""
        model_cls = _make_simple_model(in_features=256, hidden=512, out_features=128)
        model = _make_bf16_model(model_cls)

        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (8,), device="cuda")

        loss_before = self._training_step(model, x, labels)

        # Second forward (same data) should have lower loss
        logits = model(x)
        loss_after = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        ).item()

        assert loss_after < loss_before, (
            f"BF16 loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_fp8_loss_decreases(self):
        # [FUNCTION DOC]
        # Same check for the FP8 model; failure means FP8 forward, backward or update
        # propagation is broken.
        """FP8 model loss should decrease after one step (key test).

        If this fails, it means the optimizer step is not effective —
        either forward logits are wrong, backward gradients are wrong,
        or the parameter update doesn't propagate through FP8 layers.
        """
        model_cls = _make_simple_model(in_features=256, hidden=512, out_features=128)
        _, fp8_model = _make_paired_models(model_cls)

        x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (8,), device="cuda")

        loss_before = self._training_step(fp8_model, x, labels)

        logits = fp8_model(x)
        loss_after = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        ).item()

        assert loss_after < loss_before, (
            f"FP8 loss did not decrease: {loss_before:.4f} → {loss_after:.4f}. "
            f"This means FP8 training is fundamentally broken — the optimizer "
            f"step did not improve the model."
        )

    def test_fp8_multi_step_convergence(self):
        # [FUNCTION DOC]
        # Runs 10 Adam steps on a fixed 16-sample batch with the FP8 model and asserts
        # the last loss is below the first.
        """FP8 model should converge over multiple steps on a tiny dataset."""
        model_cls = _make_simple_model(in_features=256, hidden=512, out_features=128)
        _, fp8_model = _make_paired_models(model_cls)
        optimizer = torch.optim.Adam(fp8_model.parameters(), lr=1e-3)

        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (16,), device="cuda")

        losses = []
        for _ in range(10):
            fp8_model.train()
            logits = fp8_model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease overall (first > last)
        assert losses[-1] < losses[0], (
            f"FP8 did not converge over 10 steps: "
            f"first_loss={losses[0]:.4f}, last_loss={losses[-1]:.4f}. "
            f"Loss trajectory: {[f'{v:.4f}' for v in losses]}"
        )
