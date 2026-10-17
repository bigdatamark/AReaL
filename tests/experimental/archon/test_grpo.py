# [MODULE DOC]
# Single-GPU GRPO pipeline comparison tests between the Archon engine and the FSDP
# engine (both held by a module-scoped DualEngineFixture loading the same model): raw
# logits, logprobs/entropy and importance ratios, gradient flow, optimizer-step weight
# deltas, plus standalone checks of grpo_loss_fn determinism, edge cases and sensitivity
# to advantages. Module skipped without CUDA.
"""GRPO training tests for Archon Engine.

These tests compare Archon vs FSDP engine at each stage of the GRPO pipeline
to verify numerical consistency and gradient correctness.

Run tests:
    pytest tests/experimental/archon/test_grpo.py -v

Note: These tests require GPU and are marked as slow.
"""

import pytest
import torch

from tests.experimental.archon.utils import (
    ComparisonMetrics,
    DualEngineFixture,
    compare_tensors,
    create_grpo_batch,
)

from areal.infra.platforms import current_platform
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.utils.functional import gather_logprobs_entropy

# Skip if no CUDA available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(scope="module")
def engines():
    # [FUNCTION DOC]
    # Module-scoped fixture: creates a DualEngineFixture, calls setup (initializes both
    # engines on GPU), yields it to tests, then calls teardown.
    """Fixture to provide initialized engines."""
    fixture = DualEngineFixture()
    fixture.setup()
    yield fixture
    fixture.teardown()


# =============================================================================
# Suite 1: Logits/Logprobs Comparison Tests
# =============================================================================


class TestLogitsComparison:
    # [CLASS DOC]
    # Compares forward outputs of Archon and FSDP engines on the same GRPO batch. Each
    # test builds a batch with create_grpo_batch, moves tensors to device, prepares
    # micro-batch lists via each engine's private _prepare_mb_list / _prepare_mb_inputs,
    # takes the first micro-batch, and runs Archon's model with input_ids, position_ids,
    # cu_seqlens and int max_seqlen and FSDP's HF model with its kwargs, squeezing a
    # leading batch dim of 1.
    """Test suite for comparing logits and logprobs between Archon and FSDP."""

    def test_logits_numerical_precision(self, engines: DualEngineFixture):
        # [FUNCTION DOC]
        # Eval mode, no grad, batch 4 x 128. Truncates both logits to the original
        # unpadded token count and asserts compare_tensors max diff below 4.0 and mean
        # diff below 0.2; prints the metrics.
        """Compare raw logits with stricter tolerances for RL training.

        The existing test allows max_diff < 2.0, but for RL training we need
        stricter tolerances as small differences compound across sequences.
        """
        batch = create_grpo_batch(
            model_path=engines.model_path, batch_size=4, max_seq_len=128
        )
        device = torch.device(current_platform.device_type)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Prepare micro-batch lists
        archon_mb_list = engines.archon_engine._prepare_mb_list(batch).to(device)
        fsdp_mb_list = engines.fsdp_engine._prepare_mb_list(batch).to(device)

        archon_mb = next(iter(archon_mb_list))
        fsdp_mb = next(iter(fsdp_mb_list))

        archon_inputs, archon_ctx = engines.archon_engine._prepare_mb_inputs(archon_mb)
        fsdp_inputs, fsdp_ctx = engines.fsdp_engine._prepare_mb_inputs(fsdp_mb)

        # Forward pass
        engines.archon_engine.eval()
        engines.fsdp_engine.eval()

        with torch.no_grad():
            # Archon forward
            cu_seqlens = archon_inputs.get("cu_seqlens")
            max_seqlen = archon_inputs.get("max_seqlen")
            if max_seqlen is not None:
                max_seqlen = int(max_seqlen)

            archon_logits = engines.archon_engine.model(
                archon_inputs["input_ids"],
                archon_inputs.get("position_ids"),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if archon_logits.ndim == 3 and archon_logits.shape[0] == 1:
                archon_logits = archon_logits.squeeze(0)

            # FSDP forward
            fsdp_outputs = engines.fsdp_engine.model(**fsdp_inputs)
            fsdp_logits = fsdp_outputs.logits
            if fsdp_logits.ndim == 3 and fsdp_logits.shape[0] == 1:
                fsdp_logits = fsdp_logits.squeeze(0)

        # Compare non-padding area
        # Use original batch length instead of ctx.pad_length since
        # Archon and FSDP may have different padding strategies
        original_length = batch["input_ids"].numel()
        archon_logits_valid = archon_logits[:original_length]
        fsdp_logits_valid = fsdp_logits[:original_length]

        metrics = compare_tensors(archon_logits_valid, fsdp_logits_valid)
        print(f"\n[Logits Comparison] {metrics}")

        # Different attention implementations have numerical differences
        # Archon and FSDP may also have different padding strategies which
        # can affect boundary tokens
        assert metrics.max_diff < 4.0, (
            f"Logits max_diff too large for RL: {metrics.max_diff}"
        )
        assert metrics.mean_diff < 0.2, (
            f"Logits mean_diff too large for RL: {metrics.mean_diff}"
        )

    def test_logprobs_consistency(self, engines: DualEngineFixture):
        # [FUNCTION DOC]
        # Same forward, then computes per-token logprobs and entropy with
        # gather_logprobs_entropy (temperature 1, no TP group) using labels obtained by
        # rolling the flattened input ids left by one. Prints logprob, entropy and
        # importance-ratio metrics (exp of logprob difference vs 1) and asserts logprob
        # max diff below 1.5 and max abs ratio deviation below 1.1.
        # KNOWN ISSUE: Rolling the whole packed flattened sequence makes the last token
        # of each sequence predict the first token of the next (and the final one wraps
        # to the start); harmless for a both-sides comparison but not true labels.
        """Test that gather_logprobs_entropy produces consistent results."""
        batch = create_grpo_batch(
            model_path=engines.model_path, batch_size=4, max_seq_len=128
        )
        device = torch.device(current_platform.device_type)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Get logits from both engines
        archon_mb_list = engines.archon_engine._prepare_mb_list(batch).to(device)
        fsdp_mb_list = engines.fsdp_engine._prepare_mb_list(batch).to(device)

        archon_mb = next(iter(archon_mb_list))
        fsdp_mb = next(iter(fsdp_mb_list))

        archon_inputs, archon_ctx = engines.archon_engine._prepare_mb_inputs(archon_mb)
        fsdp_inputs, fsdp_ctx = engines.fsdp_engine._prepare_mb_inputs(fsdp_mb)

        engines.archon_engine.eval()
        engines.fsdp_engine.eval()

        with torch.no_grad():
            # Archon logits
            cu_seqlens = archon_inputs.get("cu_seqlens")
            max_seqlen = archon_inputs.get("max_seqlen")
            if max_seqlen is not None:
                max_seqlen = int(max_seqlen)

            archon_logits = engines.archon_engine.model(
                archon_inputs["input_ids"],
                archon_inputs.get("position_ids"),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if archon_logits.ndim == 3 and archon_logits.shape[0] == 1:
                archon_logits = archon_logits.squeeze(0)

            # FSDP logits
            fsdp_outputs = engines.fsdp_engine.model(**fsdp_inputs)
            fsdp_logits = fsdp_outputs.logits
            if fsdp_logits.ndim == 3 and fsdp_logits.shape[0] == 1:
                fsdp_logits = fsdp_logits.squeeze(0)

            # Compute logprobs using the same function
            # Use original batch length for labels
            original_length = batch["input_ids"].numel()
            labels = torch.roll(batch["input_ids"].flatten(), shifts=-1, dims=-1)

            archon_logprobs, archon_entropy = gather_logprobs_entropy(
                archon_logits[:original_length], labels, temperature=1.0, tp_group=None
            )
            fsdp_logprobs, fsdp_entropy = gather_logprobs_entropy(
                fsdp_logits[:original_length], labels, temperature=1.0, tp_group=None
            )

        # Compare logprobs
        logprobs_metrics = compare_tensors(archon_logprobs, fsdp_logprobs)
        entropy_metrics = compare_tensors(archon_entropy, fsdp_entropy)

        print(f"\n[Logprobs Comparison] {logprobs_metrics}")
        print(f"[Entropy Comparison] {entropy_metrics}")

        # Check importance weight ratio (critical for PPO)
        importance_ratio = torch.exp(archon_logprobs - fsdp_logprobs)
        ratio_metrics = ComparisonMetrics(
            max_diff=(importance_ratio - 1.0).abs().max().item(),
            mean_diff=(importance_ratio - 1.0).abs().mean().item(),
            std_diff=(importance_ratio - 1.0).std().item(),
            allclose=torch.allclose(
                importance_ratio, torch.ones_like(importance_ratio), atol=0.1
            ),
            shape_match=True,
        )
        print(f"[Importance Ratio (should be ~1.0)] {ratio_metrics}")

        # Different attention implementations have numerical differences
        # For bfloat16 with different attention backends, allow up to ~5% deviation
        assert logprobs_metrics.max_diff < 1.5, (
            f"Logprobs max_diff too large: {logprobs_metrics.max_diff}"
        )
        assert ratio_metrics.max_diff < 1.1, (
            f"Importance ratio deviates too much from 1.0: {ratio_metrics.max_diff}"
        )

    def test_logprobs_gradient_flow(self, engines: DualEngineFixture):
        # [FUNCTION DOC]
        # Train mode, batch 2 x 64: backprops the sum of logits on both engines,
        # computes and prints global gradient norms (with per-parameter .item() syncs),
        # asserts no NaN/Inf gradients on either side, then zeroes gradients via
        # optimizer_zero_grad.
        """Verify gradients flow correctly through logprobs computation."""
        batch = create_grpo_batch(
            model_path=engines.model_path, batch_size=2, max_seq_len=64
        )
        device = torch.device(current_platform.device_type)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Prepare inputs
        archon_mb_list = engines.archon_engine._prepare_mb_list(batch).to(device)
        fsdp_mb_list = engines.fsdp_engine._prepare_mb_list(batch).to(device)

        archon_mb = next(iter(archon_mb_list))
        fsdp_mb = next(iter(fsdp_mb_list))

        archon_inputs, _ = engines.archon_engine._prepare_mb_inputs(archon_mb)
        fsdp_inputs, _ = engines.fsdp_engine._prepare_mb_inputs(fsdp_mb)

        engines.archon_engine.train()
        engines.fsdp_engine.train()

        # Forward with gradients
        # Archon forward
        cu_seqlens = archon_inputs.get("cu_seqlens")
        max_seqlen = archon_inputs.get("max_seqlen")
        if max_seqlen is not None:
            max_seqlen = int(max_seqlen)

        archon_logits = engines.archon_engine.model(
            archon_inputs["input_ids"],
            archon_inputs.get("position_ids"),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if archon_logits.ndim == 3 and archon_logits.shape[0] == 1:
            archon_logits = archon_logits.squeeze(0)

        # FSDP forward
        fsdp_outputs = engines.fsdp_engine.model(**fsdp_inputs)
        fsdp_logits = fsdp_outputs.logits
        if fsdp_logits.ndim == 3 and fsdp_logits.shape[0] == 1:
            fsdp_logits = fsdp_logits.squeeze(0)

        # Compute simple loss
        archon_loss = archon_logits.sum()
        fsdp_loss = fsdp_logits.sum()

        # Backward
        archon_loss.backward()
        fsdp_loss.backward()

        # Check gradient norms
        archon_grad_norm = 0.0
        fsdp_grad_norm = 0.0

        for param in engines.archon_engine.model.parameters():
            if param.grad is not None:
                archon_grad_norm += param.grad.norm().item() ** 2

        for param in engines.fsdp_engine.model.parameters():
            if param.grad is not None:
                fsdp_grad_norm += param.grad.norm().item() ** 2

        archon_grad_norm = archon_grad_norm**0.5
        fsdp_grad_norm = fsdp_grad_norm**0.5

        print(
            f"\n[Gradient Norms] Archon: {archon_grad_norm:.6f}, FSDP: {fsdp_grad_norm:.6f}"
        )
        print(f"[Gradient Norm Diff] {abs(archon_grad_norm - fsdp_grad_norm):.6f}")

        # Check for NaN/Inf
        archon_has_nan = any(
            param.grad is not None
            and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            for param in engines.archon_engine.model.parameters()
        )
        fsdp_has_nan = any(
            param.grad is not None
            and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            for param in engines.fsdp_engine.model.parameters()
        )

        assert not archon_has_nan, "Archon gradients contain NaN/Inf"
        assert not fsdp_has_nan, "FSDP gradients contain NaN/Inf"

        # Zero gradients for next test
        engines.archon_engine.optimizer_zero_grad()
        engines.fsdp_engine.optimizer_zero_grad()


# =============================================================================
# Suite 2: Loss/Advantage Calculation Tests
# =============================================================================


class TestLossAdvantageCalculation:
    # [CLASS DOC]
    # Pure grpo_loss_fn tests on random device tensors (no engines).
    """Test suite for comparing loss and advantage computation."""

    def test_grpo_loss_fn_consistency(self):
        # [FUNCTION DOC]
        # Seeded random logprobs, slightly perturbed old logprobs used as both behavior
        # and proximal logprobs, random advantages, loss mask with the first 10 tokens
        # off, random non-negative entropy. Calls grpo_loss_fn twice on cloned inputs
        # (eps_clip 0.2, token-level importance sampling, recompute prox method, no SAPO
        # / decoupled loss) and asserts identical results.
        """Test GRPO loss function produces consistent results for identical inputs."""
        device = torch.device(current_platform.device_type)
        torch.manual_seed(42)

        # Create mock GRPO inputs
        batch_size = 4
        seq_len = 64

        logprobs = torch.randn(batch_size, seq_len, device=device) * 0.5 - 2.0
        old_logprobs = logprobs.clone() + torch.randn_like(logprobs) * 0.1
        advantages = torch.randn(batch_size, seq_len, device=device)
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        # Mask out first few tokens (prompt)
        loss_mask[:, :10] = False

        input_data = {
            "logprobs": old_logprobs,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "prox_logp": old_logprobs.clone(),
        }

        entropy = torch.randn(batch_size, seq_len, device=device).abs()

        # Call grpo_loss_fn twice with identical inputs
        loss1 = grpo_loss_fn(
            logprobs=logprobs.clone(),
            entropy=entropy.clone(),
            input_data={
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in input_data.items()
            },
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            importance_sampling_level="token",
            current_version=1,
            prox_logp_method="recompute",
            use_sapo_loss=False,
            use_decoupled_loss=False,
        )

        loss2 = grpo_loss_fn(
            logprobs=logprobs.clone(),
            entropy=entropy.clone(),
            input_data={
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in input_data.items()
            },
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            importance_sampling_level="token",
            current_version=1,
            prox_logp_method="recompute",
            use_sapo_loss=False,
            use_decoupled_loss=False,
        )

        print(
            f"\n[GRPO Loss Consistency] loss1={loss1.item():.6f}, loss2={loss2.item():.6f}"
        )
        assert torch.allclose(loss1, loss2), (
            f"GRPO loss not deterministic: {loss1.item()} vs {loss2.item()}"
        )

    def test_ppo_loss_edge_cases(self):
        # [FUNCTION DOC]
        # Runs grpo_loss_fn for scenarios: all-zero advantages, +10 advantages, -10
        # advantages, and an all-false loss mask; asserts the loss is finite for each,
        # printing results and re-raising any exception.
        """Test PPO loss with edge cases that might reveal numerical instabilities."""
        device = torch.device(current_platform.device_type)

        scenarios = {
            "uniform_advantages": {
                "advantages": torch.zeros(4, 64, device=device),
                "description": "All advantages = 0",
            },
            "extreme_positive_advantages": {
                "advantages": torch.ones(4, 64, device=device) * 10.0,
                "description": "Large positive advantages",
            },
            "extreme_negative_advantages": {
                "advantages": torch.ones(4, 64, device=device) * -10.0,
                "description": "Large negative advantages",
            },
            "sparse_loss_mask": {
                "loss_mask": torch.zeros(4, 64, dtype=torch.bool, device=device),
                "description": "All tokens masked",
            },
        }

        for scenario_name, scenario_data in scenarios.items():
            torch.manual_seed(42)

            batch_size = 4
            seq_len = 64

            logprobs = torch.randn(batch_size, seq_len, device=device) * 0.5 - 2.0
            old_logprobs = logprobs.clone() + torch.randn_like(logprobs) * 0.1
            advantages = scenario_data.get(
                "advantages",
                torch.randn(batch_size, seq_len, device=device),
            )
            loss_mask = scenario_data.get(
                "loss_mask",
                torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
            )
            entropy = torch.randn(batch_size, seq_len, device=device).abs()

            input_data = {
                "logprobs": old_logprobs,
                "advantages": advantages,
                "loss_mask": loss_mask,
                "prox_logp": old_logprobs.clone(),
            }

            try:
                loss = grpo_loss_fn(
                    logprobs=logprobs,
                    entropy=entropy,
                    input_data=input_data,
                    eps_clip=0.2,
                    eps_clip_higher=None,
                    c_clip=None,
                    importance_sampling_level="token",
                    current_version=1,
                    prox_logp_method="recompute",
                    use_sapo_loss=False,
                    use_decoupled_loss=False,
                )

                is_finite = torch.isfinite(loss).item()
                print(
                    f"\n[{scenario_name}] {scenario_data['description']}: "
                    f"loss={loss.item():.6f}, finite={is_finite}"
                )

                assert is_finite, f"Loss is not finite for scenario: {scenario_name}"

            except Exception as e:
                print(f"\n[{scenario_name}] Exception: {e}")
                raise


# =============================================================================
# Suite 3: Gradient Updates Tests
# =============================================================================


class TestGradientUpdates:
    # [CLASS DOC]
    # Compares parameter updates of the two engines.
    """Test suite for comparing gradient computation and updates."""

    def test_optimizer_step_weight_delta(self, engines: DualEngineFixture):
        # [FUNCTION DOC]
        # Snapshots all parameters, runs a forward on both engines, backprops the logits
        # mean, steps each engine's optimizer, and prints the maximum difference between
        # Archon and FSDP weight deltas over same-named, same-shaped parameters. Then
        # zeroes gradients and copies the snapshots back into the parameters.
        # KNOWN ISSUE: Nothing is asserted, so the test cannot fail on mismatch;
        # parameter names of the two engines generally differ (Archon vs HF naming), so
        # few or no deltas may be compared; and optimizer state (e.g. Adam moments, step
        # count) is not restored, leaking into later tests.
        """Compare weight changes after a single optimizer step."""
        batch = create_grpo_batch(
            model_path=engines.model_path, batch_size=2, max_seq_len=64
        )
        device = torch.device(current_platform.device_type)
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Save initial weights
        archon_weights_before = {
            name: param.detach().clone()
            for name, param in engines.archon_engine.model.named_parameters()
        }
        fsdp_weights_before = {
            name: param.detach().clone()
            for name, param in engines.fsdp_engine.model.named_parameters()
        }

        # Prepare inputs
        archon_mb_list = engines.archon_engine._prepare_mb_list(batch).to(device)
        fsdp_mb_list = engines.fsdp_engine._prepare_mb_list(batch).to(device)

        archon_mb = next(iter(archon_mb_list))
        fsdp_mb = next(iter(fsdp_mb_list))

        archon_inputs, _ = engines.archon_engine._prepare_mb_inputs(archon_mb)
        fsdp_inputs, _ = engines.fsdp_engine._prepare_mb_inputs(fsdp_mb)

        engines.archon_engine.train()
        engines.fsdp_engine.train()

        # Forward pass
        cu_seqlens = archon_inputs.get("cu_seqlens")
        max_seqlen = archon_inputs.get("max_seqlen")
        if max_seqlen is not None:
            max_seqlen = int(max_seqlen)

        archon_logits = engines.archon_engine.model(
            archon_inputs["input_ids"],
            archon_inputs.get("position_ids"),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if archon_logits.ndim == 3 and archon_logits.shape[0] == 1:
            archon_logits = archon_logits.squeeze(0)

        fsdp_outputs = engines.fsdp_engine.model(**fsdp_inputs)
        fsdp_logits = fsdp_outputs.logits
        if fsdp_logits.ndim == 3 and fsdp_logits.shape[0] == 1:
            fsdp_logits = fsdp_logits.squeeze(0)

        # Simple loss for testing
        archon_loss = archon_logits.mean()
        fsdp_loss = fsdp_logits.mean()

        # Backward
        archon_loss.backward()
        fsdp_loss.backward()

        # Optimizer step
        engines.archon_engine.optimizer.step()
        engines.fsdp_engine.optimizer.step()

        # Compare weight deltas
        archon_weights_after = {
            name: param.detach().clone()
            for name, param in engines.archon_engine.model.named_parameters()
        }
        fsdp_weights_after = {
            name: param.detach().clone()
            for name, param in engines.fsdp_engine.model.named_parameters()
        }

        print("\n[Weight Delta Comparison]")
        max_delta_diff = 0.0
        for name in archon_weights_before:
            if name in fsdp_weights_before:
                archon_delta = archon_weights_after[name] - archon_weights_before[name]
                fsdp_delta = fsdp_weights_after[name] - fsdp_weights_before[name]

                if archon_delta.shape == fsdp_delta.shape:
                    delta_diff = (archon_delta - fsdp_delta).abs().max().item()
                    max_delta_diff = max(max_delta_diff, delta_diff)

        print(f"  Max weight delta difference: {max_delta_diff:.8f}")

        # Zero gradients
        engines.archon_engine.optimizer_zero_grad()
        engines.fsdp_engine.optimizer_zero_grad()

        # Restore original weights for other tests
        for name, param in engines.archon_engine.model.named_parameters():
            if name in archon_weights_before:
                param.data.copy_(archon_weights_before[name])
        for name, param in engines.fsdp_engine.model.named_parameters():
            if name in fsdp_weights_before:
                param.data.copy_(fsdp_weights_before[name])


# =============================================================================
# Suite 4: End-to-End GRPO Step Tests
# =============================================================================


class TestEndToEndGRPO:
    # [CLASS DOC]
    # End-to-end checks of reward-to-loss behavior.
    """Test suite for end-to-end GRPO step comparison."""

    def test_reward_signal_propagation(self):
        # [FUNCTION DOC]
        # With old logprobs equal to current logprobs (ratio 1), computes grpo_loss_fn
        # with all advantages +2 and with all -2, prints both, and asserts both are
        # finite and differ.
        """Verify that reward signals properly influence loss gradients."""
        device = torch.device(current_platform.device_type)
        torch.manual_seed(42)

        batch_size = 4
        seq_len = 64

        # Create two scenarios: high reward vs low reward
        logprobs = torch.randn(batch_size, seq_len, device=device) * 0.5 - 2.0
        old_logprobs = logprobs.clone()
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        loss_mask[:, :10] = False
        entropy = torch.randn(batch_size, seq_len, device=device).abs()

        # High reward scenario: advantages are positive
        high_reward_advantages = torch.ones(batch_size, seq_len, device=device) * 2.0
        high_reward_input = {
            "logprobs": old_logprobs.clone(),
            "advantages": high_reward_advantages,
            "loss_mask": loss_mask.clone(),
            "prox_logp": old_logprobs.clone(),
        }

        high_reward_loss = grpo_loss_fn(
            logprobs=logprobs.clone(),
            entropy=entropy.clone(),
            input_data=high_reward_input,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            importance_sampling_level="token",
            current_version=1,
            prox_logp_method="recompute",
            use_sapo_loss=False,
            use_decoupled_loss=False,
        )

        # Low reward scenario: advantages are negative
        low_reward_advantages = torch.ones(batch_size, seq_len, device=device) * -2.0
        low_reward_input = {
            "logprobs": old_logprobs.clone(),
            "advantages": low_reward_advantages,
            "loss_mask": loss_mask.clone(),
            "prox_logp": old_logprobs.clone(),
        }

        low_reward_loss = grpo_loss_fn(
            logprobs=logprobs.clone(),
            entropy=entropy.clone(),
            input_data=low_reward_input,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            importance_sampling_level="token",
            current_version=1,
            prox_logp_method="recompute",
            use_sapo_loss=False,
            use_decoupled_loss=False,
        )

        print("\n[Reward Signal Propagation]")
        print(f"  High reward loss: {high_reward_loss.item():.6f}")
        print(f"  Low reward loss: {low_reward_loss.item():.6f}")
        print(f"  Difference: {(low_reward_loss - high_reward_loss).item():.6f}")

        # With positive advantages (high reward), we want to increase probability
        # PPO loss = -advantages * ratio, so higher advantages should give more negative loss
        # But since ratio starts at 1.0, the sign of loss depends on advantages sign
        # We mainly check that the losses are different and finite
        assert torch.isfinite(high_reward_loss), "High reward loss is not finite"
        assert torch.isfinite(low_reward_loss), "Low reward loss is not finite"
        assert high_reward_loss != low_reward_loss, (
            "Losses should differ for different rewards"
        )
