# [MODULE DOC]
# CPU unit tests for PPO statistics bookkeeping: infer_token_denominator (which tensor
# defines the per-token stats denominator), the n_tokens denominator passed to the stats
# tracker by the GRPO actor loss and the critic loss for packed (cu_seqlens) inputs, and
# _infer_prompt_lens recovering prompt lengths from the shifted loss mask.
from unittest.mock import MagicMock, patch

import torch

from areal.trainer.ppo.actor import _infer_prompt_lens, grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils.stats_tracker import DistributedStatsTracker


def test_infer_token_denominator_prefers_attention_mask():
    # [FUNCTION DOC]
    # When attention_mask exists, the denominator is an all-True bool tensor shaped like it,
    # ignoring input_ids and the fallback.
    input_data = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "input_ids": torch.tensor([[11, 12], [13, 14]]),
    }

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(5))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_uses_input_ids_when_attention_mask_missing():
    # [FUNCTION DOC]
    # Without attention_mask, a 2-D input_ids shape is used.
    input_data = {"input_ids": torch.tensor([[11, 12, 13], [14, 15, 16]])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(2, 3))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_for_padded_tree_input_ids():
    # [FUNCTION DOC]
    # 1-D (tree-packed, padded) input_ids are not trusted; the fallback's shape is used.
    input_data = {"input_ids": torch.tensor([11, 12, 13, 0])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(3))

    assert n_tokens.shape == torch.Size([3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_when_metadata_is_missing():
    # [FUNCTION DOC]
    # With no attention_mask or input_ids, the fallback shape is used.
    fallback = torch.zeros(4)

    n_tokens = infer_token_denominator({"logprobs": torch.zeros(2)}, fallback=fallback)

    assert n_tokens.shape == torch.Size([4])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    # [FUNCTION DOC]
    # grpo_loss_fn (stats_tracker mocked) reports an n_tokens denominator sized by the last
    # cu_seqlens entry (4) rather than the 2 provided tokens, all True.
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "logprobs": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
        "prox_logp": torch.zeros(2),
        "versions": torch.zeros(2, dtype=torch.int32),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scope = MagicMock()
        mock_tracker.scope.return_value.__enter__ = MagicMock()
        mock_tracker.scope.return_value.__exit__ = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )

    n_tokens = next(
        call.kwargs["n_tokens"]
        for call in mock_tracker.denominator.call_args_list
        if "n_tokens" in call.kwargs
    )
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_critic_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    # [FUNCTION DOC]
    # Critic ppo_loss_fn (stats_tracker mocked) likewise reports n_tokens of size 4 from
    # cu_seqlens.
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "values": torch.zeros(2),
        "returns": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.critic.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()

        ppo_loss_fn(
            value=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
        )

    n_tokens = mock_tracker.denominator.call_args.kwargs["n_tokens"]
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_packed_denominator_for_tree_vocab_stats():
    # [FUNCTION DOC]
    # With a real DistributedStatsTracker, tree-style padded input_ids and vocab min/max
    # logits, grpo_loss_fn runs without shape mismatch and exports an "n_tokens" stat.
    tracker = DistributedStatsTracker()
    input_data = {
        "input_ids": torch.tensor([11, 12, 13, 0]),
        "logprobs": torch.zeros(3),
        "advantages": torch.ones(3),
        "loss_mask": torch.ones(3, dtype=torch.bool),
        "prox_logp": torch.zeros(3),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(3),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            vocab_min_logits=torch.zeros(3),
            vocab_max_logits=torch.zeros(3),
        )

    stats = tracker.export(reset=True)
    assert "n_tokens" in stats


def _rolled_loss_mask(prompt_len: int, answer_len: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns a [1, prompt_len+answer_len] float mask (0 for prompt, 1 for answer)
    # rolled left by one position, matching the mask seen inside _ppo_update.
    """Build the loss_mask as _ppo_update sees it: rolled left by one."""
    mask = torch.tensor([[0] * prompt_len + [1] * answer_len], dtype=torch.float)
    return torch.roll(mask, shifts=-1, dims=-1)


def test_infer_prompt_lens_recovers_the_prompt_length():
    # [FUNCTION DOC]
    # _infer_prompt_lens returns the original prompt length for a rolled mask.
    prompt_len, answer_len = 5, 4
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [prompt_len]


def test_infer_prompt_lens_matches_the_sum_formula_without_rejection():
    # [FUNCTION DOC]
    # Result equals attention_mask sum minus loss_mask sum when no tokens are rejected.
    prompt_len, answer_len = 3, 7
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    legacy = attention_mask.sum(-1) - loss_mask.sum(-1)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == legacy.tolist()


def test_infer_prompt_lens_handles_a_batch_of_mixed_prompt_lengths():
    # [FUNCTION DOC]
    # Works per row for a batch with different prompt lengths.
    specs = [(2, 6), (5, 3), (1, 7)]
    masks = torch.cat([_rolled_loss_mask(p, a) for p, a in specs])
    attention_mask = torch.ones(len(specs), 8, dtype=torch.long)

    got = _infer_prompt_lens(attention_mask, masks).tolist()

    assert got == [p for p, _ in specs]


def test_infer_prompt_lens_falls_back_to_seqlen_when_nothing_is_trained():
    # [FUNCTION DOC]
    # An all-zero loss mask yields the full sequence length.
    loss_mask = torch.zeros(1, 9)
    attention_mask = torch.ones(1, 9, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [9]
