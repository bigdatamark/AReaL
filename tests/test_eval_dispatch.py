# [MODULE DOC]
# CPU tests for evaluation-batch dispatch: _pad_eval_batch (pads item lists with dummy
# items, attention_mask all zero, to a multiple of dp_size times group_size) and
# _dispatch_tensors from the train controller, make_dummy_eval_item,
# compute_total_loss_weight, reward-model (RW) pair grouping, RW loss/metrics with dummy
# pairs, and explicit padding in LMController.evaluate_lm / RWController.evaluate_rw.
from types import MethodType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import compute_total_loss_weight
from areal.infra.controller.train_controller import (
    _dispatch_tensors,
    _pad_eval_batch,
)
from areal.trainer.rw.rw_engine import (
    RWController,
    RWEngine,
    _rw_loss_weight,
    compute_rw_loss,
)
from areal.trainer.sft.lm_engine import LMController
from areal.utils.data import (
    MicroBatchList,
    concat_padded_tensors,
    make_dummy_eval_item,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.stats_tracker import DistributedStatsTracker


def _make_item(idx: int, seqlen: int = 2) -> dict[str, object]:
    # [FUNCTION DOC]
    # HELPER: Builds one batch item with input_ids filled with idx+1 of shape [1,
    # seqlen], all-True attention and loss masks, and meta {"id": idx}.
    return {
        "input_ids": torch.full((1, seqlen), idx + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "loss_mask": torch.ones((1, seqlen), dtype=torch.bool),
        "meta": {"id": idx},
    }


def _flatten_splits(splits: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    # [FUNCTION DOC]
    # HELPER: Flattens per-rank item lists into one list.
    return [item for group in splits for item in group]


def _count_dummies(items: list[dict[str, object]]) -> int:
    # [FUNCTION DOC]
    # HELPER: Counts dummy items, i.e. items whose attention_mask sums to zero.
    return sum(
        int(cast(torch.Tensor, item["attention_mask"]).sum().item() == 0)
        for item in items
    )


def _make_rw_pair(
    pair_idx: int, chosen_len: int = 3, rejected_len: int = 2
) -> tuple[dict[str, object], dict[str, object]]:
    # [FUNCTION DOC]
    # HELPER: Builds a (chosen, rejected) item pair of given lengths with distinct token
    # fills and meta recording pair index and role.
    chosen: dict[str, object] = {
        "input_ids": torch.full((1, chosen_len), pair_idx * 2 + 1, dtype=torch.long),
        "attention_mask": torch.ones((1, chosen_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "chosen"},
    }
    rejected: dict[str, object] = {
        "input_ids": torch.full((1, rejected_len), pair_idx * 2 + 2, dtype=torch.long),
        "attention_mask": torch.ones((1, rejected_len), dtype=torch.bool),
        "meta": {"pair": pair_idx, "role": "rejected"},
    }
    return chosen, rejected


def _build_rw_batch(n_pairs: int, chosen_len: int = 3, rejected_len: int = 2):
    # [FUNCTION DOC]
    # HELPER: Returns a flat list of n_pairs chosen/rejected items in interleaved order.
    items: list[dict[str, object]] = []
    for p in range(n_pairs):
        c, r = _make_rw_pair(p, chosen_len, rejected_len)
        items.extend([c, r])
    return items


def _make_rw_input(seqlens: list[int]) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Returns {"cu_seqlens": int32 tensor} with cumulative offsets for the given
    # sequence lengths.
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return {"cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32)}


class TestEvalBatchPadding:
    # [CLASS DOC]
    # Groups tests of eval-batch padding and dispatch.
    def test_pad_eval_batch_no_padding_when_divisible(self):
        # [FUNCTION DOC]
        # 8 items, dp 4: asserts no padding added.
        items = [_make_item(i) for i in range(8)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 0

    def test_pad_eval_batch_pads_when_not_divisible(self):
        # [FUNCTION DOC]
        # 7 items, dp 4: asserts padded to 8 with one dummy.
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 1

    def test_pad_eval_batch_pads_when_n_less_than_dp(self):
        # [FUNCTION DOC]
        # 2 items, dp 4: asserts padded to 4 with two dummies.
        items = [_make_item(i) for i in range(2)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 4
        assert _count_dummies(padded) == 2

    def test_dispatch_tensors_raises_when_not_divisible(self):
        # [FUNCTION DOC]
        # Asserts _dispatch_tensors raises ValueError "divisible" for 7 items over dp 4.
        items = [_make_item(i) for i in range(7)]
        with pytest.raises(ValueError, match="divisible"):
            _dispatch_tensors(items, dp_size=4)

    def test_pad_then_dispatch_end_to_end(self):
        # [FUNCTION DOC]
        # Pads 7 items then dispatches over dp 4; asserts each rank gets 2 items and
        # exactly one dummy overall.
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        splits, _ = _dispatch_tensors(padded, dp_size=4)
        assert all(len(group) == 2 for group in splits)
        assert _count_dummies(_flatten_splits(splits)) == 1

    def test_make_dummy_eval_item_schema(self):
        # [FUNCTION DOC]
        # Asserts make_dummy_eval_item keeps the template keys, uses [{}] for
        # multi_modal_input, produces [1,1] zero tensors with the template dtypes for
        # masks and input_ids, and deep-copies meta (later template mutation does not
        # leak).
        template: dict[str, object] = {
            "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[True, True, False]], dtype=torch.bool),
            "loss_mask": torch.tensor([[1, 1, 0]], dtype=torch.int32),
            "multi_modal_input": [{"image": torch.tensor([1.0])}],
            "meta": {"tag": ["x"]},
        }

        dummy = make_dummy_eval_item(template)
        assert set(dummy.keys()) == set(template.keys())
        assert dummy["multi_modal_input"] == [{}]
        torch.testing.assert_close(
            dummy["attention_mask"],
            torch.zeros((1, 1), dtype=torch.bool),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            dummy["loss_mask"],
            torch.zeros((1, 1), dtype=cast(torch.Tensor, template["loss_mask"]).dtype),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            dummy["input_ids"],
            torch.zeros((1, 1), dtype=cast(torch.Tensor, template["input_ids"]).dtype),
            rtol=0.0,
            atol=0.0,
        )
        cast(dict[str, list[str]], template["meta"])["tag"].append("y")
        assert dummy["meta"] == {"tag": ["x"]}

    def test_pad_eval_batch_keeps_multimodal_payload_aligned(self):
        # [FUNCTION DOC]
        # Pads 3 multimodal items to 4, concatenates and splits into micro-batches;
        # asserts the dummy has multi_modal_input [{}] and the multimodal list length
        # matches batch size in the batch and in every micro-batch.
        items: list[dict[str, object]] = []
        for _ in range(3):
            items.append(
                {
                    "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        [[True, True, True]], dtype=torch.bool
                    ),
                    "loss_mask": torch.tensor([[1, 1, 1]], dtype=torch.int32),
                    "multi_modal_input": [
                        {
                            "pixel_values": torch.ones((1, 2, 2), dtype=torch.float32),
                            "image_grid_thw": torch.tensor(
                                [[1, 1, 1]], dtype=torch.int32
                            ),
                        }
                    ],
                }
            )

        (padded,) = _pad_eval_batch((items,), dp_size=4)
        batched = concat_padded_tensors(padded)
        mb_list = split_padded_tensor_dict_into_mb_list(batched, MicroBatchSpec())

        assert len(padded) == 4
        assert len(batched["multi_modal_input"]) == batched["attention_mask"].shape[0]
        assert padded[-1]["multi_modal_input"] == [{}]
        for mb in mb_list.mbs:
            assert len(mb["multi_modal_input"]) == mb["attention_mask"].shape[0]

    def test_pad_eval_batch_group_size_odd_dp(self):
        # [FUNCTION DOC]
        # 2 items, dp 3, group_size 2: asserts padding to 6 with 4 dummies.
        items = [_make_item(i) for i in range(2)]
        (padded,) = _pad_eval_batch((items,), dp_size=3, group_size=2)
        assert len(padded) == 6
        assert len(padded) % 2 == 0
        assert len(padded) % 3 == 0
        assert _count_dummies(padded) == 4

    def test_pad_eval_batch_group_size_even_dp(self):
        # [FUNCTION DOC]
        # 6 items, dp 4, group_size 2: asserts padding to 8 with 2 dummies.
        items = [_make_item(i) for i in range(6)]
        (padded,) = _pad_eval_batch((items,), dp_size=4, group_size=2)
        assert len(padded) == 8
        assert len(padded) % 2 == 0
        assert len(padded) % 4 == 0
        assert _count_dummies(padded) == 2

    def test_pad_eval_batch_group_size_no_pad_needed(self):
        # [FUNCTION DOC]
        # 6 items, dp 3, group_size 2: asserts no padding.
        items = [_make_item(i) for i in range(6)]
        (padded,) = _pad_eval_batch((items,), dp_size=3, group_size=2)
        assert len(padded) == 6
        assert _count_dummies(padded) == 0

    def test_pad_eval_batch_default_group_size_unchanged(self):
        # [FUNCTION DOC]
        # Asserts default group_size behaves like plain dp padding (7 -> 8).
        items = [_make_item(i) for i in range(7)]
        (padded,) = _pad_eval_batch((items,), dp_size=4)
        assert len(padded) == 8
        assert _count_dummies(padded) == 1

    def test_compute_total_loss_weight_allows_local_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC]
        # Monkeypatches dist.all_reduce to add 3; asserts compute_total_loss_weight
        # returns 3 even though the local weight is 0 (no local-zero error).
        mb_list = MicroBatchList(
            data={},
            mb_spec=MicroBatchSpec(),
            mbs=[{"attention_mask": torch.zeros((1, 1), dtype=torch.bool)}],
            group_lens=[1],
        )

        def _mock_all_reduce(
            tensor: torch.Tensor, group: dist.ProcessGroup | None = None
        ):
            # [FUNCTION DOC]
            # HELPER: Fake all_reduce adding 3.0 in place.
            del group
            tensor.add_(3.0)

        monkeypatch.setattr(dist, "all_reduce", _mock_all_reduce)

        total_weight = compute_total_loss_weight(
            mb_list=mb_list,
            loss_weight_fn=lambda _mb: torch.tensor(0.0),
            dp_group=cast(dist.ProcessGroup, object()),
        )

        torch.testing.assert_close(total_weight, torch.tensor(3.0), rtol=0.0, atol=0.0)


class TestRWDispatchGrouping:
    # [CLASS DOC]
    # Groups tests that RW chosen/rejected pairs stay together through padding and
    # dispatch.
    def test_dispatch_group_size_preserves_rw_pairs(self):
        # [FUNCTION DOC]
        # 8 pairs, dp 4, group_size 2: asserts every shard has an even count and
        # consecutive items form same-pair (chosen, rejected) couples.
        dp_size = 4
        items = _build_rw_batch(n_pairs=8)
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=2)
        assert len(padded) == 16
        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=2)

        for shard_items in splits:
            assert len(shard_items) % 2 == 0, (
                f"Shard received odd item count {len(shard_items)}"
            )
            for j in range(0, len(shard_items), 2):
                c_meta = shard_items[j].get("meta", {})
                r_meta = shard_items[j + 1].get("meta", {})
                if c_meta.get("pair") is not None:
                    assert c_meta["pair"] == r_meta["pair"], (
                        f"Pair split across items: {c_meta} vs {r_meta}"
                    )
                    assert c_meta["role"] == "chosen"
                    assert r_meta["role"] == "rejected"

    def test_dispatch_group_size_rw_5_examples_dp4(self):
        # [FUNCTION DOC]
        # 5 pairs (10 items), dp 4: asserts padding to 16 and every shard gets an even
        # number of items.
        dp_size = 4
        items = _build_rw_batch(n_pairs=5)
        assert len(items) == 10

        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=2)
        assert len(padded) == 16
        assert len(padded) % (dp_size * 2) == 0

        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=2)
        assert len(splits) == dp_size
        for shard_items in splits:
            assert len(shard_items) % 2 == 0, (
                f"Shard received odd item count {len(shard_items)}, view(-1, 2) would fail"
            )

    def test_dispatch_group_size_not_divisible_raises(self):
        # [FUNCTION DOC]
        # Asserts _dispatch_tensors raises "divisible by group_size" for 7 items with
        # group_size 2.
        items = _build_rw_batch(n_pairs=3)
        items.append(items[0])
        with pytest.raises(ValueError, match="divisible by group_size"):
            _dispatch_tensors(items, dp_size=2, group_size=2)

    def test_dispatch_group_size_1_unchanged(self):
        # [FUNCTION DOC]
        # Asserts group_size=1 yields the same dispatch indices as the default.
        items = [_make_item(i, seqlen=i + 1) for i in range(8)]
        _, indices_default = _dispatch_tensors(items, dp_size=4)
        _, indices_gs1 = _dispatch_tensors(items, dp_size=4, group_size=1)
        assert indices_default == indices_gs1

    @pytest.mark.parametrize(
        "dp_size, group_size, n_items",
        [
            (3, 2, 12),
            (4, 2, 16),
            (6, 2, 24),
            (4, 3, 24),
            (3, 2, 6),
            (4, 2, 8),
        ],
    )
    def test_pad_and_dispatch_group_matrix(
        self, dp_size: int, group_size: int, n_items: int
    ):
        # [FUNCTION DOC]
        # Parametrized dp/group/n combos: asserts padded length is a multiple of
        # dp*group_size and every shard length is a multiple of group_size.
        items = [_make_item(i, seqlen=i % 5 + 1) for i in range(n_items)]
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=group_size)
        assert len(padded) % (dp_size * group_size) == 0

        splits, _ = _dispatch_tensors(padded, dp_size=dp_size, group_size=group_size)
        assert len(splits) == dp_size
        for shard in splits:
            assert len(shard) % group_size == 0

    @pytest.mark.parametrize(
        "dp_size, group_size, n_raw",
        [
            (3, 2, 5),
            (4, 2, 7),
            (4, 3, 10),
            (6, 2, 3),
        ],
    )
    def test_pad_aligns_non_divisible_input(
        self, dp_size: int, group_size: int, n_raw: int
    ):
        # [FUNCTION DOC]
        # Parametrized non-divisible inputs: asserts padded length is a multiple of
        # dp*group_size and not shorter than the input.
        items = [_make_item(i) for i in range(n_raw)]
        (padded,) = _pad_eval_batch((items,), dp_size=dp_size, group_size=group_size)
        target = dp_size * group_size
        assert len(padded) % target == 0
        assert len(padded) >= n_raw


class TestRWDummyPairSemantics:
    # [CLASS DOC]
    # Groups tests that dummy (empty) RW pairs do not affect loss or metrics.
    def test_rw_loss_weight_counts_only_valid_pairs(self):
        # [FUNCTION DOC]
        # Seqlens [5,4,0,0]: asserts _rw_loss_weight counts only the one valid pair.
        input_data = _make_rw_input([5, 4, 0, 0])

        loss_weight = _rw_loss_weight(input_data)

        torch.testing.assert_close(loss_weight, torch.tensor(1.0), rtol=0.0, atol=0.0)

    def test_compute_rw_loss_ignores_dummy_pairs_in_loss_and_metrics(self):
        # [FUNCTION DOC]
        # With the stats tracker patched, scores for one valid pair (3.0 vs 1.0 at
        # sequence ends): asserts loss is -logsigmoid(2) and n_pairs, correct_ratio,
        # pos/neg score and loss stats reflect only the valid pair.
        tracker = DistributedStatsTracker()
        input_data = _make_rw_input([5, 4, 0, 0])
        scores = torch.tensor([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        expected_loss = -F.logsigmoid(torch.tensor(2.0))

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            loss = compute_rw_loss(scores, input_data)

        stats = tracker.export(reset=True)

        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)
        assert stats["n_pairs"] == 1.0
        assert stats["correct_ratio/avg"] == 1.0
        assert stats["pos_score/avg"] == 3.0
        assert stats["neg_score/avg"] == 1.0
        torch.testing.assert_close(
            torch.tensor(stats["loss/avg"]),
            expected_loss,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_compute_rw_loss_returns_zero_for_all_dummy_pairs(self):
        # [FUNCTION DOC]
        # All-dummy pairs with empty scores: asserts loss 0, n_pairs 0 and no loss/avg
        # stat.
        tracker = DistributedStatsTracker()
        input_data = _make_rw_input([0, 0])

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            loss = compute_rw_loss(torch.tensor([], dtype=torch.float32), input_data)

        stats = tracker.export(reset=True)

        torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)
        assert stats["n_pairs"] == 0.0
        assert "loss/avg" not in stats


class TestExplicitEvalDispatchControllers:
    # [CLASS DOC]
    # Groups tests that controllers pad eval batches explicitly before dispatch.
    def test_lm_controller_evaluate_lm_explicitly_pads_batch(self):
        # [FUNCTION DOC]
        # Creates an LMController without __init__ (dp 4), replaces
        # _custom_function_call with a capturer; asserts evaluate_lm on 7 items calls
        # "evaluate_lm" with 8 padded items and rpc_meta broadcast True.
        controller = LMController.__new__(LMController)
        cast(Any, controller).train_alloc = SimpleNamespace(
            parallel=SimpleNamespace(dp_size=4)
        )
        captured: dict[str, Any] = {}

        def _capture_call(self, method: str, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records method name, args and kwargs; returns None.
            captured["method"] = method
            captured["args"] = args
            captured["kwargs"] = kwargs
            return None

        controller._custom_function_call = MethodType(_capture_call, controller)

        items = [_make_item(i) for i in range(7)]
        controller.evaluate_lm(items)

        assert captured["method"] == "evaluate_lm"
        padded_items = cast(list[dict[str, object]], captured["args"][0])
        assert len(padded_items) == 8
        assert captured["kwargs"] == {"rpc_meta": {"broadcast": True}}

    def test_rw_controller_evaluate_rw_explicitly_pads_pairs(self):
        # [FUNCTION DOC]
        # Same capture approach for RWController; asserts evaluate_rw on 5 pairs calls
        # "evaluate_rw" with group_size 2 and 16 padded items.
        controller = RWController.__new__(RWController)
        cast(Any, controller).train_alloc = SimpleNamespace(
            parallel=SimpleNamespace(dp_size=4)
        )
        captured: dict[str, Any] = {}

        def _capture_call(self, method: str, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records method name, args and kwargs; returns None.
            captured["method"] = method
            captured["args"] = args
            captured["kwargs"] = kwargs
            return None

        controller._custom_function_call = MethodType(_capture_call, controller)

        items = _build_rw_batch(n_pairs=5)
        controller.evaluate_rw(items)

        assert captured["method"] == "evaluate_rw"
        assert captured["kwargs"]["group_size"] == 2
        padded_items = cast(list[dict[str, object]], captured["args"][0])
        assert len(padded_items) == 16

    def test_evaluate_rw_logs_zero_pair_denominator_for_all_dummy_local_batch(self):
        # [FUNCTION DOC]
        # Runs RWEngine._evaluate_rw on an all-dummy input with a no-op engine and
        # patched tracker; asserts n_pairs logged as 0 and no loss/avg.
        tracker = DistributedStatsTracker()

        class _DummyEngine:
            # [CLASS DOC]
            # HELPER: Fake train engine with no-op eval and eval_batch.
            def eval(self) -> None:
                # [FUNCTION DOC]
                # HELPER: No-op.
                return None

            def eval_batch(self, **kwargs) -> None:
                # [FUNCTION DOC]
                # HELPER: Ignores kwargs; returns None.
                del kwargs
                return None

        with patch("areal.trainer.rw.rw_engine.stats_tracker", tracker):
            RWEngine(_DummyEngine())._evaluate_rw(_make_rw_input([0, 0]))

        stats = tracker.export(reset=True)

        assert stats["n_pairs"] == 0.0
        assert "loss/avg" not in stats
