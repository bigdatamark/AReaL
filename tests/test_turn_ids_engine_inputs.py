# [MODULE DOC]
# Unit tests that FSDPEngine and ArchonEngine _prepare_mb_inputs drop algorithm-only
# fields (turn_ids) from the model inputs while keeping them in the returned context's
# mb_input. Engines are created with __new__ and only the attributes the method reads
# are set.
from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import MicroBatchItem


def _make_microbatch() -> MicroBatchItem:
    # [FUNCTION DOC]
    # HELPER: Builds a MicroBatchItem whose orig and padded data are one 3-token
    # sequence with input_ids, all-True attention_mask and turn_ids [-1, 0, 0]; no
    # padding, no old cu_seqlens.
    data = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "turn_ids": torch.tensor([[-1, 0, 0]], dtype=torch.int32),
    }
    return MicroBatchItem(
        orig_mb=data,
        padded_mb=data,
        padding_length=0,
        old_cu_seqlens=None,
    )


def test_fsdp_prepare_inputs_strips_turn_ids_without_mutating_context():
    # [FUNCTION DOC]
    # FSDP engine with sp_size 1: asserts turn_ids is absent from inputs but present in
    # context.mb_input.
    """FSDP forwards model fields only while retaining algorithm metadata."""
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.parallel_helper = SimpleNamespace(sp_size=1)

    inputs, context = engine._prepare_mb_inputs(_make_microbatch())

    assert "turn_ids" not in inputs
    assert "turn_ids" in context.mb_input


def test_archon_prepare_inputs_strips_turn_ids_without_mutating_context():
    # [FUNCTION DOC]
    # Skips without triton. Archon engine with tree training and CP disabled: asserts
    # turn_ids is absent from inputs but present in context.mb_input.
    """Archon forwards model fields only while retaining algorithm metadata."""
    pytest.importorskip("triton", reason="Archon import requires Triton")
    from areal.experimental.engine.archon_engine import ArchonEngine

    engine = ArchonEngine.__new__(ArchonEngine)
    engine.enable_tree_training = False
    engine.parallel_dims = SimpleNamespace(cp_enabled=False)

    inputs, context = engine._prepare_mb_inputs(_make_microbatch())

    assert "turn_ids" not in inputs
    assert "turn_ids" in context.mb_input
