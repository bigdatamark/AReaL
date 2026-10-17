# [MODULE DOC]
# CPU-only unit tests for Megatron VLM support: unwrap_to_gpt_model, vision-tensor
# extraction and sequence-layout preparation in
# areal.engine.megatron_utils.packed_context_parallel (wrapper THD, padded BSHD, model-
# owned THD), packed_context_parallel_forward argument construction (mpu functions
# monkeypatched), AReaL chunked LM-head helper functions in
# areal.engine.megatron_engine, vision model detection helpers, mcore-to-HF parameter-
# name converters for Qwen2.5-VL, Qwen3-VL and Qwen3-VL-MoE, and remove_padding for VLM
# vocab tensors. Imports are done inside each test; requires megatron.core installed but
# no GPU.
"""CPU-only unit tests for Megatron engine VLM (Vision-Language Model) support.

Distributed integration tests live in
``tests/test_megatron_engine_vlm_distributed.py`` — they launch torchrun
subprocesses and require GPUs.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


class TestUnwrapToGptModel:
    # [CLASS DOC]
    # Tests unwrap_to_gpt_model, which strips DDP-style .module wrappers and VLM
    # .language_model attributes until reaching a GPTModel.
    """Test unwrap_to_gpt_model with VLM model structures."""

    def test_plain_gpt_model_unwraps(self):
        # [FUNCTION DOC]
        # A GPTModel-classed mock is returned unchanged.
        """Plain GPTModel (no wrapping) should return itself."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        result = unwrap_to_gpt_model(mock_gpt)
        assert result is mock_gpt

    def test_ddp_wrapped_gpt_model_unwraps(self):
        # [FUNCTION DOC]
        # A wrapper whose .module is the GPTModel unwraps to it.
        """GPTModel wrapped in DDP (via .module) should unwrap correctly."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = mock_gpt
        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_vlm_model_with_language_model_unwraps(self):
        # [FUNCTION DOC]
        # A VLM-like object without .module but with .language_model unwraps to the
        # language model.
        """VLM model (e.g. Qwen2_5VLModel) with .language_model attribute unwraps."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        # VLM model: no .module, but has .language_model
        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module  # ensure no .module attribute
        vlm_model.language_model = mock_gpt

        result = unwrap_to_gpt_model(vlm_model)
        assert result is mock_gpt

    def test_ddp_wrapped_vlm_model_unwraps(self):
        # [FUNCTION DOC]
        # A wrapper around a VLM unwraps through both .module and .language_model.
        """VLM model wrapped in DDP should unwrap to language_model."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module
        vlm_model.language_model = mock_gpt

        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = vlm_model

        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_unsupported_model_raises_type_error(self):
        # [FUNCTION DOC]
        # An object with neither GPTModel type, .module nor .language_model raises
        # TypeError ("could not be unwrapped").
        """Model with neither GPTModel nor .language_model should raise TypeError."""
        from areal.models.mcore.registry import unwrap_to_gpt_model

        unsupported = MagicMock()
        unsupported.__class__ = type("RandomModel", (), {})
        del unsupported.module
        del unsupported.language_model

        with pytest.raises(TypeError, match="could not be unwrapped"):
            unwrap_to_gpt_model(unsupported)


class TestExtractVisionFromMultiModal:
    # [CLASS DOC]
    # Tests extract_vision_from_multi_modal(mb, padded_mb): pops multi_modal_input from
    # both dicts and concatenates per-sample pixel_values, image_grid_thw and
    # video_grid_thw onto padded_mb only.
    """Test _extract_vision_from_multi_modal helper."""

    def test_extracts_pixel_values_and_grid(self):
        # [FUNCTION DOC]
        # Two samples (10 and 5 patches): padded_mb gets 15 pixel rows and 2 grids; mb
        # gets no vision keys; multi_modal_input removed from both.
        """Vision tensors should land on padded_mb only, not duplicated on mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(10, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                {
                    "pixel_values": torch.randn(5, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        # Forward side (padded_mb) gets the concatenated vision tensors.
        assert padded_mb["pixel_values"].shape[0] == 15  # 10 + 5
        assert padded_mb["image_grid_thw"].shape[0] == 2  # 2 grids
        # Loss side (mb) does not carry them.
        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb
        # multi_modal_input is consumed and removed from both sides.
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_handles_video_grid_thw(self):
        # [FUNCTION DOC]
        # video_grid_thw is extracted onto padded_mb and absent from mb.
        """Should extract video_grid_thw onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(4, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert padded_mb["video_grid_thw"].shape[0] == 1
        assert "video_grid_thw" not in mb

    def test_no_multi_modal_input_is_noop(self):
        # [FUNCTION DOC]
        # Without multi_modal_input nothing is added.
        """Should do nothing if multi_modal_input not in dict."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"input_ids": torch.tensor([1, 2, 3])}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb

    def test_empty_multi_modal_input_is_noop(self):
        # [FUNCTION DOC]
        # A multi_modal_input of empty dicts adds no vision keys but is still popped
        # from both dicts.
        """Should not add keys if multi_modal_input items have no vision data,
        but should still pop multi_modal_input itself."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"multi_modal_input": [{}]}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "pixel_values" not in padded_mb
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_falls_back_to_padded_mb(self):
        # [FUNCTION DOC]
        # When only padded_mb has multi_modal_input, its pixel_values are still
        # concatenated onto padded_mb and the key removed.
        """When mb lacks multi_modal_input but padded_mb has it, the fallback
        branch should still concatenate vision tensors onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        pixel_values = [torch.randn(2, 4)]
        mb: dict[str, Any] = {"input_ids": torch.ones(2, dtype=torch.long)}
        padded_mb: dict[str, Any] = {
            "input_ids": torch.ones(2, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": pixel_values[0]}],
        }

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb
        assert "pixel_values" not in mb
        assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))


class TestPackedContextParallelForward:
    # [CLASS DOC]
    # Tests sequence-layout preparation, packed_context_parallel_forward kwargs, and
    # AReaL LM-head helper functions from megatron_engine.
    def test_wrapper_thd_padding_keeps_sequence_offsets_tp_aligned(self):
        # [FUNCTION DOC]
        # pad_packed_tensor_dict with pad_to_length 16 and seq_align_to 4 on two
        # sequences (3 and 2 tokens) yields per-sequence lengths [4, 4, 248], all
        # multiples of 4 (sequence offsets stay TP-aligned; the trailing pad sequence
        # absorbs the remainder).
        from areal.utils.data import pad_packed_tensor_dict

        padded, _, _, _ = pad_packed_tensor_dict(
            {
                "input_ids": torch.tensor([10, 11, 12, 20, 21]),
                "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
                "max_seqlen": 3,
            },
            pad_to_length=16,
            seq_align_to=4,
        )

        seq_lens = padded["cu_seqlens"][1:] - padded["cu_seqlens"][:-1]
        assert seq_lens.tolist() == [4, 4, 248]
        assert torch.all(seq_lens % 4 == 0)

    @pytest.mark.parametrize(
        ("sequence_packing_mode", "expected_cu_seqlens", "expected_bshd_shape"),
        [
            ("wrapper_thd", [0, 14_010, 14_080], None),
            ("padded", [0, 14_010], (1, 14_010)),
            ("model_thd", [0, 14_010], (1, 14_010)),
        ],
    )
    def test_sequence_layout_controls_batch_padding(
        self, sequence_packing_mode, expected_cu_seqlens, expected_bshd_shape
    ):
        # [FUNCTION DOC]
        # Parametrized over sequence packing modes for one 14010-token sequence:
        # wrapper_thd pads by appending a pad sequence (cu_seqlens [0, 14010, 14080]);
        # padded and model_thd keep cu_seqlens [0, 14010] and reconstruct to a (1,
        # 14010) BSHD batch via _reconstruct_padded_2d.
        from areal.api.cli_args import MicroBatchSpec
        from areal.engine.core.model import SequencePackingMode
        from areal.engine.megatron_utils.packed_context_parallel import (
            _reconstruct_padded_2d,
            prepare_microbatches_for_sequence_layout,
        )
        from areal.utils.data import MicroBatchList

        input_ids = torch.arange(14_010)
        mb = {
            "input_ids": input_ids,
            "cu_seqlens": torch.tensor([0, 14_010], dtype=torch.int32),
            "max_seqlen": 14_010,
        }
        mb_list = MicroBatchList(
            data=mb,
            mb_spec=MicroBatchSpec(),
            mbs=[mb],
            group_lens=[14_010],
        )

        prepared = prepare_microbatches_for_sequence_layout(
            mb_list,
            sequence_packing_mode=SequencePackingMode(sequence_packing_mode),
            pad_to_maximum=False,
            seq_align_to=1,
        )

        assert prepared.padded_mbs is not None
        padded_mb = prepared.padded_mbs[0]
        assert padded_mb["cu_seqlens"].tolist() == expected_cu_seqlens
        if expected_bshd_shape is None:
            return

        reconstructed, _, seq_lens, _ = _reconstruct_padded_2d(
            padded_mb["input_ids"],
            padded_mb["cu_seqlens"],
            padded_mb["max_seqlen"],
        )
        assert seq_lens.tolist() == [14_010]
        assert reconstructed.shape == expected_bshd_shape

    def test_bshd_pad_to_maximum_preserves_legacy_packed_axis_behavior(self):
        # [FUNCTION DOC]
        # PADDED mode with pad_to_maximum True and max_tokens_per_mb 14080 still appends
        # the legacy pad segment (cu_seqlens [0, 14010, 14080]).
        from areal.api.cli_args import MicroBatchSpec
        from areal.engine.core.model import SequencePackingMode
        from areal.engine.megatron_utils.packed_context_parallel import (
            prepare_microbatches_for_sequence_layout,
        )
        from areal.utils.data import MicroBatchList

        input_ids = torch.arange(14_010)
        mb = {
            "input_ids": input_ids,
            "cu_seqlens": torch.tensor([0, 14_010], dtype=torch.int32),
            "max_seqlen": 14_010,
        }
        mb_list = MicroBatchList(
            data=mb,
            mb_spec=MicroBatchSpec(max_tokens_per_mb=14_080),
            mbs=[mb],
            group_lens=[14_010],
        )

        prepared = prepare_microbatches_for_sequence_layout(
            mb_list,
            sequence_packing_mode=SequencePackingMode.PADDED,
            pad_to_maximum=True,
            seq_align_to=1,
        )

        assert prepared.padded_mbs is not None
        assert prepared.padded_mbs[0]["cu_seqlens"].tolist() == [0, 14_010, 14_080]

    @pytest.mark.parametrize(
        ("use_padded_seq", "expected_mask"),
        [
            (False, None),
            (True, [[True, True, True], [True, True, False]]),
        ],
    )
    def test_padded_vlm_preserves_model_specific_mask_semantics(
        self, monkeypatch, use_padded_seq, expected_mask
    ):
        # [FUNCTION DOC]
        # Vision model forward with padded BSHD: input_ids are reshaped to
        # [[10,11,12],[20,21,0]], position_ids and packed_seq_params are None,
        # attention_mask is None unless use_padded_seq (then a boolean valid-token
        # mask), and output is repacked to 5 tokens x 4.
        from areal.engine.megatron_utils import packed_context_parallel

        model = MagicMock(return_value=torch.ones(2, 3, 4))
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "is_pipeline_last_stage",
            lambda **_kwargs: True,
        )

        output = packed_context_parallel.packed_context_parallel_forward(
            model,
            {
                "input_ids": torch.tensor([10, 11, 12, 20, 21]),
                "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
                "max_seqlen": 3,
            },
            is_vision_model=True,
            use_padded_seq=use_padded_seq,
        )

        call = model.call_args.kwargs
        assert call["input_ids"].tolist() == [[10, 11, 12], [20, 21, 0]]
        if expected_mask is None:
            assert call["attention_mask"] is None
        else:
            assert call["attention_mask"].tolist() == expected_mask
        assert call["position_ids"] is None
        assert call["packed_seq_params"] is None
        assert output.shape == (5, 4)

    def test_model_thd_passes_padded_inputs_and_packed_metadata(self, monkeypatch):
        # [FUNCTION DOC]
        # use_model_packed_seq: model receives padded input_ids, a boolean attention
        # mask, no position_ids, and THD packed_seq_params with cu_seqlens_q [0, 3, 5];
        # output repacked to 5 x 3.
        from areal.engine.megatron_utils import packed_context_parallel

        model = MagicMock(return_value=torch.ones(1, 5, 3))
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "is_pipeline_last_stage",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "get_context_parallel_world_size",
            lambda: 1,
        )

        output = packed_context_parallel.packed_context_parallel_forward(
            model,
            {
                "input_ids": torch.tensor([10, 11, 12, 20, 21]),
                "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
                "max_seqlen": 3,
            },
            is_vision_model=True,
            use_model_packed_seq=True,
        )

        call = model.call_args.kwargs
        assert call["input_ids"].tolist() == [[10, 11, 12], [20, 21, 0]]
        assert call["attention_mask"].tolist() == [
            [True, True, True],
            [True, True, False],
        ]
        assert call["position_ids"] is None
        assert call["packed_seq_params"].qkv_format == "thd"
        assert call["packed_seq_params"].cu_seqlens_q.tolist() == [0, 3, 5]
        assert output.shape == (5, 3)

    def test_hidden_state_mode_bypasses_post_process_and_restores_model(self):
        # [FUNCTION DOC]
        # With return_hidden_states True, the model's post_process flag is False during
        # forward and restored to True afterwards; the raw hidden-state shape (2, 1, 3)
        # is returned.
        from areal.engine.megatron_utils import packed_context_parallel

        class HiddenModel(torch.nn.Module):
            # [CLASS DOC]
            # HELPER: Tiny module recording the post_process value seen at forward time
            # and returning ones(2, 1, 3).
            def __init__(self):
                # [FUNCTION DOC]
                # HELPER: Sets post_process and seen_post_process to True.
                super().__init__()
                self.post_process = True
                self.seen_post_process = True

            def forward(self, **_kwargs):
                # [FUNCTION DOC]
                # HELPER: Records post_process into seen_post_process and returns
                # ones(2, 1, 3); kwargs ignored.
                self.seen_post_process = self.post_process
                return torch.ones(2, 1, 3)

        model = HiddenModel()
        output = packed_context_parallel.packed_context_parallel_forward(
            model,
            {"input_ids": torch.ones(2, dtype=torch.long)},
            return_hidden_states=True,
        )

        assert output.shape == (2, 1, 3)
        assert model.seen_post_process is False
        assert model.post_process is True

    def test_padded_lm_head_labels_and_outputs_preserve_sequence_order(self):
        # [FUNCTION DOC]
        # _padded_lm_head_labels builds [S, B] next-token labels from packed ids
        # (shifted within each sequence, pad 0); _repack_padded_lm_head_output flattens
        # an [S, B] output, drops the pad slot and restores sequence-major packed order.
        from areal.engine.megatron_engine import (
            _padded_lm_head_labels,
            _repack_padded_lm_head_output,
        )

        input_ids = torch.tensor([10, 11, 12, 20, 21])
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

        labels = _padded_lm_head_labels(input_ids, cu_seqlens, max_seqlen=3)
        assert labels.tolist() == [[11, 21], [12, 0], [10, 20]]

        # Chunked LM Head flattens [S, B] token outputs. Repacking must remove
        # the padded slot and restore sequence-major packed order.
        padded_outputs = torch.tensor([100, 200, 101, 201, 102, 0])
        repacked = _repack_padded_lm_head_output(
            padded_outputs, cu_seqlens, max_seqlen=3
        )
        assert repacked.tolist() == [100, 101, 102, 200, 201]

    def test_repack_padded_lm_head_output_rejects_wrong_token_count(self):
        # [FUNCTION DOC]
        # _repack_padded_lm_head_output raises ValueError when the token count does not
        # match the BSHD layout.
        from areal.engine.megatron_engine import _repack_padded_lm_head_output

        with pytest.raises(ValueError, match="does not match the BSHD token layout"):
            _repack_padded_lm_head_output(
                torch.ones(5),
                torch.tensor([0, 3, 5], dtype=torch.int32),
                max_seqlen=3,
            )

    @pytest.mark.parametrize(
        (
            "enable_chunked_logits",
            "model_dtype",
            "expected",
        ),
        [
            (True, torch.bfloat16, False),
            (True, torch.float16, False),
            (True, torch.float32, None),
            (False, torch.bfloat16, None),
        ],
    )
    def test_float16_wrapper_override_follows_areal_lm_head(
        self,
        enable_chunked_logits,
        model_dtype,
        expected,
    ):
        # [FUNCTION DOC]
        # Parametrized: _float16_wrapper_fp32_output returns False for chunked logits
        # with bf16/fp16 models, and None (no override) for fp32 models or when chunked
        # logits are disabled.
        from areal.engine.megatron_engine import _float16_wrapper_fp32_output

        assert (
            _float16_wrapper_fp32_output(
                enable_chunked_logits,
                model_dtype,
            )
            is expected
        )

    @pytest.mark.parametrize(
        (
            "enable_chunked_logits",
            "enable_fp32_lm_head",
            "cross_entropy_loss_fusion",
            "expected",
        ),
        [
            (True, True, False, {}),
            (False, False, False, {}),
            (False, True, False, {"enable_fp32_lm_head": True}),
            (True, False, True, {"cross_entropy_loss_fusion": True}),
            (True, True, True, {"cross_entropy_loss_fusion": True}),
            (
                False,
                True,
                True,
                {
                    "enable_fp32_lm_head": True,
                    "cross_entropy_loss_fusion": True,
                },
            ),
        ],
    )
    def test_mbridge_precision_args_preserve_native_fallback(
        self,
        enable_chunked_logits,
        enable_fp32_lm_head,
        cross_entropy_loss_fusion,
        expected,
    ):
        # [FUNCTION DOC]
        # Parametrized: _mbridge_precision_args returns a kwargs dict that includes
        # enable_fp32_lm_head only when chunked logits are disabled and fp32 head
        # requested, and cross_entropy_loss_fusion only when enabled.
        from areal.engine.megatron_engine import _mbridge_precision_args

        assert (
            _mbridge_precision_args(
                enable_chunked_logits,
                enable_fp32_lm_head,
                cross_entropy_loss_fusion,
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("enable_chunked_logits", "entropy_requires_grad", "expected"),
        [
            (False, True, False),
            (False, False, False),
            (True, True, False),
            (True, False, True),
        ],
    )
    def test_logits_reuse_respects_entropy_gradient_setting(
        self,
        enable_chunked_logits,
        entropy_requires_grad,
        expected,
    ):
        # [FUNCTION DOC]
        # Parametrized: _reuse_chunked_logits_storage is True only when chunked logits
        # are enabled and entropy does not require grad.
        from areal.engine.megatron_engine import _reuse_chunked_logits_storage

        assert (
            _reuse_chunked_logits_storage(
                enable_chunked_logits,
                entropy_requires_grad,
            )
            is expected
        )

    @pytest.mark.parametrize(
        (
            "global_rank",
            "is_critic",
            "enable_chunked_logits",
            "entropy_requires_grad",
            "warns",
        ),
        [
            (0, False, True, False, True),
            (1, False, True, False, False),
            (0, True, True, False, False),
            (0, False, False, False, False),
            (0, False, True, True, False),
        ],
    )
    def test_areal_lm_head_warns_when_entropy_is_nondifferentiable(
        self,
        global_rank,
        is_critic,
        enable_chunked_logits,
        entropy_requires_grad,
        warns,
    ):
        # [FUNCTION DOC]
        # Parametrized: the warning about non-differentiable entropy is logged once only
        # on global rank 0, for actors (not critics), with chunked logits enabled and
        # entropy not requiring grad.
        from areal.engine.megatron_engine import (
            _warn_if_areal_lm_head_entropy_is_nondifferentiable,
        )

        logger = MagicMock()
        _warn_if_areal_lm_head_entropy_is_nondifferentiable(
            logger,
            global_rank=global_rank,
            is_critic=is_critic,
            enable_chunked_logits=enable_chunked_logits,
            entropy_requires_grad=entropy_requires_grad,
        )

        if warns:
            logger.warning.assert_called_once()
            assert "entropy is non-differentiable" in logger.warning.call_args.args[0]
        else:
            logger.warning.assert_not_called()

    @pytest.mark.parametrize(
        (
            "enable_chunked_logits",
            "enable_tree_training",
            "npu_available",
            "error_match",
        ),
        [
            (False, True, True, None),
            (True, False, False, None),
            (True, True, False, "tree training"),
            (True, False, True, "NPU training"),
        ],
    )
    def test_areal_lm_head_rejects_unsupported_training_modes(
        self,
        enable_chunked_logits,
        enable_tree_training,
        npu_available,
        error_match,
    ):
        # [FUNCTION DOC]
        # Parametrized: _validate_areal_lm_head_compatibility raises NotImplementedError
        # for chunked logits combined with tree training or NPU, and passes otherwise.
        from areal.engine.megatron_engine import (
            _validate_areal_lm_head_compatibility,
        )

        if error_match is None:
            _validate_areal_lm_head_compatibility(
                enable_chunked_logits,
                enable_tree_training=enable_tree_training,
                npu_available=npu_available,
            )
            return

        with pytest.raises(NotImplementedError, match=error_match):
            _validate_areal_lm_head_compatibility(
                enable_chunked_logits,
                enable_tree_training=enable_tree_training,
                npu_available=npu_available,
            )

    @pytest.mark.parametrize("fp32_output", [False, True])
    def test_forwards_explicit_fp32_output(self, monkeypatch, fp32_output):
        # [FUNCTION DOC]
        # Parametrized: an explicit fp32_output value is forwarded unchanged to the
        # model call.
        from areal.engine.megatron_utils import packed_context_parallel

        model = MagicMock(return_value=torch.ones(1, 2, 3))
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "is_pipeline_last_stage",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "get_context_parallel_world_size",
            lambda: 1,
        )

        packed_context_parallel.packed_context_parallel_forward(
            model,
            {"input_ids": torch.ones(2, dtype=torch.long)},
            fp32_output=fp32_output,
        )

        assert model.call_args.kwargs["fp32_output"] is fp32_output

    def test_omits_unspecified_fp32_output(self, monkeypatch):
        # [FUNCTION DOC]
        # When fp32_output is not given, the kwarg is omitted from the model call.
        from areal.engine.megatron_utils import packed_context_parallel

        model = MagicMock(return_value=torch.ones(1, 2, 3))
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "is_pipeline_last_stage",
            lambda **_kwargs: True,
        )
        monkeypatch.setattr(
            packed_context_parallel.mpu,
            "get_context_parallel_world_size",
            lambda: 1,
        )

        packed_context_parallel.packed_context_parallel_forward(
            model,
            {"input_ids": torch.ones(2, dtype=torch.long)},
        )

        assert "fp32_output" not in model.call_args.kwargs


class TestPrepareMbListRebindCallerSafety:
    # [CLASS DOC]
    # Pins the contract that multimodal payload keys are dropped by rebinding
    # mb_list.data to a filtered copy rather than mutating the caller's dict in place.
    """Verify _prepare_mb_list rebinds mb_list.data to a filtered copy so the
    caller's input dict survives across repeated forward() calls.

    `split_padded_tensor_dict_into_mb_list` constructs `MicroBatchList(data=input_)`
    where `mb_list.data` is the *same object* as the caller's input dict (not a
    copy). An earlier draft did `_drop_multi_modal_payload(mb_list.data)` in-place,
    which mutated the caller's dict and broke `engine.forward(input_)` when the
    same `input_` was forwarded a second time (e.g. the save/load round-trip
    test). The current implementation rebinds `mb_list.data = {filtered copy}`
    instead — this test pins that contract.
    """

    def test_rebind_does_not_mutate_caller_input(self):
        # [FUNCTION DOC]
        # Builds an input dict with multimodal keys, wraps it in a MicroBatchList
        # (asserting data aliases it), rebinds data to a copy filtered by
        # _is_multi_modal_payload_key, and asserts the caller's keys are unchanged while
        # the new data lacks multimodal keys but keeps input_ids.
        # KNOWN ISSUE: The test performs the rebinding itself instead of calling the
        # engine's _prepare_mb_list, so a regression in the engine code would not be
        # caught.
        from areal.engine.megatron_utils.packed_context_parallel import (
            _is_multi_modal_payload_key,
        )
        from areal.utils.data import MicroBatchList, MicroBatchSpec

        input_ = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": torch.randn(2, 4)}],
            "pixel_values": torch.randn(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        snapshot_keys = set(input_.keys())

        mb_list = MicroBatchList(
            data=input_,
            mb_spec=MicroBatchSpec(),
            mbs=[],
            group_lens=[],
        )
        assert mb_list.data is input_, (
            "MicroBatchList.data must alias caller's input dict to mirror "
            "split_padded_tensor_dict_into_mb_list semantics."
        )

        mb_list.data = {
            k: v for k, v in mb_list.data.items() if not _is_multi_modal_payload_key(k)
        }

        assert set(input_.keys()) == snapshot_keys, (
            "Rebind must not mutate the caller's input dict — regression to "
            "in-place `_drop_multi_modal_payload(mb_list.data)` would fail here."
        )
        assert "multi_modal_input" not in mb_list.data
        assert "pixel_values" not in mb_list.data
        assert "image_grid_thw" not in mb_list.data
        assert "input_ids" in mb_list.data
        assert mb_list.data is not input_


class TestVisionModelDetection:
    # [CLASS DOC]
    # Tests is_valid_vision_model and the Qwen3-VL family helpers on model_type strings.
    """Test is_valid_vision_model detection."""

    def test_qwen2_5_vl_detected(self):
        # [FUNCTION DOC]
        # "qwen2_5_vl" is a vision model.
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_5_vl") is True

    def test_qwen2_vl_detected(self):
        # [FUNCTION DOC]
        # "qwen2_vl" is a vision model.
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_vl") is True

    def test_qwen3_vl_detected(self):
        # [FUNCTION DOC]
        # "qwen3_vl" is a vision model.
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen3_vl") is True

    def test_qwen3_vl_moe_detected(self):
        # [FUNCTION DOC]
        # "qwen3_vl_moe" is a vision model and a Qwen3-VL-MoE; the family helper
        # is_qwen3_vl_model accepts both dense and MoE while the MoE helper rejects
        # "qwen3_vl".
        from areal.engine.core.model import (
            is_qwen3_vl_model,
            is_qwen3_vl_moe_model,
            is_valid_vision_model,
        )

        assert is_valid_vision_model("qwen3_vl_moe") is True
        assert is_qwen3_vl_moe_model("qwen3_vl_moe") is True
        # Family-level helper covers both dense and MoE.
        assert is_qwen3_vl_model("qwen3_vl_moe") is True
        assert is_qwen3_vl_model("qwen3_vl") is True
        assert is_qwen3_vl_moe_model("qwen3_vl") is False

    def test_qwen3_not_detected(self):
        # [FUNCTION DOC]
        # "qwen3" is not a vision model.
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen3") is False

    def test_text_model_not_detected(self):
        # [FUNCTION DOC]
        # "llama" is not a vision model.
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("llama") is False


class TestConvertQwen25VLToHF:
    # [CLASS DOC]
    # Tests mcore-to-HF parameter conversion for Qwen2.5-VL (convert_qwen2_5_vl_to_hf
    # and convert_to_hf dispatch), returning lists of (hf_name, tensor).
    """Test mcore→HF weight name conversion for Qwen2.5-VL."""

    @pytest.fixture()
    def tf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock TransformerConfig with Qwen2.5-VL-3B text dims (hidden
        # 2048, 16 heads, 2 query groups, kv_channels 128).
        """Mock TransformerConfig with Qwen2.5-VL-3B values."""
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 16
        cfg.num_query_groups = 2
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock HF config with vision_config num_heads 16 and hidden_size
        # 1280.
        """Mock HF config with vision_config for Qwen2.5-VL-3B."""
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1280
        return cfg

    # --- Registry dispatch ---

    def test_registry_dispatches_qwen2_5_vl_before_qwen2(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # convert_to_hf with "qwen2_5_vl" routes to the VLM converter (vision final
        # layernorm maps to visual.merger.ln_q.weight) rather than the qwen2 converter.
        """convert_to_hf with model_name='qwen2_5_vl' must use the VLM converter."""
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(2048)
        result = convert_to_hf(
            tf_config,
            "qwen2_5_vl",
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    # --- Language model delegation ---

    def test_language_model_embedding_delegates(self, tf_config):
        # [FUNCTION DOC]
        # Language embedding is delegated to the qwen2 converter: one entry named
        # model.embed_tokens.weight.
        """language_model.embedding should strip prefix and delegate to qwen2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert len(result) == 1
        assert result[0][0] == "model.embed_tokens.weight"

    def test_language_model_output_layer_delegates(self, tf_config):
        # [FUNCTION DOC]
        # Language output_layer maps to lm_head.weight.
        """language_model.output_layer should map to lm_head."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.output_layer.weight",
            param,
        )
        assert result[0][0] == "lm_head.weight"

    def test_language_model_decoder_layer_qkv_delegates(self, tf_config):
        # [FUNCTION DOC]
        # Language fused QKV for layer 5 splits into q_proj, k_proj and v_proj names.
        """language_model decoder QKV should split into q/k/v projections."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        # Qwen2.5-VL-3B: num_query_groups=2, value_num_per_group=8, kv_channels=128
        # QKV weight shape: (num_query_groups * (value_num_per_group + 2) * kv_channels, hidden)
        qkv_dim = tf_config.num_query_groups * (8 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.5.self_attn.q_proj.weight" in names
        assert "model.layers.5.self_attn.k_proj.weight" in names
        assert "model.layers.5.self_attn.v_proj.weight" in names

    def test_language_model_mlp_fc1_splits_gate_up(self, tf_config):
        # [FUNCTION DOC]
        # Language MLP linear_fc1 splits into gate_proj and up_proj.
        """language_model MLP fc1 should split into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(22016, 2048)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.up_proj.weight" in names

    # --- Vision model direct mappings ---

    def test_vision_patch_embed(self, tf_config):
        # [FUNCTION DOC]
        # Vision patch_embed maps 1:1 to visual.patch_embed.proj.weight with the same
        # tensor.
        """vision_model.patch_embed maps to visual.patch_embed."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3, 2, 14, 14)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.weight",
            param,
        )
        assert result == [("visual.patch_embed.proj.weight", param)]

    def test_vision_merger_ln(self, tf_config):
        # [FUNCTION DOC]
        # Vision final layernorm maps to visual.merger.ln_q.weight.
        """vision final layernorm maps to visual.merger.ln_q."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    def test_vision_merger_mlp(self, tf_config):
        # [FUNCTION DOC]
        # Vision projection encoder linear_fc1 maps to visual.merger.mlp.0.weight.
        """vision projection MLP maps to visual.merger.mlp."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(5120, 5120)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.projection.encoder.linear_fc1.weight",
            param,
        )
        assert result[0][0] == "visual.merger.mlp.0.weight"

    # --- Vision model per-layer params ---

    def test_vision_layer_attn_qkv(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # Vision layer QKV maps to visual.blocks.7.attn.qkv.weight with the same shape
        # (rows reordered using hf_config head counts).
        """Vision attention QKV maps to visual.blocks.<idx>.attn.qkv with reordering."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(3840, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape  # same shape, different ordering

    def test_vision_layer_attn_proj(self, tf_config):
        # [FUNCTION DOC]
        # Vision attention linear_proj maps to visual.blocks.3.attn.proj.weight.
        """Vision attention proj maps to visual.blocks.<idx>.attn.proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.3.self_attention.linear_proj.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.3.attn.proj.weight"

    def test_vision_layer_norm1(self, tf_config):
        # [FUNCTION DOC]
        # Vision QKV layer_norm_weight maps to visual.blocks.0.norm1.weight.
        """Vision input layernorm maps to visual.blocks.<idx>.norm1."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.0.norm1.weight"

    def test_vision_layer_mlp_fc1_splits_gate_up(self, tf_config):
        # [FUNCTION DOC]
        # Vision MLP linear_fc1 splits into gate_proj and up_proj (gated MLP in
        # Qwen2.5-VL).
        """Vision MLP fc1 splits into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(6840, 1280)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.2.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "visual.blocks.2.mlp.gate_proj.weight" in names
        assert "visual.blocks.2.mlp.up_proj.weight" in names

    def test_vision_layer_mlp_fc2(self, tf_config):
        # [FUNCTION DOC]
        # Vision MLP linear_fc2 maps to mlp.down_proj.
        """Vision MLP fc2 maps to visual.blocks.<idx>.mlp.down_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3420)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.10.mlp.linear_fc2.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.10.mlp.down_proj.weight"

    def test_vision_layer_norm2(self, tf_config):
        # [FUNCTION DOC]
        # Vision MLP fc1 layer_norm_weight maps to norm2.weight.
        """Vision post-attention layernorm maps to visual.blocks.<idx>.norm2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.1.norm2.weight"

    # --- Error handling ---

    def test_unknown_param_raises_value_error(self, tf_config):
        # [FUNCTION DOC]
        # Unknown name raises ValueError ("Unknown parameter name").
        """Unknown parameter name should raise ValueError."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        with pytest.raises(ValueError, match="Unknown parameter name"):
            convert_qwen2_5_vl_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestConvertQwen3VLToHF:
    # [CLASS DOC]
    # Tests mcore-to-HF conversion for dense Qwen3-VL (convert_qwen3_vl_to_hf), whose HF
    # names use model.language_model.* and model.visual.* prefixes.
    """Test mcore→HF weight name conversion for Qwen3-VL (dense)."""

    @pytest.fixture()
    def tf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock TransformerConfig with Qwen3-VL-2B text dims (hidden 2048,
        # 16 heads, 8 query groups, kv_channels 128).
        """Mock TransformerConfig matching Qwen/Qwen3-VL-2B-Instruct text_config:
        hidden=2048, attn_heads=16, kv_heads=8, head_dim=128.
        """
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 16
        cfg.num_query_groups = 8
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock HF config with vision num_heads 16 and hidden_size 1024.
        """Mock HF config matching Qwen/Qwen3-VL-2B-Instruct vision_config:
        hidden=1024, num_heads=16 (head_dim=64).
        """
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1024
        return cfg

    # --- Registry dispatch ---

    def test_registry_dispatches_qwen3_vl_before_qwen3(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # convert_to_hf with "qwen3_vl" uses the VLM converter (merger patch_norm maps
        # to model.visual.merger.norm.weight) instead of falling through to qwen3.
        """convert_to_hf with model_name='qwen3_vl' must use the VLM converter,
        not fall through to qwen3 (substring match order)."""
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(1024)
        result = convert_to_hf(
            tf_config,
            "qwen3_vl",
            "module.module.vision_model.merger.patch_norm.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.merger.norm.weight"

    # --- Language model (Qwen3-VL has model.language_model.* prefix) ---

    def test_language_model_embedding_uses_language_model_prefix(self, tf_config):
        # [FUNCTION DOC]
        # Embedding maps to model.language_model.embed_tokens.weight with the same
        # tensor.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert result == [("model.language_model.embed_tokens.weight", param)]

    def test_language_model_final_norm(self, tf_config):
        # [FUNCTION DOC]
        # Final layernorm maps to model.language_model.norm.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.final_layernorm.weight",
            param,
        )
        assert result == [("model.language_model.norm.weight", param)]

    def test_language_model_output_layer(self, tf_config):
        # [FUNCTION DOC]
        # Output layer maps to lm_head.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.output_layer.weight",
            param,
        )
        assert result == [("lm_head.weight", param)]

    def test_language_model_qkv_split(self, tf_config):
        # [FUNCTION DOC]
        # Fused QKV weight splits into q/k/v_proj under model.language_model.layers.5.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        # value_num_per_group = 16/8 = 2, head_dim = 128
        # qkv_dim = num_query_groups * (value_num_per_group + 2) * head_dim
        qkv_dim = tf_config.num_query_groups * (2 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.5.self_attn.q_proj.weight" in names
        assert "model.language_model.layers.5.self_attn.k_proj.weight" in names
        assert "model.language_model.layers.5.self_attn.v_proj.weight" in names

    def test_language_model_qkv_bias_split(self, tf_config):
        # [FUNCTION DOC]
        # Fused QKV bias splits into q/k/v_proj biases.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        qkv_dim = tf_config.num_query_groups * (2 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.bias",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.self_attn.q_proj.bias" in names
        assert "model.language_model.layers.0.self_attn.k_proj.bias" in names
        assert "model.language_model.layers.0.self_attn.v_proj.bias" in names

    def test_language_model_q_norm(self, tf_config):
        # [FUNCTION DOC]
        # q_layernorm maps to self_attn.q_norm.weight.
        """Qwen3-specific q_layernorm → self_attn.q_norm."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.self_attention.q_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.self_attn.q_norm.weight", param)
        ]

    def test_language_model_k_norm(self, tf_config):
        # [FUNCTION DOC]
        # k_layernorm maps to self_attn.k_norm.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.self_attention.k_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.self_attn.k_norm.weight", param)
        ]

    def test_language_model_o_proj(self, tf_config):
        # [FUNCTION DOC]
        # linear_proj maps to self_attn.o_proj.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_proj.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.self_attn.o_proj.weight", param)
        ]

    def test_language_model_input_layernorm(self, tf_config):
        # [FUNCTION DOC]
        # QKV layer_norm_weight maps to input_layernorm.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.input_layernorm.weight", param)
        ]

    def test_language_model_post_attention_layernorm(self, tf_config):
        # [FUNCTION DOC]
        # MLP fc1 layer_norm_weight maps to post_attention_layernorm.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.post_attention_layernorm.weight", param)
        ]

    def test_language_model_mlp_fc1_splits_gate_up(self, tf_config):
        # [FUNCTION DOC]
        # Language MLP fc1 splits into gate_proj and up_proj.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(12288, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

    def test_language_model_mlp_fc2(self, tf_config):
        # [FUNCTION DOC]
        # Language MLP fc2 maps to mlp.down_proj.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048, 6144)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc2.weight",
            param,
        )
        assert result == [("model.language_model.layers.0.mlp.down_proj.weight", param)]

    # --- Vision model: direct mappings (model.visual.* prefix) ---

    def test_vision_patch_embed_weight(self, tf_config):
        # [FUNCTION DOC]
        # Vision patch_embed weight maps to model.visual.patch_embed.proj.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(1024, 3, 2, 16, 16)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.weight",
            param,
        )
        assert result == [("model.visual.patch_embed.proj.weight", param)]

    def test_vision_patch_embed_bias(self, tf_config):
        # [FUNCTION DOC]
        # Vision patch_embed bias maps to model.visual.patch_embed.proj.bias.
        """Qwen3-VL adds patch_embed bias (Qwen2.5-VL was bias=False)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.bias",
            param,
        )
        assert result == [("model.visual.patch_embed.proj.bias", param)]

    def test_vision_pos_embed(self, tf_config):
        # [FUNCTION DOC]
        # Vision pos_embed maps to model.visual.pos_embed.weight.
        """Qwen3-VL has pos_embed (new vs Qwen2.5-VL)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2304, 1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.pos_embed.weight",
            param,
        )
        assert result == [("model.visual.pos_embed.weight", param)]

    def test_vision_merger_patch_norm(self, tf_config):
        # [FUNCTION DOC]
        # Merger patch_norm weight/bias map to model.visual.merger.norm.weight/bias.
        """merger.patch_norm.{weight,bias} → merger.norm.{weight,bias}."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.merger.patch_norm.{kind}",
                param,
            )
            assert result == [(f"model.visual.merger.norm.{kind}", param)]

    def test_vision_merger_linear_fc1_fc2(self, tf_config):
        # [FUNCTION DOC]
        # Merger linear_fc1/linear_fc2 weight/bias map 1:1 under model.visual.merger.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for fc in ("linear_fc1", "linear_fc2"):
            for kind in ("weight", "bias"):
                param = (
                    torch.randn(4096, 4096) if kind == "weight" else torch.randn(4096)
                )
                result = convert_qwen3_vl_to_hf(
                    tf_config,
                    f"module.module.vision_model.merger.{fc}.{kind}",
                    param,
                )
                assert result == [(f"model.visual.merger.{fc}.{kind}", param)]

    # --- Vision model: per-layer params ---

    def test_vision_block_qkv_weight_reordered(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # Vision block QKV weight maps to model.visual.blocks.7.attn.qkv.weight, shape
        # preserved (reordered).
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        # 3 * hidden_vision = 3 * 1024 = 3072
        param = torch.randn(3072, 1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape

    def test_vision_block_qkv_bias_reordered(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # Vision block QKV bias maps to attn.qkv.bias, shape preserved.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(3072)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.bias",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.0.attn.qkv.bias"
        assert result[0][1].shape == param.shape

    def test_vision_block_qkv_requires_hf_config(self, tf_config):
        # [FUNCTION DOC]
        # Vision QKV conversion without hf_config raises ValueError mentioning
        # vision_config.num_heads.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(3072, 1024)
        with pytest.raises(ValueError, match="vision_config.num_heads"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.weight",
                param,
            )

    def test_vision_block_attn_proj(self, tf_config):
        # [FUNCTION DOC]
        # Vision attention linear_proj weight/bias map to attn.proj.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (1024, 1024)), ("bias", (1024,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.3.self_attention.linear_proj.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.3.attn.proj.{kind}", param)]

    def test_vision_block_norm1_weight_and_bias(self, tf_config):
        # [FUNCTION DOC]
        # QKV layer_norm weight/bias map to norm1 weight/bias.
        """Qwen3-VL adds norm1.bias (Qwen2.5-VL was weight-only)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.0.norm1.{kind}", param)]

    def test_vision_block_norm2_weight_and_bias(self, tf_config):
        # [FUNCTION DOC]
        # MLP fc1 layer_norm weight/bias map to norm2 weight/bias.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.1.mlp.linear_fc1.layer_norm_{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.1.norm2.{kind}", param)]

    def test_vision_block_mlp_is_NOT_gated(self, tf_config):
        # [FUNCTION DOC]
        # Vision MLP fc1 weight/bias map 1:1 to mlp.linear_fc1 (not split; the Qwen3-VL
        # vision MLP is not gated).
        """Regression guard: Qwen3-VL vision MLP is NOT gated.

        linear_fc1 must map 1:1 (no chunk into gate_proj/up_proj).
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (4096, 1024)), ("bias", (4096,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.2.mlp.linear_fc1.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.2.mlp.linear_fc1.{kind}", param)]

    def test_vision_block_mlp_fc2(self, tf_config):
        # [FUNCTION DOC]
        # Vision MLP fc2 weight/bias map 1:1 to mlp.linear_fc2.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (1024, 4096)), ("bias", (1024,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.10.mlp.linear_fc2.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.10.mlp.linear_fc2.{kind}", param)]

    # --- Deepstack mergers (new in Qwen3-VL) ---

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_deepstack_merger_norm(self, tf_config, idx):
        # [FUNCTION DOC]
        # Parametrized idx 0-2: deepstack merger patch_norm weight/bias map to
        # deepstack_merger_list.idx.norm.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.deepstack_merger_list.{idx}.patch_norm.{kind}",
                param,
            )
            assert result == [
                (f"model.visual.deepstack_merger_list.{idx}.norm.{kind}", param)
            ]

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_deepstack_merger_linear_fc(self, tf_config, idx):
        # [FUNCTION DOC]
        # Parametrized idx 0-2: deepstack merger linear_fc1/fc2 weight/bias map 1:1.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for fc in ("linear_fc1", "linear_fc2"):
            for kind, shape in (("weight", (4096, 4096)), ("bias", (4096,))):
                param = torch.randn(*shape)
                result = convert_qwen3_vl_to_hf(
                    tf_config,
                    f"module.module.vision_model.decoder.deepstack_merger_list.{idx}.{fc}.{kind}",
                    param,
                )
                assert result == [
                    (f"model.visual.deepstack_merger_list.{idx}.{fc}.{kind}", param)
                ]

    # --- Error handling ---

    def test_unknown_lang_param_raises(self, tf_config):
        # [FUNCTION DOC]
        # Unknown language-model param raises ValueError ("Unknown Qwen3-VL language-
        # model").
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL language-model"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.unknown.weight",
                torch.randn(10),
            )

    def test_unknown_param_raises(self, tf_config):
        # [FUNCTION DOC]
        # Unknown top-level param raises ValueError ("Unknown Qwen3-VL parameter").
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestConvertQwen3VLMoEToHF:
    # [CLASS DOC]
    # Tests convert_qwen3_vl_moe_to_hf: MoE experts, router and pre-MLP layernorm,
    # registry ordering against dense, and shared attention/vision paths.
    """Test mcore→HF weight name conversion for Qwen3-VL-MoE.

    Qwen3-VL-MoE shares vision tower + language attention with dense
    Qwen3-VL; this class focuses on MoE-specific tensors (experts, router,
    pre_mlp_layernorm) plus registry-ordering and a few cross-cutting
    regression guards.
    """

    @pytest.fixture()
    def tf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock TransformerConfig with Qwen3-VL-30B-A3B text dims (hidden
        # 2048, 32 heads, 4 query groups, kv_channels 128).
        """Mock TransformerConfig matching Qwen/Qwen3-VL-30B-A3B-Instruct text_config:
        hidden=2048, attn_heads=32, kv_heads=4, head_dim=128.
        """
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 32
        cfg.num_query_groups = 4
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        # [FUNCTION DOC]
        # Fixture: MagicMock HF config with vision num_heads 16 and hidden_size 1152.
        """Mock HF config matching Qwen/Qwen3-VL-30B-A3B-Instruct vision_config."""
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1152
        return cfg

    # --- Registry dispatch (must hit MoE converter before dense) ---

    def test_registry_dispatches_qwen3_vl_moe_before_qwen3_vl(
        self, tf_config, hf_config
    ):
        # [FUNCTION DOC]
        # convert_to_hf with "qwen3_vl_moe" routes to the MoE converter: per-expert fc1
        # "weight5" splits into experts.5 gate_proj and up_proj.
        """``qwen3_vl_moe`` model_name must dispatch to the MoE converter, not
        fall through to ``qwen3_vl`` / ``qwen3_moe`` / ``qwen3`` (substring match
        order). MoE-specific names like ``mlp.experts.linear_fc1.weight0`` would
        otherwise raise ``Unknown Qwen3-VL language-model parameter``.
        """
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(1536, 2048)  # 2 * expert_dim x hidden
        result = convert_to_hf(
            tf_config,
            "qwen3_vl_moe",
            "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight5",
            param,
            hf_config=hf_config,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.experts.5.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.experts.5.up_proj.weight" in names

    def test_registry_dense_still_routes_to_dense(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # convert_to_hf with "qwen3_vl" still routes to the dense converter (fc1 splits
        # into gate/up).
        """Sanity: ``qwen3_vl`` model_name still dispatches to dense converter
        even after MoE entry is inserted. Dense MLP fc1 must chunk gate/up.
        """
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(12288, 2048)
        result = convert_to_hf(
            tf_config,
            "qwen3_vl",
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
            hf_config=hf_config,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

    # --- Shared global / attention paths (regression guards on factoring) ---

    def test_lm_global_embedding_shared_with_dense(self, tf_config):
        # [FUNCTION DOC]
        # Embedding conversion matches the dense path.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert result == [("model.language_model.embed_tokens.weight", param)]

    def test_lm_attention_qkv_split_shared_with_dense(self, tf_config):
        # [FUNCTION DOC]
        # QKV split for layer 4 matches the dense path.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # value_num_per_group = 32/4 = 8, head_dim = 128
        qkv_dim = tf_config.num_query_groups * (8 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.4.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.4.self_attn.q_proj.weight" in names
        assert "model.language_model.layers.4.self_attn.k_proj.weight" in names
        assert "model.language_model.layers.4.self_attn.v_proj.weight" in names

    def test_lm_attention_q_norm_shared_with_dense(self, tf_config):
        # [FUNCTION DOC]
        # q_layernorm conversion matches the dense path.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.7.self_attention.q_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.7.self_attn.q_norm.weight", param)
        ]

    # --- MoE-specific: experts, router, pre_mlp_layernorm ---

    def test_expert_fc1_chunks_gate_up_per_expert(self, tf_config):
        # [FUNCTION DOC]
        # Expert fc1 "weight12" splits into exactly two tensors: experts.12 gate_proj
        # and up_proj.
        """Per-expert ``linear_fc1.weight{idx}`` chunks into gate_proj and
        up_proj at the per-expert HF path.
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(1536, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.mlp.experts.linear_fc1.weight12",
            param,
        )
        assert len(result) == 2
        names = [n for n, _ in result]
        assert "model.language_model.layers.3.mlp.experts.12.gate_proj.weight" in names
        assert "model.language_model.layers.3.mlp.experts.12.up_proj.weight" in names

    def test_expert_fc2_passthrough_per_expert(self, tf_config):
        # [FUNCTION DOC]
        # Expert fc2 "weight12" maps 1:1 to experts.12.down_proj.weight.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(2048, 768)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.mlp.experts.linear_fc2.weight12",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.mlp.experts.12.down_proj.weight", param)
        ]

    def test_router_renamed_to_mlp_gate(self, tf_config):
        # [FUNCTION DOC]
        # mlp.router.weight maps to mlp.gate.weight.
        """``mlp.router.weight`` → HF ``mlp.gate.weight`` (NOT ``mlp.router.weight``)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(128, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.router.weight",
            param,
        )
        assert result == [("model.language_model.layers.0.mlp.gate.weight", param)]

    def test_pre_mlp_layernorm_renamed_to_post_attention(self, tf_config):
        # [FUNCTION DOC]
        # pre_mlp_layernorm.weight maps to post_attention_layernorm.weight.
        """``pre_mlp_layernorm.weight`` → HF ``post_attention_layernorm.weight``."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.pre_mlp_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.5.post_attention_layernorm.weight", param)
        ]

    def test_dense_mlp_fallback_for_mixed_sparse_step(self, tf_config):
        # [FUNCTION DOC]
        # Dense MLP fallback in the MoE converter: fc1 splits gate/up, fc1
        # layer_norm_weight maps to post_attention_layernorm, fc2 maps to down_proj.
        """Layers left non-sparse by ``decoder_sparse_step > 1`` keep the dense
        MLP path. Qwen3-VL-30B-A3B-Instruct uses ``decoder_sparse_step=1`` so
        these branches are dead for that checkpoint, but they must exist for
        future variants with mixed dense/sparse layouts.
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # linear_fc1: chunk gate/up
        param = torch.randn(12288, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

        # linear_fc1.layer_norm_weight: post_attention_layernorm
        ln_param = torch.randn(2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
            ln_param,
        )
        assert result == [
            ("model.language_model.layers.0.post_attention_layernorm.weight", ln_param)
        ]

        # linear_fc2: down_proj passthrough
        fc2_param = torch.randn(2048, 6144)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc2.weight",
            fc2_param,
        )
        assert result == [
            ("model.language_model.layers.0.mlp.down_proj.weight", fc2_param)
        ]

    # --- Vision tower (delegates to shared helper) ---

    def test_vision_block_qkv_reordered_via_shared_helper(self, tf_config, hf_config):
        # [FUNCTION DOC]
        # Vision QKV goes through the shared helper: name
        # model.visual.blocks.7.attn.qkv.weight, shape preserved.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # 3 * hidden_vision = 3 * 1152 = 3456
        param = torch.randn(3456, 1152)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape

    def test_deepstack_merger_via_shared_helper(self, tf_config):
        # [FUNCTION DOC]
        # Deepstack merger patch_norm goes through the shared helper.
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(1152)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.vision_model.decoder.deepstack_merger_list.1.patch_norm.weight",
            param,
        )
        assert result == [("model.visual.deepstack_merger_list.1.norm.weight", param)]

    # --- Error handling ---

    def test_unknown_lang_param_raises(self, tf_config):
        # [FUNCTION DOC]
        # Unknown language-model param raises ValueError ("Unknown Qwen3-VL-MoE
        # language-model").
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL-MoE language-model"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.unknown.weight",
                torch.randn(10),
            )

    def test_unknown_expert_kind_raises(self, tf_config):
        # [FUNCTION DOC]
        # Unknown expert kind (linear_fc99) raises ValueError ("Unknown Qwen3-VL-MoE
        # expert").
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL-MoE expert"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc99.weight0",
                torch.randn(10),
            )

    def test_unknown_top_level_param_raises(self, tf_config):
        # [FUNCTION DOC]
        # Unknown top-level name raises via the shared vision helper ("Unknown Qwen3-VL
        # parameter").
        """Non-LM, non-vision names should raise via the shared vision helper."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestRemovePaddingVLM:
    # [CLASS DOC]
    # Tests remove_padding(name, tensor, vocab_size) for VLM language_model-prefixed
    # vocab tensors.
    """Test remove_padding handles VLM language_model prefixed params."""

    def test_vlm_embedding_padding_removed(self):
        # [FUNCTION DOC]
        # A padded language_model embedding is trimmed to vocab_size rows.
        """VLM language_model embedding should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)  # padded for TP alignment
        result = remove_padding(
            "module.module.language_model.embedding.word_embeddings.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_vlm_output_layer_padding_removed(self):
        # [FUNCTION DOC]
        # A padded language_model output_layer is trimmed to vocab_size rows.
        """VLM language_model output_layer should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)
        result = remove_padding(
            "module.module.language_model.output_layer.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_non_embedding_param_unchanged(self):
        # [FUNCTION DOC]
        # A non-vocab parameter is returned as the same object.
        """Non-embedding params should pass through unchanged."""
        from areal.engine.megatron_utils.megatron import remove_padding

        param = torch.randn(1280)
        result = remove_padding(
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            151936,
        )
        assert result is param
