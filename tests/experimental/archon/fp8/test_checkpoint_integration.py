# [MODULE DOC]
# Integration tests chaining the FP8 checkpoint helpers: _detect_fp8_checkpoint,
# _prepare_fp8_state_dict (swap BF16 placeholders for FP8 weights plus float32 scale_inv
# tensors) and dequant_fp8_state_dict (convert back to a target dtype and drop scale
# keys). CPU only, uses tmp_path.
import json
from unittest.mock import patch

import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    _detect_fp8_checkpoint,
    _prepare_fp8_state_dict,
    dequant_fp8_state_dict,
)


class TestFP8CheckpointFlow:
    # [CLASS DOC]
    # Covers the full prepare-then-dequant flow for index-based and single-file
    # checkpoints, BF16 passthrough, and the no-index case.
    def test_full_flow_fp8_to_bf16(self, tmp_path):
        # [FUNCTION DOC]
        # Writes an index listing a weight and its scale_inv; prepares a BF16 [256,512]
        # state dict and asserts the weight became FP8 and the scale key was added; then
        # fills real FP8 zeros and ones scales and dequantizes, asserting the weight is
        # BF16 and the scale key removed.
        weight_key = "model.layers.0.q_proj.weight"
        scale_key = "model.layers.0.q_proj.weight_scale_inv"
        index = {
            "weight_map": {
                weight_key: "model-00001.safetensors",
                scale_key: "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        sd = {weight_key: torch.randn(256, 512, dtype=torch.bfloat16)}
        sd = _prepare_fp8_state_dict(sd, str(tmp_path))
        assert sd[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in sd

        sd[weight_key] = torch.zeros(256, 512, dtype=torch.float8_e4m3fn)
        sd[scale_key] = torch.ones(2, 4, dtype=torch.float32)

        sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)
        assert sd[weight_key].dtype == torch.bfloat16
        assert scale_key not in sd

    def test_full_flow_single_file_fp8_to_bf16(self, tmp_path):
        # [FUNCTION DOC]
        # Same flow using a single model.safetensors (no index): saves FP8 weight and
        # [2,4] scale, asserts detection is True, prepares placeholders, substitutes the
        # saved tensors, dequantizes to BF16 and checks dtype and scale-key removal.
        from safetensors.torch import save_file

        weight_key = "model.layers.0.q_proj.weight"
        scale_key = "model.layers.0.q_proj.weight_scale_inv"
        weight_fp8 = torch.randn(256, 512, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 4, dtype=torch.float32)
        save_file(
            {weight_key: weight_fp8, scale_key: scale_inv},
            tmp_path / "model.safetensors",
        )

        assert _detect_fp8_checkpoint(str(tmp_path)) is True

        sd = {weight_key: torch.empty(256, 512, dtype=torch.bfloat16)}
        sd = _prepare_fp8_state_dict(sd, str(tmp_path))
        assert sd[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in sd

        sd[weight_key] = weight_fp8
        sd[scale_key] = scale_inv

        sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)
        assert sd[weight_key].dtype == torch.bfloat16
        assert scale_key not in sd

    def test_bf16_checkpoint_passthrough(self, tmp_path):
        # [FUNCTION DOC]
        # Index with no scale keys: asserts detection is False and that
        # dequant_fp8_state_dict returns the very same dict object with BF16 weights
        # unchanged.
        index = {
            "weight_map": {
                "model.layers.0.q_proj.weight": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        sd = {
            "model.layers.0.q_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16)
        }
        assert not _detect_fp8_checkpoint(str(tmp_path))
        result = dequant_fp8_state_dict(sd)
        assert result is sd
        assert result["model.layers.0.q_proj.weight"].dtype == torch.bfloat16

    def test_no_index_file_prepare_not_called(self, tmp_path):
        # [FUNCTION DOC]
        # Patches the module-level _prepare_fp8_state_dict with a mock, calls detection
        # on an empty directory, and asserts it returns False and the mock was never
        # called.
        # KNOWN ISSUE: The test only calls detection directly, which never invokes
        # prepare anyway, so the assertion on the mock is trivially satisfied and does
        # not exercise any loader gating logic.
        with patch(
            "areal.experimental.models.archon.fp8_checkpoint._prepare_fp8_state_dict"
        ) as mock_prepare:
            is_fp8 = _detect_fp8_checkpoint(str(tmp_path))
            assert not is_fp8
            mock_prepare.assert_not_called()
