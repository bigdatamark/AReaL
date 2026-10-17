# [MODULE DOC]
# Unit tests for _prepare_fp8_state_dict, which reads FP8 checkpoint metadata and
# replaces BF16 placeholder weights with FP8 e4m3 tensors plus float32 scale_inv tensors
# of shape ceil(rows/128) by ceil(cols/128). CPU only, uses tmp_path.
import json
from unittest.mock import patch

import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    _prepare_fp8_state_dict,
)


class TestPrepareFP8StateDict:
    # [CLASS DOC]
    # Tests placeholder creation, non-aligned scale shapes, passthrough without
    # checkpoint metadata, missing-weight warnings, and single-file safetensors input.
    def _make_index_and_state_dict(self, tmp_path, dim0=256, dim1=512):
        # [FUNCTION DOC]
        # HELPER: Writes a model.safetensors.index.json mapping a q_proj weight and its
        # weight_scale_inv to one shard, and builds a state dict with an empty BF16
        # weight of shape [dim0, dim1] (defaults 256x512). Returns (state_dict,
        # weight_key, scale_key). Side effect: writes the index file into tmp_path.
        weight_key = "model.layers.0.self_attn.q_proj.weight"
        scale_key = "model.layers.0.self_attn.q_proj.weight_scale_inv"
        index = {
            "weight_map": {
                weight_key: "model-00001.safetensors",
                scale_key: "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        hf_state_dict = {
            weight_key: torch.empty(dim0, dim1, dtype=torch.bfloat16),
        }
        return hf_state_dict, weight_key, scale_key

    def test_creates_correct_placeholders(self, tmp_path):
        # [FUNCTION DOC]
        # Asserts the weight becomes FP8 and a float32 scale of shape (2, 4) is added
        # for a 256x512 weight.
        hf_sd, weight_key, scale_key = self._make_index_and_state_dict(
            tmp_path, dim0=256, dim1=512
        )
        result = _prepare_fp8_state_dict(hf_sd, str(tmp_path))
        assert result[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in result
        assert result[scale_key].dtype == torch.float32
        assert result[scale_key].shape == (2, 4)  # 256/128, 512/128

    def test_scale_shape_non_aligned(self, tmp_path):
        # [FUNCTION DOC]
        # For a 300x500 weight asserts the scale shape is (3, 4), i.e. ceil division by
        # 128.
        hf_sd, _, scale_key = self._make_index_and_state_dict(
            tmp_path, dim0=300, dim1=500
        )
        result = _prepare_fp8_state_dict(hf_sd, str(tmp_path))
        assert result[scale_key].shape == (3, 4)  # ceil(300/128), ceil(500/128)

    def test_no_index_file_passthrough(self, tmp_path):
        # [FUNCTION DOC]
        # With no index or safetensors file the function must return the same dict
        # object.
        original = {"some.weight": torch.empty(128, 128, dtype=torch.bfloat16)}
        result = _prepare_fp8_state_dict(original, str(tmp_path))
        assert result is original

    def test_missing_weight_key_warns(self, tmp_path):
        # [FUNCTION DOC]
        # Index lists only a scale key whose weight is absent from the state dict;
        # patches the module logger and asserts unrelated keys survive and a warning
        # containing "no matching weight" was logged.
        scale_key = "model.layers.0.missing.weight_scale_inv"
        index = {"weight_map": {scale_key: "model-00001.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        hf_sd = {"other.weight": torch.randn(128, 128, dtype=torch.bfloat16)}
        with patch(
            "areal.experimental.models.archon.fp8_checkpoint.logger"
        ) as mock_logger:
            result = _prepare_fp8_state_dict(hf_sd, str(tmp_path))
        assert "other.weight" in result
        warning_msgs = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("no matching weight" in msg for msg in warning_msgs)

    def test_from_single_file_safetensors(self, tmp_path):
        # [FUNCTION DOC]
        # Saves an FP8 weight and float32 scale in a single model.safetensors; asserts
        # prepare converts the BF16 placeholder to FP8 and adds a float32 scale key.
        from safetensors.torch import save_file

        weight_key = "model.layers.0.self_attn.q_proj.weight"
        scale_key = "model.layers.0.self_attn.q_proj.weight_scale_inv"
        tensors = {
            weight_key: torch.zeros(256, 512, dtype=torch.float8_e4m3fn),
            scale_key: torch.ones(2, 4, dtype=torch.float32),
        }
        save_file(tensors, tmp_path / "model.safetensors")
        hf_sd = {weight_key: torch.empty(256, 512, dtype=torch.bfloat16)}
        result = _prepare_fp8_state_dict(hf_sd, str(tmp_path))

        assert result[weight_key].dtype == torch.float8_e4m3fn
        assert scale_key in result
        assert result[scale_key].dtype == torch.float32
