# [MODULE DOC]
# CPU unit tests for the fp32 LM-head patch in areal.models.mcore.registry:
# _is_lm_head_module_name (name matcher), _fp32_lm_head_forward (replacement forward
# that computes logits in float32 and routes tensor-parallel collectives through the
# module's own tp_group), and _enable_fp32_lm_head_forward (patches matching submodules,
# returns how many were patched). megatron tensor_parallel helpers are replaced with
# identity fakes. Running the file directly invokes pytest on itself.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fp32 lm-head forward patch."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from areal.models.mcore.registry import (
    _enable_fp32_lm_head_forward,
    _fp32_lm_head_forward,
    _is_lm_head_module_name,
)


class TestLmHeadModuleName:
    # [CLASS DOC]
    # Tests _is_lm_head_module_name.
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("output_layer", True),
            ("lm_head", True),
            ("decoder.output_layer", True),
            ("module.module.lm_head", True),
            ("output_layer_extra", False),
            ("decoder.layers.0.mlp", False),
            ("", False),
        ],
    )
    def test_matches_only_lm_head_modules(self, name, expected):
        # [FUNCTION DOC]
        # Parametrized: names whose last component is exactly "output_layer" or
        # "lm_head" (with any prefix) match; suffixed names, other layers and empty
        # string do not.
        assert _is_lm_head_module_name(name) is expected


class _Head(torch.nn.Module):
    # [CLASS DOC]
    # HELPER: Minimal fake Megatron ColumnParallelLinear-style LM head: 4x3 weight of
    # ones, no bias, skip_bias_add/sequence_parallel/explicit_expert_comm/gather_output
    # False, and a configurable tp_group attribute.
    def __init__(self, tp_group):
        # [FUNCTION DOC]
        # HELPER: Stores tp_group and initializes the fixed attributes described on the
        # class.
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 3))
        self.bias = None
        self.skip_bias_add = False
        self.sequence_parallel = False
        self.explicit_expert_comm = False
        self.gather_output = False
        self.tp_group = tp_group


class TestForwardForwardsTheTensorParallelGroup:
    # [CLASS DOC]
    # Checks that the fp32 forward passes the module's tp_group to the TP collectives
    # and returns float32 output.
    def test_collectives_receive_the_modules_tp_group(self):
        # [FUNCTION DOC]
        # With gather_output True and a sentinel group object, patches tensor_parallel
        # with recording fakes, runs _fp32_lm_head_forward on a 2x3 input, and asserts
        # both copy_to and gather_from received that exact group.
        group = object()
        head = _Head(group)
        seen = {}

        def copy_to(input_, group=None):
            # [FUNCTION DOC]
            # HELPER: Fake copy_to_tensor_model_parallel_region that records the group
            # and returns input unchanged.
            seen["copy_to"] = group
            return input_

        def gather_from(input_, group=None):
            # [FUNCTION DOC]
            # HELPER: Fake gather_from_tensor_model_parallel_region that records the
            # group and returns input unchanged.
            seen["gather_from"] = group
            return input_

        head.gather_output = True
        with mock.patch(
            "areal.models.mcore.registry.tensor_parallel",
            SimpleNamespace(
                copy_to_tensor_model_parallel_region=copy_to,
                gather_from_tensor_model_parallel_region=gather_from,
                gather_from_sequence_parallel_region=lambda *a, **k: a[0],
            ),
        ):
            _fp32_lm_head_forward(head, torch.ones(2, 3))

        assert seen["copy_to"] is group, (
            "copy_to_tensor_model_parallel_region ran on the default group; "
            "mcore forwards self.tp_group here"
        )
        assert seen["gather_from"] is group, (
            "gather_from_tensor_model_parallel_region ran on the default group; "
            "mcore forwards self.tp_group here"
        )

    def test_output_is_float32(self):
        # [FUNCTION DOC]
        # Given a bfloat16 input and identity collectives, the forward returns an
        # (output, bias) pair whose output dtype is float32.
        head = _Head(None)
        with mock.patch(
            "areal.models.mcore.registry.tensor_parallel",
            SimpleNamespace(
                copy_to_tensor_model_parallel_region=lambda i, group=None: i,
                gather_from_tensor_model_parallel_region=lambda i, group=None: i,
                gather_from_sequence_parallel_region=lambda *a, **k: a[0],
            ),
        ):
            output, _bias = _fp32_lm_head_forward(head, torch.ones(2, 3).bfloat16())

        assert output.dtype is torch.float32


class TestEnablePatch:
    # [CLASS DOC]
    # Tests _enable_fp32_lm_head_forward(modules, enabled).
    def test_disabled_patches_nothing(self):
        # [FUNCTION DOC]
        # enabled=False patches nothing and returns 0.
        head = _Head(None)
        assert _enable_fp32_lm_head_forward([head], enabled=False) == 0

    def test_patch_is_idempotent(self):
        # [FUNCTION DOC]
        # A model with an "output_layer" child is patched once (returns 1); a second
        # call returns 0 because already-patched modules are skipped.
        model = torch.nn.Module()
        model.add_module("output_layer", _Head(None))

        first = _enable_fp32_lm_head_forward([model], enabled=True)
        second = _enable_fp32_lm_head_forward([model], enabled=True)

        assert first == 1
        assert second == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
