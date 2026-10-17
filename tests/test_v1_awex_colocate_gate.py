# [MODULE DOC]
# Unit tests for PPOTrainer._is_v1_awex_colocate, the gate that decides whether the
# v1 AWEX weight-update colocation path is used. Verifies it only fires for controller
# v1 with weight_update_mode "awex" regardless of scheduling strategy, and statically
# scans rl_trainer.py source (via the ast module) for ungated "awex" comparisons.
# No GPU, network or process side effects.
# SPDX-License-Identifier: Apache-2.0
"""Guards that v1 AWEX colocation never activates for controller v2."""

import ast
import pathlib

import pytest

import areal.trainer.rl_trainer as rl_trainer_module
from areal.api.cli_args import SchedulingStrategy
from areal.trainer.rl_trainer import PPOTrainer


def _config(version: str, mode: str, colocated: bool):
    # [FUNCTION DOC]
    # HELPER: Builds a duck-typed fake config object. Inputs: version (actor._version,
    # e.g. "v1"/"v2"), mode (actor.weight_update_mode), colocated (if True actor gets a
    # colocation SchedulingStrategy targeting "rollout", else a separation strategy with
    # empty target). Returns an object with .actor and .rollout (rollout strategy None).
    actor = type(
        "Actor",
        (),
        {
            "_version": version,
            "weight_update_mode": mode,
            "scheduling_strategy": SchedulingStrategy(
                type="colocation" if colocated else "separation",
                target="rollout" if colocated else "",
            ),
        },
    )()
    rollout = type("Rollout", (), {"scheduling_strategy": None})()
    return type("Cfg", (), {"actor": actor, "rollout": rollout})()


class TestV1AwexColocateGate:
    # [CLASS DOC]
    # Behavioural tests of the AWEX colocation gate over version/mode/strategy combos.
    @pytest.mark.parametrize(
        "version,mode,colocated,expected",
        [
            ("v1", "awex", True, True),
            ("v1", "awex", False, True),
            ("v2", "awex", True, False),
            ("v2", "awex", False, False),
            ("v1", "xccl", True, False),
            ("v1", "disk", True, False),
        ],
    )
    def test_gate_selects_v1_awex_whatever_the_strategy(
        self, version, mode, colocated, expected
    ):
        # [FUNCTION DOC]
        # Parametrized: creates a PPOTrainer without running __init__ (object.__new__), builds
        # a fake config, and asserts the gate returns `expected` - True only for v1+awex,
        # independent of colocation; False for v2 or non-awex modes.
        trainer = object.__new__(PPOTrainer)
        cfg = _config(version, mode, colocated)

        assert PPOTrainer._is_v1_awex_colocate(trainer, cfg) is expected

    def test_the_default_separation_strategy_still_selects_v1_awex(self):
        # [FUNCTION DOC]
        # Asserts the default SchedulingStrategy is "separation" and that the gate still
        # returns True for a v1 awex config with a separation strategy.
        """AWEX runs opt in with weight_update_mode, not with a strategy.

        examples/math/gsm8k_grpo.yaml leaves actor and rollout on the dataclass
        default, so requiring colocation here skips the meta-server handoff and
        every worker starts its own; the run then waits on 'infer_conf' forever.
        """
        trainer = object.__new__(PPOTrainer)
        default = SchedulingStrategy()

        assert default.type == "separation"
        assert (
            PPOTrainer._is_v1_awex_colocate(
                trainer, _config("v1", "awex", colocated=False)
            )
            is True
        )


class TestNoUngatedAwexChecks:
    # [CLASS DOC]
    # Static source check guarding against new ungated "awex" comparisons in rl_trainer.
    def test_weight_update_mode_awex_is_only_compared_in_the_meta_dispatch(self):
        # [FUNCTION DOC]
        # Reads the rl_trainer module source, parses it, locates the _is_v1_awex_colocate
        # function line range, then collects line numbers of Compare nodes whose left side is
        # an attribute named weight_update_mode compared to the constant "awex" outside that
        # gate. Asserts at most one such comparison exists (the allowed meta dispatch one).
        # KNOWN ISSUE: only inspects the left operand of comparisons and only FunctionDef (not
        # AsyncFunctionDef) for the gate; reversed comparisons would slip through.
        source = pathlib.Path(rl_trainer_module.__file__).read_text()
        tree = ast.parse(source)

        gate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_is_v1_awex_colocate"
        )
        gate_lines = range(gate.lineno, gate.end_lineno + 1)

        bare = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not (
                isinstance(node.left, ast.Attribute)
                and node.left.attr == "weight_update_mode"
            ):
                continue
            if node.lineno in gate_lines:
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == "awex":
                    bare.append(node.lineno)

        # The meta dispatch keeps its comparison: it sits in an elif chain that
        # controller v2 already short-circuits.
        assert len(bare) <= 1, (
            'ungated `weight_update_mode == "awex"` checks at lines '
            f"{bare}; use _is_v1_awex_colocate so controller v2 separation "
            "runs are unaffected"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
