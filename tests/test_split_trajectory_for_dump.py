# [MODULE DOC]
# Unit tests for WorkflowExecutor static helpers: _split_trajectory_for_dump (splits
# token ids by loss mask into prompt/completion text and, for multi-turn, role segments)
# and _compute_output_versions (head/tail version and run-length encoding over generated
# tokens).
"""Tests for WorkflowExecutor._split_trajectory_for_dump and _compute_output_versions."""

import pytest

from areal.infra.workflow_executor import WorkflowExecutor


class _FakeTokenizer:
    # [CLASS DOC]
    # HELPER: Tokenizer stub whose decode returns "[N tokens]" for N ids.
    """Minimal tokenizer that returns a deterministic string for testing."""

    def decode(self, ids: list[int], **kwargs) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns "[<len(ids)> tokens]"; ignores kwargs.
        return f"[{len(ids)} tokens]"


class TestSplitTrajectoryForDump:
    # [CLASS DOC]
    # Tests _split_trajectory_for_dump(ids, mask, tokenizer).
    @pytest.fixture
    def tokenizer(self):
        # [FUNCTION DOC]
        # Fixture: returns a _FakeTokenizer.
        return _FakeTokenizer()

    def test_single_turn_no_segments(self, tokenizer):
        # [FUNCTION DOC]
        # Single prompt run then one gen run: asserts prompt_end 3, 3-token
        # prompt/completion text, segments None.
        ids = [1, 2, 3, 4, 5, 6]
        mask = [0, 0, 0, 1, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 3
        assert result["prompt_text"] == "[3 tokens]"
        assert result["completion_text"] == "[3 tokens]"
        assert result["segments"] is None

    def test_multi_turn_has_segments(self, tokenizer):
        # [FUNCTION DOC]
        # Alternating prompt/gen/context/gen runs: asserts prompt_end 2 and four
        # segments with those roles each of length 2.
        ids = list(range(8))
        mask = [0, 0, 1, 1, 0, 0, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 2
        assert result["segments"] is not None
        assert len(result["segments"]) == 4
        roles = [s["role"] for s in result["segments"]]
        assert roles == ["prompt", "gen", "context", "gen"]
        lengths = [s["len"] for s in result["segments"]]
        assert lengths == [2, 2, 2, 2]

    def test_all_zeros_prompt_only(self, tokenizer):
        # [FUNCTION DOC]
        # All-zero mask: asserts whole sequence is prompt, empty completion, no
        # segments.
        ids = [1, 2, 3]
        mask = [0, 0, 0]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 3
        assert result["prompt_text"] == "[3 tokens]"
        assert result["completion_text"] == "[0 tokens]"
        assert result["segments"] is None

    def test_all_ones_gen_only(self, tokenizer):
        # [FUNCTION DOC]
        # All-one mask: asserts prompt_end 0, empty prompt, 3-token completion, no
        # segments.
        ids = [10, 20, 30]
        mask = [1, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 0
        assert result["prompt_text"] == "[0 tokens]"
        assert result["completion_text"] == "[3 tokens]"
        assert result["segments"] is None

    def test_three_gen_runs(self, tokenizer):
        # [FUNCTION DOC]
        # Three gen runs: asserts roles prompt, gen, context, gen, context, gen.
        ids = list(range(12))
        mask = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 2
        assert result["segments"] is not None
        roles = [s["role"] for s in result["segments"]]
        assert roles == ["prompt", "gen", "context", "gen", "context", "gen"]

    def test_length_mismatch_raises_valueerror(self, tokenizer):
        # [FUNCTION DOC]
        # Asserts ValueError "ids length 2 != mask length 1" on length mismatch.
        with pytest.raises(ValueError, match="ids length 2 != mask length 1"):
            WorkflowExecutor._split_trajectory_for_dump([1, 2], [0], tokenizer)

    def test_single_token_gen(self, tokenizer):
        # [FUNCTION DOC]
        # Single generated token: asserts prompt_end 0 and no segments.
        ids = [42]
        mask = [1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 0
        assert result["segments"] is None


class TestComputeOutputVersions:
    # [CLASS DOC]
    # Tests _compute_output_versions(versions, mask) -> (head, tail, rle) over masked
    # positions, ignoring -1 placeholders; -1/-1/[] when nothing is generated.
    def test_filters_negative_one_placeholders(self):
        # [FUNCTION DOC]
        # Asserts placeholders are skipped: head 5, tail 6, rle [[5,2],[6,1]].
        versions = [-1, -1, 5, 5, 6]
        mask = [0, 0, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 5
        assert tail == 6
        assert rle == [[5, 2], [6, 1]]

    def test_single_version(self):
        # [FUNCTION DOC]
        # Asserts a single version gives head=tail=3 and rle [[3,3]].
        versions = [-1, 3, 3, 3]
        mask = [0, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 3
        assert tail == 3
        assert rle == [[3, 3]]

    def test_multiple_version_transitions(self):
        # [FUNCTION DOC]
        # Asserts transitions 2->3->4 give rle [[2,2],[3,2],[4,1]].
        versions = [-1, 2, 2, 3, 3, 4]
        mask = [0, 1, 1, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 2
        assert tail == 4
        assert rle == [[2, 2], [3, 2], [4, 1]]

    def test_all_masked_out(self):
        # [FUNCTION DOC]
        # Asserts all-masked-out input yields head -1, tail -1, empty rle.
        versions = [1, 2, 3]
        mask = [0, 0, 0]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == -1
        assert tail == -1
        assert rle == []

    def test_interleaved_multi_turn(self):
        # [FUNCTION DOC]
        # Asserts interleaved turns give head 5, tail 6, rle [[5,2],[6,2]].
        versions = [-1, -1, 5, 5, -1, -1, 6, 6]
        mask = [0, 0, 1, 1, 0, 0, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 5
        assert tail == 6
        assert rle == [[5, 2], [6, 2]]
