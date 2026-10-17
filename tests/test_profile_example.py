# [MODULE DOC]
# Tests for the examples/profile tooling: the synthetic repeated SFT dataset used for
# profiling, the Qwen3-30B-A3B profile launcher script, pipeline-rank selection helpers,
# and postprocess_profile (kernel trace views, memory snapshot archiving and summaries).
# Relative paths assume the repo root as CWD.
from __future__ import annotations

import json
from pathlib import Path

from examples.profile.postprocess_profile import postprocess_profile
from examples.profile.profile_rank_utils import (
    pp_rank0_ranks_from_backend,
    resolve_profile_ranks,
)
from examples.profile.train_sft_profile import RepeatedProfileSFTDataset


class _FakeTokenizer:
    # [CLASS DOC]
    # HELPER: Minimal tokenizer stub with eos id 2, pad id 0 and a deterministic encode.
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Maps each non-whitespace character to (ord mod 97) + 3; ignores
        # add_special_tokens. Returns a list of ints.
        del add_special_tokens
        return [(ord(ch) % 97) + 3 for ch in text if not ch.isspace()]


def _write_jsonl(path: Path, events: list[dict]) -> None:
    # [FUNCTION DOC]
    # HELPER: Writes each event dict as one JSON line to path.
    with path.open("w", encoding="utf-8") as fout:
        for event in events:
            json.dump(event, fout)
            fout.write("\n")


def test_repeated_profile_sft_dataset_builds_fixed_loss_span() -> None:
    # [FUNCTION DOC]
    # RepeatedProfileSFTDataset with seq_len 32, size 3 and loss_start_ratio 0.5 yields
    # samples whose input_ids and loss_mask are length 32, with loss only on the second
    # half, and distinct tensor storage per sample.
    dataset = RepeatedProfileSFTDataset(
        _FakeTokenizer(),
        seq_len=32,
        dataset_size=3,
        loss_start_ratio=0.5,
    )

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample["input_ids"].shape == sample["loss_mask"].shape == (32,)
    assert sample["loss_mask"][:16].sum().item() == 0
    assert sample["loss_mask"][16:].sum().item() == 16
    assert dataset[0]["input_ids"].data_ptr() != dataset[1]["input_ids"].data_ptr()


def test_profile_launcher_passes_logprobs_chunk_size_as_cli() -> None:
    # [FUNCTION DOC]
    # Reads the launcher shell script and checks it defaults LOGPROBS_CHUNK_SIZE to 128,
    # passes it as actor.logprobs_chunk_size on the CLI, and does not use the
    # AREAL_LOGPROBS_CHUNK_SIZE env var.
    """The profile launcher must configure chunking explicitly through the CLI."""
    script = Path("examples/profile/run_qwen3_30b_a3b_sft_profile.sh").read_text(
        encoding="utf-8"
    )

    assert "LOGPROBS_CHUNK_SIZE=${LOGPROBS_CHUNK_SIZE:-128}" in script
    assert 'actor.logprobs_chunk_size="${LOGPROBS_CHUNK_SIZE}"' in script
    assert "AREAL_LOGPROBS_CHUNK_SIZE" not in script


def test_profile_rank_utils_resolves_pp_rank0(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # pp_rank0_ranks_from_backend returns global ranks of pipeline stage 0 ("0" for
    # d2p1t2; "0,4" for the attn/ffn hybrid backend); resolve_profile_ranks expands
    # "pp_rank0" by reading actor.backend from a YAML config and passes other strings through.
    config = tmp_path / "profile.yaml"
    config.write_text(
        'actor:\n  backend: "megatron:(attn:d1p2t2c2|ffn:d1p2e4)"\n',
        encoding="utf-8",
    )

    assert pp_rank0_ranks_from_backend("megatron:d2p1t2") == "0"
    assert pp_rank0_ranks_from_backend("megatron:(attn:d1p2t2c2|ffn:d1p2e4)") == "0,4"
    assert resolve_profile_ranks("pp_rank0", config) == "0,4"
    assert resolve_profile_ranks("", config) == ""
    assert resolve_profile_ranks("0,2-4", config) == "0,2-4"


def test_postprocess_profile_writes_kernel_views_and_summary(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Builds a fake log dir (one kernel trace, one memory snapshot, trainer log with memory
    # lines, nvidia-smi CSV) and checks postprocess_profile counts traces/snapshots, takes
    # the peak nvidia-smi MiB, parses trainer memory, writes JSON and Markdown summaries,
    # emits gpu_only trace views both in place and under run_dir/kernel_traces, and copies
    # snapshots into run_dir.
    log_dir = tmp_path / "logs"
    trace_dir = log_dir / "perf_tracer" / "actor"
    snapshot_dir = log_dir / "memory_snapshots" / "step_1"
    run_dir = tmp_path / "run"
    trace_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "snapshot_rank00_p0d0c0t0.pickle").write_bytes(b"snapshot")
    trainer_log = log_dir / "trainer.log"
    trainer_log.write_text(
        "memory allocated (GB): 1.25/2.0, memory reserved (GB): 2.50/3.0\n"
        "device memory used/total (GB): 3.75/80.0\n",
        encoding="utf-8",
    )
    nvidia_smi = run_dir / "nvidia_smi.csv"
    run_dir.mkdir()
    nvidia_smi.write_text(
        "timestamp,index,memory.used [MiB]\nnow,0,123\nnow,0,456\n",
        encoding="utf-8",
    )
    _write_jsonl(
        trace_dir / "traces-r0.jsonl",
        [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "dur": 1,
                "args": {"rank": 0, "correlation": 7},
            },
            {
                "ph": "s",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 7,
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "args": {"rank": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "profile_kernel",
                "pid": 1,
                "tid": -3,
                "ts": 2,
                "dur": 2,
                "args": {"rank": 0, "correlation": 7, "stream": 0},
            },
            {
                "ph": "f",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 7,
                "pid": 1,
                "tid": -3,
                "ts": 2,
                "args": {"rank": 0},
            },
        ],
    )

    summary = postprocess_profile(
        log_dir=log_dir,
        run_dir=run_dir,
        profile_kind="kernel",
        profile_step=1,
        trainer_log=trainer_log,
        nvidia_smi_csv=nvidia_smi,
    )

    assert summary["trace_file_count"] == 1
    assert summary["memory_snapshot_count"] == 1
    assert summary["peak_nvidia_smi_mib"] == 456
    assert summary["trainer_memory_gb"]["memory allocated"] == 1.25
    assert len(summary["archived_memory_snapshots"]) == 1
    assert (run_dir / "profile_summary.json").exists()
    assert (run_dir / "profile_summary.md").exists()
    assert (trace_dir / "traces-r0.gpu_only.chrome.json").exists()
    assert (
        run_dir / "kernel_traces" / "actor" / "traces-r0.gpu_only.chrome.json"
    ).exists()
    assert (
        run_dir / "memory_snapshots" / "step_1" / "snapshot_rank00_p0d0c0t0.pickle"
    ).exists()
