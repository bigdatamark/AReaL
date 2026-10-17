# [MODULE DOC]
# Tests for areal.tools.perf_trace_converter, which turns per-rank JSONL perf-tracer
# event files into Chrome trace JSON. Covers single-file / directory / glob inputs,
# pid/tid remapping and process/thread metadata, the CLI main() output locations, the
# split trace view (separating CPU, CUDA API and GPU kernel lanes while keeping ac2g flow
# events), and the kernel-profile multi-view writer. All I/O is under tmp_path.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from areal.tools.perf_trace_converter import (
    convert_jsonl_to_chrome_trace,
    convert_jsonl_to_split_trace,
    main,
    write_kernel_profile_trace_views,
)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    # [FUNCTION DOC]
    # HELPER: Writes each event dict as one JSON line to path (UTF-8), overwriting the file.
    with path.open("w", encoding="utf-8") as fout:
        for event in events:
            json.dump(event, fout)
            fout.write("\n")


def _extract_metadata(trace_events: list[dict], name: str) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Returns the metadata events (ph == "M") whose name equals the given name.
    return [
        event
        for event in trace_events
        if event.get("ph") == "M" and event.get("name") == name
    ]


def test_convert_single_file_basic(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Converting one file: output uses displayTimeUnit "ms", input metadata is replaced by a
    # "[Rank 0, Process 5]..." process name and "[Thread N]" thread names, payload events are
    # sorted by ts, and original pid/tid are remapped to compact ids (pid 1, tids 1 and 2).
    # String and int rank values are treated alike.
    source = tmp_path / "rank0.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "thread_name",
                "ph": "M",
                "pid": 5,
                "tid": 7,
                "args": {"name": "ignored"},
            },
            {
                "name": "compute",
                "ph": "X",
                "pid": 5,
                "tid": 7,
                "ts": 3,
                "args": {"rank": 0, "payload": "a"},
            },
            {
                "name": "io",
                "ph": "X",
                "pid": 5,
                "tid": 8,
                "ts": 2,
                "args": {"rank": "0"},
            },
        ],
    )

    result = convert_jsonl_to_chrome_trace(source)
    assert result["displayTimeUnit"] == "ms"
    events = result["traceEvents"]

    process_meta = _extract_metadata(events, "process_name")
    assert process_meta and process_meta[0]["args"]["name"].startswith(
        "[Rank 0, Process 5]"
    )

    thread_meta = _extract_metadata(events, "thread_name")
    assert {item["args"]["name"] for item in thread_meta} == {
        "[Thread 7]",
        "[Thread 8]",
    }

    payload_events = [event for event in events if event.get("ph") != "M"]
    assert [event["ts"] for event in payload_events] == [2, 3]
    assert {event["pid"] for event in payload_events} == {1}
    assert {event["tid"] for event in payload_events} == {1, 2}


def test_convert_directory_multi_rank_with_shared_ids(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # Converting a directory of two rank files whose raw pid/tid collide: each rank gets its
    # own process metadata (ranks 0, 1) and thread metadata, events are merged sorted by ts,
    # and remapped pids/tids are distinct per rank (rank 0 -> tid 1, rank 1 -> tid 2).
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 3,
                "ts": 5,
                "args": {"rank": 0},
            }
        ],
    )
    _write_jsonl(
        dir_path / "rank1.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 3,
                "ts": 4,
                "args": {"rank": 1},
            }
        ],
    )

    result = convert_jsonl_to_chrome_trace(dir_path)
    events = result["traceEvents"]

    process_meta = _extract_metadata(events, "process_name")
    assert [item["args"]["rank"] for item in process_meta] == [0, 1]

    thread_meta = _extract_metadata(events, "thread_name")
    assert {
        (item["args"]["rank"], item["args"].get("name")) for item in thread_meta
    } == {
        (0, "[Thread 3]"),
        (1, "[Thread 3]"),
    }

    payload_events = [event for event in events if event.get("ph") != "M"]
    assert [event["ts"] for event in payload_events] == [4, 5]
    assert {event["pid"] for event in payload_events} == {1, 2}
    assert {event["tid"] for event in payload_events} == {1, 2}
    by_rank_tid = {event["args"]["rank"]: event["tid"] for event in payload_events}
    assert by_rank_tid == {0: 1, 1: 2}


def test_convert_glob_not_found(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # A glob matching no files must raise FileNotFoundError.
    pattern = str(tmp_path / "missing" / "*.jsonl")
    with pytest.raises(FileNotFoundError):
        convert_jsonl_to_chrome_trace(pattern)


def test_main_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC]
    # main() with only an input file returns 0 and writes "<stem>.json" next to the input.
    source = tmp_path / "rank.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 2,
                "tid": 4,
                "ts": 1,
                "args": {"rank": 0},
            }
        ],
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main([str(source)])
    assert exit_code == 0

    # With new behavior, output should be in same directory with .json extension
    output_path = tmp_path / "rank.json"
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert payload["traceEvents"]


def test_main_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # [FUNCTION DOC]
    # main() with output "-" returns 0 and prints the trace JSON to stdout.
    source = tmp_path / "rank.jsonl"
    _write_jsonl(
        source,
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 2,
                "tid": 4,
                "ts": 1,
                "args": {"rank": 0},
            }
        ],
    )

    exit_code = main([str(source), "-"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["traceEvents"]


def test_main_default_output_directory(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # main() with a directory input writes "<dir>/traces.json".
    """Test that when input is a directory, output goes to <dir>/traces.json"""
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 3,
                "args": {"rank": 0},
            }
        ],
    )

    exit_code = main([str(dir_path)])
    assert exit_code == 0

    # Output should be in the same directory
    output_path = dir_path / "traces.json"
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert payload["traceEvents"]


def test_main_default_output_multiple_files(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # main() with a glob matching several files writes traces.json in their common parent
    # containing both payload events.
    """Test that when input is a glob matching multiple files, output goes to common parent"""
    dir_path = tmp_path / "traces"
    dir_path.mkdir()

    _write_jsonl(
        dir_path / "rank0.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 3,
                "args": {"rank": 0},
            }
        ],
    )
    _write_jsonl(
        dir_path / "rank1.jsonl",
        [
            {
                "name": "compute",
                "ph": "X",
                "pid": 1,
                "tid": 2,
                "ts": 4,
                "args": {"rank": 1},
            }
        ],
    )

    # Use glob pattern
    pattern = str(dir_path / "*.jsonl")
    exit_code = main([pattern])
    assert exit_code == 0

    # Output should be in the common parent directory
    output_path = dir_path / "traces.json"
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as fin:
        payload = json.load(fin)
    assert len([e for e in payload["traceEvents"] if e.get("ph") != "M"]) == 2


def test_convert_split_trace_keeps_cpu_to_gpu_flow(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # convert_jsonl_to_split_trace keeps both ac2g flow events with one shared id; the flow
    # start lands in a process named with "CUDA API" and the finish in a "GPU kernels/memcpy"
    # process with tid "stream 7" (taken from the kernel's stream arg).
    source = tmp_path / "traces-r0.jsonl"
    _write_jsonl(
        source,
        [
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "_GroupedLinear",
                "pid": 10,
                "tid": 11,
                "ts": 1,
                "dur": 20,
                "args": {"rank": 0, "role": "master"},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 10,
                "tid": -2,
                "ts": 2,
                "dur": 1,
                "args": {"rank": 0, "role": "master", "correlation": 42},
            },
            {
                "ph": "s",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 42,
                "pid": 10,
                "tid": -2,
                "ts": 2,
                "args": {"rank": 0, "role": "master"},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "nvjet_qqtst",
                "pid": 10,
                "tid": -3,
                "ts": 5,
                "dur": 7,
                "args": {
                    "rank": 0,
                    "role": "master",
                    "correlation": 42,
                    "stream": 7,
                },
            },
            {
                "ph": "f",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 42,
                "pid": 10,
                "tid": -3,
                "ts": 5,
                "bp": "e",
                "args": {"rank": 0, "role": "master"},
            },
        ],
    )

    result = convert_jsonl_to_split_trace(source)
    events = result["traceEvents"]
    process_names = {
        event["pid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    flows = [event for event in events if event.get("cat") == "ac2g"]

    assert len(flows) == 2
    assert {event["id"] for event in flows} == {flows[0]["id"]}
    start = next(event for event in flows if event["ph"] == "s")
    finish = next(event for event in flows if event["ph"] == "f")
    assert "CUDA API" in process_names[start["pid"]]
    assert "GPU kernels/memcpy" in process_names[finish["pid"]]
    assert finish["tid"] == "stream 7"


def test_write_kernel_profile_trace_views(tmp_path: Path) -> None:
    # [FUNCTION DOC]
    # write_kernel_profile_trace_views writes five view files (chrome, split_clean, gpu_only,
    # cpu_only, cuda_api_only) next to the source, the split_clean path is
    # "<stem>.split_clean.chrome.json", all exist, and split_clean retains ac2g flow events.
    source = tmp_path / "traces-r0.jsonl"
    _write_jsonl(
        source,
        [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "dur": 1,
                "args": {"rank": 0, "correlation": 9},
            },
            {
                "ph": "s",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 9,
                "pid": 1,
                "tid": -2,
                "ts": 1,
                "args": {"rank": 0},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "kernel",
                "pid": 1,
                "tid": -3,
                "ts": 3,
                "dur": 2,
                "args": {"rank": 0, "correlation": 9, "stream": 0},
            },
            {
                "ph": "f",
                "cat": "ac2g",
                "name": "ac2g",
                "id": 9,
                "pid": 1,
                "tid": -3,
                "ts": 3,
                "args": {"rank": 0},
            },
        ],
    )

    outputs = write_kernel_profile_trace_views(source)

    assert set(outputs) == {
        "chrome",
        "split_clean",
        "gpu_only",
        "cpu_only",
        "cuda_api_only",
    }
    assert outputs["split_clean"] == tmp_path / "traces-r0.split_clean.chrome.json"
    assert all(path.exists() for path in outputs.values())
    with outputs["split_clean"].open("r", encoding="utf-8") as fin:
        split_payload = json.load(fin)
    assert any(event.get("cat") == "ac2g" for event in split_payload["traceEvents"])
