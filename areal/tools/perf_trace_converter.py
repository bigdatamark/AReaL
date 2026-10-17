# [MODULE DOC]
# CLI and library for converting AReaL PerfTracer JSONL output (one Chrome-trace event
# dict per line, possibly from many ranks/roles, with args.rank and args.role) into
# Chrome Trace JSON viewable in Perfetto/chrome://tracing. Two views: a merged view in
# which (rank, role, original pid/tid) are remapped to unique sequential ids with
# generated process/thread name metadata, and a "split" view that places CPU scopes,
# CUDA API calls and GPU work (kernels/memcpy/memset, keyed by stream) on separate
# process tracks per (rank, role), keeping CPU-to-GPU "ac2g" flow arrows. Flow ids and
# args.correlation values are renumbered per (rank, role) to avoid cross-rank
# collisions. Category sets and track labels are module constants.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from glob import glob
from pathlib import Path

_CPU_TRACE_CATEGORIES = frozenset(
    {
        "cpu_op",
        "user_annotation",
        "fwdbwd",
        "io",
        "compute",
        "misc",
        "instr",
    }
)
_CUDA_API_TRACE_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
_GPU_TRACE_CATEGORIES = frozenset(
    {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
)
_CPU_TO_GPU_FLOW_CATEGORIES = frozenset({"ac2g"})

_SPLIT_TRACK_CPU = "cpu"
_SPLIT_TRACK_CUDA_API = "cuda_api"
_SPLIT_TRACK_GPU = "gpu"

_SPLIT_TRACK_LABELS = {
    _SPLIT_TRACK_CPU: "CPU scopes",
    _SPLIT_TRACK_CUDA_API: "CUDA API",
    _SPLIT_TRACK_GPU: "GPU kernels/memcpy",
}

_SPLIT_TRACK_SORT_INDEX = {
    _SPLIT_TRACK_CPU: 0,
    _SPLIT_TRACK_CUDA_API: 1,
    _SPLIT_TRACK_GPU: 2,
}


def _load_events(path: Path) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Reads a JSONL file (UTF-8), skipping blank lines, and returns the parsed
    # events. Raises ValueError naming the line on JSON decode errors.
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid payload
                raise ValueError(
                    f"Failed to parse JSONL at line {lineno} in {path}: {exc}"
                ) from exc
    return events


def _extract_rank(event: dict) -> str | int | None:
    # [FUNCTION DOC]
    # HELPER: Returns args.rank as an int when numeric or a numeric string, the stripped
    # string for other non-empty strings, str(value) for other types, and None when args
    # is missing, rank is absent/empty, boolean, or a non-convertible number.
    """Best-effort extraction of the rank identifier from a trace event."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    rank = args.get("rank")
    if rank is None:
        return None
    if isinstance(rank, bool):  # guard against bool subclassing int
        return None
    if isinstance(rank, int | float):
        try:
            return int(rank)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(rank, str):
        text = rank.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return text
    return str(rank)


def _extract_role(event: dict) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns stripped args.role if it is a non-empty string, else None.
    """Best-effort extraction of the role identifier from a trace event."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    role = args.get("role")
    if role is None or not isinstance(role, str):
        return None
    return role.strip() or None


def _format_rank(rank: str | int) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the rank as a string.
    return str(rank)


def _rank_sort_key(rank: str | int | None) -> tuple[int, object]:
    # [FUNCTION DOC]
    # HELPER: Sort key ordering int ranks first (numerically), then string ranks, then
    # None.
    if rank is None:
        return (2, 0)
    if isinstance(rank, int):
        return (0, rank)
    return (1, str(rank))


def _role_sort_key(role: str | None) -> tuple[int, str]:
    # [FUNCTION DOC]
    # HELPER: Sort key putting named roles (alphabetically) before None.
    if role is None:
        return (1, "")
    return (0, role)


def _value_sort_key(value: object) -> tuple[int, object]:
    # [FUNCTION DOC]
    # HELPER: Total-order sort key for heterogeneous values: bools, ints, floats,
    # strings, then anything else by repr.
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (1, value)
    if isinstance(value, float):
        return (2, value)
    if isinstance(value, str):
        return (3, value)
    return (4, repr(value))


def _metadata_name_sort_key(name: object) -> int:
    # [FUNCTION DOC]
    # HELPER: Orders metadata events: process_name, process_sort_index, thread_name,
    # others.
    if name == "process_name":
        return 0
    if name == "process_sort_index":
        return 1
    if name == "thread_name":
        return 2
    return 3


def _tid_sort_key(value: object) -> tuple[int, object]:
    # [FUNCTION DOC]
    # HELPER: Sort key for thread ids: non-negative ints ascending, then negative ints
    # by magnitude, then floats/strings/others (bools first).
    """Sort key for TIDs: positive ints < negative ints < others."""
    prio, val = _value_sort_key(value)
    if prio == 1:  # int
        if isinstance(val, int) and val < 0:
            return (2, -val)
        return (1, val)
    if prio >= 2:
        return (prio + 1, val)
    return (prio, val)


def _remap_process_and_thread_ids(
    events: list[dict],
    existing_process_names: dict[tuple[str | int, str | None, object], str]
    | None = None,
    existing_thread_names: dict[tuple[str | int, str | None, object, object], str]
    | None = None,
) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Mutates events in place. Considers only events with a rank and pid.
    # Assigns new pids 1..N to sorted unique (rank, role, original pid) keys, and for
    # each new pid assigns thread ids starting at the pid value and incrementing over
    # sorted (rank, role, pid, tid) keys. Rewrites pid/tid of qualifying events (tid set
    # to None if unmapped, which cannot normally happen). Returns generated metadata
    # events: process_name (from existing_process_names or "[role] Rank r, Process p" /
    # "[Rank r, Process p]"), process_sort_index (= new pid), thread_name (existing or
    # "[Thread t]") and thread_sort_index (= new tid), each carrying rank/role in args.
    # KNOWN ISSUE: events without a rank keep their original pid/tid, which may collide
    # with the newly assigned small integer ids.
    """Remap pid/tid to be unique and return metadata events.

    This function modifies the `events` list in-place by replacing `pid` and
    `tid` values. It returns a new list of generated metadata events.
    """
    if existing_process_names is None:
        existing_process_names = {}
    if existing_thread_names is None:
        existing_thread_names = {}

    # pid_keys: (rank, role, original_pid)
    pid_keys: set[tuple[str | int, str | None, object]] = set()
    # tid_keys: (rank, role, original_pid, original_tid)
    tid_keys: set[tuple[str | int, str | None, object, object]] = set()

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        role = _extract_role(event)
        original_pid = event.get("pid")
        if original_pid is None:
            continue
        pid_keys.add((rank, role, original_pid))

        original_tid = event.get("tid")
        if original_tid is not None:
            tid_keys.add((rank, role, original_pid, original_tid))

    sorted_pid_keys = sorted(
        pid_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
        ),
    )

    pid_map: dict[tuple[str | int, str | None, object], int] = {}
    pid_labels: dict[int, tuple[str | int, str | None, object]] = {}
    for new_pid, key in enumerate(sorted_pid_keys):
        pid_map[key] = new_pid + 1
        pid_labels[new_pid + 1] = key

    tid_counters: dict[int, int] = {}
    tid_map: dict[tuple[str | int, str | None, object, object], int] = {}
    tid_labels: dict[tuple[int, int], tuple[str | int, str | None, object]] = {}

    sorted_tid_keys = sorted(
        tid_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
            _tid_sort_key(item[3]),
        ),
    )

    for key in sorted_tid_keys:
        rank, role, original_pid, original_tid = key
        new_pid = pid_map[(rank, role, original_pid)]
        next_tid = tid_counters.get(new_pid, new_pid)
        tid_counters[new_pid] = next_tid + 1
        tid_map[key] = next_tid
        tid_labels[(new_pid, next_tid)] = (rank, role, original_tid)

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue

        role = _extract_role(event)
        original_pid = event.get("pid")
        if original_pid is None:
            continue
        new_pid = pid_map[(rank, role, original_pid)]
        event["pid"] = new_pid

        original_tid = event.get("tid")
        if original_tid is not None:
            tid_key = (rank, role, original_pid, original_tid)
            if tid_key in tid_map:
                event["tid"] = tid_map[tid_key]
            else:
                # Defensive: leave event["tid"] as is, or set to None, or log warning
                event["tid"] = None

    metadata_events: list[dict] = []
    for pid, (rank, role, original_pid) in pid_labels.items():
        rank_text = _format_rank(rank)
        process_name = existing_process_names.get((rank, role, original_pid))
        if process_name is None:
            if role:
                process_name = f"[{role}] Rank {rank_text}, Process {original_pid}"
            else:
                process_name = f"[Rank {rank_text}, Process {original_pid}]"

        args: dict = {"name": process_name, "rank": rank}
        if role is not None:
            args["role"] = role

        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": args,
            }
        )
        sort_args: dict = {"sort_index": pid, "rank": rank}
        if role is not None:
            sort_args["role"] = role
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": sort_args,
            }
        )

    for (pid, tid), (rank, role, original_tid) in tid_labels.items():
        # Retrieve the correct original_pid for this new_pid
        _, _, original_pid = pid_labels[pid]

        rank_text = _format_rank(rank)
        thread_name = existing_thread_names.get(
            (rank, role, original_pid, original_tid)
        )
        if thread_name is None:
            thread_name = f"[Thread {original_tid}]"

        thread_args: dict = {"name": thread_name, "rank": rank}
        if role is not None:
            thread_args["role"] = role

        metadata_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": thread_args,
            }
        )
        metadata_events.append(
            {
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"sort_index": tid, "rank": rank},
            }
        )

    return metadata_events


def _resolve_trace_files(source: Path) -> list[Path]:
    # [FUNCTION DOC]
    # HELPER: Returns [source] if it is a file, all *.jsonl files under it recursively
    # (sorted) if it is a directory, otherwise the sorted files matching it as a
    # recursive glob pattern (possibly empty).
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*.jsonl") if p.is_file())
    matches = [Path(p) for p in glob(str(source), recursive=True)]
    files = [p for p in matches if p.is_file()]
    return sorted(files)


def _load_trace_events(input_path: str | os.PathLike[str]) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Resolves input files and concatenates their events in sorted path order.
    # Raises FileNotFoundError if nothing matches.
    sources = _resolve_trace_files(Path(input_path))
    if not sources:
        raise FileNotFoundError(f"No trace files matched input path: {input_path}")

    events: list[dict] = []
    for path in sources:
        events.extend(_load_events(path))
    return events


def _clone_events(events: list[dict]) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Returns shallow copies of events, also copying each args dict, so later
    # in-place remapping does not affect the source list.
    cloned_events: list[dict] = []
    for event in events:
        cloned_event = dict(event)
        args = event.get("args")
        if isinstance(args, dict):
            cloned_event["args"] = dict(args)
        cloned_events.append(cloned_event)
    return cloned_events


def _write_chrome_trace(path: Path, chrome_trace: dict) -> None:
    # [FUNCTION DOC]
    # HELPER: Creates the parent directory if needed and writes the trace dict as JSON
    # (UTF-8, non-ASCII preserved).
    if path.parent != Path(".") and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(chrome_trace, fout, ensure_ascii=False)


def _remap_flow_ids(events: list[dict]) -> None:
    # [FUNCTION DOC]
    # HELPER: In place, renumbers flow event ids (phases s/t/f) and args.correlation
    # values to sequential integers from 1, keyed by (rank, role, value) so that flow
    # ids and correlations with the same value within a rank/role map to the same
    # number. Events without a rank are untouched.
    # Collect all unique flow IDs / correlations to remap them sequentially.
    # flow_id_keys: (rank, role, flow_id)
    flow_id_keys: set[tuple[str | int, str | None, object]] = set()
    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue
        role = _extract_role(event)

        # Collect from flow events.
        if event.get("ph") in ("s", "t", "f") and "id" in event:
            flow_id_keys.add((rank, role, event["id"]))

        # Collect from args.correlation.
        args = event.get("args")
        if isinstance(args, dict) and "correlation" in args:
            flow_id_keys.add((rank, role, args["correlation"]))

    sorted_flow_keys = sorted(
        flow_id_keys,
        key=lambda item: (
            _rank_sort_key(item[0]),
            _role_sort_key(item[1]),
            _value_sort_key(item[2]),
        ),
    )
    flow_id_map = {key: i for i, key in enumerate(sorted_flow_keys, start=1)}

    for event in events:
        rank = _extract_rank(event)
        if rank is None:
            continue
        role = _extract_role(event)

        if event.get("ph") in ("s", "t", "f") and "id" in event:
            key = (rank, role, event["id"])
            if key in flow_id_map:
                event["id"] = flow_id_map[key]

        args = event.get("args")
        if isinstance(args, dict) and "correlation" in args:
            key = (rank, role, args["correlation"])
            if key in flow_id_map:
                args["correlation"] = flow_id_map[key]


def _build_chrome_trace(events: list[dict], display_time_unit: str) -> dict:
    # [FUNCTION DOC]
    # HELPER: Builds the merged-view trace from mutable events: harvests existing
    # process_name/thread_name metadata (keyed by rank, role, pid[, tid]) and drops all
    # process/thread name and sort-index metadata events, renumbers flow ids, remaps
    # pids/tids and generates fresh metadata, sorts metadata by rank/role/kind/pid/tid
    # and other events by (ts, pid, tid), and returns {"traceEvents": metadata + events,
    # "displayTimeUnit": display_time_unit}.
    """Build a Chrome trace payload from mutable trace event dictionaries."""

    existing_process_names: dict[tuple[str | int, str | None, object], str] = {}
    existing_thread_names: dict[tuple[str | int, str | None, object, object], str] = {}

    filtered_events: list[dict] = []
    ignored_metadata = {
        "process_name",
        "thread_name",
        "process_sort_index",
        "thread_sort_index",
    }
    for event in events:
        rank = _extract_rank(event)
        role = _extract_role(event)
        if event.get("ph") == "M":
            name = event.get("name")
            args = event.get("args", {})
            pid = event.get("pid")
            tid = event.get("tid")

            if rank is not None and pid is not None:
                if name == "process_name" and isinstance(args, dict):
                    pname = args.get("name")
                    if pname:
                        existing_process_names[(rank, role, pid)] = str(pname)
                elif (
                    name == "thread_name" and tid is not None and isinstance(args, dict)
                ):
                    tname = args.get("name")
                    if tname:
                        existing_thread_names[(rank, role, pid, tid)] = str(tname)

            if name in ignored_metadata:
                continue
        filtered_events.append(event)

    events = filtered_events

    _remap_flow_ids(events)

    metadata_events = _remap_process_and_thread_ids(
        events,
        existing_process_names=existing_process_names,
        existing_thread_names=existing_thread_names,
    )

    metadata_events.sort(
        key=lambda event: (
            _rank_sort_key(event.get("args", {}).get("rank")),
            _role_sort_key(event.get("args", {}).get("role")),
            _metadata_name_sort_key(event.get("name")),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events.sort(
        key=lambda event: (
            event.get("ts", 0),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
        )
    )

    events = metadata_events + events

    chrome_trace = {
        "traceEvents": events,
        "displayTimeUnit": display_time_unit,
    }
    return chrome_trace


def convert_jsonl_to_chrome_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
) -> dict:
    # [FUNCTION DOC]
    # Public API: loads events from a file, directory or glob, builds the merged Chrome
    # trace on cloned events, writes it to output_path if given, and returns the trace
    # dict. Raises FileNotFoundError if no input matches, ValueError on bad JSON.
    """Convert newline-delimited trace events into Chrome Trace JSON.

    The ``input_path`` may point to a single JSONL file, a directory containing
    per-rank JSONL files, or a glob pattern. All matching files are concatenated
    in lexical order before emitting the Chrome trace payload.
    """

    events = _load_trace_events(input_path)
    chrome_trace = _build_chrome_trace(_clone_events(events), display_time_unit)

    if output_path is not None:
        _write_chrome_trace(Path(output_path), chrome_trace)
    return chrome_trace


def _split_track_for_category(category: object) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Maps an event category to the cpu, cuda_api or gpu track name, or None for
    # other categories.
    if category in _CPU_TRACE_CATEGORIES:
        return _SPLIT_TRACK_CPU
    if category in _CUDA_API_TRACE_CATEGORIES:
        return _SPLIT_TRACK_CUDA_API
    if category in _GPU_TRACE_CATEGORIES:
        return _SPLIT_TRACK_GPU
    return None


def _event_stream_tid(event: dict) -> str | object:
    # [FUNCTION DOC]
    # HELPER: Returns "stream <id>" when args.stream is set (GPU events), else the
    # event's tid.
    args = event.get("args")
    if isinstance(args, dict) and args.get("stream") is not None:
        return f"stream {args['stream']}"
    return event.get("tid")


def _trace_identity(event: dict) -> tuple[str | int | None, str | None]:
    # [FUNCTION DOC]
    # HELPER: Returns (rank, role) extracted from an event.
    return (_extract_rank(event), _extract_role(event))


def _split_process_key(
    event: dict, track: str
) -> tuple[str | int | None, str | None, str]:
    # [FUNCTION DOC]
    # HELPER: Returns the split-view process key (rank, role, track).
    rank, role = _trace_identity(event)
    return (rank, role, track)


def _split_process_sort_key(
    key: tuple[str | int | None, str | None, str],
) -> tuple[tuple[int, object], tuple[int, str], int]:
    # [FUNCTION DOC]
    # HELPER: Sort key for split process keys: rank, then role, then track order (cpu,
    # cuda_api, gpu).
    rank, role, track = key
    return (
        _rank_sort_key(rank),
        _role_sort_key(role),
        _SPLIT_TRACK_SORT_INDEX[track],
    )


def _build_flow_endpoint_maps(
    events: list[dict],
) -> dict[tuple[str | int | None, str | None, object], dict[str, object]]:
    # [FUNCTION DOC]
    # HELPER: Scans complete ("X") CUDA API and GPU events carrying args.correlation and
    # returns a map from (rank, role, correlation) to the endpoint tids: the CUDA API
    # event's tid and the GPU event's stream-based tid.
    endpoints: dict[tuple[str | int | None, str | None, object], dict[str, object]] = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category not in _CUDA_API_TRACE_CATEGORIES | _GPU_TRACE_CATEGORIES:
            continue
        args = event.get("args")
        if not isinstance(args, dict) or "correlation" not in args:
            continue

        key = (*_trace_identity(event), args["correlation"])
        endpoint = endpoints.setdefault(key, {})
        if category in _CUDA_API_TRACE_CATEGORIES:
            endpoint[_SPLIT_TRACK_CUDA_API] = event.get("tid")
        elif category in _GPU_TRACE_CATEGORIES:
            endpoint[_SPLIT_TRACK_GPU] = _event_stream_tid(event)
    return endpoints


def _fallback_flow_track(event: dict) -> str:
    # [FUNCTION DOC]
    # HELPER: Guesses a flow event's track from its original tid when no endpoint is
    # known: -3 means GPU, -2 means CUDA API, otherwise CPU (PyTorch profiler
    # conventions).
    original_tid = event.get("tid")
    if original_tid == -3:
        return _SPLIT_TRACK_GPU
    if original_tid == -2:
        return _SPLIT_TRACK_CUDA_API
    return _SPLIT_TRACK_CPU


def _flow_endpoint(
    event: dict,
    flow_endpoints: dict[
        tuple[str | int | None, str | None, object], dict[str, object]
    ],
) -> tuple[str, object]:
    # [FUNCTION DOC]
    # HELPER: Chooses the (track, tid) where an ac2g flow event should be attached: flow
    # end ("f") goes to the GPU endpoint, flow start ("s") to the CUDA API endpoint,
    # otherwise whichever endpoint is known (CUDA API preferred), else the fallback
    # track with a stream or original tid. Lookup uses the event id as correlation.
    key = (*_trace_identity(event), event.get("id"))
    endpoint = flow_endpoints.get(key, {})
    phase = event.get("ph")

    if phase == "f" and _SPLIT_TRACK_GPU in endpoint:
        return (_SPLIT_TRACK_GPU, endpoint[_SPLIT_TRACK_GPU])
    if phase == "s" and _SPLIT_TRACK_CUDA_API in endpoint:
        return (_SPLIT_TRACK_CUDA_API, endpoint[_SPLIT_TRACK_CUDA_API])
    if _SPLIT_TRACK_CUDA_API in endpoint:
        return (_SPLIT_TRACK_CUDA_API, endpoint[_SPLIT_TRACK_CUDA_API])
    if _SPLIT_TRACK_GPU in endpoint:
        return (_SPLIT_TRACK_GPU, endpoint[_SPLIT_TRACK_GPU])

    track = _fallback_flow_track(event)
    tid = _event_stream_tid(event) if track == _SPLIT_TRACK_GPU else event.get("tid")
    return (track, tid)


def _keep_split_event(
    event: dict,
    *,
    include_cpu: bool,
    include_cuda_api: bool,
    include_gpu: bool,
    include_cpu_to_gpu_flows: bool,
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Decides whether an event appears in a split view: metadata events never;
    # CPU, CUDA API and GPU categories according to their include flags; ac2g flows only
    # when include_cpu_to_gpu_flows; everything else is dropped.
    if event.get("ph") == "M":
        return False

    category = event.get("cat")
    if category in _CPU_TRACE_CATEGORIES:
        return include_cpu
    if category in _CUDA_API_TRACE_CATEGORIES:
        return include_cuda_api
    if category in _GPU_TRACE_CATEGORIES:
        return include_gpu
    return bool(include_cpu_to_gpu_flows and category in _CPU_TO_GPU_FLOW_CATEGORIES)


def _build_split_metadata_events(
    process_keys: set[tuple[str | int | None, str | None, str]],
    process_map: dict[tuple[str | int | None, str | None, str], int],
) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Generates process_name ("[role] <track label> Rank r") and
    # process_sort_index (= pid) metadata for each split process key in sorted order,
    # including rank/role in args when present.
    metadata_events: list[dict] = []
    for key in sorted(process_keys, key=_split_process_sort_key):
        rank, role, track = key
        pid = process_map[key]
        rank_label = f" Rank {rank}" if rank is not None else ""
        role_label = f"[{role}] " if role else ""
        args: dict = {"name": f"{role_label}{_SPLIT_TRACK_LABELS[track]}{rank_label}"}
        if rank is not None:
            args["rank"] = rank
        if role is not None:
            args["role"] = role

        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": args,
            }
        )
        metadata_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "args": {
                    "sort_index": pid,
                    **({"rank": rank} if rank is not None else {}),
                    **({"role": role} if role is not None else {}),
                },
            }
        )
    return metadata_events


def _build_split_trace(
    events: list[dict],
    *,
    display_time_unit: str,
    include_cpu: bool,
    include_cuda_api: bool,
    include_gpu: bool,
    include_cpu_to_gpu_flows: bool,
) -> dict:
    # [FUNCTION DOC]
    # HELPER: Builds a split-view trace: computes flow endpoints, filters events by the
    # include flags, copies kept events, assigns each to a track (flows via endpoint
    # lookup; GPU events use stream tids), sets or removes tid accordingly, assigns pids
    # 1..N to sorted (rank, role, track) keys, renumbers flow ids, generates metadata,
    # sorts events by (ts, pid, tid, ph) and returns the trace dict with
    # displayTimeUnit. Events without a rank share the rank-None processes.
    flow_endpoints = _build_flow_endpoint_maps(events)
    filtered_events: list[dict] = []
    process_keys: set[tuple[str | int | None, str | None, str]] = set()

    for event in events:
        if not _keep_split_event(
            event,
            include_cpu=include_cpu,
            include_cuda_api=include_cuda_api,
            include_gpu=include_gpu,
            include_cpu_to_gpu_flows=include_cpu_to_gpu_flows,
        ):
            continue

        event = dict(event)
        args = event.get("args")
        if isinstance(args, dict):
            event["args"] = dict(args)

        category = event.get("cat")
        if category in _CPU_TO_GPU_FLOW_CATEGORIES:
            track, tid = _flow_endpoint(event, flow_endpoints)
        else:
            track = _split_track_for_category(category)
            if track is None:
                continue
            tid = (
                _event_stream_tid(event)
                if track == _SPLIT_TRACK_GPU
                else event.get("tid")
            )

        process_key = _split_process_key(event, track)
        process_keys.add(process_key)
        event["_split_process_key"] = process_key
        if tid is not None:
            event["tid"] = tid
        elif "tid" in event:
            del event["tid"]
        filtered_events.append(event)

    process_map = {
        key: index + 1
        for index, key in enumerate(sorted(process_keys, key=_split_process_sort_key))
    }
    for event in filtered_events:
        event["pid"] = process_map[event.pop("_split_process_key")]

    _remap_flow_ids(filtered_events)
    metadata_events = _build_split_metadata_events(process_keys, process_map)
    filtered_events.sort(
        key=lambda event: (
            event.get("ts", 0),
            _value_sort_key(event.get("pid")),
            _value_sort_key(event.get("tid")),
            _value_sort_key(event.get("ph")),
        )
    )

    return {
        "traceEvents": metadata_events + filtered_events,
        "displayTimeUnit": display_time_unit,
    }


def convert_jsonl_to_split_trace(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
    include_cpu: bool = True,
    include_cuda_api: bool = True,
    include_gpu: bool = True,
    include_cpu_to_gpu_flows: bool = True,
) -> dict:
    # [FUNCTION DOC]
    # Public API: loads events, builds the split view with the given include flags (all
    # True by default), writes it to output_path when given, and returns the trace dict.
    """Convert PerfTracer JSONL into a split CPU/CUDA/GPU Chrome trace view.

    The split view puts CPU scopes, CUDA API calls, and GPU work on separate
    process tracks. CPU-to-GPU ``ac2g`` flow events are retained by default, so
    Perfetto can still draw links from CUDA launches to the corresponding GPU
    kernels.
    """

    events = _load_trace_events(input_path)
    chrome_trace = _build_split_trace(
        _clone_events(events),
        display_time_unit=display_time_unit,
        include_cpu=include_cpu,
        include_cuda_api=include_cuda_api,
        include_gpu=include_gpu,
        include_cpu_to_gpu_flows=include_cpu_to_gpu_flows,
    )
    if output_path is not None:
        _write_chrome_trace(Path(output_path), chrome_trace)
    return chrome_trace


def _kernel_profile_output_prefix(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None,
) -> Path:
    # [FUNCTION DOC]
    # HELPER: Resolves input files (FileNotFoundError if none) and returns the output
    # path prefix: directory = output_dir, or the single file's parent, or the common
    # parent of all files; stem = the single file's stem or "traces" for several files.
    # KNOWN ISSUE: callers append suffixes with Path.with_suffix, so a stem that itself
    # contains a dot (e.g. "trace.rank0") loses its last component and different inputs
    # can map to the same output names.
    source = Path(input_path)
    files = _resolve_trace_files(source)
    if not files:
        raise FileNotFoundError(f"No trace files matched input path: {input_path}")

    if output_dir is None:
        if len(files) == 1:
            parent = files[0].parent
        else:
            parent = Path(os.path.commonpath([p.parent for p in files]))
    else:
        parent = Path(output_dir)

    if len(files) == 1:
        stem = files[0].stem
    else:
        stem = "traces"
    return parent / stem


def write_kernel_profile_trace_views(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    display_time_unit: str = "ms",
) -> dict[str, Path]:
    # [FUNCTION DOC]
    # Public API: loads events once and writes five views next to the computed prefix:
    # .chrome.json (merged view), .split_clean.chrome.json (all split tracks plus ac2g
    # flows), .gpu_only, .cpu_only and .cuda_api_only split views. Returns a dict
    # mapping view name to written path. File I/O only.
    """Write standard Chrome trace views for a kernel profile JSONL trace.

    The generated split view intentionally keeps CPU-to-GPU ``ac2g`` flow links.
    """

    events = _load_trace_events(input_path)
    output_prefix = _kernel_profile_output_prefix(input_path, output_dir)
    outputs = {
        "chrome": output_prefix.with_suffix(".chrome.json"),
        "split_clean": output_prefix.with_suffix(".split_clean.chrome.json"),
        "gpu_only": output_prefix.with_suffix(".gpu_only.chrome.json"),
        "cpu_only": output_prefix.with_suffix(".cpu_only.chrome.json"),
        "cuda_api_only": output_prefix.with_suffix(".cuda_api_only.chrome.json"),
    }

    _write_chrome_trace(
        outputs["chrome"],
        _build_chrome_trace(_clone_events(events), display_time_unit),
    )
    _write_chrome_trace(
        outputs["split_clean"],
        _build_split_trace(
            _clone_events(events),
            display_time_unit=display_time_unit,
            include_cpu=True,
            include_cuda_api=True,
            include_gpu=True,
            include_cpu_to_gpu_flows=True,
        ),
    )
    _write_chrome_trace(
        outputs["gpu_only"],
        _build_split_trace(
            _clone_events(events),
            display_time_unit=display_time_unit,
            include_cpu=False,
            include_cuda_api=False,
            include_gpu=True,
            include_cpu_to_gpu_flows=False,
        ),
    )
    _write_chrome_trace(
        outputs["cpu_only"],
        _build_split_trace(
            _clone_events(events),
            display_time_unit=display_time_unit,
            include_cpu=True,
            include_cuda_api=False,
            include_gpu=False,
            include_cpu_to_gpu_flows=False,
        ),
    )
    _write_chrome_trace(
        outputs["cuda_api_only"],
        _build_split_trace(
            _clone_events(events),
            display_time_unit=display_time_unit,
            include_cpu=False,
            include_cuda_api=True,
            include_gpu=False,
            include_cpu_to_gpu_flows=False,
        ),
    )
    return outputs


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    # [FUNCTION DOC]
    # HELPER: Builds the CLI parser (positional input, optional output path or "-" for
    # stdout, --display-time-unit default "ms", --kernel-profile-views flag) and parses
    # argv.
    parser = argparse.ArgumentParser(
        description="Convert PerfTracer JSONL output into Chrome Trace JSON format.",
    )
    parser.add_argument(
        "input",
        type=str,
        help=(
            "Path, directory, or glob pattern for PerfTracer JSONL files "
            "(per-rank outputs allowed)"
        ),
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help=(
            "Optional output path for the Chrome Trace JSON file. "
            "If not specified, the output location is inferred from input: "
            "for a directory, outputs to <dir>/traces.json; "
            "for a file, outputs to same dir with .json extension; "
            "for a glob, outputs to common parent dir/traces.json. "
            "Pass '-' to write to stdout."
        ),
    )
    parser.add_argument(
        "--display-time-unit",
        type=str,
        default="ms",
        help="Value for the displayTimeUnit field in the Chrome trace output",
    )
    parser.add_argument(
        "--kernel-profile-views",
        action="store_true",
        help=(
            "Write standard kernel profile views next to the input trace: "
            ".chrome.json, .split_clean.chrome.json, .gpu_only.chrome.json, "
            ".cpu_only.chrome.json, and .cuda_api_only.chrome.json. "
            "The split_clean view keeps CPU-to-GPU flow links."
        ),
    )
    return parser.parse_args(argv)


def _infer_output_path(input_path: str) -> Path:
    # [FUNCTION DOC]
    # HELPER: Infers the merged-view output path when none is given: <dir>/traces.json
    # for a directory; same name with .json for a file (replacing .jsonl); for a glob,
    # the single match's .json path or <common parent>/traces.json (cwd/traces.json if
    # no common path); for an unmatched glob, the parent of the text before the first
    # wildcard plus traces.json; otherwise cwd/traces.json.
    """Infer output path based on input path when output is not specified.

    Rules:
    - If input is a directory: output to <dir>/traces.json
    - If input is a file: output to same dir with .json extension
    - If input is a glob pattern: output to common parent dir/traces.json
    """
    input_as_path = Path(input_path)

    # Case 1: Input is an existing directory
    if input_as_path.is_dir():
        return input_as_path / "traces.json"

    # Case 2: Input is an existing file
    if input_as_path.is_file():
        # Replace .jsonl extension with .json, or just add .json
        if input_as_path.suffix.lower() == ".jsonl":
            return input_as_path.with_suffix(".json")
        else:
            return input_as_path.parent / f"{input_as_path.stem}.json"

    # Case 3: Input might be a glob pattern or non-existent path
    # Try to resolve it and find common parent
    resolved = _resolve_trace_files(input_as_path)
    if resolved:
        # Find common parent directory of all matched files
        if len(resolved) == 1:
            # Single file matched - same as Case 2
            matched_file = resolved[0]
            if matched_file.suffix.lower() == ".jsonl":
                return matched_file.with_suffix(".json")
            else:
                return matched_file.parent / f"{matched_file.stem}.json"
        else:
            # Multiple files - find common parent
            try:
                common_parent = Path(os.path.commonpath([p.parent for p in resolved]))
                return common_parent / "traces.json"
            except ValueError:
                # No common path (e.g., files on different drives on Windows)
                return Path.cwd() / "traces.json"

    # Fallback: treat as a potential directory or use parent
    if "*" in input_path or "?" in input_path:
        # It's a glob pattern - extract the base directory
        base = input_path.split("*")[0].split("?")[0]
        base_path = Path(base).parent if base else Path.cwd()
        return base_path / "traces.json"

    # Default fallback to current directory
    return Path.cwd() / "traces.json"


def main(argv: Sequence[str] | None = None) -> int:
    # [FUNCTION DOC]
    # CLI entry. With --kernel-profile-views writes the five views (output used as the
    # output directory unless missing or "-"; with "-" prints a JSON map of written
    # paths to stdout) and returns 0. Otherwise converts to the merged view, writing to
    # the given or inferred path, or printing the JSON to stdout when output is "-".
    # Returns 0; errors propagate as exceptions.
    args = _parse_args(argv)
    if args.kernel_profile_views:
        output_dir = None if args.output in (None, "-") else args.output
        outputs = write_kernel_profile_trace_views(
            args.input,
            output_dir,
            display_time_unit=args.display_time_unit,
        )
        if args.output == "-":
            json.dump({key: str(path) for key, path in outputs.items()}, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return 0

    emit_stdout = args.output == "-"
    if args.output is None:
        destination: str | os.PathLike[str] | None = _infer_output_path(args.input)
    elif emit_stdout:
        destination = None
    else:
        destination = args.output
    chrome_trace = convert_jsonl_to_chrome_trace(
        args.input,
        destination,
        display_time_unit=args.display_time_unit,
    )
    if emit_stdout:
        json.dump(chrome_trace, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
