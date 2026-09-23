#!/usr/bin/env python3
"""Audit L3 dual-GPU allocation and integer/index capacity without running CUDA.

The audit deliberately keeps three evidence classes separate:

* STATIC_*: derived from the archived source/build command and audited graph
  metadata (or a conservative allocation upper bound).
* RUNTIME_*: observed in an already existing A100 log.
* NOT_RUN / NOT_PROVEN: no execution or no diagnostic capable of proving the
  property.  In particular, final L2 conservation is not peak ring occupancy.

This script never invokes a GPU program.  It reads build source archives,
commands, stage0 manifests, a single CSR row-end value for a rejected graph,
and existing logs, then writes a JSON manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
INT_BYTES = 4
U32_BYTES = 4
U64_BYTES = 8
NODE_BYTES = 8  # node_struct: int id + int VALUE_TYPE; confirmed by L2 logs.
MEM_BLOCK_SIZE = 512
BUCKETS = 16
L2_BATCH_SIZE = 8
BULK_INBOX_SLOTS = 2
BULK_L3_BATCH = 128
WARP_SIZE = 32
WORKER_THREADS = 512
WORKER_WARPS = WORKER_THREADS // WARP_SIZE
A100_SMS = 108
MAX_WORK_BLOCKS = A100_SMS - 1
ACK_SLOTS = MAX_WORK_BLOCKS * WORKER_WARPS
GPU_MEMORY_BUDGET = INT32_MAX


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def command_defines(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for arg in command:
        if not arg.startswith("-D"):
            continue
        item = arg[2:]
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = item, "1"
        result[key] = value
    return result


def tar_member_text(archive: Path, member: str) -> str:
    with tarfile.open(archive, "r:gz") as source:
        extracted = source.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"missing {member} in {archive}")
        return extracted.read().decode("utf-8")


def member_sha256(archive: Path, member: str) -> str:
    with tarfile.open(archive, "r:gz") as source:
        extracted = source.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"missing {member} in {archive}")
        return hashlib.sha256(extracted.read()).hexdigest()


def require_fragments(text: str, member: str, fragments: Iterable[str]) -> None:
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"source contract drift in {member}: missing {missing}")


@dataclass(frozen=True)
class BuildSnapshot:
    name: str
    command_path: Path
    source_archive: Path
    compact: bool


@dataclass(frozen=True)
class Partition:
    gpu: int
    begin: int
    end: int
    vertices: int
    edges: int


GRAPH_RE = re.compile(r"number of vertices (\d+) number of edges (\d+)")
PARTITION_RE = re.compile(
    r"GPU(\d+) partition: \[(\d+), (\d+)\) v_local=(\d+) nedges=(\d+)"
)
BOUNDARY_RE = re.compile(
    r"L3_BOUNDARY_INDEX gpu=(\d+) dense_words=(\d+) boundary_words=(\d+) "
    r"indexed=(\d+)"
)
COMPACT_RE = re.compile(
    r"L3_COMPACT_CANDIDATES gpu=(\d+) peer_vertices=(\d+) "
    r"boundary_vertices=(\d+) compact=(\d+) words=(\d+)"
)
L2_CAPACITY_RE = re.compile(
    r"L2_CAPACITY budget=(\d+) record_bytes=(\d+) buckets=(\d+) "
    r"per_bucket=(\d+) allocated_records=(\d+) counter_bits=(\d+)"
)
L2_FINAL_RE = re.compile(
    r"L2_FINAL gpu=(\d+) buckets=(\d+) reads=(\d+) writes=(\d+) completed=(\d+)"
    r"(?: max_bucket_writes=(\d+) per_bucket_capacity=(\d+) "
    r"counter_bits=(\d+) no_wrap=(\d+))?"
)
ACK_RE = re.compile(
    r"L3_WORKER_ACK gpu=(\d+) active_slots=(\d+) capacity=(\d+) "
    r"work_blocks=(\d+) warps_per_block=(\d+)"
)
WIDE_RE = re.compile(r"WIDE_ORACLE vertices=(\d+) correct=(\d+)")
MAX_WEIGHT_RE = re.compile(r"maximum weight ([0-9]+(?:\.[0-9]+)?)")


def l2_constants() -> dict[str, int]:
    allocated_records = GPU_MEMORY_BUDGET // NODE_BYTES
    allocated_blocks = allocated_records // MEM_BLOCK_SIZE
    per_bucket = (
        allocated_records // BUCKETS // MEM_BLOCK_SIZE * MEM_BLOCK_SIZE
    )
    usable_records = per_bucket * BUCKETS
    requested = {
        "data": allocated_records * NODE_BYTES,
        "block_write_done": allocated_blocks * INT_BYTES,
        "five_bucket_counter_arrays": 5 * BUCKETS * INT_BYTES,
        "five_queue_scalar_counters": 5 * INT_BYTES,
        "ml_queue_run_begin": INT_BYTES,
    }
    return {
        "budget_bytes": GPU_MEMORY_BUDGET,
        "record_bytes": NODE_BYTES,
        "allocated_records": allocated_records,
        "allocated_block_counters": allocated_blocks,
        "buckets": BUCKETS,
        "per_bucket_records": per_bucket,
        "usable_ring_records": usable_records,
        "unused_tail_records": allocated_records - usable_records,
        "explicit_device_bytes": sum(requested.values()),
        **{f"bytes_{key}": value for key, value in requested.items()},
    }


L2 = l2_constants()


def parse_runtime_log(path: Path, root: Path, label: str, build: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    graph_match = GRAPH_RE.search(text)
    if graph_match is None:
        raise RuntimeError(f"missing graph header in {path}")
    vertices, edges = map(int, graph_match.groups())

    partitions = [
        Partition(*map(int, match.groups())) for match in PARTITION_RE.finditer(text)
    ]
    if sorted(partition.gpu for partition in partitions) != [0, 1]:
        raise RuntimeError(f"expected two partitions in {path}")

    boundaries: dict[int, dict[str, Any]] = {}
    for match in BOUNDARY_RE.finditer(text):
        gpu, dense, sparse, indexed = map(int, match.groups())
        boundaries[gpu] = {
            "dense_words": dense,
            "boundary_words": sparse,
            "indexed": bool(indexed),
            "allocated_bytes": sparse * INT_BYTES if indexed else 0,
            "evidence_class": "RUNTIME_LOG_VALIDATION",
        }

    compact: dict[int, dict[str, Any]] = {}
    for match in COMPACT_RE.finditer(text):
        gpu, peer_vertices, boundary_vertices, enabled, words = map(
            int, match.groups()
        )
        compact[gpu] = {
            "peer_vertices": peer_vertices,
            "boundary_vertices": boundary_vertices,
            "compact_selected": bool(enabled),
            "scan_words": words,
            "lookup_bytes": peer_vertices * INT_BYTES if enabled else 0,
            "reverse_bytes": boundary_vertices * INT_BYTES if enabled else 0,
            "evidence_class": "RUNTIME_LOG_VALIDATION",
        }

    capacity_rows = [
        tuple(map(int, match.groups())) for match in L2_CAPACITY_RE.finditer(text)
    ]
    expected_capacity = (
        L2["budget_bytes"],
        L2["record_bytes"],
        L2["buckets"],
        L2["per_bucket_records"],
        L2["allocated_records"],
        32,
    )
    capacity_consistent = len(capacity_rows) == 2 and all(
        row == expected_capacity for row in capacity_rows
    )

    finals: dict[int, list[dict[str, int]]] = {0: [], 1: []}
    exact_final_schema_count = 0
    for match in L2_FINAL_RE.finditer(text):
        gpu, buckets, reads, writes, completed = map(int, match.groups()[:5])
        exact_groups = match.groups()[5:]
        exact = None
        if all(value is not None for value in exact_groups):
            max_bucket, per_bucket_capacity, counter_bits, no_wrap = map(
                int, exact_groups
            )
            exact = {
                "max_bucket_writes": max_bucket,
                "per_bucket_capacity": per_bucket_capacity,
                "counter_bits": counter_bits,
                "no_wrap": no_wrap,
            }
            exact_final_schema_count += 1
        finals.setdefault(gpu, []).append(
            {
                "buckets": buckets,
                "reads": reads,
                "writes": writes,
                "completed": completed,
                "exact_no_wrap": exact,
            }
        )
    all_final_equal = bool(finals[0] and finals[1]) and all(
        sample["buckets"] == BUCKETS
        and sample["reads"] == sample["writes"] == sample["completed"]
        for samples in finals.values()
        for sample in samples
    )
    final_line_count = sum(len(rows) for rows in finals.values())

    l2_by_gpu: dict[str, dict[str, Any]] = {}
    for gpu in (0, 1):
        samples = finals[gpu]
        maximum = max((sample["writes"] for sample in samples), default=None)
        aggregate_below_ring = (
            maximum is not None and maximum < L2["per_bucket_records"]
        )
        exact_rows = [sample["exact_no_wrap"] for sample in samples]
        exact_schema_complete = bool(samples) and all(row is not None for row in exact_rows)
        exact_no_wrap = exact_schema_complete and all(
            sample["exact_no_wrap"]["no_wrap"] == 1
            and sample["exact_no_wrap"]["max_bucket_writes"] <= sample["writes"]
            and sample["exact_no_wrap"]["max_bucket_writes"]
            <= sample["exact_no_wrap"]["per_bucket_capacity"]
            and sample["exact_no_wrap"]["per_bucket_capacity"]
            == L2["per_bucket_records"]
            and sample["exact_no_wrap"]["counter_bits"] == 32
            and sample["writes"] <= INT32_MAX
            for sample in samples
            if sample["exact_no_wrap"] is not None
        )
        if exact_no_wrap:
            no_wrap_status = "RUNTIME_PROVEN_FROM_EXACT_PER_BUCKET_FINAL"
            no_wrap_reason = (
                "The post-sync host audit reports max_bucket_writes <= "
                "per_bucket_capacity and no_wrap=1 for every sample."
            )
        elif aggregate_below_ring:
            no_wrap_status = "RUNTIME_PROVEN_FROM_AGGREGATE_TOTAL"
            no_wrap_reason = (
                "Each bucket writes no more than the all-bucket total, and the "
                "all-bucket total is below one bucket capacity."
            )
        elif exact_schema_complete:
            no_wrap_status = "RUNTIME_EXACT_NO_WRAP_GATE_FAILED"
            no_wrap_reason = (
                "The exact post-sync per-bucket no-wrap fields are present but do "
                "not satisfy the archived capacity contract."
            )
        else:
            no_wrap_status = "NOT_PROVEN_AGGREGATE_EXCEEDS_PER_BUCKET_CAPACITY"
            no_wrap_reason = (
                "This is a legacy L2_FINAL row. Only the sum over 16 buckets is "
                "logged; neither per-bucket writes nor peak outstanding occupancy is logged."
            )
        l2_by_gpu[str(gpu)] = {
            "query_count": len(samples),
            "total_writes_per_query": [sample["writes"] for sample in samples],
            "maximum_total_writes": maximum,
            "exact_no_wrap_schema_complete": exact_schema_complete,
            "exact_no_wrap_rows": exact_rows,
            "all_total_writes_below_int32_max": bool(
                samples and all(sample["writes"] <= INT32_MAX for sample in samples)
            ),
            "no_bucket_wrap_status": no_wrap_status,
            "no_bucket_wrap_reason": no_wrap_reason,
        }

    # Preserve compatibility with archived legacy rows, while treating the
    # newer exact schema as an all-or-nothing runtime contract.  A mixed log is
    # neither a valid legacy record nor a complete exact record.
    exact_schema_unused = exact_final_schema_count == 0
    exact_schema_complete_for_log = (
        final_line_count > 0 and exact_final_schema_count == final_line_count
    )
    exact_schema_valid = exact_schema_complete_for_log and all(
        row["no_bucket_wrap_status"]
        == "RUNTIME_PROVEN_FROM_EXACT_PER_BUCKET_FINAL"
        for row in l2_by_gpu.values()
    )
    l2_final_schema_valid = exact_schema_unused or exact_schema_valid

    acknowledgements: dict[int, list[dict[str, int]]] = {0: [], 1: []}
    for match in ACK_RE.finditer(text):
        gpu, active, capacity, blocks, warps = map(int, match.groups())
        acknowledgements[gpu].append(
            {
                "active_slots": active,
                "capacity": capacity,
                "work_blocks": blocks,
                "warps_per_block": warps,
            }
        )
    ack_logged = bool(acknowledgements[0] and acknowledgements[1])
    ack_rows_valid = ack_logged and all(
        row["active_slots"] <= row["capacity"] == ACK_SLOTS
        and row["work_blocks"] <= MAX_WORK_BLOCKS
        and row["warps_per_block"] == WORKER_WARPS
        for rows in acknowledgements.values()
        for row in rows
    )

    wide_rows = [tuple(map(int, match.groups())) for match in WIDE_RE.finditer(text)]
    correctness_lines = text.count("mlmq sssp correct!")
    error_tokens = (
        "Error at node",
        "CUDA API failed",
        "kernel launch error",
        "manager launch error",
        "BENCH_FAIL",
    )
    error_lines = [
        line
        for line in text.splitlines()
        if any(token in line for token in error_tokens)
    ]
    runtime_valid = (
        capacity_consistent
        and all_final_equal
        and l2_final_schema_valid
        and all(
            sample["writes"] <= INT32_MAX
            for samples in finals.values()
            for sample in samples
        )
        and bool(wide_rows)
        and all(count == vertices and correct == 1 for count, correct in wide_rows)
        and correctness_lines == len(wide_rows)
        and not error_lines
    )
    max_weight_match = MAX_WEIGHT_RE.search(text)

    return {
        "label": label,
        "build": build,
        "path": relpath(path, root),
        "sha256": sha256_path(path),
        "graph_vertices": vertices,
        "graph_edges": edges,
        "maximum_weight_reported": (
            int(float(max_weight_match.group(1))) if max_weight_match else None
        ),
        "partitions": [partition.__dict__ for partition in partitions],
        "boundary_index": {str(key): value for key, value in sorted(boundaries.items())},
        "compact_candidates": {str(key): value for key, value in sorted(compact.items())},
        "l2_capacity_line_count": len(capacity_rows),
        "l2_capacity_matches_archived_source_formula": capacity_consistent,
        "l2_final_line_count": final_line_count,
        "l2_final_exact_no_wrap_schema_count": exact_final_schema_count,
        "l2_final_schema": (
            "EXACT_NO_WRAP"
            if exact_schema_valid
            else "LEGACY"
            if exact_schema_unused
            else "INVALID_OR_MIXED"
        ),
        "l2_final_schema_valid": l2_final_schema_valid,
        "l2_final_per_bucket_equality_asserted_before_each_print": all_final_equal,
        "l2": l2_by_gpu,
        "peak_per_bucket_occupancy_logged": False,
        "ack": {
            "status": "RUNTIME_VALIDATED" if ack_rows_valid else "NOT_RUN_IN_THIS_LOG",
            "rows": {
                str(key): value for key, value in sorted(acknowledgements.items())
            },
        },
        "wide_oracle_query_count": len(wide_rows),
        "wide_oracle_all_correct": bool(wide_rows) and all(row[1] == 1 for row in wide_rows),
        "correctness_line_count": correctness_lines,
        "error_lines": error_lines,
        "status": (
            "RUNTIME_VALIDATED_FOR_RECORDED_QUERIES"
            if runtime_valid
            else "RUNTIME_LOG_VALIDATION_FAILED"
        ),
        "scope_note": (
            "Successful recorded queries validate those executions only. They do not "
            "prove unlogged peak L2 occupancy, unchecked int32 additions, or arbitrary inputs."
        ),
    }


def read_cut_edges(record: dict[str, Any], cut: int) -> tuple[int, str]:
    """Read one audited GR uint64 row-end; no edge scan and no GPU execution."""
    graph = Path(record["realpath"])
    row_offset = int(record["layout"]["row_end_offset"])
    if cut <= 0:
        return 0, "STATIC_FROM_EMPTY_PREFIX"
    with graph.open("rb") as stream:
        stream.seek(row_offset + (cut - 1) * U64_BYTES)
        payload = stream.read(U64_BYTES)
    if len(payload) != U64_BYTES:
        raise RuntimeError(f"short row-end read from {graph}")
    value = struct.unpack("<Q", payload)[0]
    return value, "STATIC_FROM_AUDITED_GR_ROW_END"


def boundary_upper_bytes(peer_vertices: int) -> int:
    dense_words = ceil_div(peer_vertices, 32)
    # Allocation happens only when count < dense_words/2.  Integer count's
    # largest possible value is floor((dense_words-1)/2).
    return ((dense_words - 1) // 2) * INT_BYTES if dense_words else 0


def compact_upper_bytes(peer_vertices: int) -> int:
    # If compact mode is selected, lookup is dense and reverse has strictly
    # fewer than peer_vertices/2 entries.  Otherwise neither allocation exists.
    reverse_max = (peer_vertices - 1) // 2 if peer_vertices else 0
    return peer_vertices * INT_BYTES + reverse_max * INT_BYTES


def memory_components(
    *,
    graph_vertices: int,
    local_vertices: int,
    local_edges: int,
    peer_vertices: int,
    boundary_bytes: int,
    compact_bytes: int,
) -> dict[str, int]:
    local_mark_words = ceil_div(local_vertices, 32)
    local_hint_words = ceil_div(local_mark_words, 32)
    peer_mark_words = ceil_div(peer_vertices, 32)
    peer_hint_words = ceil_div(peer_mark_words, 32)
    peer_hint2_words = ceil_div(peer_hint_words, 32)
    components = {
        "csr_row": (local_vertices + 1) * INT_BYTES,
        "csr_destination": local_edges * INT_BYTES,
        "csr_weight": local_edges * INT_BYTES,
        "authoritative_distance": (local_vertices + 1) * INT_BYTES,
        "global_exit": INT_BYTES,
        "dirty_bitmap": local_mark_words * U32_BYTES,
        "dirty_hint": local_hint_words * U32_BYTES,
        "last_processed": (local_vertices + 1) * INT_BYTES,
        "local_idle": INT_BYTES,
        "remote_effective_counter": U64_BYTES,
        "candidate_values_dense": (peer_vertices + 1) * INT_BYTES,
        "candidate_mark": peer_mark_words * U32_BYTES,
        "candidate_hint": peer_hint_words * U32_BYTES,
        "candidate_hint2": peer_hint2_words * U32_BYTES,
        "peer_cache_dense": (peer_vertices + 1) * INT_BYTES,
        "bulk_send_list_dense": (peer_vertices + 1) * NODE_BYTES,
        "bulk_inbox_two_slots": (
            (local_vertices + 1) * NODE_BYTES * BULK_INBOX_SLOTS
        ),
        "bulk_inbox_metadata": 5 * BULK_INBOX_SLOTS * INT_BYTES,
        "quiesce_flags": 2 * INT_BYTES,
        "termination_flags": 2 * INT_BYTES,
        "worker_ack_slots": ACK_SLOTS * INT_BYTES,
        "phase_seed_flags_and_count": 4 * INT_BYTES,
        "seed_list_global_capacity": (graph_vertices + 1) * NODE_BYTES,
        "boundary_word_index": boundary_bytes,
        "compact_lookup_and_reverse": compact_bytes,
        "persistent_l2_query_workspace": L2["explicit_device_bytes"],
    }
    components["total_explicit_requested_bytes"] = sum(components.values())
    return components


def memory_case(
    *,
    name: str,
    partitions: list[Partition],
    graph_vertices: int,
    boundary: dict[int, dict[str, Any]] | None,
    compact: dict[int, dict[str, Any]] | None,
    gpu_total_bytes: int,
    evidence_class: str,
    runtime_log: str | None,
) -> dict[str, Any]:
    per_gpu: list[dict[str, Any]] = []
    for partition in sorted(partitions, key=lambda item: item.gpu):
        peer = next(item for item in partitions if item.gpu != partition.gpu)
        boundary_entry = (boundary or {}).get(partition.gpu)
        if boundary_entry is None:
            boundary_bytes = boundary_upper_bytes(peer.vertices)
            boundary_kind = "STATIC_CONSERVATIVE_UPPER_BOUND"
        else:
            boundary_bytes = int(boundary_entry["allocated_bytes"])
            boundary_kind = "RUNTIME_EXACT_ALLOCATION_INPUT"

        compact_entry = (compact or {}).get(partition.gpu)
        if compact is None:
            compact_bytes = 0
            compact_kind = "CONTROL_BUILD_DISABLED"
        elif compact_entry is None:
            compact_bytes = compact_upper_bytes(peer.vertices)
            compact_kind = "STATIC_CONSERVATIVE_UPPER_BOUND_NOT_RUN"
        else:
            compact_bytes = int(compact_entry["lookup_bytes"]) + int(
                compact_entry["reverse_bytes"]
            )
            compact_kind = "RUNTIME_EXACT_ALLOCATION_INPUT"

        components = memory_components(
            graph_vertices=graph_vertices,
            local_vertices=partition.vertices,
            local_edges=partition.edges,
            peer_vertices=peer.vertices,
            boundary_bytes=boundary_bytes,
            compact_bytes=compact_bytes,
        )
        total = components["total_explicit_requested_bytes"]
        per_gpu.append(
            {
                "gpu": partition.gpu,
                "local_vertices": partition.vertices,
                "local_edges": partition.edges,
                "peer_vertices": peer.vertices,
                "boundary_bytes_basis": boundary_kind,
                "compact_bytes_basis": compact_kind,
                "components_bytes": components,
                "total_explicit_requested_bytes": total,
                "total_explicit_requested_gib": round(total / (1 << 30), 6),
                "a100_total_bytes": gpu_total_bytes,
                "fraction_of_a100_total": round(total / gpu_total_bytes, 9),
                "raw_request_fit_status": (
                    "STATIC_RAW_REQUEST_BOUND_PASS"
                    if total < gpu_total_bytes
                    else "STATIC_RAW_REQUEST_BOUND_FAIL"
                ),
            }
        )
    return {
        "name": name,
        "evidence_class": evidence_class,
        "runtime_log": runtime_log,
        "per_gpu": per_gpu,
        "status": (
            "STATIC_RAW_REQUEST_BOUND_PASS"
            if all(
                item["raw_request_fit_status"] == "STATIC_RAW_REQUEST_BOUND_PASS"
                for item in per_gpu
            )
            else "STATIC_RAW_REQUEST_BOUND_FAIL"
        ),
        "scope_note": (
            "Sum covers explicit cudaMalloc requests in the audited current path. "
            "CUDA context/module/allocator overhead and device symbols are not bounded; "
            "successful existing runs are separate runtime evidence."
        ),
    }


def partitions_from_log(log: dict[str, Any]) -> list[Partition]:
    return [Partition(**row) for row in log["partitions"]]


def static_partitions(record: dict[str, Any]) -> tuple[list[Partition], str]:
    vertices = int(record["header"]["vertices"])
    edges = int(record["header"]["directed_edges"])
    cut = vertices // 2
    first_edges, basis = read_cut_edges(record, cut)
    if first_edges > edges:
        raise RuntimeError(f"cut edge prefix exceeds graph edge count for {record['realpath']}")
    return (
        [
            Partition(0, 0, cut, cut, first_edges),
            Partition(1, cut, vertices, vertices - cut, edges - first_edges),
        ],
        basis,
    )


def audit_build(snapshot: BuildSnapshot, root: Path) -> dict[str, Any]:
    command = load_json(snapshot.command_path)
    defines = command_defines(command)
    expected = {
        "WORK_COUNT": "false",
        "MLMQ_WORKER_THREADS": str(WORKER_THREADS),
        "L3_COOPERATIVE_COLLECT": "true",
        "L3_DIRECT_RX": "true",
        "L3_RETAIN_TX": "true",
        "L3_WINDOW_MODE": "2",
        "L3_WORKER_RECOVERY": "true",
        "L3_TERM_WAIT_ACK": "true",
        "L3_ACK_SCAN": "true",
        "L3_ACK_WIDE_SCAN": "true",
        "L3_BOUNDARY_INDEX": "true",
        "L3_L2_FINAL_COUNTS": "true",
        "L3_CHAIN_SHORTCUTS": "false",
    }
    mismatches = {
        key: {"expected": value, "observed": defines.get(key)}
        for key, value in expected.items()
        if defines.get(key) != value
    }
    compact_observed = defines.get("L3_COMPACT_CANDIDATES", "false") == "true"
    if compact_observed != snapshot.compact:
        mismatches["L3_COMPACT_CANDIDATES"] = {
            "expected": str(snapshot.compact).lower(),
            "observed": str(compact_observed).lower(),
        }

    members = {
        "SSSP/sssp.cuh": tar_member_text(snapshot.source_archive, "SSSP/sssp.cuh"),
        "SSSP/sssp_run.cu": tar_member_text(
            snapshot.source_archive, "SSSP/sssp_run.cu"
        ),
        "SSSP/graph_partition.h": tar_member_text(
            snapshot.source_archive, "SSSP/graph_partition.h"
        ),
        "SSSP/csr_graph.h": tar_member_text(
            snapshot.source_archive, "SSSP/csr_graph.h"
        ),
        "SSSP/l3/l3_boundary_index.cuh": tar_member_text(
            snapshot.source_archive, "SSSP/l3/l3_boundary_index.cuh"
        ),
        "SSSP/l3/l3_bulk.cuh": tar_member_text(
            snapshot.source_archive, "SSSP/l3/l3_bulk.cuh"
        ),
        "core/cu_delta_queue/cu_delta_queue.cuh": tar_member_text(
            snapshot.source_archive, "core/cu_delta_queue/cu_delta_queue.cuh"
        ),
        "core/src/ml_queue.cu": tar_member_text(
            snapshot.source_archive, "core/src/ml_queue.cu"
        ),
        "core/include/common.h": tar_member_text(
            snapshot.source_archive, "core/include/common.h"
        ),
        "core/include/GPU_setup.h": tar_member_text(
            snapshot.source_archive, "core/include/GPU_setup.h"
        ),
    }
    require_fragments(members["SSSP/sssp.cuh"], "SSSP/sssp.cuh", [
        "#define GPU_MEMORY INT_MAX",
        "#define BULK_L3_BATCH 128",
        "#define BULK_INBOX_SLOTS 2",
    ])
    require_fragments(members["core/include/common.h"], "core/include/common.h", [
        "#define VALUE_TYPE int",
        "#define BNUM 16",
        "#define l2_batch_size 8",
    ])
    require_fragments(members["SSSP/csr_graph.h"], "SSSP/csr_graph.h", [
        "typedef int index_type",
        "index_type nnodes, nedges",
    ])
    require_fragments(
        members["core/cu_delta_queue/cu_delta_queue.cuh"],
        "core/cu_delta_queue/cu_delta_queue.cuh",
        [
            "Not considered: When read_pos / write_reserve / write_done overflow.",
            "total_size = all_total_size / bucketNum / MEM_BLOCK_SIZE * MEM_BLOCK_SIZE;",
            "(current_reserve + write_pos) % total_size",
        ],
    )
    require_fragments(members["SSSP/sssp_run.cu"], "SSSP/sssp_run.cu", [
        "cudaMalloc(&remote_cand, sizeof(VALUE_TYPE) * (peer_v_local + 1))",
        "cudaMalloc(&bulk_send_list, sizeof(NODE_TYPE) * (peer_v_local + 1))",
        "sizeof(NODE_TYPE) * (local_v + 1) * BULK_INBOX_SLOTS",
        "cudaMalloc(&gctx[gpu_id].seed_list, sizeof(NODE_TYPE) * (gctx[gpu_id].m + 1))",
        "mlmq.init_host(GPU_MEMORY, init_limits, setup)",
        "new_dist = large_snapshot + edge_data",
        "new_dist = source_snapshot + edge_data",
    ])
    require_fragments(
        members["SSSP/l3/l3_boundary_index.cuh"],
        "SSSP/l3/l3_boundary_index.cuh",
        [
            "const int words=(peer_size+31)/32",
            "index.size()<size_t(words)/2",
            "lookup.size()*sizeof(int)",
            "reverse.size()*sizeof(int)",
            "arrays retain their old allocation size",
        ],
    )
    require_fragments(members["SSSP/l3/l3_bulk.cuh"], "SSSP/l3/l3_bulk.cuh", [
        "if (count > peer_v_local) count = peer_v_local",
        "peer_inbox + slot * (peer_v_local + 1)",
    ])
    require_fragments(members["SSSP/graph_partition.h"], "SSSP/graph_partition.h", [
        "cudaMalloc((void **)&row_start_d, (v_local + 1) * sizeof(int))",
        "cudaMalloc((void **)&col_idx_d, nedges_local * sizeof(int))",
    ])

    return {
        "name": snapshot.name,
        "command_path": relpath(snapshot.command_path, root),
        "command_sha256": sha256_path(snapshot.command_path),
        "source_archive": relpath(snapshot.source_archive, root),
        "source_archive_sha256": sha256_path(snapshot.source_archive),
        "source_member_sha256": {
            member: member_sha256(snapshot.source_archive, member)
            for member in members
        },
        "defines": defines,
        "compact_candidates": compact_observed,
        "contract_check": "PASS" if not mismatches else "FAIL",
        "macro_mismatches": mismatches,
    }


def parse_hardware(preflight: Path, root: Path) -> dict[str, Any]:
    text = preflight.read_text(encoding="utf-8")
    rows = [
        tuple(map(int, match))
        for match in re.findall(
            r"CUDA_DEVICE id=(\d+).*?sm=(\d+) total_bytes=(\d+) free_bytes=(\d+)",
            text,
        )
    ]
    if len(rows) != 2:
        raise RuntimeError(f"expected two CUDA devices in {preflight}")
    if any(sm != A100_SMS for _, sm, _, _ in rows):
        raise RuntimeError(f"unexpected SM count in {preflight}: {rows}")
    total_bytes = min(row[2] for row in rows)
    return {
        "evidence_class": "RUNTIME_PREFLIGHT_LOG",
        "path": relpath(preflight, root),
        "sha256": sha256_path(preflight),
        "devices": [
            {
                "gpu": gpu,
                "sm_count": sm,
                "total_bytes": total,
                "free_bytes_at_preflight": free,
            }
            for gpu, sm, total, free in rows
        ],
        "minimum_total_bytes": total_bytes,
    }


def graph_name_from_record(record: dict[str, Any]) -> str:
    return Path(record["requested_path"]).stem


def numeric_audit(
    *,
    name: str,
    vertices: int,
    minimum_weight: int | None,
    maximum_weight: int | None,
    negative_count: int | None,
    oracle: dict[str, Any] | None,
) -> dict[str, Any]:
    if negative_count and negative_count > 0:
        return {
            "status": "REJECTED_NEGATIVE_WEIGHT_INPUT",
            "negative_weight_count": negative_count,
            "reason": "The current Dijkstra/SSSP acceptance contract rejects negative weights.",
            "oracle": oracle,
        }
    if maximum_weight is None or minimum_weight is None:
        return {
            "status": "NOT_PROVEN_WEIGHT_RANGE_INCOMPLETE_AND_UNCHECKED_ADDITION",
            "minimum_weight": minimum_weight,
            "maximum_weight": maximum_weight,
            "reason": (
                "The consumed evidence does not contain a full minimum/negative-weight "
                "scan for this input, and the GPU source also performs unchecked signed "
                "int candidate additions."
            ),
            "oracle": oracle,
        }
    candidate_add_bound = vertices * maximum_weight
    static_safe = minimum_weight >= 0 and candidate_add_bound <= INT32_MAX
    if static_safe:
        status = "STATIC_PROVEN_FOR_ACCEPTED_RELAXATIONS"
        reason = (
            "For nonnegative weights, every accepted improving label corresponds to a "
            "simple path; one additional relaxation is bounded by n*max_weight, which fits int32."
        )
    else:
        status = "NOT_PROVEN_UNCHECKED_INT32_CANDIDATE_ADDITION"
        reason = (
            "The source performs signed int VALUE_TYPE additions without a checked/wide "
            "candidate-add diagnostic. An int64 oracle and correct final int32 distances "
            "do not prove every transient GPU addition stayed in range."
        )
    return {
        "status": status,
        "minimum_weight": minimum_weight,
        "maximum_weight": maximum_weight,
        "n_times_max_weight_bound": candidate_add_bound,
        "int32_max": INT32_MAX,
        "reason": reason,
        "oracle": oracle,
        "graph": name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/l3_30h_20260923/stage0/capacity_manifest.json"),
    )
    parser.add_argument(
        "--stdout", action="store_true", help="print JSON instead of writing --output"
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()

    dataset_path = root / "evidence/l3_30h_20260923/stage0/dataset_manifest.json"
    oracle_path = root / "evidence/l3_30h_20260923/stage0/oracle_manifest.json"
    preflight_path = (
        root
        / "evidence/l3_30h_20260923/stage0/gpu_preflight_38122/cuda_preflight.log"
    )
    dataset = load_json(dataset_path)
    oracle_manifest = load_json(oracle_path)
    hardware = parse_hardware(preflight_path, root)
    gpu_total_bytes = int(hardware["minimum_total_bytes"])

    snapshots = [
        BuildSnapshot(
            "stage0_baseline_control",
            root / "evidence/l3_30h_20260923/stage0/build_base/dual_build/command.json",
            root / "evidence/l3_30h_20260923/stage0/build_base/dual_build/source.tgz",
            False,
        ),
        BuildSnapshot(
            "latest_r96v2_control",
            root / "tmp/l3_30h_20260923/variant_control_r96v2/dual_build/command.json",
            root / "tmp/l3_30h_20260923/variant_control_r96v2/dual_build/source.tgz",
            False,
        ),
        BuildSnapshot(
            "latest_r96v2_compact",
            root / "tmp/l3_30h_20260923/variant_compact_r96v2/dual_build/command.json",
            root / "tmp/l3_30h_20260923/variant_compact_r96v2/dual_build/source.tgz",
            True,
        ),
    ]
    build_audits = {snapshot.name: audit_build(snapshot, root) for snapshot in snapshots}
    if any(item["contract_check"] != "PASS" for item in build_audits.values()):
        raise RuntimeError("one or more build/source contract checks failed")

    log_specs = [
        (
            "stage0_delaunay_n20_smoke",
            "delaunay_n20",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/smoke_delaunay_n20_job38129/round000_1_dual_l3.log",
        ),
        (
            "stage0_delaunay_n23_baseline",
            "delaunay_n23",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/baselines_job38131/delaunay_n23/round000_1_dual_l3.log",
        ),
        (
            "stage0_rgg_baseline",
            "rgg_n_2_20_s0",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/baselines_job38131/rgg_n_2_20_s0/round000_1_dual_l3.log",
        ),
        (
            "stage0_atmosmodm_baseline",
            "atmosmodm",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/baselines_job38131/atmosmodm/round000_1_dual_l3.log",
        ),
        (
            "stage0_rmat22_baseline",
            "rmat22",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/baselines_job38131/rmat22/round000_1_dual_l3.log",
        ),
        (
            "stage1_usa_baseline",
            "usa_gplus",
            "stage0_baseline_control",
            "evidence/l3_30h_20260923/stage1/baselines_job38131/usa_gplus/round000_1_dual_l3.log",
        ),
        (
            "latest_rgg_control_job38180",
            "rgg_n_2_20_s0",
            "latest_r96v2_control",
            "evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/control/rgg_n_2_20_s0_d32_b107_compact/round000_1_dual_l3.log",
        ),
        (
            "latest_usa_control_job38180",
            "usa_gplus",
            "latest_r96v2_control",
            "evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/control/usa_gplus_s11973673_d400000_b107_compact/round000_1_dual_l3.log",
        ),
        (
            "latest_rgg_compact_job38180",
            "rgg_n_2_20_s0",
            "latest_r96v2_compact",
            "evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/compact/rgg_n_2_20_s0_d32_b107_compact/round000_1_dual_l3.log",
        ),
        (
            "latest_usa_compact_job38180",
            "usa_gplus",
            "latest_r96v2_compact",
            "evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/compact/usa_gplus_s11973673_d400000_b107_compact/round000_1_dual_l3.log",
        ),
    ]
    runtime_logs: dict[str, dict[str, Any]] = {}
    graph_logs: dict[str, list[str]] = {}
    for label, graph, build, relative in log_specs:
        parsed = parse_runtime_log(root / relative, root, label, build)
        runtime_logs[label] = parsed
        graph_logs.setdefault(graph, []).append(label)

    oracle_by_graph = {
        str(row["graph"]): row for row in oracle_manifest["records"]
    }
    baseline_by_graph = {
        "delaunay_n20": "stage0_delaunay_n20_smoke",
        "delaunay_n23": "stage0_delaunay_n23_baseline",
        "rgg_n_2_20_s0": "stage0_rgg_baseline",
        "atmosmodm": "stage0_atmosmodm_baseline",
        "rmat22": "stage0_rmat22_baseline",
    }
    latest_control_by_graph = {
        "rgg_n_2_20_s0": "latest_rgg_control_job38180",
        "usa_gplus": "latest_usa_control_job38180",
    }
    latest_compact_by_graph = {
        "rgg_n_2_20_s0": "latest_rgg_compact_job38180",
        "usa_gplus": "latest_usa_compact_job38180",
    }

    graph_records: list[dict[str, Any]] = []
    for record in dataset["records"]:
        name = graph_name_from_record(record)
        vertices = int(record["header"]["vertices"])
        edges = int(record["header"]["directed_edges"])
        base_label = baseline_by_graph.get(name)
        partition_basis: str
        if base_label:
            partitions = partitions_from_log(runtime_logs[base_label])
            partition_basis = "RUNTIME_LOG_VALIDATION"
        else:
            partitions, partition_basis = static_partitions(record)
        if sum(part.vertices for part in partitions) != vertices:
            raise RuntimeError(f"partition vertex mismatch for {name}")
        if sum(part.edges for part in partitions) != edges:
            raise RuntimeError(f"partition edge mismatch for {name}")

        memory_cases: list[dict[str, Any]] = []
        if base_label:
            base_log = runtime_logs[base_label]
            boundary = {
                int(key): value for key, value in base_log["boundary_index"].items()
            }
            memory_cases.append(
                memory_case(
                    name="stage0_baseline_control_exact_inputs",
                    partitions=partitions,
                    graph_vertices=vertices,
                    boundary=boundary,
                    compact=None,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_FORMULA_PLUS_RUNTIME_ALLOCATION_INPUTS",
                    runtime_log=base_label,
                )
            )
        else:
            memory_cases.append(
                memory_case(
                    name="stage0_control_static_upper_bound_not_run",
                    partitions=partitions,
                    graph_vertices=vertices,
                    boundary=None,
                    compact=None,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_CONSERVATIVE_UPPER_BOUND_NOT_RUN",
                    runtime_log=None,
                )
            )

        current_control_label = latest_control_by_graph.get(name)
        if current_control_label:
            current_log = runtime_logs[current_control_label]
            current_parts = partitions_from_log(current_log)
            current_boundary = {
                int(key): value
                for key, value in current_log["boundary_index"].items()
            }
            memory_cases.append(
                memory_case(
                    name="latest_r96v2_control_exact_inputs",
                    partitions=current_parts,
                    graph_vertices=vertices,
                    boundary=current_boundary,
                    compact=None,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_FORMULA_PLUS_RUNTIME_ALLOCATION_INPUTS",
                    runtime_log=current_control_label,
                )
            )

        current_compact_label = latest_compact_by_graph.get(name)
        if current_compact_label:
            current_log = runtime_logs[current_compact_label]
            current_parts = partitions_from_log(current_log)
            current_boundary = {
                int(key): value
                for key, value in current_log["boundary_index"].items()
            }
            current_compact = {
                int(key): value
                for key, value in current_log["compact_candidates"].items()
            }
            memory_cases.append(
                memory_case(
                    name="latest_r96v2_compact_exact_inputs",
                    partitions=current_parts,
                    graph_vertices=vertices,
                    boundary=current_boundary,
                    compact=current_compact,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_FORMULA_PLUS_RUNTIME_ALLOCATION_INPUTS",
                    runtime_log=current_compact_label,
                )
            )
        else:
            memory_cases.append(
                memory_case(
                    name="latest_r96v2_compact_static_upper_bound_not_run",
                    partitions=partitions,
                    graph_vertices=vertices,
                    boundary=None,
                    compact={},
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_CONSERVATIVE_UPPER_BOUND_NOT_RUN",
                    runtime_log=None,
                )
            )

        oracle = oracle_by_graph.get(name)
        graph_records.append(
            {
                "graph": name,
                "stage0_dataset": True,
                "path": record["realpath"],
                "sha256": record["sha256"],
                "vertices": vertices,
                "edges": edges,
                "partition_basis": partition_basis,
                "partitions": [part.__dict__ for part in partitions],
                "all_graph_and_partition_int_fields_fit_int32": all(
                    0 <= value <= INT32_MAX
                    for value in [
                        vertices,
                        edges,
                        *[part.vertices for part in partitions],
                        *[part.edges for part in partitions],
                        *[part.begin for part in partitions],
                        *[part.end for part in partitions],
                    ]
                ),
                "numeric": numeric_audit(
                    name=name,
                    vertices=vertices,
                    minimum_weight=int(record["weights"]["minimum"]),
                    maximum_weight=int(record["weights"]["maximum"]),
                    negative_count=int(record["weights"]["negative_count"]),
                    oracle=oracle,
                ),
                "memory_cases": memory_cases,
                "runtime_logs": graph_logs.get(name, []),
                "gpu_execution_status": (
                    "RUNTIME_VALIDATED_FOR_RECORDED_QUERIES"
                    if base_label
                    else "NOT_RUN_REJECTED_NEGATIVE_WEIGHT_INPUT"
                ),
            }
        )

    # USA G+ is the frozen stage1 target rather than a stage0 dataset record.
    usa_control = runtime_logs["latest_usa_control_job38180"]
    usa_compact = runtime_logs["latest_usa_compact_job38180"]
    usa_parts = partitions_from_log(usa_control)
    usa_vertices = int(usa_control["graph_vertices"])
    usa_edges = int(usa_control["graph_edges"])
    usa_boundary = {
        int(key): value for key, value in usa_control["boundary_index"].items()
    }
    usa_compact_rows = {
        int(key): value for key, value in usa_compact["compact_candidates"].items()
    }
    usa_manifest = load_json(
        root
        / "evidence/l3_30h_20260923/stage1/route_b2_compact_r96_job38180/compact/usa_gplus_s11973673_d400000_b107_compact/manifest.json"
    )
    usa_graph_input = usa_manifest["inputs"]["graph"]
    graph_records.append(
        {
            "graph": "usa_gplus",
            "stage0_dataset": False,
            "path": usa_graph_input["path"],
            "sha256": usa_graph_input["sha256"],
            "vertices": usa_vertices,
            "edges": usa_edges,
            "partition_basis": "RUNTIME_LOG_VALIDATION_CUT60",
            "partitions": [part.__dict__ for part in usa_parts],
            "all_graph_and_partition_int_fields_fit_int32": all(
                0 <= value <= INT32_MAX
                for value in [
                    usa_vertices,
                    usa_edges,
                    *[part.vertices for part in usa_parts],
                    *[part.edges for part in usa_parts],
                    *[part.begin for part in usa_parts],
                    *[part.end for part in usa_parts],
                ]
            ),
            "numeric": numeric_audit(
                name="usa_gplus",
                vertices=usa_vertices,
                minimum_weight=None,
                maximum_weight=int(usa_compact["maximum_weight_reported"]),
                negative_count=None,
                oracle={
                    "status": "RUNTIME_WIDE_ORACLE_CORRECT_FOR_RECORDED_QUERIES",
                    "path": usa_manifest["inputs"]["oracle"]["path"],
                    "sha256": usa_manifest["inputs"]["oracle"]["sha256"],
                    "note": "No max-distance/checked transient-add manifest is available here.",
                },
            ),
            "memory_cases": [
                memory_case(
                    name="latest_r96v2_control_exact_inputs",
                    partitions=usa_parts,
                    graph_vertices=usa_vertices,
                    boundary=usa_boundary,
                    compact=None,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_FORMULA_PLUS_RUNTIME_ALLOCATION_INPUTS",
                    runtime_log="latest_usa_control_job38180",
                ),
                memory_case(
                    name="latest_r96v2_compact_exact_inputs",
                    partitions=partitions_from_log(usa_compact),
                    graph_vertices=usa_vertices,
                    boundary={
                        int(key): value
                        for key, value in usa_compact["boundary_index"].items()
                    },
                    compact=usa_compact_rows,
                    gpu_total_bytes=gpu_total_bytes,
                    evidence_class="STATIC_FORMULA_PLUS_RUNTIME_ALLOCATION_INPUTS",
                    runtime_log="latest_usa_compact_job38180",
                ),
            ],
            "runtime_logs": graph_logs["usa_gplus"],
            "gpu_execution_status": "RUNTIME_VALIDATED_FOR_RECORDED_QUERIES",
        }
    )

    all_memory_cases = [
        case for graph in graph_records for case in graph["memory_cases"]
    ]
    all_raw_fit = all(case["status"] == "STATIC_RAW_REQUEST_BOUND_PASS" for case in all_memory_cases)
    non_proven_l2_logs = [
        label
        for label, log in runtime_logs.items()
        if any(
            not row["no_bucket_wrap_status"].startswith("RUNTIME_PROVEN")
            for row in log["l2"].values()
        )
    ]
    numeric_not_proven = [
        graph["graph"]
        for graph in graph_records
        if graph["numeric"]["status"].startswith("NOT_PROVEN")
    ]

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/multigpu/audit_l3_capacity.py",
        "execution_policy": "CPU/read-only audit; no GPU program or scheduler command invoked",
        "status_vocabulary": {
            "STATIC": "Derived from archived source/build commands and audited input metadata.",
            "RUNTIME": "Validated only for the queries in an existing log.",
            "NOT_RUN": "No matching execution was performed or consumed.",
            "NOT_PROVEN": "Available evidence cannot establish the claimed bound.",
        },
        "inputs": {
            "dataset_manifest": {
                "path": relpath(dataset_path, root),
                "sha256": sha256_path(dataset_path),
            },
            "oracle_manifest": {
                "path": relpath(oracle_path, root),
                "sha256": sha256_path(oracle_path),
            },
            "hardware": hardware,
            "builds": build_audits,
        },
        "constants": {
            "int32_min": INT32_MIN,
            "int32_max": INT32_MAX,
            "node_record_bytes": NODE_BYTES,
            "worker_threads": WORKER_THREADS,
            "worker_warps_per_block": WORKER_WARPS,
            "a100_sms": A100_SMS,
            "maximum_work_blocks": MAX_WORK_BLOCKS,
            "ack_slots": ACK_SLOTS,
            "bulk_inbox_slots": BULK_INBOX_SLOTS,
            "retained_tx_records": WARP_SIZE * BULK_L3_BATCH,
            "retained_tx_shared_bytes": (
                WARP_SIZE * BULK_L3_BATCH * (INT_BYTES + INT_BYTES)
            ),
            "l2": L2,
        },
        "static_contracts": [
            {
                "component": "CSR and graph integer domain",
                "status": "STATIC_PROVEN_FOR_AUDITED_INPUTS",
                "proof": "CSR row, destination, edge counts, partition bounds, and graph sizes are int; every audited value is checked below INT32_MAX.",
                "source": ["SSSP/csr_graph.h:22,115-119", "SSSP/graph_partition.h:107-165"],
            },
            {
                "component": "distance arithmetic",
                "status": "MIXED_SEE_PER_GRAPH",
                "proof": "VALUE_TYPE is signed int and hot-path additions are unchecked. Unit-weight graphs have a sufficient n*max_weight bound; atmosmodm, rmat22, and USA G+ do not.",
                "source": ["core/include/common.h:11-19", "SSSP/sssp_run.cu:2446,2621,4604"],
            },
            {
                "component": "L2 allocation and addressing",
                "status": "STATIC_ALLOCATION_BOUND_PROVEN_RUNTIME_PEAK_MIXED",
                "proof": "The INT_MAX-byte request rounds down to 16 rings of 16,776,704 records; modulo addresses remain inside 268,435,455 allocated records. The implementation explicitly does not handle signed counter overflow and has no fullness guard.",
                "source": ["SSSP/sssp.cuh:6-7", "core/cu_delta_queue/cu_delta_queue.cuh:83-165,430-487"],
            },
            {
                "component": "candidate and bitmap hierarchy",
                "status": "STATIC_PROVEN_FOR_CURRENT_N2_DOMAIN",
                "proof": "Candidate/cache arrays have peer_vertices+1 entries; candidate IDs are 1..peer_vertices and mark/hint indexes are ceil-divided from 0-based peer IDs. Compact mode retains dense candidate allocations.",
                "source": ["SSSP/sssp_run.cu:331-394", "SSSP/l3/l3_candidate.cuh:7-44", "SSSP/l3/l3_boundary_index.cuh:23-33,149-174"],
            },
            {
                "component": "boundary index",
                "status": "STATIC_PROVEN_WITH_RUNTIME_EXACT_OR_STATIC_UPPER_BYTES",
                "proof": "Only peer-local destinations in [0,peer_size) enter the index. Sparse storage is allocated only below half the dense word count; dense mode allocates no index buffer.",
                "source": ["SSSP/l3/l3_boundary_index.cuh:64-124"],
            },
            {
                "component": "inbox and dense TX",
                "status": "STATIC_PROVEN_FOR_AUDITED_PARTITION_SIZES",
                "proof": "Each receiver owns two (local_vertices+1)-record slots. Published count is clamped to peer_vertices, which equals the receiver local size for n=2; slot offsets fit int for every audited partition.",
                "source": ["SSSP/sssp_run.cu:422-445", "SSSP/l3/l3_bulk.cuh:433-506"],
            },
            {
                "component": "retained TX journal",
                "status": "STATIC_PROVEN_SHARED_MEMORY_BOUND",
                "proof": "The retained journal is the same 32*128 shared entries (32,768 bytes for ID and distance arrays), not a second global allocation. Extraction clamps l3_cnt to 4,096 and leaves overflow marked for a later scan.",
                "source": ["SSSP/sssp_run.cu:9012-9015,9026-9028,9695-9697,9842-9848"],
            },
            {
                "component": "query workspace",
                "status": "STATIC_EXPLICIT_ALLOCATION_ACCOUNTED_RUNTIME_PEAK_NOT_LOGGED",
                "proof": "Exactly one selected queue workspace per GPU allocates the L2 queue and reuses it across sources. WORK_COUNT=false removes the histogram/counter allocations; QUERY_WORKSPACE_DIAG is not enabled, so free-memory peak is NOT_RUN.",
                "source": ["SSSP/sssp_run.cu:1125-1224", "core/src/ml_queue.cu:3-98"],
            },
            {
                "component": "worker ACK slots",
                "status": "STATIC_PROVEN_AND_LATEST_RUNTIME_VALIDATED",
                "proof": "A100 allocates (108-1)*16=1,712 slots and runtime work blocks are clamped to at most 107. Latest logs report active_slots=capacity=1,712 on both GPUs.",
                "source": ["SSSP/sssp_run.cu:295-318,11099-11110"],
            },
        ],
        "formal_runner_recommendation": {
            "legacy_log_gate": {
                "status": "RECOMMENDED_STRICT_SUFFICIENT_GATE",
                "rule": (
                    "For every dual-GPU sample and GPU, parse the matching L2_CAPACITY "
                    "and require L2_FINAL writes < per_bucket, plus reads == writes == "
                    "completed."
                ),
                "proof": (
                    "Every physical bucket's cumulative writes are at most the aggregate "
                    "writes across all buckets. If the aggregate is below one bucket's "
                    "capacity, no bucket can execute even one modulo wrap."
                ),
                "failure_semantics": (
                    "aggregate writes >= per_bucket is NOT_PROVEN, not evidence that an "
                    "overwrite occurred."
                ),
            },
            "preferred_exact_source_gate": {
                "status": "RECOMMENDED_HOST_ONLY_DIAGNOSTIC",
                "rule": (
                    "In report_final_l2_counts, compute max_bucket_writes from the already "
                    "copied write_reserve array and emit per_bucket_capacity, counter_bits, "
                    "and no_wrap=(max_bucket_writes <= q.total_size). Have the formal runner "
                    "require no_wrap=1. Equality is safe: exactly capacity writes address "
                    "indices 0..capacity-1 once."
                ),
                "hot_path_impact": "none; post-sync host reduction only",
                "terminology": (
                    "Call this no_wrap, not universal no_overwrite: a future run may safely "
                    "wrap after consumption, which needs a peak-outstanding diagnostic."
                ),
                "counter_scope": (
                    "write_total <= INT32_MAX proves the aggregate read_done/debug totals, "
                    "not every signed counter in arbitrary runs. For a blanket counter claim, "
                    "also bound/log read_ptr and first_pos; write_total <= INT32_MAX/BNUM is "
                    "a simple conservative current-build first_pos gate."
                ),
            },
            "peak_outstanding_extension": {
                "status": "NOT_RUN_OPTIONAL_FOR_WRAPPING_WORKLOADS",
                "rule": (
                    "If max_bucket_writes exceeds capacity but the workload must still be "
                    "accepted, use an isolated diagnostic build to record a conservative "
                    "per-bucket maximum of write_reserve - bucket_read_done at reservation."
                ),
            },
        },
        "graphs": graph_records,
        "runtime_logs": runtime_logs,
        "gates": {
            "explicit_device_allocation_raw_request_fit": {
                "status": "PASS" if all_raw_fit else "FAIL",
                "scope": "All reported exact cases and conservative NOT_RUN upper-bound cases fit the preflight A100 total bytes. CUDA runtime/module overhead is not statically bounded.",
            },
            "index_ranges_for_current_n2_build": {
                "status": "PASS",
                "scope": "Audited graph/partition, candidate, bitmap, inbox, boundary, retained journal, and ACK indexes only.",
            },
            "l2_ring_no_overwrite": {
                "status": "NOT_PROVEN_FOR_ALL_RECORDED_RUNS",
                "logs_requiring_per_bucket_peak_or_write_diagnostics": non_proven_l2_logs,
                "reason": "Delaunay_n23 and latest compact USA have all-bucket totals above one bucket capacity; aggregate final conservation cannot establish peak per-bucket occupancy.",
            },
            "signed_counter_overflow": {
                "status": "RUNTIME_TOTALS_IN_RANGE_GENERAL_PROOF_ABSENT",
                "reason": "Logged final totals are below INT32_MAX, but the queue source explicitly excludes counter-overflow handling and no max read_ptr/write_reserve diagnostic is logged.",
            },
            "checked_int32_candidate_addition": {
                "status": "NOT_PROVEN_FOR_ALL_ADMITTED_GRAPHS",
                "graphs": numeric_not_proven,
                "reason": "Correct final output and int64 oracle agreement do not prove every transient signed int addition avoided overflow.",
            },
            "compact_stage0_execution": {
                "status": "NOT_RUN_EXCEPT_RGG_AND_USA_GPLUS",
                "reason": "Other stage0 compact rows are conservative static memory bounds only.",
            },
            "overall_capacity_and_numeric_contract": {
                "status": "NOT_PROVEN",
                "blocking_items": [
                    "Per-bucket peak L2 occupancy/write totals are absent where aggregate totals exceed one bucket capacity.",
                    "Checked or wide transient candidate-add evidence is absent for atmosmodm, rmat22, and USA G+.",
                ],
                "non_blocking_positive_evidence": [
                    "All explicit device allocation requests fit with large raw-request margin.",
                    "Current n=2 index formulas fit all audited graph and partition sizes.",
                    "Recorded valid runs end with per-bucket L2 conservation assertions and wide-oracle correctness.",
                ],
            },
        },
    }

    rendered = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    if args.stdout:
        print(rendered, end="")
    else:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, output)
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
