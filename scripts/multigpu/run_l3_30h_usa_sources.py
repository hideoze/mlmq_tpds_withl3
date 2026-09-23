#!/usr/bin/env python3
"""Run the two frozen additional USA sources from one fresh clean-HEAD pair.

This is the robustness companion to the primary formal run.  It deliberately
does not select a source and it does not accept externally built binaries.
The wrapper validates the committed source manifest, creates one fresh
single/dual pair with the canonical builder, then runs both additional sources
serially through ``run_l3_30h.py`` in AB/BA order.  A result below 1.20 remains
a valid measurement; only an execution, correctness, provenance, or evidence
contract failure makes the wrapper fail.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from typing import Any

from run_l3_30h import (
    FORMAL_INPUT_CONTRACT,
    QUEUE_TYPE_IDS,
    build_formal_pair,
    formal_pair_integrity_snapshot,
    formal_runtime_environment,
    gpu_preflight,
    parse_dual_contract,
    parse_oracle_contract,
    read_gr_header,
    sample_statistics,
    validate_formal_pair_build,
)
from run_usa_road_matrix import parse_bench, sha256


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/multigpu/run_l3_30h.py"
PAIR_BUILDER = ROOT / "scripts/multigpu/build_l3_30h_pair.sh"
SOURCE_MANIFEST = (
    ROOT / "evidence/l3_30h_20260923/final/usa_source_manifest.json"
)

SOURCE_ROLES = ("fixed_additional_source_1", "fixed_additional_source_2")
EXPECTED_SOURCE_IDS = {
    "fixed_additional_source_1": {
        "old_logical_id": 7982448,
        "new_layout_id": 18266241,
        "oracle_i32_sha256":
            "f86e1b6509badf703eb50125156f3479938a928b2abbb2430910673bcf9ca9e3",
    },
    "fixed_additional_source_2": {
        "old_logical_id": 15964897,
        "new_layout_id": 6146689,
        "oracle_i32_sha256":
            "84664767003586c247a6657450796b0ee2ffbfde8ba84445f70ba05b9c008294",
    },
}
EXPECTED_GRAPH_SHA256 = (
    "85c273900a89422369a06f5524f784b58ed91ceef36addb4ae3710dff5a1d6eb"
)
EXPECTED_PRIMARY_SOURCE = 11973673
EXPECTED_VERTICES = 23947347
EXPECTED_EDGES = 66684784
DELTA = 400000
CUT_PERCENT = 60
BLOCKS = 107
WORKERS = 512
QUEUE = "L1SLF_L2DQ"
WINDOW_MODE = 2
WINDOW_MIN = 25000
WINDOW_MAX = 25000
IDLE_BACKOFF = 0
WARMUPS = 1
REPEATS = 5
ROUNDS = 2
PROCESS_SAMPLES = WARMUPS + REPEATS
FORMAL_SAMPLES_PER_CONFIGURATION = ROUNDS * REPEATS
TARGET = 1.20
EXPECTED_PROCESS_ORDER = (
    (0, 0, "single_no_l3", 1),
    (0, 1, "dual_l3", 2),
    (1, 0, "dual_l3", 2),
    (1, 1, "single_no_l3", 1),
)
FAILURE_MARKERS = (
    "Error at node",
    "BENCH_REQUIRE_FAILURE",
    "ORACLE_MISMATCH",
    "ORACLE_SIZE_ERROR",
    "FINAL_AUDIT_ERROR",
    "L2_FINAL_UNSUPPORTED",
    "CUDA error",
    "illegal memory access",
)


class ContractError(RuntimeError):
    """A frozen-input, clean-build, execution, or evidence contract failed."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {description} {path}: {error}") from error


def git_bytes(*arguments: str) -> bytes:
    process = subprocess.run(
        ["/usr/bin/git", "-C", str(ROOT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if process.returncode:
        detail = process.stderr.decode(errors="replace").strip()
        raise ContractError(
            f"git {' '.join(arguments)} failed with rc={process.returncode}: {detail}"
        )
    return process.stdout


def clean_head(expected: str | None = None) -> dict[str, Any]:
    head = git_bytes("rev-parse", "HEAD").decode().strip()
    status = git_bytes(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).decode().rstrip()
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "invalid git HEAD")
    require(not status, f"USA-source run requires a clean worktree:\n{status}")
    if expected is not None:
        require(head == expected, f"git HEAD changed: {head} != {expected}")
    branch = git_bytes("branch", "--show-current").decode().strip()
    return {"head": head, "branch": branch, "status_porcelain": []}


def committed_file(path: Path) -> dict[str, str]:
    require(not path.is_symlink(), f"committed input must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    require(resolved.is_file(),
            f"committed input is not a regular file: {resolved}")
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise ContractError(f"committed input is outside repository: {resolved}") from error
    raw = resolved.read_bytes()
    expected = git_bytes("show", f"HEAD:{relative.as_posix()}")
    require(raw == expected, f"{relative} differs from clean HEAD")
    return {
        "path": str(resolved),
        "head_path": relative.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def regular_file(path: Path, description: str) -> Path:
    require(not path.is_symlink(), f"{description} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve {description} {path}: {error}") from error
    require(resolved.is_file(), f"{description} is not a regular file: {resolved}")
    return resolved


def resolve_manifest_path(raw: Any, description: str) -> Path:
    require(isinstance(raw, str) and raw, f"{description} must be a path string")
    requested = Path(raw)
    if not requested.is_absolute():
        requested = ROOT / requested
    return regular_file(requested, description)


def load_formal_configuration() -> dict[str, Any]:
    identity = committed_file(ROOT / FORMAL_INPUT_CONTRACT)
    payload = read_json(Path(identity["path"]), "formal input contract")
    expected = {
        "schema": 1,
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "source": EXPECTED_PRIMARY_SOURCE,
        "delta": DELTA,
        "cut_percent": CUT_PERCENT,
        "blocks": BLOCKS,
        "queue": QUEUE,
        "window_mode": WINDOW_MODE,
        "window_min": WINDOW_MIN,
        "window_max": WINDOW_MAX,
        "idle_backoff": IDLE_BACKOFF,
        "warmups_per_process": WARMUPS,
        "repeats_per_process": REPEATS,
        "rounds": ROUNDS,
        "vertices": EXPECTED_VERTICES,
        "edges": EXPECTED_EDGES,
    }
    differences = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items() if payload.get(key) != value
    }
    require(not differences,
            f"formal input contract differs from frozen USA configuration: {differences}")
    return {"identity": identity, "payload": payload}


def load_source_manifest() -> dict[str, Any]:
    identity = committed_file(SOURCE_MANIFEST)
    payload = read_json(SOURCE_MANIFEST, "USA source manifest")
    require(isinstance(payload, dict) and payload.get("schema") == 2,
            "USA source manifest must be a schema-2 object")
    require(payload.get("selection_status") == "FROZEN_BEFORE_FORMAL_SAMPLING",
            "USA source selection was not frozen before formal sampling")

    graph_row = payload.get("frozen_augmented_input")
    require(isinstance(graph_row, dict), "USA manifest lacks frozen_augmented_input")
    graph = resolve_manifest_path(graph_row.get("path"), "frozen USA graph")
    expected_graph = {
        "sha256": EXPECTED_GRAPH_SHA256,
        "vertices": EXPECTED_VERTICES,
        "directed_edges": EXPECTED_EDGES,
        "cut_percent": CUT_PERCENT,
        "cut_vertex": EXPECTED_VERTICES * CUT_PERCENT // 100,
    }
    observed_graph = {
        "sha256": graph_row.get("sha256"),
        "vertices": graph_row.get("vertices"),
        "directed_edges": graph_row.get("directed_edges"),
        "cut_percent": graph_row.get("cut_percent"),
        "cut_vertex": graph_row.get("cut_vertex"),
    }
    require(observed_graph == expected_graph,
            f"USA graph contract differs: {observed_graph} != {expected_graph}")
    header = read_gr_header(graph)
    require(header["vertices"] == EXPECTED_VERTICES and
            header["edges"] == EXPECTED_EDGES,
            f"USA graph header differs from frozen dimensions: {header}")
    require(sha256(graph) == EXPECTED_GRAPH_SHA256,
            "frozen USA graph SHA256 mismatch")

    raw_sources = payload.get("sources")
    require(isinstance(raw_sources, list) and len(raw_sources) == 3,
            "USA source manifest must contain exactly three frozen sources")
    by_role: dict[str, dict[str, Any]] = {}
    for row in raw_sources:
        require(isinstance(row, dict) and isinstance(row.get("role"), str),
                "USA source manifest contains a malformed source row")
        role = row["role"]
        require(role not in by_role, f"duplicate USA source role: {role}")
        by_role[role] = row
    require(set(by_role) == {
        "development_and_formal_primary", *SOURCE_ROLES,
    }, f"unexpected USA source roles: {sorted(by_role)}")
    require(by_role["development_and_formal_primary"].get("new_layout_id") ==
            EXPECTED_PRIMARY_SOURCE, "primary USA source ID differs")

    selected = []
    for role in SOURCE_ROLES:
        row = by_role[role]
        expected = EXPECTED_SOURCE_IDS[role]
        for key, value in expected.items():
            require(row.get(key) == value,
                    f"{role} {key}={row.get(key)!r}, expected {value!r}")
        require(row.get("reached") == EXPECTED_VERTICES and
                row.get("unreachable") == 0 and row.get("over_int32") == 0,
                f"{role} oracle reachability/range contract differs")
        oracle = resolve_manifest_path(row.get("oracle_i32_path"), f"{role} oracle")
        require(oracle.stat().st_size == EXPECTED_VERTICES * 4,
                f"{role} oracle byte count is not vertices*4")
        require(row.get("oracle_i32_bytes") == oracle.stat().st_size,
                f"{role} manifest oracle byte count differs")
        require(sha256(oracle) == expected["oracle_i32_sha256"],
                f"{role} oracle SHA256 mismatch")
        selected.append({
            "role": role,
            "old_logical_id": row["old_logical_id"],
            "source": row["new_layout_id"],
            "graph": graph,
            "oracle": oracle,
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "oracle_sha256": expected["oracle_i32_sha256"],
            "vertices": EXPECTED_VERTICES,
            "edges": EXPECTED_EDGES,
        })
    return {
        "identity": identity,
        "payload": payload,
        "graph": graph,
        "header": header,
        "sources": selected,
    }


def input_snapshot(source_manifest: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {"graph": source_manifest["graph"]}
    for source in source_manifest["sources"]:
        paths[source["role"] + "_oracle"] = source["oracle"]
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }


def source_command(source: dict[str, Any], pair: Path, out: Path,
                   timeout: int) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        str(RUNNER.resolve()),
        "--sampling", "exploratory",
        "--dual-binary", str((pair / "dual_build/mlmq").resolve()),
        "--single-binary", str((pair / "single_build/mlmq").resolve()),
        "--pair-build", str(pair.resolve()),
        "--graph", str(source["graph"]),
        "--oracle", str(source["oracle"]),
        "--source", str(source["source"]),
        "--delta", str(DELTA),
        "--cut-percent", str(CUT_PERCENT),
        "--blocks", str(BLOCKS),
        "--warmups", str(WARMUPS),
        "--repeats", str(REPEATS),
        "--rounds", str(ROUNDS),
        "--timeout", str(timeout),
        "--out", str(out),
        "--queue", QUEUE,
        "--final-audit", "all",
        "--expected-window-mode", str(WINDOW_MODE),
        "--expected-window-min", str(WINDOW_MIN),
        "--expected-window-max", str(WINDOW_MAX),
        "--expected-idle-backoff", str(IDLE_BACKOFF),
    ]


def run_child(command: list[str], log: Path, environment: dict[str, str],
              timeout: int) -> dict[str, Any]:
    started = now_utc()
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
        rc, raw, timed_out = process.returncode, process.stdout, False
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        raw += f"\nOUTER_TIMEOUT_SECONDS={timeout}\n"
        rc, timed_out = 124, True
    except OSError as error:
        raw = f"OUTER_LAUNCH_ERROR={error!r}\n"
        rc, timed_out = 125, False
    log.write_text(raw, encoding="utf-8")
    return {
        "command": command,
        "started_utc": started,
        "finished_utc": now_utc(),
        "rc": rc,
        "timed_out": timed_out,
        "log": str(log),
        "log_sha256": sha256(log),
        "timeout_seconds": timeout,
    }


def close_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def validate_capacity_lines(raw: str, gpu_count: int) -> list[str]:
    errors: list[str] = []
    pattern = re.compile(
        r"^L2_CAPACITY budget=(\d+) record_bytes=(\d+) buckets=(\d+) "
        r"per_bucket=(\d+) allocated_records=(\d+) counter_bits=(\d+)$",
        re.MULTILINE,
    )
    rows = [tuple(map(int, match.groups()))
            for match in pattern.finditer(raw)]
    if len(rows) != gpu_count:
        errors.append(f"expected {gpu_count} L2_CAPACITY rows, got {len(rows)}")
    for row in rows:
        budget, record_bytes, buckets, per_bucket, allocated, bits = row
        expected_allocated = budget // record_bytes if record_bytes else -1
        expected_per_bucket = (
            expected_allocated // buckets // 512 * 512 if buckets else -1
        )
        if (budget <= 0 or record_bytes <= 0 or buckets <= 0 or
                allocated != expected_allocated or
                per_bucket != expected_per_bucket or per_bucket <= 0 or bits != 32):
            errors.append(f"invalid L2_CAPACITY row: {row}")
    if gpu_count == 2 and len(rows) == 2 and rows[0] != rows[1]:
        errors.append("dual GPUs reported different L2_CAPACITY rows")
    return errors


def validate_raw_log(path: Path, *, gpu_count: int, source: int,
                     vertices: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return {"valid": False, "errors": [f"cannot read raw log: {error}"]}
    rows = parse_bench(raw)
    if len(rows) != PROCESS_SAMPLES:
        errors.append(f"expected {PROCESS_SAMPLES} BENCH rows, got {len(rows)}")
    for index, row in enumerate(rows):
        expected = {
            "algorithm": "MLMQ",
            "gpu_count": str(gpu_count),
            "source": str(source),
            "repeat": str(index),
            "warmup": str(int(index < WARMUPS)),
            "queue": QUEUE,
            "correct": "1",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(
                    f"BENCH row {index} {key}={row.get(key)!r}, expected {value!r}"
                )
        for field in ("solve_ms", "query_wall_ms"):
            try:
                value = float(row[field])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError
            except (KeyError, ValueError):
                errors.append(f"BENCH row {index} invalid {field}")

    oracle = parse_oracle_contract(raw, PROCESS_SAMPLES, vertices)
    errors.extend(oracle["errors"])
    errors.extend(validate_capacity_lines(raw, gpu_count))
    found = [marker for marker in FAILURE_MARKERS if marker in raw]
    if found:
        errors.append(f"failure markers present: {found}")
    if len(re.findall(r"^RUN_RC=0$", raw, re.MULTILINE)) != 1:
        errors.append("raw log does not contain exactly one RUN_RC=0 line")

    partitions = [tuple(map(int, match.groups())) for match in re.finditer(
        r"GPU(\d+) partition: \[(\d+), (\d+)\)", raw)]
    if gpu_count == 1:
        if partitions:
            errors.append(f"independent single logged GPU partitions: {partitions}")
        configs = re.findall(
            r"^NO_L3_CONFIG workers=(\d+) delta=(\d+) queue=(\S+)$",
            raw, re.MULTILINE)
        launches = [tuple(map(int, match.groups())) for match in re.finditer(
            r"^NO_L3_LAUNCH work_blocks=(\d+) delta=(\d+) repeat=(\d+)$",
            raw, re.MULTILINE)]
        if configs != [(str(WORKERS), str(DELTA), QUEUE)]:
            errors.append(f"invalid NO_L3_CONFIG rows: {configs}")
        expected_launches = [(BLOCKS, DELTA, index)
                             for index in range(PROCESS_SAMPLES)]
        if launches != expected_launches:
            errors.append(f"invalid NO_L3_LAUNCH rows: {launches}")
        if re.search(r"^L3_CONFIG ", raw, re.MULTILINE):
            errors.append("independent single unexpectedly logged L3_CONFIG")
        if re.search(r"^FINAL_AUDIT ", raw, re.MULTILINE):
            errors.append("independent single unexpectedly logged FINAL_AUDIT")
        dual = None
    else:
        cut = vertices * CUT_PERCENT // 100
        expected_partitions = [(0, 0, cut), (1, cut, vertices)]
        if sorted(set(partitions)) != expected_partitions:
            errors.append(
                f"effective partitions={sorted(set(partitions))}, "
                f"expected={expected_partitions}"
            )
        dual = parse_dual_contract(
            raw, PROCESS_SAMPLES, BLOCKS, DELTA, QUEUE_TYPE_IDS[QUEUE],
            WINDOW_MODE, WINDOW_MIN, WINDOW_MAX, IDLE_BACKOFF,
            require_l2_final=True,
        )
        errors.extend(dual["errors"])
        expected_rx = {
            "rx_priority_bootstrap": 0,
            "rx_express": 0,
            "rx_express_enabled": 0,
            "rx_express_slots": 64,
            "rx_express_batch": 32,
            "rx_l2_pull": 0,
            "rx_l2_pull_claim": 0,
            "rx_l2_pull_enabled": 0,
        }
        for sample_index, sample in enumerate(dual.get("samples", [])):
            for row in sample.get("l3_config", []):
                if row is None:
                    continue
                mismatches = {
                    key: {"actual": row[key], "expected": value}
                    for key, value in expected_rx.items()
                    if row[key] != value
                }
                if mismatches:
                    errors.append(
                        f"sample {sample_index} gpu {row['gpu']} RX config "
                        f"differs: {mismatches}"
                    )
        audits = [tuple(map(int, match.groups())) for match in re.finditer(
            r"^FINAL_AUDIT mismatches=(\d+) residual_edges=(\d+) "
            r"cross_residual_edges=(\d+)$", raw, re.MULTILINE)]
        if audits != [(0, 0, 0)] * PROCESS_SAMPLES:
            errors.append(
                f"expected {PROCESS_SAMPLES} all-zero FINAL_AUDIT rows, got {audits}"
            )
        if re.search(r"^NO_L3_CONFIG ", raw, re.MULTILINE):
            errors.append("dual log unexpectedly contains NO_L3_CONFIG")

    return {
        "valid": not errors,
        "errors": errors,
        "bench_rows": rows,
        "oracle_contract": oracle,
        "dual_contract": dual,
        "partitions": partitions,
        "sha256": sha256(path),
        "failure_markers": found,
    }


def validate_source_output(source: dict[str, Any], case_out: Path,
                           command: list[str], child: dict[str, Any],
                           *, head: str, pair: Path,
                           timeout: int, slurm_job_id: str) -> dict[str, Any]:
    errors: list[str] = []

    def load(name: str, expected_type: type) -> Any:
        try:
            value = read_json(case_out / name, f"source {name}")
        except ContractError as error:
            errors.append(str(error))
            return None
        if not isinstance(value, expected_type):
            errors.append(
                f"{name} has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
            return None
        return value

    if child["rc"] != 0:
        errors.append(f"runner exited with rc={child['rc']}")
    if child["timed_out"]:
        errors.append("outer runner timed out")
    manifest = load("manifest.json", dict)
    records = load("records.json", list)
    samples = load("samples.json", list)
    summary = load("summary.json", dict)

    expected_configuration = {
        "sampling": "exploratory",
        "source": source["source"],
        "delta": DELTA,
        "cut_percent": CUT_PERCENT,
        "blocks": BLOCKS,
        "warmups_per_process": WARMUPS,
        "formal_repeats_per_process": REPEATS,
        "rounds": ROUNDS,
        "timeout_seconds": timeout,
        "queue": QUEUE,
        "final_audit": "all",
        "target_speedup": TARGET,
    }
    binary_paths = {
        "single_no_l3": (pair / "single_build/mlmq").resolve(),
        "dual_l3": (pair / "dual_build/mlmq").resolve(),
    }
    binary_hashes = {key: sha256(path) for key, path in binary_paths.items()}
    if manifest is not None:
        for key, value in {
            "status": "measurement_valid",
            "measurement_valid": True,
            "all_processes_valid": True,
            "process_count": len(EXPECTED_PROCESS_ORDER),
            "slurm_job_id": slurm_job_id,
        }.items():
            if manifest.get(key) != value:
                errors.append(
                    f"manifest {key}={manifest.get(key)!r}, expected {value!r}"
                )
        if manifest.get("command") != command[1:]:
            errors.append("runner manifest command differs from wrapper command")
        configuration = manifest.get("configuration", {})
        for key, value in expected_configuration.items():
            if configuration.get(key) != value:
                errors.append(
                    f"configuration {key}={configuration.get(key)!r}, expected {value!r}"
                )
        if configuration.get("gpu_contract") != {
            "single_no_l3": 1, "dual_l3": 2, "fallback_allowed": False,
        }:
            errors.append("runner GPU contract is not strict 1/2 without fallback")
        if configuration.get("expected_window") != {
            "mode": WINDOW_MODE,
            "min_cycles": WINDOW_MIN,
            "max_cycles": WINDOW_MAX,
            "idle_backoff": IDLE_BACKOFF,
        }:
            errors.append("runner expected-window contract differs")
        git = manifest.get("git", {})
        if git.get("head") != {"rc": 0, "output": head} or \
                git.get("status") != {"rc": 0, "output": ""}:
            errors.append("runner did not observe the same clean HEAD")
        inputs = manifest.get("inputs", {})
        expected_inputs = {
            "single_binary": (
                str(binary_paths["single_no_l3"]), binary_hashes["single_no_l3"]),
            "dual_binary": (
                str(binary_paths["dual_l3"]), binary_hashes["dual_l3"]),
            "graph": (str(source["graph"]), source["graph_sha256"]),
            "oracle": (str(source["oracle"]), source["oracle_sha256"]),
            "runner": (str(RUNNER.resolve()), sha256(RUNNER)),
        }
        for label, (path, digest) in expected_inputs.items():
            row = inputs.get(label, {})
            if row.get("path") != path or row.get("sha256") != digest:
                errors.append(f"runner input identity differs for {label}: {row}")
        if inputs.get("oracle", {}).get("bytes") != source["vertices"] * 4:
            errors.append("runner oracle byte count differs")
        preflight = manifest.get("preflight", {})
        if preflight.get("visible_gpu_count") != 2 or \
                preflight.get("compute_apps_before") != []:
            errors.append("runner preflight did not observe two idle GPUs")
        gpu_rows = preflight.get("visible_gpus")
        if not isinstance(gpu_rows, list) or len(gpu_rows) != 2 or any(
                not isinstance(row, dict) or "A100" not in str(row.get("name")) or
                str(row.get("compute_capability")) != "8.0" for row in gpu_rows):
            errors.append(f"runner preflight lacks exact two A100 CC8.0 rows: {gpu_rows}")

    formal: dict[str, list[dict[str, str]]] = {
        "single_no_l3": [], "dual_l3": [],
    }
    raw_evidence = []
    if records is not None:
        if len(records) != len(EXPECTED_PROCESS_ORDER):
            errors.append(
                f"records.json has {len(records)} rows, "
                f"expected {len(EXPECTED_PROCESS_ORDER)}"
            )
        for index, expected in enumerate(EXPECTED_PROCESS_ORDER):
            if index >= len(records) or not isinstance(records[index], dict):
                errors.append(f"records.json lacks object row {index}")
                continue
            round_id, sequence, label, gpu_count = expected
            record = records[index]
            prefix = case_out / f"round{round_id:03d}_{sequence}_{label}"
            for key, value in {
                "round": round_id,
                "sequence_in_round": sequence,
                "configuration": label,
                "requested_gpu_count": gpu_count,
                "prefix": str(prefix),
                "rc": 0,
                "valid": True,
                "reason": "ok",
            }.items():
                if record.get(key) != value:
                    errors.append(
                        f"record {index} {key}={record.get(key)!r}, expected {value!r}"
                    )
            expected_solver = [
                str(binary_paths[label]), "-i", str(source["graph"]),
                "-n", str(gpu_count), "-d", str(DELTA),
            ]
            if record.get("command") != expected_solver:
                errors.append(f"record {index} solver command differs")
            individual = prefix.with_suffix(".json")
            if not individual.is_file() or read_json(
                    individual, "individual process record") != record:
                errors.append(f"record {index} differs from {individual.name}")
            for suffix in (".gpu_before.log", ".gpu_after.log"):
                snapshot = prefix.with_suffix(suffix)
                if not snapshot.is_file() or snapshot.stat().st_size == 0:
                    errors.append(f"missing/empty GPU snapshot {snapshot}")
            raw = validate_raw_log(
                prefix.with_suffix(".log"), gpu_count=gpu_count,
                source=source["source"], vertices=source["vertices"],
            )
            raw_evidence.append({
                "record_index": index,
                "round": round_id,
                "sequence": sequence,
                "configuration": label,
                **raw,
            })
            errors.extend(f"record {index}: {error}" for error in raw["errors"])
            record_samples = record.get("samples")
            if record_samples != raw.get("bench_rows"):
                errors.append(f"record {index} samples differ from raw BENCH rows")
                record_samples = []
            effective = record.get("effective_contract")
            if not isinstance(effective, dict) or effective.get("errors") != []:
                errors.append(f"record {index} effective contract is invalid")
            elif gpu_count == 2:
                dual = effective.get("dual_contract", {})
                if (not isinstance(dual, dict) or dual.get("valid") is not True or
                        dual.get("l2_final_present") is not True or
                        dual.get("l2_capacity_present") is not True or
                        dual.get("errors") != []):
                    errors.append(f"record {index} serialized dual contract is invalid")
            for row in record_samples:
                if row.get("warmup") == "0":
                    formal[label].append(row)

    if samples is not None:
        expected_count = len(EXPECTED_PROCESS_ORDER) * PROCESS_SAMPLES
        if len(samples) != expected_count:
            errors.append(f"samples.json has {len(samples)} rows, expected {expected_count}")
        if any(not isinstance(row, dict) or row.get("run_valid") is not True
               for row in samples):
            errors.append("samples.json contains an invalid-run sample")

    metrics: dict[str, Any] = {}
    if all(len(rows) == FORMAL_SAMPLES_PER_CONFIGURATION
           for rows in formal.values()):
        try:
            single_stats = sample_statistics(formal["single_no_l3"], "solve_ms")
            dual_stats = sample_statistics(formal["dual_l3"], "solve_ms")
            single_wall = sample_statistics(
                formal["single_no_l3"], "query_wall_ms")
            dual_wall = sample_statistics(formal["dual_l3"], "query_wall_ms")
            speedup = single_stats["median_ms"] / dual_stats["median_ms"]
            wall_speedup = single_wall["median_ms"] / dual_wall["median_ms"]
            metrics = {
                "T1_single_no_l3_solve": single_stats,
                "T2_dual_l3_solve": dual_stats,
                "T1_single_no_l3_query_wall": single_wall,
                "T2_dual_l3_query_wall": dual_wall,
                "S_solve_T1_over_T2": speedup,
                "S_query_wall_T1_over_T2": wall_speedup,
                "target": TARGET,
                "target_met": speedup >= TARGET,
            }
        except (KeyError, ValueError, ZeroDivisionError) as error:
            errors.append(f"cannot recompute source statistics: {error}")
    else:
        errors.append(
            "formal sample counts differ: " +
            repr({key: len(value) for key, value in formal.items()})
        )

    if summary is not None and metrics:
        if summary.get("sampling") != "exploratory" or \
                summary.get("measurement_valid") is not True:
            errors.append("runner summary is not a valid exploratory measurement")
        combined = summary.get("combined", {})
        if not close_number(
                combined.get("S_solve_T1_over_T2"),
                metrics["S_solve_T1_over_T2"]):
            errors.append("runner solve speedup differs from raw samples")
        if not close_number(
                combined.get("S_query_wall_T1_over_T2"),
                metrics["S_query_wall_T1_over_T2"]):
            errors.append("runner query-wall speedup differs from raw samples")
        if summary.get("target_met") is not metrics["target_met"] or \
                summary.get("numeric_target_met") is not metrics["target_met"]:
            errors.append("runner target classification differs from raw samples")
        rounds = summary.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != ROUNDS or any(
                not isinstance(row, dict) or row.get("valid") is not True
                for row in rounds):
            errors.append("runner summary lacks two valid reverse-order rounds")

    measurement_valid = not errors
    target_met = metrics.get("target_met") if measurement_valid else None
    return {
        "role": source["role"],
        "old_logical_id": source["old_logical_id"],
        "source": source["source"],
        "measurement_valid": measurement_valid,
        "target_met": target_met,
        "target_failure_invalidates_measurement": False,
        "errors": errors,
        "metrics": metrics,
        "raw_process_evidence": raw_evidence,
        "runner": child,
        "case_output": str(case_out),
        "case_manifest_sha256": sha256(case_out / "manifest.json")
            if (case_out / "manifest.json").is_file() else None,
        "case_summary_sha256": sha256(case_out / "summary.json")
            if (case_out / "summary.json").is_file() else None,
    }


def artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"output contains a symlink: {path}")
        if path.is_file() and path.name != "artifact_sha256.json":
            hashes[path.relative_to(root).as_posix()] = sha256(path)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path,
        help="new evidence directory outside the repository",
    )
    parser.add_argument(
        "--timeout", type=int, default=400,
        help="seconds per single/dual solver process (default: 400)",
    )
    parser.add_argument(
        "--outer-slack", type=int, default=1200,
        help="extra seconds around each four-process source runner",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.outer_slack <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        initial_git = clean_head()
        committed = {
            "wrapper": committed_file(Path(__file__)),
            "runner": committed_file(RUNNER),
            "pair_builder": committed_file(PAIR_BUILDER),
            "source_manifest": committed_file(SOURCE_MANIFEST),
        }
        formal_configuration = load_formal_configuration()
        source_manifest = load_source_manifest()
        initial_inputs = input_snapshot(source_manifest)

        requested = args.out.absolute()
        require(not requested.exists() and not requested.is_symlink(),
                f"output already exists: {requested}")
        requested.parent.mkdir(parents=True, exist_ok=True)
        out = requested.parent.resolve(strict=True) / requested.name
        require(out != ROOT and ROOT not in out.parents,
                "USA-source output must be outside the repository")
        out.mkdir()
    except ContractError as error:
        print(f"USA_SOURCE_CONTRACT_ERROR {error}", file=sys.stderr)
        return 2

    write_json(out / "status.json", {
        "status": "preflight_pending",
        "measurement_valid": False,
        "created_utc": now_utc(),
    })
    try:
        preflight = gpu_preflight(out, require_a100=True)
        pair, build_record = build_formal_pair(ROOT, out)
        clean_head(initial_git["head"])
        dual = (pair / "dual_build/mlmq").resolve(strict=True)
        single = (pair / "single_build/mlmq").resolve(strict=True)
        try:
            pair_validation = validate_formal_pair_build(
                pair, dual, single, initial_git["head"], ROOT)
        except SystemExit as error:
            raise ContractError(f"fresh pair validation failed: {error}") from error
        pair_start = formal_pair_integrity_snapshot(pair, ROOT)

        environment, cleared = formal_runtime_environment()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        run_manifest = {
            "schema": 1,
            "purpose": "fixed additional USA-source robustness measurement",
            "created_utc": now_utc(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "repository": {"root": str(ROOT), **initial_git},
            "committed_files": committed,
            "formal_configuration": formal_configuration,
            "source_manifest": {
                "identity": source_manifest["identity"],
                "selection_status": source_manifest["payload"]["selection_status"],
                "roles": list(SOURCE_ROLES),
            },
            "configuration": {
                "sources": [row["source"] for row in source_manifest["sources"]],
                "delta": DELTA,
                "cut_percent": CUT_PERCENT,
                "blocks": BLOCKS,
                "workers": WORKERS,
                "queue": QUEUE,
                "window_mode": WINDOW_MODE,
                "window_min": WINDOW_MIN,
                "window_max": WINDOW_MAX,
                "idle_backoff": IDLE_BACKOFF,
                "round_order": ["single then dual", "dual then single"],
                "warmups_per_process": WARMUPS,
                "formal_per_process": REPEATS,
                "rounds": ROUNDS,
                "target": TARGET,
                "target_failure_invalidates_measurement": False,
                "gpu_execution": "strictly serial",
                "external_pair_allowed": False,
            },
            "preflight": preflight,
            "fresh_pair_build_record": build_record,
            "pair_validation": pair_validation,
            "pair_integrity_start": pair_start,
            "input_integrity_start": initial_inputs,
            "runtime_environment": environment,
            "cleared_environment_variables": cleared,
        }
        write_json(out / "run_manifest.json", run_manifest)
        write_json(out / "status.json", {
            "status": "sampling",
            "measurement_valid": False,
            "started_utc": now_utc(),
            "completed_sources": 0,
            "expected_sources": len(SOURCE_ROLES),
        })

        results = []
        driver_dir = out / "driver_logs"
        driver_dir.mkdir()
        case_root = out / "sources"
        case_root.mkdir()
        runner_hash = committed["runner"]["sha256"]
        source_manifest_hash = committed["source_manifest"]["sha256"]
        child_timeout = args.timeout * len(EXPECTED_PROCESS_ORDER) + args.outer_slack
        for index, source in enumerate(source_manifest["sources"]):
            clean_head(initial_git["head"])
            require(sha256(RUNNER) == runner_hash, "canonical runner changed")
            require(sha256(SOURCE_MANIFEST) == source_manifest_hash,
                    "USA source manifest changed")
            require(formal_pair_integrity_snapshot(pair, ROOT) == pair_start,
                    "pair artifacts changed before a source run")
            case_out = case_root / source["role"]
            command = source_command(source, pair, case_out, args.timeout)
            command_record = {
                "role": source["role"],
                "source": source["source"],
                "command": command,
                "cwd": str(ROOT),
                "outer_timeout_seconds": child_timeout,
            }
            write_json(driver_dir / f"{source['role']}.command.json", command_record)
            child = run_child(
                command, driver_dir / f"{source['role']}.log",
                environment, child_timeout,
            )
            result = validate_source_output(
                source, case_out, command, child,
                head=initial_git["head"], pair=pair,
                timeout=args.timeout,
                slurm_job_id=os.environ["SLURM_JOB_ID"],
            )
            results.append(result)
            write_json(out / "source_summaries" / f"{source['role']}.json", result)
            write_json(out / "source_index.json", results)
            write_json(out / "status.json", {
                "status": "sampling",
                "measurement_valid": False,
                "updated_utc": now_utc(),
                "completed_sources": len(results),
                "expected_sources": len(SOURCE_ROLES),
                "invalid_sources_so_far": sum(
                    row["measurement_valid"] is not True for row in results
                ),
            })
            print(
                f"USA_SOURCE_DONE {index + 1}/{len(SOURCE_ROLES)} "
                f"role={source['role']} source={source['source']} "
                f"measurement_valid={int(result['measurement_valid'])} "
                f"target_met={result['target_met']}",
                flush=True,
            )

        final_git = clean_head(initial_git["head"])
        pair_end = formal_pair_integrity_snapshot(pair, ROOT)
        require(pair_end == pair_start, "pair artifacts changed during sampling")
        final_inputs = input_snapshot(source_manifest)
        require(final_inputs == initial_inputs,
                "graph or oracle inputs changed during sampling")
        require(sha256(RUNNER) == runner_hash, "canonical runner changed")
        require(sha256(SOURCE_MANIFEST) == source_manifest_hash,
                "USA source manifest changed")
        try:
            validate_formal_pair_build(
                pair, dual, single, initial_git["head"], ROOT)
        except SystemExit as error:
            raise ContractError(f"post-run pair validation failed: {error}") from error

        measurement_valid = (
            len(results) == len(SOURCE_ROLES) and
            all(row["measurement_valid"] for row in results)
        )
        summary = {
            "measurement_valid": measurement_valid,
            "measurement_validity_is_independent_of_target": True,
            "target": TARGET,
            "target_met_count": sum(row["target_met"] is True for row in results),
            "target_evaluated_count": sum(row["target_met"] is not None
                                          for row in results),
            "all_sources_target_met": (
                len(results) == len(SOURCE_ROLES) and
                all(row["target_met"] is True for row in results)
            ),
            "sources": results,
            "repository_end": final_git,
            "pair_integrity_end": pair_end,
            "input_integrity_end": final_inputs,
            "completed_utc": now_utc(),
        }
        write_json(out / "summary.json", summary)
        write_json(out / "complete.json", {
            "measurement_valid": measurement_valid,
            "sources": len(results),
            "expected_processes": len(SOURCE_ROLES) * len(EXPECTED_PROCESS_ORDER),
            "expected_queries": (
                len(SOURCE_ROLES) * len(EXPECTED_PROCESS_ORDER) * PROCESS_SAMPLES
            ),
            "expected_formal_queries": (
                len(SOURCE_ROLES) * 2 * FORMAL_SAMPLES_PER_CONFIGURATION
            ),
            "target_met_count": summary["target_met_count"],
            "all_sources_target_met": summary["all_sources_target_met"],
            "target_result_affects_exit_status": False,
        })
        write_json(out / "status.json", {
            "status": "measurement_valid" if measurement_valid
                      else "measurement_invalid",
            "measurement_valid": measurement_valid,
            "completed_sources": len(results),
            "expected_sources": len(SOURCE_ROLES),
            "target_met_count": summary["target_met_count"],
            "all_sources_target_met": summary["all_sources_target_met"],
            "completed_utc": summary["completed_utc"],
        })
        write_json(out / "artifact_sha256.json", artifact_hashes(out))
        print(json.dumps({
            "measurement_valid": measurement_valid,
            "target_met_count": summary["target_met_count"],
            "all_sources_target_met": summary["all_sources_target_met"],
            "output": str(out),
        }, indent=2, sort_keys=True), flush=True)
        return 0 if measurement_valid else 1
    except BaseException as error:
        failure = {
            "status": "driver_failed",
            "measurement_valid": False,
            "failure_type": type(error).__name__,
            "failure": repr(error),
            "failed_utc": now_utc(),
        }
        write_json(out / "failure.json", failure)
        write_json(out / "status.json", failure)
        try:
            write_json(out / "artifact_sha256.json", artifact_hashes(out))
        except Exception:
            pass
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f"USA_SOURCE_RUN_FAILED {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
