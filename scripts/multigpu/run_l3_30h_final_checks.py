#!/usr/bin/env python3
"""Run the clean-SHA L3 correctness gates inside one two-A100 allocation.

This is deliberately not a performance runner.  It validates the same clean
pair contract accepted by ``run_l3_30h.py``, then recompiles the exact archived
dual source with the exact recorded command before running one correctness
query on each of the frozen eight G+ inputs.  Every command and complete stdout
is archived.

The sequential-source executable is a diagnostic derivative of the archived
dual source: its only source change is a host-side test hook that selects a
different source/oracle before each query.  A fresh, unmodified archived-source
rebuild is used for the eight-graph regression.
"""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import time

from run_l3_30h import (
    FORMAL_TOOL_PATH,
    QUEUE_TYPE_IDS,
    _git_head_file,
    formal_pair_integrity_snapshot,
    formal_runtime_environment,
    git_snapshot,
    parse_dual_contract,
    parse_gpu_query,
    parse_oracle_contract,
    read_gr_header,
    run_one_with_canonical_tools,
    safe_extract_regular_archive,
    slurm_gpu_evidence,
    validate_formal_pair_build,
)
from run_usa_road_matrix import parse_bench, sha256


ROOT = Path(__file__).resolve().parents[2]
GRAPH_NAMES = ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA")
CANONICAL_GRAPH_MANIFEST = Path(
    "evidence/l3_latest_rerun_38082/matrix/manifest.json")
PREFIX_ENV = ("MLMQ_", "BENCH_", "L3_SUPPLEMENT_")
SEQUENTIAL_DIAGNOSTIC_DEFINE = "-DQUERY_WORKSPACE_DIAG=true"
L2_CAPACITY_RE = re.compile(
    r"^L2_CAPACITY budget=(\d+) record_bytes=(\d+) buckets=(\d+) "
    r"per_bucket=(\d+) allocated_records=(\d+) counter_bits=(\d+)$")
REUSE_RE = re.compile(
    r"^QUERY_REUSE gpu([01]) src=(\d+) reset=(\d+) "
    r"free=([0-9.]+)MiB min=([0-9.]+)MiB max=([0-9.]+)MiB$")
SEQUENTIAL_RE = re.compile(
    r"^SEQUENTIAL_L3_QUERY sample=(\d+) source=(\d+) gpu_count=(\d+)$")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def selected_environment(environment):
    keys = sorted(key for key in environment
                  if key.startswith(("SLURM_", "CUDA_", *PREFIX_ENV)))
    return {key: environment[key] for key in keys}


def command_record(out, name, command, *, environment=None, timeout=600,
                   required=False):
    """Execute without a shell and retain command, stdout, rc, and hashes."""
    command = [str(item) for item in command]
    if environment is None:
        environment, _ = formal_runtime_environment()
    environment = dict(environment)
    command_path = out / f"{name}.command.json"
    log_path = out / f"{name}.log"
    result_path = out / f"{name}.result.json"
    started = now_utc()
    write_json(command_path, {
        "command": command,
        "cwd": str(ROOT),
        "environment": selected_environment(environment),
        "started_utc": started,
        "timeout_seconds": timeout,
    })
    begin = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc, raw = process.returncode, process.stdout
    except subprocess.TimeoutExpired as error:
        timed_out = True
        rc, raw = 124, error.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        raw += f"\nTIMEOUT_SECONDS={timeout}\n"
    except OSError as error:
        rc, raw = 127, f"EXEC_ERROR={error!r}\n"
    elapsed = time.monotonic() - begin
    log_path.write_text(
        raw + f"\nRUN_RC={rc}\nHOST_ELAPSED_SECONDS={elapsed:.6f}\n")
    result = {
        "command": command,
        "command_sha256": sha256(command_path),
        "finished_utc": now_utc(),
        "host_elapsed_seconds": elapsed,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "rc": rc,
        "timed_out": timed_out,
    }
    write_json(result_path, result)
    if required and rc != 0:
        raise RuntimeError(f"{name} failed with rc={rc}; see {log_path}")
    return result, raw


def safe_extract(archive, destination):
    return safe_extract_regular_archive(archive, destination)


def patch_once(path, old, new, description):
    before = path.read_text()
    count = before.count(old)
    if count != 1:
        raise RuntimeError(
            f"{description}: expected one patch anchor in {path}, got {count}")
    after = before.replace(old, new)
    path.write_text(after)
    return before, after


def write_patch(path, before, after, label):
    path.write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"{label}.clean", tofile=f"{label}.diagnostic")))


def clean_benchmark_environment():
    environment, _ = formal_runtime_environment()
    return {key: value for key, value in environment.items()
            if not key.startswith(PREFIX_ENV)}


def validate_small_fixture(name, result, raw):
    errors = []
    if result["rc"] != 0:
        errors.append(f"rc={result['rc']}")
    if name == "primitives":
        for gpu in (0, 1):
            if not re.search(
                    rf"^CANDIDATE gpu={gpu} controlled=96 racing=32 "
                    r"updates_per_race=32768 PASS$", raw, re.MULTILINE):
                errors.append(f"missing candidate PASS for gpu {gpu}")
        for source, target in ((0, 1), (1, 0)):
            pattern = (
                rf"^TRANSPORT source={source} target={target} epochs=256 "
                r"backpressure_retries=\d+ precommit_checks=256 l2_writes=8192 "
                r"delayed_ack_fired=1 pending=0 PASS$")
            if not re.search(pattern, raw, re.MULTILINE):
                errors.append(f"missing transport PASS for {source}->{target}")
        if "L3_SUPPLEMENT_PRIMITIVES PASS" not in raw:
            errors.append("missing final primitive PASS")
    elif name == "dq_publication":
        if not re.search(
                r"^DQ_PUBLICATION tests=400 clamp_diag=[01] PASS$",
                raw, re.MULTILINE):
            errors.append("missing 400-case DQ publication PASS")
    elif name == "capacity_full":
        if "CAPACITY_BOUNDARY accepted=1024 capacity=1024 PASS" not in raw:
            errors.append("missing exact-full acceptance PASS")
    elif name == "capacity_overflow":
        if ("CAPACITY_BOUNDARY rejected=1056 capacity=1024 "
                "expected_device_assert=1 PASS") not in raw:
            errors.append("missing immediate-overflow rejection PASS")
    else:
        errors.append(f"unknown small fixture {name}")
    return {"valid": not errors, "errors": errors, **result}


def parse_capacity_bound(raw, dual_contract):
    """Cross-check the exact per-query no-wrap fields with allocation logs."""
    rows = []
    errors = []
    for line in raw.splitlines():
        if not line.startswith("L2_CAPACITY "):
            continue
        match = L2_CAPACITY_RE.fullmatch(line)
        if not match:
            errors.append(f"malformed L2_CAPACITY line: {line!r}")
            continue
        values = tuple(map(int, match.groups()))
        rows.append({
            "budget": values[0], "record_bytes": values[1],
            "buckets": values[2], "per_bucket": values[3],
            "allocated_records": values[4], "counter_bits": values[5],
        })
    if len(rows) != 2:
        errors.append(f"expected two L2_CAPACITY lines, got {len(rows)}")
    if rows and any(row != rows[0] for row in rows[1:]):
        errors.append("the two GPUs reported different L2 capacities")
    if rows and (rows[0]["per_bucket"] <= 0 or rows[0]["counter_bits"] != 32):
        errors.append("invalid per-bucket capacity/counter width")
    capacity = rows[0]["per_bucket"] if rows else None
    buckets = rows[0]["buckets"] if rows else None
    counter_bits = rows[0]["counter_bits"] if rows else None
    proofs = []
    if capacity is not None:
        for sample_index, sample in enumerate(dual_contract.get("samples", [])):
            for row in sample.get("l2_final", []):
                if row is None:
                    continue
                valid = (
                    row["buckets"] == buckets and
                    row["per_bucket_capacity"] == capacity and
                    row["total_capacity"] == buckets * capacity and
                    row["total_capacity"] <= rows[0]["allocated_records"] and
                    row["total_capacity"] <= 2147483647 and
                    row["counter_bits"] == counter_bits == 32 and
                    row["guarded_writes"] == row["writes"] and
                    row["max_bucket_writes"] <= capacity and
                    row["overflow_guard"] == 1 and
                    row["overflow_detected"] == 0 and
                    row["no_wrap"] == 1)
                proofs.append({
                    "sample": sample_index, "gpu": row["gpu"],
                    "total_writes": row["writes"],
                    "max_bucket_writes": row["max_bucket_writes"],
                    "per_bucket_capacity": capacity,
                    "total_capacity": row["total_capacity"],
                    "allocated_records": rows[0]["allocated_records"],
                    "counter_bits": row["counter_bits"],
                    "guarded_writes": row["guarded_writes"],
                    "overflow_guard": row["overflow_guard"],
                    "overflow_detected": row["overflow_detected"],
                    "no_wrap": row["no_wrap"],
                    "matches_allocation": valid,
                })
                if not valid:
                    errors.append(
                        f"sample {sample_index} gpu {row['gpu']} exact no-wrap "
                        "fields differ from the L2 allocation or fail the gate")
    return {
        "valid": not errors,
        "argument": (
            "every query reports guarded_writes, max_bucket_writes, "
            "per_bucket_capacity, total_capacity, counter_bits, overflow_guard, "
            "overflow_detected, and no_wrap; these exact fields must agree with "
            "the initial L2 allocation, while the isolated guarded fixture "
            "also tests exact-full acceptance and first-overflow rejection"),
        "capacity_rows": rows,
        "proofs": proofs,
        "errors": errors,
    }


def strict_dual_contract(raw, *, samples, vertices, blocks, delta, queue,
                         window_mode, window_min, window_max, idle_backoff):
    oracle = parse_oracle_contract(raw, samples, vertices)
    dual = parse_dual_contract(
        raw, samples, blocks, delta, QUEUE_TYPE_IDS[queue], window_mode,
        window_min, window_max, idle_backoff, require_l2_final=True)
    capacity = parse_capacity_bound(raw, dual)
    errors = list(oracle["errors"]) + list(dual["errors"]) + list(capacity["errors"])
    for marker in ("Error at node", "ORACLE_MISMATCH", "ORACLE_SIZE_ERROR"):
        if marker in raw:
            errors.append(f"solver reported {marker}")
    return {
        "valid": not errors,
        "oracle": oracle,
        "dual": dual,
        "capacity_no_wrap": capacity,
        "errors": errors,
    }


def fixture_graph(path):
    n = 2052
    adjacency = [[] for _ in range(n)]
    for vertex in range(2049):
        adjacency[vertex].append((vertex + 1, 1 + vertex % 3))
        adjacency[vertex + 1].append((vertex, 1 + vertex % 5))
    for vertex in range(1, 513):
        adjacency[0].append((vertex, 1000 + vertex))
        adjacency[vertex].append((1025, vertex % 7))
        adjacency[1025].append((vertex, 2 + vertex % 5))
    for vertex in (5, 1030):
        adjacency[vertex].append((vertex + 1, 0))
        adjacency[vertex + 1].append((vertex, 0))
    ends, destinations, weights = [], [], []
    count = 0
    for edges in adjacency:
        count += len(edges)
        ends.append(count)
        for target, weight in edges:
            destinations.append(target)
            weights.append(weight)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4Q", 1, 4, n, count))
        stream.write(struct.pack(f"<{n}Q", *ends))
        stream.write(struct.pack(f"<{count}I", *destinations))
        if count % 2:
            stream.write(b"\0" * 4)
        stream.write(struct.pack(f"<{count}i", *weights))
    return {"vertices": n, "edges": count, "sha256": sha256(path)}


def translated_compile_command(pair, source, binary, nvcc, *, workspace_diag):
    original = json.loads((pair / "dual_build/command.json").read_text())
    if not isinstance(original, list) or not all(isinstance(item, str) for item in original):
        raise RuntimeError("dual command.json is not a string array")
    translated = []
    skip_output = False
    sources = {"main.cu", "csr_graph.cu", "sssp_run.cu"}
    for index, argument in enumerate(original):
        if index == 0:
            translated.append(nvcc)
            continue
        if skip_output:
            translated.append(str(binary))
            skip_output = False
            continue
        if argument == "-o":
            translated.append(argument)
            skip_output = True
            continue
        candidate = Path(argument)
        if candidate.name in sources and candidate.parent.name == "SSSP":
            translated.append(str(source / "SSSP" / candidate.name))
            continue
        if argument.startswith("-I"):
            include = Path(argument[2:])
            if include.name == "include" and include.parent.name == "core":
                translated.append("-I" + str(source / "core/include"))
                continue
        translated.append(argument)
    if skip_output:
        raise RuntimeError("dual command ends immediately after -o")
    existing_workspace_defines = [
        argument for argument in translated
        if argument.startswith("-DQUERY_WORKSPACE_DIAG=")]
    if existing_workspace_defines:
        raise RuntimeError(
            "pair command unexpectedly fixes QUERY_WORKSPACE_DIAG: "
            f"{existing_workspace_defines}")
    if workspace_diag:
        translated.append(SEQUENTIAL_DIAGNOSTIC_DEFINE)
        if translated.count(SEQUENTIAL_DIAGNOSTIC_DEFINE) != 1:
            raise RuntimeError("sequential diagnostic define is not unique")
    return original, translated


def diagnostic_compile_command(pair, source, binary, nvcc):
    return translated_compile_command(
        pair, source, binary, nvcc, workspace_diag=True)


def patch_sequential_source(source, out):
    path = source / "SSSP/main.cu"
    before = path.read_text()
    include_anchor = "#include <vector>\n"
    loop_anchor = "\tfor (int sample = 0; sample < repeats + warmups; ++sample) {\n"
    if before.count(include_anchor) != 1 or before.count(loop_anchor) != 1:
        raise RuntimeError("sequential-source host hook anchors changed")
    after = before.replace(include_anchor, include_anchor + "#include <sstream>\n")
    hook = r'''
	std::vector<int> l3_30h_sources;
	if (const char *sequence = std::getenv("L3_SUPPLEMENT_SOURCES")) {
		std::stringstream stream(sequence);
		std::string token;
		while (std::getline(stream, token, ',')) {
			int value = std::stoi(token);
			if (value < 0 || value >= g.nnodes) return 2;
			l3_30h_sources.push_back(value);
		}
		if (l3_30h_sources.size() != size_t(repeats + warmups)) return 2;
	}
	for (int sample = 0; sample < repeats + warmups; ++sample) {
	if (!l3_30h_sources.empty()) {
		src = l3_30h_sources[sample];
		const char *directory = std::getenv("L3_SUPPLEMENT_ORACLE_DIR");
		if (!directory) return 2;
		std::string path = std::string(directory) + "/s" + std::to_string(src) + ".i32";
		setenv("L3_SUPPLEMENT_ORACLE", path.c_str(), 1);
		FILE *oracle = fopen(path.c_str(), "rb");
		if (!oracle) return 2;
		bool loaded = fread(node_data_base, sizeof(int), g.nnodes, oracle) == size_t(g.nnodes)
		           && fgetc(oracle) == EOF;
		fclose(oracle);
		if (!loaded) return 2;
		printf("SEQUENTIAL_L3_QUERY sample=%d source=%d gpu_count=%d\n", sample, src, n_gpu);
	}
'''
    after = after.replace(loop_anchor, hook)
    path.write_text(after)
    patch_path = out / "sequential_host_hook.patch"
    write_patch(patch_path, before, after, "SSSP/main.cu")
    return {
        "source": str(path),
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": sha256(path),
        "patch": str(patch_path),
        "patch_sha256": sha256(patch_path),
        "scope": "host-only source/oracle selection before each query",
    }


def validate_sequential(raw, *, vertices, sources, blocks, delta, queue,
                        window_mode, window_min, window_max, idle_backoff):
    contract = strict_dual_contract(
        raw, samples=len(sources), vertices=vertices, blocks=blocks,
        delta=delta, queue=queue, window_mode=window_mode,
        window_min=window_min, window_max=window_max,
        idle_backoff=idle_backoff)
    errors = list(contract["errors"])
    rows = parse_bench(raw)
    if len(rows) != len(sources):
        errors.append(f"expected {len(sources)} BENCH rows, got {len(rows)}")
    for index, row in enumerate(rows[:len(sources)]):
        expected = {
            "algorithm": "MLMQ", "gpu_count": "2",
            "source": str(sources[index]), "repeat": str(index),
            "warmup": "0", "queue": queue, "correct": "1",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(
                    f"BENCH {index} {key}={row.get(key)!r}, expected {value!r}")
    selected = [tuple(map(int, match.groups())) for line in raw.splitlines()
                if (match := SEQUENTIAL_RE.fullmatch(line))]
    expected_selected = [(index, source, 2)
                         for index, source in enumerate(sources)]
    if selected != expected_selected:
        errors.append(
            f"source selection lines {selected}, expected {expected_selected}")
    reuse = {0: [], 1: []}
    for line in raw.splitlines():
        match = REUSE_RE.fullmatch(line)
        if match:
            gpu, source, reset = map(int, match.groups()[:3])
            reuse[gpu].append((source, reset))
    expected_reuse = [(source, index + 1)
                      for index, source in enumerate(sources)]
    for gpu in (0, 1):
        if reuse[gpu] != expected_reuse:
            errors.append(
                f"gpu {gpu} reset sequence {reuse[gpu]}, expected {expected_reuse}")
    contract.update({
        "valid": not errors,
        "errors": errors,
        "bench_rows": rows,
        "selected_sources": selected,
        "query_reuse": reuse,
    })
    return contract


def load_graphs(path):
    payload = json.loads(path.read_text())
    graphs = payload.get("graphs")
    if not isinstance(graphs, dict) or tuple(graphs) != GRAPH_NAMES:
        raise RuntimeError(
            f"graph manifest must contain exactly {GRAPH_NAMES} in order")
    resolved = {}
    for name in GRAPH_NAMES:
        row = graphs[name]
        graph = Path(row["augmented"]).resolve(strict=True)
        oracle = Path(row["oracle"]).resolve(strict=True)
        expected = row.get("hashes", {})
        actual = {"augmented": sha256(graph), "oracle": sha256(oracle)}
        if actual != {key: expected.get(key) for key in actual}:
            raise RuntimeError(
                f"{name} graph/oracle hash mismatch: expected={expected} actual={actual}")
        header = read_gr_header(graph)
        if header["vertices"] != row["vertices"]:
            raise RuntimeError(f"{name} vertex count differs from manifest")
        if oracle.stat().st_size != 4 * header["vertices"]:
            raise RuntimeError(f"{name} oracle byte count is invalid")
        resolved[name] = {
            "graph": graph, "oracle": oracle, "source": int(row["source"]),
            "cut_percent": int(row["cut_percent"]), "header": header,
            "hashes": actual,
        }
    return resolved


def graph_input_integrity(path, graphs):
    return {
        "manifest": {"path": str(path), "sha256": sha256(path)},
        "graphs": {
            name: {
                "graph": {"path": str(row["graph"]),
                          "sha256": sha256(row["graph"])},
                "oracle": {"path": str(row["oracle"]),
                           "sha256": sha256(row["oracle"])},
            }
            for name, row in graphs.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-build", required=True, type=Path,
                        help="successful clean-HEAD build_l3_30h_pair.sh output")
    parser.add_argument("--graph-manifest", required=True, type=Path,
                        help="frozen eight-G+ matrix manifest")
    parser.add_argument("--out", required=True, type=Path,
                        help="new evidence directory outside the repository")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cxx", default="/usr/bin/g++")
    parser.add_argument(
        "--boost-include", type=Path,
        default=Path("/a100-data/wyh/boost_1_87_0"))
    parser.add_argument("--blocks", type=int, default=107)
    parser.add_argument("--delta", type=int, default=200000)
    parser.add_argument("--queue", choices=("L1SLF_L2DQ", "L1V_L2DQ"),
                        default="L1SLF_L2DQ")
    parser.add_argument("--window-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--window-min", type=int, default=25000)
    parser.add_argument("--window-max", type=int, default=25000)
    parser.add_argument("--idle-backoff", type=int, choices=(0, 1), default=0)
    parser.add_argument("--timeout", type=int, default=400,
                        help="seconds per GPU executable")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("checks must run inside a Slurm allocation")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    if (args.blocks <= 0 or args.delta <= 0 or args.window_min <= 0 or
            args.window_max < args.window_min or args.timeout <= 0):
        parser.error("blocks/delta/windows/timeout are invalid")
    return args


def main():
    args = parse_args()
    git_info = git_snapshot(ROOT)
    if (git_info["head"]["rc"] != 0 or git_info["status"]["rc"] != 0 or
            git_info["status"]["output"]):
        raise SystemExit(
            "final checks require a clean readable worktree; status is:\n" +
            git_info["status"]["output"])
    output = args.out.resolve(strict=False)
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("final-check output must be outside the repository")
    pair = args.pair_build.resolve(strict=True)
    graph_manifest = args.graph_manifest.resolve(strict=True)
    canonical_graph_manifest = (ROOT / CANONICAL_GRAPH_MANIFEST).resolve(strict=True)
    if graph_manifest != canonical_graph_manifest:
        raise SystemExit(
            "final checks require the canonical frozen eight-graph manifest: "
            f"{canonical_graph_manifest}")
    if graph_manifest.read_bytes() != _git_head_file(
            ROOT, CANONICAL_GRAPH_MANIFEST):
        raise SystemExit("frozen eight-graph manifest differs from clean HEAD")
    matrix_contract = json.loads(graph_manifest.read_text())
    expected_configuration = {
        "blocks": int(matrix_contract.get("blocks", -1)),
        "delta": int(matrix_contract.get("delta", -1)),
        "queue": "L1SLF_L2DQ",
        "window_mode": 2,
        "window_min": 25000,
        "window_max": 25000,
        "idle_backoff": 0,
    }
    requested_configuration = {
        "blocks": args.blocks, "delta": args.delta, "queue": args.queue,
        "window_mode": args.window_mode, "window_min": args.window_min,
        "window_max": args.window_max, "idle_backoff": args.idle_backoff,
    }
    if requested_configuration != expected_configuration:
        raise SystemExit(
            "final-check configuration differs from the frozen canonical "
            f"contract: requested={requested_configuration} "
            f"expected={expected_configuration}")
    dual_binary = (pair / "dual_build/mlmq").resolve(strict=True)
    single_binary = (pair / "single_build/mlmq").resolve(strict=True)
    try:
        pair_evidence = validate_formal_pair_build(
            pair, dual_binary, single_binary,
            git_info["head"]["output"], ROOT)
    except SystemExit as error:
        raise SystemExit(f"pair-build validation failed: {error}") from error
    pair_integrity_before = formal_pair_integrity_snapshot(pair, ROOT)
    requested_nvcc = shutil.which(str(args.nvcc), path=FORMAL_TOOL_PATH)
    if requested_nvcc is None:
        raise SystemExit(f"cannot resolve --nvcc executable: {args.nvcc}")
    requested_nvcc = Path(requested_nvcc).resolve(strict=True)
    pair_nvcc = Path(pair_evidence["compiler"]["path"]).resolve(strict=True)
    if (requested_nvcc != pair_nvcc or
            sha256(requested_nvcc) != pair_evidence["compiler"]["sha256"]):
        raise SystemExit(
            "final checks must use the exact compiler recorded by the clean pair: "
            f"requested={requested_nvcc} pair={pair_nvcc}")
    args.nvcc = str(pair_nvcc)
    requested_boost = args.boost_include.resolve(strict=True)
    pair_boost = Path(
        pair_evidence["version"]["boost_include_dir"]).resolve(strict=True)
    if requested_boost != pair_boost:
        raise SystemExit(
            "final checks must use the exact Boost include recorded by the clean "
            f"pair: requested={requested_boost} pair={pair_boost}")
    args.boost_include = pair_boost
    requested_cxx = shutil.which(str(args.cxx), path=FORMAL_TOOL_PATH)
    canonical_cxx = Path("/usr/bin/g++").resolve(strict=True)
    if requested_cxx is None or Path(requested_cxx).resolve(strict=True) != canonical_cxx:
        raise SystemExit(
            f"final checks require canonical C++ compiler {canonical_cxx}")
    args.cxx = str(canonical_cxx)
    graphs = load_graphs(graph_manifest)
    graph_integrity_before = graph_input_integrity(graph_manifest, graphs)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    shutil.copy2(__file__, output / "runner.py")
    effective_environment, cleared_environment = formal_runtime_environment()
    manifest = {
        "schema": 1,
        "status": "starting",
        "created_utc": now_utc(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "allocation": slurm_gpu_evidence(os.environ),
        "launch_environment": selected_environment(os.environ),
        "effective_environment": effective_environment,
        "cleared_environment_variables": cleared_environment,
        "git": git_info,
        "pair_build": pair_evidence,
        "pair_integrity_before": pair_integrity_before,
        "graph_integrity_before": graph_integrity_before,
        "configuration": {
            "blocks": args.blocks, "delta": args.delta, "queue": args.queue,
            "window_mode": args.window_mode, "window_min": args.window_min,
            "window_max": args.window_max, "idle_backoff": args.idle_backoff,
            "warmups": 0, "repeats_per_graph": 1,
            "purpose": "correctness only; no performance conclusion",
            "nvcc": str(pair_nvcc),
            "nvcc_sha256": sha256(pair_nvcc),
            "boost_include": str(pair_boost),
            "cxx": str(canonical_cxx),
            "cxx_sha256": sha256(canonical_cxx),
        },
        "inputs": {
            "runner": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256(Path(__file__).resolve())},
            "graph_manifest": {"path": str(graph_manifest),
                               "sha256": sha256(graph_manifest)},
            "dual_binary": {"path": str(dual_binary),
                            "sha256": sha256(dual_binary)},
            "test_sources": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    ROOT / "scripts/multigpu/test_l3_supplement.cu",
                    ROOT / "scripts/multigpu/test_dq_publication.cu",
                    ROOT / "scripts/multigpu/test_dq_capacity.cu",
                    ROOT / "scripts/multigpu/supplement_graph.cpp",
                )
            },
            "graphs": {
                name: {
                    "graph": str(row["graph"]), "oracle": str(row["oracle"]),
                    "source": row["source"], "cut_percent": row["cut_percent"],
                    "header": row["header"], "hashes": row["hashes"],
                } for name, row in graphs.items()
            },
        },
    }
    write_json(output / "manifest.json", manifest)
    problems = []

    preflight = {}
    for name, command, strict in (
            ("nvidia_smi", ["nvidia-smi"], True),
            ("gpu_query", ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap",
                           "--format=csv,noheader,nounits"], True),
            ("topology", ["nvidia-smi", "topo", "-m"], True),
            ("compute_apps", ["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"], True),
            ("slurm_job", ["scontrol", "show", "job", "-dd",
                           os.environ["SLURM_JOB_ID"]], False)):
        record, raw = command_record(
            output, name, command, timeout=30, required=False)
        preflight[name] = record
        if strict and record["rc"] != 0:
            problems.append(f"preflight {name} failed with rc={record['rc']}")
        if name == "gpu_query":
            try:
                record["visible_gpus"] = parse_gpu_query(
                    raw, require_a100=True)
            except RuntimeError as error:
                problems.append(str(error))
        if name == "compute_apps" and raw.strip():
            problems.append("GPU compute application present before checks")
    manifest["preflight"] = preflight
    if problems:
        manifest.update(status="preflight_failed", problems=problems)
        write_json(output / "manifest.json", manifest)
        write_json(output / "results.json", {"valid": False, "problems": problems})
        return 2

    source_snapshot = output / "dual_source"
    safe_extract(pair / "dual_build/source.tgz", source_snapshot)
    guarded_core = output / "guarded_core"
    shutil.copytree(source_snapshot / "core", guarded_core)
    dq = guarded_core / "cu_delta_queue/cu_delta_queue.cuh"
    guard_anchor = (
        "current_reserve = atomicAdd(&write_reserve[dst_bucket_id], "
        "write_bucket_num);")
    guarded_line = (
        guard_anchor +
        "\n                assert(current_reserve >= 0 && "
        "(long long)current_reserve + write_bucket_num <= total_size);")
    before, after = patch_once(
        dq, guard_anchor, guarded_line, "isolated DQ capacity guard")
    guard_patch = output / "capacity_guard.patch"
    write_patch(guard_patch, before, after, "core/cu_delta_queue/cu_delta_queue.cuh")
    manifest["capacity_guard"] = {
        "production_source_modified": False,
        "source_archive": str(pair / "dual_build/source.tgz"),
        "source_archive_sha256": sha256(pair / "dual_build/source.tgz"),
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": sha256(dq),
        "patch": str(guard_patch),
        "patch_sha256": sha256(guard_patch),
        "claim": "controlled boundary acceptance/rejection only",
    }

    binaries = output / "binaries"
    binaries.mkdir()
    production_binary = binaries / "archived_dual"
    original_production_command, production_command = translated_compile_command(
        pair, source_snapshot, production_binary, args.nvcc,
        workspace_diag=False)
    production_build, _ = command_record(
        output, "build_archived_dual", production_command,
        timeout=1200, required=True)
    production_build.update(
        binary=str(production_binary),
        binary_sha256=sha256(production_binary),
        source_archive=str(pair / "dual_build/source.tgz"),
        source_archive_sha256=sha256(pair / "dual_build/source.tgz"),
        original_pair_command=original_production_command,
        derivation=(
            "exact validated clean-HEAD dual archive and canonical command; "
            "only source/include/output paths were relocated"),
    )
    manifest["archived_dual_rebuild"] = production_build
    common = [
        "-O3", "-std=c++17", "-rdc=true",
        "-gencode=arch=compute_80,code=sm_80",
        "-I" + str(args.boost_include.resolve()),
    ]
    compile_specs = {
        "test_primitives": [
            args.nvcc, ROOT / "scripts/multigpu/test_l3_supplement.cu",
            "-o", binaries / "test_primitives", *common,
            "-I" + str(source_snapshot / "core/include"),
            "-DL3_DIRECT_RX=true", "-DL3_FAULT_INJECT_ACK_DELAY=true",
            "-DBULK_NO_CACHE=true",
        ],
        "test_dq": [
            args.nvcc, ROOT / "scripts/multigpu/test_dq_publication.cu",
            "-o", binaries / "test_dq", *common,
            "-I" + str(source_snapshot / "core/include"),
        ],
        "test_capacity": [
            args.nvcc, ROOT / "scripts/multigpu/test_dq_capacity.cu",
            "-o", binaries / "test_capacity", *common,
            "-I" + str(guarded_core / "include"),
        ],
    }
    builds = {}
    for name, command in compile_specs.items():
        record, _ = command_record(
            output, "build_" + name, command, timeout=900, required=True)
        binary = binaries / name
        record["binary"] = str(binary)
        record["binary_sha256"] = sha256(binary)
        builds[name] = record
    manifest["fixture_builds"] = builds
    write_json(output / "manifest.json", manifest)

    small_results = {}
    for name, command in (
            ("primitives", [binaries / "test_primitives"]),
            ("dq_publication", [binaries / "test_dq"]),
            ("capacity_full", [binaries / "test_capacity"]),
            ("capacity_overflow", [binaries / "test_capacity", "overflow"])):
        result, raw = command_record(
            output, name, command, timeout=args.timeout)
        checked = validate_small_fixture(name, result, raw)
        small_results[name] = checked
        if not checked["valid"]:
            problems.extend(f"{name}: {error}" for error in checked["errors"])
        write_json(output / "small_results.json", small_results)

    exporter = binaries / "export_graph"
    export_build, _ = command_record(
        output, "build_export_graph",
        [args.cxx, "-std=c++17", "-O3",
         ROOT / "scripts/multigpu/supplement_graph.cpp", "-o", exporter],
        timeout=300, required=True)
    export_build.update(binary=str(exporter), binary_sha256=sha256(exporter))
    fixture_dir = output / "sequential_fixture"
    fixture_dir.mkdir()
    fixture = fixture_dir / "fixture.gr"
    fixture_info = fixture_graph(fixture)
    exports = {}
    for source in (0, 1030, 2050):
        record, _ = command_record(
            output, f"export_source_{source}",
            [exporter, fixture, str(source), str(fixture_info["vertices"] // 2),
             fixture_dir / f"s{source}.gr", fixture_dir / f"s{source}.i32"],
            timeout=120, required=True)
        graph = fixture_dir / f"s{source}.gr"
        oracle = fixture_dir / f"s{source}.i32"
        record["outputs"] = {
            "graph": {"path": str(graph), "sha256": sha256(graph)},
            "oracle": {"path": str(oracle), "sha256": sha256(oracle)},
        }
        exports[str(source)] = record

    sequential_patch = patch_sequential_source(source_snapshot, output)
    sequential_binary = binaries / "sequential_dual"
    original_command, sequential_command = diagnostic_compile_command(
        pair, source_snapshot, sequential_binary, args.nvcc)
    if sequential_command.count(SEQUENTIAL_DIAGNOSTIC_DEFINE) != 1:
        raise RuntimeError(
            "sequential build must enable QUERY_WORKSPACE_DIAG exactly once")
    sequential_build, _ = command_record(
        output, "build_sequential_dual", sequential_command,
        timeout=1200, required=True)
    sequential_build.update(
        binary=str(sequential_binary), binary_sha256=sha256(sequential_binary),
        original_pair_command=original_command,
        diagnostic_compile_defines=[SEQUENTIAL_DIAGNOSTIC_DEFINE],
        query_workspace_diag_enabled=True,
        derivation="archived dual source plus recorded host-only query selector")
    sources = (0, 1030, 2050, 0)
    sequential_env = clean_benchmark_environment()
    sequential_env.update({
        "MLMQ_BENCH": "1", "MLMQ_WORK_BLOCKS": str(args.blocks),
        "MLMQ_CUT_PERCENT": "50", "MLMQ_FINAL_AUDIT": "failure",
        "BENCH_DELTA": str(args.delta), "BENCH_SOURCE": "0",
        "BENCH_WARMUPS": "0", "BENCH_REPEATS": str(len(sources)),
        "BENCH_QUEUE": args.queue,
        "L3_SUPPLEMENT_SOURCES": ",".join(map(str, sources)),
        "L3_SUPPLEMENT_ORACLE_DIR": str(fixture_dir),
    })
    sequential_result, sequential_raw = command_record(
        output, "sequential_reset",
        [sequential_binary, "-i", fixture_dir / "s0.gr", "-n", "2",
         "-d", str(args.delta)], environment=sequential_env,
        timeout=args.timeout)
    sequential_contract = validate_sequential(
        sequential_raw, vertices=fixture_info["vertices"], sources=sources,
        blocks=args.blocks, delta=args.delta, queue=args.queue,
        window_mode=args.window_mode, window_min=args.window_min,
        window_max=args.window_max, idle_backoff=args.idle_backoff)
    if sequential_result["rc"] != 0:
        sequential_contract["valid"] = False
        sequential_contract["errors"].append(
            f"sequential executable rc={sequential_result['rc']}")
    if not sequential_contract["valid"]:
        problems.extend("sequential_reset: " + error
                        for error in sequential_contract["errors"])
    sequential = {
        "build": sequential_build,
        "patch": sequential_patch,
        "fixture": fixture_info,
        "exports": exports,
        "run": sequential_result,
        "contract": sequential_contract,
    }
    write_json(output / "sequential_result.json", sequential)

    graph_records = []
    for name in GRAPH_NAMES:
        row = graphs[name]
        env = clean_benchmark_environment()
        env.update({
            "MLMQ_BENCH": "1", "MLMQ_WORK_BLOCKS": str(args.blocks),
            "MLMQ_CUT_PERCENT": str(row["cut_percent"]),
            "MLMQ_FINAL_AUDIT": "failure", "BENCH_DELTA": str(args.delta),
            "BENCH_SOURCE": str(row["source"]), "BENCH_WARMUPS": "0",
            "BENCH_REPEATS": "1", "BENCH_QUEUE": args.queue,
            "L3_SUPPLEMENT_ORACLE": str(row["oracle"]),
        })
        prefix = output / f"graph_{name}"
        record = run_one_with_canonical_tools(
            production_binary, row["graph"], "MLMQ", 2, prefix, env,
            row["source"], args.queue, 0, 1, args.timeout, args.delta)
        raw = prefix.with_suffix(".log").read_text(errors="replace")
        contract = strict_dual_contract(
            raw, samples=1, vertices=row["header"]["vertices"],
            blocks=args.blocks, delta=args.delta, queue=args.queue,
            window_mode=args.window_mode, window_min=args.window_min,
            window_max=args.window_max, idle_backoff=args.idle_backoff)
        cut = row["header"]["vertices"] * row["cut_percent"] // 100
        partitions = [tuple(map(int, match.groups())) for match in re.finditer(
            r"GPU(\d+) partition: \[(\d+), (\d+)\)", raw)]
        expected_partitions = [(0, 0, cut),
                               (1, cut, row["header"]["vertices"])]
        for expected in expected_partitions:
            if expected not in partitions:
                contract["errors"].append(f"missing effective partition {expected}")
        if record["rc"] != 0 or not record["valid"]:
            contract["errors"].append(
                f"base runner invalid rc={record['rc']} reason={record['reason']}")
        contract["valid"] = not contract["errors"]
        record.update({
            "graph": name, "source": row["source"],
            "cut_percent": row["cut_percent"],
            "environment": selected_environment(env),
            "input_hashes": row["hashes"],
            "strict_contract": contract,
            "partitions": partitions,
            "expected_partitions": expected_partitions,
            "log_sha256": sha256(prefix.with_suffix(".log")),
            "gpu_before_sha256": sha256(prefix.with_suffix(".gpu_before.log")),
            "gpu_after_sha256": sha256(prefix.with_suffix(".gpu_after.log")),
        })
        record["valid"] = contract["valid"]
        record["reason"] = "ok" if record["valid"] else "; ".join(contract["errors"])
        write_json(prefix.with_suffix(".json"), record)
        graph_records.append(record)
        write_json(output / "graph_records.json", graph_records)
        print(
            f"FINAL_CORRECTNESS graph={name} rc={record['rc']} "
            f"valid={int(record['valid'])}", flush=True)
        if not record["valid"]:
            problems.extend(f"{name}: {error}" for error in contract["errors"])

    pair_integrity_after = None
    graph_integrity_after = None
    try:
        pair_integrity_after = formal_pair_integrity_snapshot(pair, ROOT)
        if pair_integrity_after != pair_integrity_before:
            problems.append(
                "pair artifacts or repository HEAD/status changed during final checks")
        validate_formal_pair_build(
            pair, dual_binary, single_binary,
            git_info["head"]["output"], ROOT)
    except (OSError, RuntimeError, SystemExit) as error:
        problems.append(f"post-check pair validation failed: {error}")
    try:
        graph_integrity_after = graph_input_integrity(graph_manifest, graphs)
        if graph_integrity_after != graph_integrity_before:
            problems.append("graph manifest, graph, or oracle changed during final checks")
    except OSError as error:
        problems.append(f"post-check graph/oracle validation failed: {error}")
    valid = not problems
    results = {
        "valid": valid,
        "purpose": "correctness only; timings are not accepted as performance data",
        "small_fixtures": small_results,
        "sequential_reset_valid": sequential_contract["valid"],
        "eight_graphs_valid": all(record["valid"] for record in graph_records),
        "eight_graph_count": len(graph_records),
        "problems": problems,
        "finished_utc": now_utc(),
    }
    write_json(output / "results.json", results)
    manifest.update(
        status="complete" if valid else "failed",
        fixture_builds=builds,
        archived_dual_rebuild=production_build,
        pair_integrity_after=pair_integrity_after,
        graph_integrity_after=graph_integrity_after,
        export_build=export_build,
        sequential_build=sequential_build,
        results=str(output / "results.json"),
        finished_utc=results["finished_utc"],
        problems=problems)
    write_json(output / "manifest.json", manifest)
    print(
        f"L3_30H_FINAL_CHECKS valid={int(valid)} small={len(small_results)} "
        f"sequential={int(sequential_contract['valid'])} "
        f"graphs={sum(record['valid'] for record in graph_records)}/8",
        flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
