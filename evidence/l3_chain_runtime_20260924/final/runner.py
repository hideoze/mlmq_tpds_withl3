#!/usr/bin/env python3
"""Run the bounded original-G L3 runtime-chain validation inside one Slurm allocation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import struct
import subprocess
import sys

from run_usa_road_matrix import run_one
from run_l3_30h import (
    QUEUE_TYPE_IDS,
    parse_dual_contract,
    parse_l2_capacity_contract,
    parse_oracle_contract,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_CONTRACT = ROOT / "evidence/l3_chain_runtime_20260924/formal_contract.json"
EXPECTED_ADDS_SHA256 = "511e88f56762914829a055aea0215a6c8e3c0d1dfc1c078f5f2247a559638ec2"
QUEUE = "L1SLF_L2DQ"
BLOCKS = 107
FROZEN_INPUTS = {
    "USA": {
        "vertices": 23947347, "edges": 57708624, "source": 11973673,
        "delta": 400000, "cut_percent": 60,
        "graph_sha256": "7a4e9747b3e1febc4ad452e1b0156e636882482b34d2462f8308e70de533b8d7",
        "oracle_sha256": "19b674d79f5bc48d26107facb6327c8f8661d40a5bb80db67dd8feb260757a44",
    },
    "COL": {
        "vertices": 435666, "edges": 1057066, "source": 217833,
        "delta": 200000, "cut_percent": 55,
        "graph_sha256": "7158d69e3e6c8fb9b9b699de1a423c0a21eab146793ea865cafb25a4d59b3f02",
        "oracle_sha256": "095a726b9de16780d6ab9ccc7e55bf2876035ab2398e51df41566e4b13f9e018",
    },
}
SPEC_FIELDS = (
    "NAME", "GRAPH", "ORACLE", "SOURCE", "DELTA", "CUT_PERCENT",
    "GRAPH_SHA256", "ORACLE_SHA256",
)
CHAIN_COUNTER_RE = re.compile(
    r"L3_CHAIN_PARTITION gpu=(?P<gpu>\d+) closures=(?P<closures>\d+) "
    r"materialized=(?P<materialized>\d+) tails=(?P<tails>\d+) "
    r"rx_sources=(?P<rx_sources>\d+) route_reads=(?P<route_reads>\d+) "
    r"unique_segments=(?P<unique_segments>\d+) "
    r"repeated_closures=(?P<repeated_closures>\d+) "
    r"max_segment_visits=(?P<max_segment_visits>\d+)")
CHAIN_SETUP_RE = re.compile(
    r"L3_CHAIN_PARTITION_SETUP gpu=(?P<gpu>\d+) segments=(?P<segments>\d+) "
    r"interiors=(?P<interiors>\d+) overflow=(?P<overflow>\d+) "
    r"build_ms=(?P<build_ms>[0-9.]+)")
CHAIN_UPLOAD_RE = re.compile(
    r"L3_CHAIN_PARTITION_UPLOAD gpu=(?P<gpu>\d+) bytes=(?P<bytes>\d+) "
    r"upload_ms=(?P<upload_ms>[0-9.]+)")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gr_header(path):
    with path.open("rb") as stream:
        raw = stream.read(32)
    if len(raw) != 32:
        raise ValueError(f"short GR header: {path}")
    version, edge_bytes, vertices, edges = struct.unpack("<QQQQ", raw)
    if version != 1 or edge_bytes != 4 or not vertices or vertices >= (1 << 31):
        raise ValueError(f"unsupported GR header: {path}: {(version, edge_bytes, vertices, edges)}")
    return {
        "version": version,
        "edge_payload_bytes": edge_bytes,
        "vertices": vertices,
        "edges": edges,
    }


def load_specs(raw_specs):
    if not raw_specs or len(raw_specs) != 2:
        raise ValueError("exactly two --spec entries are required: USA first, then the preselected road graph")
    specs = []
    for raw in raw_specs:
        name, graph_s, oracle_s, source_s, delta_s, cut_s, graph_sha, oracle_sha = raw
        graph = Path(graph_s).resolve(strict=True)
        oracle = Path(oracle_s).resolve(strict=True)
        source, delta, cut = map(int, (source_s, delta_s, cut_s))
        header = read_gr_header(graph)
        if source < 0 or source >= header["vertices"] or delta <= 0 or not 1 <= cut <= 99:
            raise ValueError(f"invalid source/delta/cut for {name}")
        actual_graph = sha256(graph)
        actual_oracle = sha256(oracle)
        if actual_graph != graph_sha or actual_oracle != oracle_sha:
            raise ValueError(
                f"input hash mismatch for {name}: graph={actual_graph}, oracle={actual_oracle}")
        expected_oracle_bytes = header["vertices"] * 4
        if oracle.stat().st_size != expected_oracle_bytes:
            raise ValueError(
                f"oracle size mismatch for {name}: {oracle.stat().st_size} != {expected_oracle_bytes}")
        specs.append({
            "name": name,
            "graph": graph,
            "oracle": oracle,
            "source": source,
            "delta": delta,
            "cut_percent": cut,
            "header": header,
            "graph_sha256": actual_graph,
            "oracle_sha256": actual_oracle,
            "input_kind": "original CSR G in frozen reordered layout; no augmented shortcut edges",
        })
    if [spec["name"] for spec in specs] != ["USA", "COL"]:
        raise ValueError("the frozen formal order is exactly USA then COL")
    for spec in specs:
        frozen = FROZEN_INPUTS[spec["name"]]
        observed = {
            "vertices": spec["header"]["vertices"],
            "edges": spec["header"]["edges"],
            "source": spec["source"], "delta": spec["delta"],
            "cut_percent": spec["cut_percent"],
            "graph_sha256": spec["graph_sha256"],
            "oracle_sha256": spec["oracle_sha256"],
        }
        if observed != frozen:
            raise ValueError(
                f"{spec['name']} differs from the committed frozen contract: "
                f"observed={observed} expected={frozen}")
    return specs


def validate_committed_contract():
    contract = json.loads(FORMAL_CONTRACT.read_text())
    observed = {}
    for name in ("USA", "COL"):
        row = contract["inputs"][name]
        observed[name] = {
            "vertices": row["vertices"], "edges": row["edges"],
            "source": row["source"], "delta": row["delta"],
            "cut_percent": row["cut_percent"],
            "graph_sha256": row["graph_sha256"],
            "oracle_sha256": row["oracle_sha256"],
        }
    if observed != FROZEN_INPUTS:
        raise ValueError("runner constants differ from the committed formal contract")
    if (contract["sampling"]["rounds"] != 2 or
            contract["sampling"]["warmups_per_process"] != 1 or
            contract["sampling"]["formal_samples_per_process"] != 5 or
            contract["primary_metric"] !=
            "query_wall_ms including complete distance D2H materialization"):
        raise ValueError("unsupported committed sampling/timing contract")
    return contract


def validate_build(build_dir):
    build_dir = build_dir.resolve(strict=True)
    manifest = json.loads((build_dir / "manifest.json").read_text())
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True)
    if status:
        raise ValueError("formal chain run requires a clean worktree")
    if manifest.get("status") != "complete" or manifest.get("head") != head:
        raise ValueError("chain build does not match current clean HEAD")
    if manifest.get("git_status"):
        raise ValueError(f"chain build was not made from a clean tree: {manifest['git_status']}")
    roles = {}
    for role, relative in manifest["roles"].items():
        path = (build_dir / relative).resolve(strict=True)
        expected = manifest["artifacts"][relative]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"build artifact changed for {role}: {actual} != {expected}")
        roles[role] = path
    for relative in (
            "tools/test_l3_chain_partition",
            "tools/analyze_l3_chain_partitions",
            "tools/test_l3_chain_partition.log",
            "bc_command_contract.json"):
        path = (build_dir / relative).resolve(strict=True)
        expected = manifest["artifacts"][relative]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"build evidence changed: {relative}")
    bc_contract = json.loads((build_dir / "bc_command_contract.json").read_text())
    if not bc_contract.get("valid") or not bc_contract.get("only_treatment_macro_differs"):
        raise ValueError("B/C compiler attribution contract failed")
    analyzer = (build_dir / "tools/analyze_l3_chain_partitions").resolve(strict=True)
    return build_dir, manifest, roles, analyzer


def command_capture(output, name, command, timeout=60):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout)
    (output / f"{name}.log").write_text(
        result.stdout + f"\nRUN_RC={result.returncode}\n")
    return result


def slurm_preflight(output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("chain trial must run inside Slurm")
    query = command_capture(
        output, "gpu_query",
        ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap",
         "--format=csv,noheader,nounits"])
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if (query.returncode or len(rows) != 2 or
            any("A100" not in row or not row.endswith(", 8.0") for row in rows)):
        raise ValueError(f"expected exactly two allocated A100 GPUs, got {rows}")
    if os.environ.get("SLURM_JOB_PARTITION") != "a100":
        raise ValueError(
            f"expected Slurm a100 partition, got {os.environ.get('SLURM_JOB_PARTITION')!r}")
    uuids = [row.split(",")[1].strip() for row in rows]
    if len(set(uuids)) != 2 or any(not value.startswith("GPU-") for value in uuids):
        raise ValueError(f"expected two distinct GPU UUIDs, got {uuids}")
    apps = command_capture(
        output, "gpu_apps_initial",
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if apps.returncode or apps.stdout.strip():
        raise ValueError(f"GPU compute process present before trial: {apps.stdout!r}")
    for name, command in (
        ("nvidia_smi_initial", ["nvidia-smi"]),
        ("gpu_topology", ["nvidia-smi", "topo", "-m"]),
        ("slurm_job_running", ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]]),
    ):
        result = command_capture(output, name, command)
        if result.returncode:
            raise ValueError(f"preflight command failed: {name}")
    return {"job_id": os.environ["SLURM_JOB_ID"], "gpus": rows}


def clean_environment():
    environment = {
        key: value for key, value in os.environ.items()
        if key.startswith("SLURM_") or key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "TZ", "TMPDIR")
    }
    environment.update({
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "MLMQ_BENCH": "1",
        "MLMQ_WORK_BLOCKS": str(BLOCKS),
        "BENCH_QUEUE": QUEUE,
        "MLMQ_FINAL_AUDIT": "failure",
    })
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    return environment


def write_gr(path, rows):
    destinations = []
    weights = []
    ends = []
    for edges in rows:
        for target, weight in edges:
            destinations.append(target)
            weights.append(weight)
        ends.append(len(destinations))
    payload = bytearray(struct.pack("<QQQQ", 1, 4, len(rows), len(destinations)))
    payload.extend(struct.pack(f"<{len(ends)}Q", *ends))
    if destinations:
        payload.extend(struct.pack(f"<{len(destinations)}I", *destinations))
    if len(destinations) & 1:
        payload.extend(b"\0\0\0\0")
    if weights:
        payload.extend(struct.pack(f"<{len(weights)}i", *weights))
    path.write_bytes(payload)


def chain_fixtures():
    fixtures = {}
    for mode in ("reciprocal", "zero", "branches_concurrent"):
        vertices = 2051
        half = vertices // 2
        rows = [[] for _ in range(vertices)]
        starts = [0, half, 200, half + 200]
        for index, start in enumerate(starts):
            for offset in range(101):
                vertex = start + offset
                if offset:
                    rows[vertex].append((vertex - 1, 0 if mode == "zero" else 7))
                if offset < 100:
                    rows[vertex].append((vertex + 1, 0 if mode == "zero" else 1))
            rows[start + 100].append(
                (starts[(index + 1) % 4], 0 if mode == "zero" else 3))
            if mode == "branches_concurrent":
                rows[start + 45].extend([(start + 45, 0), (start + 65, 40)])
                rows[start + 75].append((start + 76, 2))
                rows[start + 25].append((starts[(index + 1) % 4] + 55, 900))
        fixtures[mode] = rows

    # The adversarial non-strict topology that the old aggregate reciprocal
    # count could accept and then abort.  Correct behavior is ordinary CSR.
    rows = [[] for _ in range(2051)]
    rows[0] = [(1, 2), (1, 3)]
    rows[1] = [(0, 5), (2, 7)]
    rows[2] = [(3, 11)]
    rows[3] = [(2, 13)]
    fixtures["parallel_fallback"] = rows

    # Cumulative weights hit INT_MAX, so the index must reject this segment
    # and preserve the ordinary CSR path.
    rows = [[] for _ in range(2051)]
    for vertex in range(100):
        rows[vertex].append((vertex + 1, 1 << 30))
        rows[vertex + 1].append((vertex, 1 << 30))
    fixtures["overflow_fallback"] = rows
    return fixtures


def parse_chain_evidence(raw):
    counters = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in CHAIN_COUNTER_RE.finditer(raw)
    ]
    setup = []
    for match in CHAIN_SETUP_RE.finditer(raw):
        row = match.groupdict()
        setup.append({
            "gpu": int(row["gpu"]), "segments": int(row["segments"]),
            "interiors": int(row["interiors"]), "overflow": int(row["overflow"]),
            "build_ms": float(row["build_ms"]),
        })
    upload = []
    for match in CHAIN_UPLOAD_RE.finditer(raw):
        row = match.groupdict()
        upload.append({
            "gpu": int(row["gpu"]), "bytes": int(row["bytes"]),
            "upload_ms": float(row["upload_ms"]),
        })
    return {"counters": counters, "setup": setup, "upload": upload}


def run_configuration(*, output, graph_name, graph, oracle, source, delta,
                      cut_percent, role, binary, algorithm, gpu_count,
                      round_id, warmups, repeats, timeout, environment,
                      performance_allowed, records):
    environment = dict(environment)
    environment.update({
        "BENCH_SOURCE": str(source),
        "BENCH_WARMUPS": str(warmups),
        "BENCH_REPEATS": str(repeats),
        "MLMQ_CUT_PERCENT": str(cut_percent),
    })
    if oracle is None:
        environment.pop("L3_SUPPLEMENT_ORACLE", None)
    else:
        environment["L3_SUPPLEMENT_ORACLE"] = str(oracle)
    prefix = output / f"r{round_id}_{graph_name}_{role}_s{source}"
    result = run_one(
        binary, graph, algorithm, gpu_count, prefix, environment, source,
        QUEUE, warmups, repeats, timeout, delta if algorithm == "MLMQ" else None)
    raw = prefix.with_suffix(".log").read_text()
    chain = parse_chain_evidence(raw)
    expected_samples = warmups + repeats
    contracts = {}
    contract_errors = []
    if algorithm == "MLMQ" and oracle is not None:
        contracts["oracle"] = parse_oracle_contract(
            raw, expected_samples, read_gr_header(graph)["vertices"])
        contract_errors.extend(contracts["oracle"]["errors"])
    for marker in ("ORACLE_MISMATCH", "ORACLE_SIZE_ERROR", "FINAL_AUDIT_ERROR"):
        if marker in raw:
            contract_errors.append(f"solver reported {marker}")
    if algorithm == "MLMQ" and gpu_count == 2:
        contracts["dual"] = parse_dual_contract(
            raw, expected_samples, BLOCKS, delta, QUEUE_TYPE_IDS[QUEUE],
            expected_window_mode=2, expected_window_min=25000,
            expected_window_max=25000, expected_idle_backoff=0,
            require_l2_final=True)
        contract_errors.extend(contracts["dual"]["errors"])
    elif algorithm == "MLMQ":
        contracts["single_l2_capacity"] = parse_l2_capacity_contract(
            raw, 1, require_frozen=True)
        contract_errors.extend(contracts["single_l2_capacity"]["errors"])
    record = {
        "graph": graph_name,
        "role": role,
        "round": round_id,
        "source": source,
        "warmups": warmups,
        "repeats": repeats,
        "performance_allowed": performance_allowed,
        "log": str(prefix.with_suffix(".log")),
        "chain": chain,
        "contracts": contracts,
        **result,
    }
    if contract_errors:
        record["valid"] = False
        record["reason"] = (
            record.get("reason", "unknown") + "; contract: " +
            "; ".join(contract_errors))
    records.append(record)
    write_json(output / "records.json", records)
    if not record["valid"] or record["rc"] != 0:
        raise RuntimeError(f"retained failed run {prefix}: {record['reason']}")
    if role.startswith("C_"):
        if len(chain["setup"]) != 2 or len(chain["upload"]) != 2:
            raise RuntimeError(f"missing two-GPU chain setup/upload evidence: {prefix}")
    elif chain["setup"] or chain["upload"] or chain["counters"]:
        raise RuntimeError(f"non-chain role emitted chain evidence: {prefix}")
    if role == "C_diagnostic":
        expected_counters = 2 * (warmups + repeats)
        if len(chain["counters"]) != expected_counters:
            raise RuntimeError(
                f"diagnostic counter count {len(chain['counters'])} != {expected_counters}")
        for row in chain["counters"]:
            if row["closures"] != row["unique_segments"] + row["repeated_closures"]:
                raise RuntimeError("diagnostic closure accounting mismatch")
    elif chain["counters"]:
        raise RuntimeError(f"timed role unexpectedly emitted hot-path diagnostics: {prefix}")
    print(json.dumps({
        "graph": graph_name, "role": role, "round": round_id,
        "valid": True,
        "solve_ms": [row["solve_ms"] for row in result["samples"]],
    }), flush=True)
    return record


def distribution(values):
    values = [float(value) for value in values]
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    center = statistics.median(ordered)
    return {
        "samples": values,
        "count": len(values),
        "median": center,
        "q1": quartiles[0],
        "q3": quartiles[2],
        "iqr": quartiles[2] - quartiles[0],
        "mad": statistics.median(abs(value - center) for value in ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def summarize(records, specs, sampling):
    result = {"sampling": sampling, "graphs": {}}
    roles = ("A_mlmq", "A_adds", "D_best", "B_control", "C_chain")
    for spec in specs:
        graph = spec["name"]
        graph_result = {"roles": {}}
        for role in roles:
            selected = [
                record for record in records
                if record["graph"] == graph and record["role"] == role
                and record["performance_allowed"]
            ]
            if len(selected) != 2:
                raise RuntimeError(f"expected two retained performance processes for {graph}/{role}")
            formal = [
                row for record in selected for row in record["samples"]
                if row.get("warmup") == "0"
            ]
            graph_result["roles"][role] = {
                "solve_ms": distribution(row["solve_ms"] for row in formal),
                "query_wall_ms": distribution(row["query_wall_ms"] for row in formal),
                "round_solve_medians": [
                    statistics.median(
                        float(row["solve_ms"]) for row in record["samples"]
                        if row.get("warmup") == "0")
                    for record in selected
                ],
                "round_query_wall_medians": [
                    statistics.median(
                        float(row["query_wall_ms"]) for row in record["samples"]
                        if row.get("warmup") == "0")
                    for record in selected
                ],
            }
        roles_summary = graph_result["roles"]
        ratios = {}
        for metric in ("solve_ms", "query_wall_ms"):
            value = lambda role: roles_summary[role][metric]["median"]
            ratios[metric] = {
                "chain_net_B_over_C": value("B_control") / value("C_chain"),
                "same_graph_single_A_over_C": value("A_mlmq") / value("C_chain"),
                "adds_A_over_C": value("A_adds") / value("C_chain"),
                "current_best_D_over_C": value("D_best") / value("C_chain"),
                "target_1_20_met": value("A_mlmq") / value("C_chain") >= 1.20,
            }
        diagnostics = [
            record for record in records
            if record["graph"] == graph and record["role"] == "C_diagnostic"
        ]
        if len(diagnostics) != 1:
            raise RuntimeError(f"expected one road diagnostic process for {graph}")
        counters = diagnostics[0]["chain"]["counters"]
        totals = {
            key: sum(row[key] for row in counters)
            for key in (
                "closures", "materialized", "tails", "rx_sources", "route_reads",
                "unique_segments", "repeated_closures")
        }
        totals["max_segment_visits"] = max(
            (row["max_segment_visits"] for row in counters), default=0)
        totals["materialized_per_closure"] = (
            totals["materialized"] / totals["closures"] if totals["closures"] else None)
        totals["closures_per_rx_source"] = (
            totals["closures"] / totals["rx_sources"] if totals["rx_sources"] else None)
        totals["repeated_closure_share"] = (
            totals["repeated_closures"] / totals["closures"] if totals["closures"] else None)
        graph_result["ratios"] = ratios
        graph_result["diagnostics"] = totals
        graph_result["chain_setup"] = diagnostics[0]["chain"]["setup"]
        graph_result["chain_upload"] = diagnostics[0]["chain"]["upload"]
        result["graphs"][graph] = graph_result
    result["all_correct"] = all(record["valid"] for record in records)
    result["target_met_on_all_graphs"] = all(
        row["ratios"]["query_wall_ms"]["target_1_20_met"]
        for row in result["graphs"].values())
    result["chain_faster_than_matched_control_on_all_graphs"] = all(
        row["ratios"]["query_wall_ms"]["chain_net_B_over_C"] > 1.0
        for row in result["graphs"].values())
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--adds", required=True, type=Path)
    parser.add_argument("--spec", action="append", nargs=8, metavar=SPEC_FIELDS)
    parser.add_argument("--sampling", choices=("screen", "formal"), default="formal")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"output already exists: {args.out}")
    try:
        args.specs = load_specs(args.spec)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    committed_contract = validate_committed_contract()
    build_dir, build_manifest, binaries, analyzer = validate_build(args.build_dir)
    adds = args.adds.resolve(strict=True)
    if sha256(adds) != EXPECTED_ADDS_SHA256:
        raise SystemExit("ADDS binary hash does not match the frozen official artifact")
    output = args.out.resolve(strict=False)
    output.mkdir(parents=True)
    allocation = slurm_preflight(output)
    environment = clean_environment()
    warmups, repeats, rounds = (1, 5, 2) if args.sampling == "formal" else (1, 3, 2)
    manifest = {
        "schema": 1,
        "status": "running",
        "created_utc": now_utc(),
        "host": socket.gethostname(),
        "invocation": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "allocation": allocation,
        "git_head": build_manifest["head"],
        "git_branch": build_manifest["branch"],
        "formal_contract": {
            "path": str(FORMAL_CONTRACT),
            "sha256": sha256(FORMAL_CONTRACT),
            "decision_rule": committed_contract["decision_rule"],
        },
        "build_dir": str(build_dir),
        "build_manifest_sha256": sha256(build_dir / "manifest.json"),
        "binaries": {
            role: {"path": str(path), "sha256": sha256(path)}
            for role, path in binaries.items()
        } | {"A_adds": {"path": str(adds), "sha256": sha256(adds)}},
        "configuration": {
            "sampling": args.sampling,
            "warmups_per_process": warmups,
            "formal_samples_per_process": repeats,
            "rounds": rounds,
            "round_1_reversed": True,
            "blocks": BLOCKS,
            "queue": QUEUE,
            "worker_threads": 512,
            "B_and_C_term_only_worker": True,
            "D_current_best_term_only_worker": False,
            "C_degree_gate": True,
            "timed_diagnostics": False,
            "primary_timing": "query_wall_ms including full distance D2H; solve_ms is secondary",
        },
        "inputs": [{
            **{key: value for key, value in spec.items()
               if key not in ("graph", "oracle")},
            "graph": str(spec["graph"]),
            "oracle": str(spec["oracle"]),
        } for spec in args.specs],
        "selection": {
            "first": "USA required by L3_chained.md",
            "second": (
                "COL fixed before current speed runs after the exact runtime builder scanned "
                "all eight road graphs: it had the highest indexed-interior coverage "
                "(19.1867%) with representative multi-interior segment lengths"
            ),
        },
        "historical_boundary": (
            "Stage230/231 already found this route correct but slower; this run is a bounded "
            "current-BNUM8/current-data-plane revalidation, not an unbounded search"
        ),
    }
    write_json(output / "manifest.json", manifest)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())

    topology = {}
    for spec in args.specs:
        result = subprocess.run(
            [str(analyzer), str(spec["graph"]), str(spec["cut_percent"])],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
        (output / f"topology_{spec['name']}.log").write_text(
            result.stdout + f"\nRUN_RC={result.returncode}\n")
        if result.returncode:
            raise RuntimeError(f"topology analyzer failed for {spec['name']}")
        topology[spec["name"]] = json.loads(result.stdout)
    write_json(output / "topology.json", topology)

    records = []
    fixtures_dir = output / "fixtures"
    fixtures_dir.mkdir()
    fixture_manifest = {}
    for name, rows in chain_fixtures().items():
        graph = fixtures_dir / f"{name}.gr"
        write_gr(graph, rows)
        fixture_manifest[name] = {
            "path": str(graph), "vertices": len(rows), "sha256": sha256(graph)}
        if name == "parallel_fallback":
            sources = (0,)
        elif name == "overflow_fallback":
            # Keep the overflow segment unreachable: this fixture validates
            # safe index fallback, not the base solver's signed-distance limit.
            sources = (len(rows) // 2,)
        else:
            sources = (0, len(rows) // 2)
        for source in sources:
            record = run_configuration(
                output=output, graph_name=f"fixture_{name}", graph=graph,
                oracle=None, source=source, delta=200000, cut_percent=50,
                role="C_diagnostic", binary=binaries["C_diagnostic"],
                algorithm="MLMQ", gpu_count=2, round_id=-1,
                warmups=1, repeats=1, timeout=args.timeout,
                environment=environment, performance_allowed=False, records=records)
            counters = record["chain"]["counters"]
            fallback = name in ("parallel_fallback", "overflow_fallback")
            if not fallback and not any(row["closures"] for row in counters):
                raise RuntimeError(f"fixture {name} did not trigger a chain closure")
            if fallback and any(row["closures"] for row in counters):
                raise RuntimeError(f"fallback fixture {name} unexpectedly triggered a closure")
            if (name == "overflow_fallback" and
                    not any(row["overflow"] for row in record["chain"]["setup"])):
                raise RuntimeError("overflow fixture did not exercise index overflow fallback")
    write_json(output / "fixtures.json", fixture_manifest)

    # Diagnostics are separate from timing and run only once per road graph.
    for spec in args.specs:
        run_configuration(
            output=output, graph_name=spec["name"], graph=spec["graph"],
            oracle=spec["oracle"], source=spec["source"], delta=spec["delta"],
            cut_percent=spec["cut_percent"], role="C_diagnostic",
            binary=binaries["C_diagnostic"], algorithm="MLMQ", gpu_count=2,
            round_id=-1, warmups=0, repeats=1, timeout=args.timeout,
            environment=environment, performance_allowed=False, records=records)

    configurations = (
        ("A_mlmq", binaries["A_mlmq"], "MLMQ", 1),
        ("A_adds", adds, "ADDS", 1),
        ("D_best", binaries["D_best"], "MLMQ", 2),
        ("B_control", binaries["B_control"], "MLMQ", 2),
        ("C_chain", binaries["C_chain"], "MLMQ", 2),
    )
    for round_id in range(rounds):
        graph_order = args.specs if round_id == 0 else list(reversed(args.specs))
        config_order = configurations if round_id == 0 else tuple(reversed(configurations))
        for spec in graph_order:
            for role, binary, algorithm, gpu_count in config_order:
                run_configuration(
                    output=output, graph_name=spec["name"], graph=spec["graph"],
                    oracle=spec["oracle"], source=spec["source"], delta=spec["delta"],
                    cut_percent=spec["cut_percent"], role=role, binary=binary,
                    algorithm=algorithm, gpu_count=gpu_count, round_id=round_id,
                    warmups=warmups, repeats=repeats, timeout=args.timeout,
                    environment=environment, performance_allowed=True, records=records)

    summary = summarize(records, args.specs, args.sampling)
    write_json(output / "summary.json", summary)
    apps = command_capture(
        output, "gpu_apps_final",
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if apps.returncode or apps.stdout.strip():
        raise RuntimeError(f"GPU process remained after trial: {apps.stdout!r}")
    command_capture(output, "nvidia_smi_final", ["nvidia-smi"])
    integrity_errors = []
    final_head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    final_status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True).splitlines()
    if final_head != build_manifest["head"]:
        integrity_errors.append(
            f"repository HEAD changed: {final_head} != {build_manifest['head']}")
    if final_status:
        integrity_errors.append(f"repository became dirty: {final_status}")
    final_build_manifest_sha = sha256(build_dir / "manifest.json")
    if final_build_manifest_sha != manifest["build_manifest_sha256"]:
        integrity_errors.append("build manifest changed during trial")
    final_contract_sha = sha256(FORMAL_CONTRACT)
    if final_contract_sha != manifest["formal_contract"]["sha256"]:
        integrity_errors.append("committed formal contract changed during trial")
    artifact_hashes = {
        role: sha256(Path(details["path"]))
        for role, details in manifest["binaries"].items()
    }
    for role, actual in artifact_hashes.items():
        if actual != manifest["binaries"][role]["sha256"]:
            integrity_errors.append(f"binary changed during trial: {role}")
    input_hashes = {}
    for spec in args.specs:
        graph_hash = sha256(spec["graph"])
        oracle_hash = sha256(spec["oracle"])
        input_hashes[spec["name"]] = {
            "graph_sha256": graph_hash, "oracle_sha256": oracle_hash}
        if graph_hash != spec["graph_sha256"] or oracle_hash != spec["oracle_sha256"]:
            integrity_errors.append(f"input changed during trial: {spec['name']}")
    integrity = {
        "valid": not integrity_errors,
        "git_head": final_head,
        "git_status": final_status,
        "build_manifest_sha256": final_build_manifest_sha,
        "formal_contract_sha256": final_contract_sha,
        "artifact_hashes": artifact_hashes,
        "input_hashes": input_hashes,
        "errors": integrity_errors,
    }
    write_json(output / "final_integrity.json", integrity)
    if integrity_errors:
        raise RuntimeError("final integrity gate failed: " + "; ".join(integrity_errors))
    manifest.update(
        status="complete", completed_utc=now_utc(), record_count=len(records),
        query_count=sum(len(record["samples"]) for record in records),
        all_correct=summary["all_correct"])
    write_json(output / "manifest.json", manifest)
    write_json(output / "complete.json", {
        "status": "complete", "valid": summary["all_correct"],
        "target_met_on_all_graphs": summary["target_met_on_all_graphs"],
        "chain_faster_than_matched_control_on_all_graphs":
            summary["chain_faster_than_matched_control_on_all_graphs"],
    })
    print(
        "L3_CHAIN_TRIAL_COMPLETE "
        f"records={len(records)} queries={manifest['query_count']} "
        f"target={int(summary['target_met_on_all_graphs'])}",
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
