#!/usr/bin/env python3
"""Run a matched USA-road L3/ADDS timing matrix inside one Slurm allocation.

The script deliberately keeps failed and timed-out runs in the output tree.  A
configuration is summarized only when every requested sample is present and
passes the solver's correctness checks.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from statistics import median
import time
import re


GRAPH_NAMES = ("BAY", "CAL", "COL", "E", "FLA", "NY", "USA", "W")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bench(stdout):
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("BENCH "):
            continue
        row = {}
        for token in line.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def gpu_snapshot():
    snapshot = subprocess.check_output(
        ["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=30)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, stderr=subprocess.STDOUT, timeout=30).strip()
    return snapshot, apps


def measured(rows, warmups):
    return [row for row in rows if row.get("warmup") == "0" and
            int(row.get("repeat", "-1")) >= warmups]


def validate(rows, expected_algorithm, expected_gpus, source, queue,
             warmups, repeats):
    if len(rows) != warmups + repeats:
        return False, f"expected {warmups + repeats} BENCH rows, got {len(rows)}"
    for index, row in enumerate(rows):
        expected = {
            "algorithm": expected_algorithm,
            "gpu_count": str(expected_gpus),
            "source": str(source),
            "repeat": str(index),
            "warmup": str(int(index < warmups)),
            "correct": "1",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                return False, f"row {index} {key}={row.get(key)!r}, expected {value!r}"
        if expected_algorithm == "MLMQ" and row.get("queue") != queue:
            return False, f"row {index} queue={row.get('queue')!r}, expected {queue!r}"
        for key in ("solve_ms", "query_wall_ms"):
            try:
                if float(row[key]) <= 0:
                    return False, f"row {index} invalid {key}={row[key]!r}"
            except (KeyError, ValueError):
                return False, f"row {index} missing {key}"
    return True, "ok"


def run_one(binary, graph, algorithm, gpu_count, out_path, env, source,
            queue, warmups, repeats, timeout_seconds, delta=None):
    snapshot = ""
    apps = ""
    try:
        snapshot, apps = gpu_snapshot()
        out_path.with_suffix(".gpu_before.log").write_text(snapshot)
        if apps:
            raise RuntimeError("GPU process detected before run: " + apps)
        command = ([str(binary), str(graph)] if algorithm == "ADDS" else
                   [str(binary), "-i", str(graph), "-n", str(gpu_count)])
        if algorithm == "MLMQ" and delta is not None:
            command += ["-d", str(delta)]
        started = time.monotonic()
        try:
            result = subprocess.run(
                command, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=timeout_seconds)
            rc = result.returncode
            stdout = result.stdout
        except subprocess.TimeoutExpired as error:
            rc = 124
            stdout = error.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            stdout += f"\nTIMEOUT_SECONDS={timeout_seconds}\n"
        elapsed = time.monotonic() - started
        out_path.with_suffix(".log").write_text(
            stdout + f"\nRUN_RC={rc}\nHOST_ELAPSED_SECONDS={elapsed:.6f}\n")
        try:
            after_snapshot, after_apps = gpu_snapshot()
            out_path.with_suffix(".gpu_after.log").write_text(after_snapshot)
        except Exception as error:  # retain the run even if post-check is unavailable
            after_apps = "post-check-error:" + repr(error)
            out_path.with_suffix(".gpu_after.log").write_text(after_apps + "\n")
        rows = parse_bench(stdout)
        valid, reason = (False, "nonzero return code") if rc != 0 else validate(
            rows, algorithm, gpu_count, source, queue, warmups, repeats)
        if "Error at node" in stdout:
            valid, reason = False, "solver reported Error at node"
        if after_apps:
            valid, reason = False, "GPU process remained after run: " + after_apps
        record = {
            "command": command,
            "rc": rc,
            "valid": valid,
            "reason": reason,
            "samples": rows,
            "host_elapsed_seconds": elapsed,
        }
    except Exception as error:
        out_path.with_suffix(".log").write_text(
            f"RUN_EXCEPTION={error!r}\nRUN_RC=125\n")
        record = {"command": [], "rc": 125, "valid": False,
                  "reason": repr(error), "samples": []}
    out_path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def config_summary(record, warmups):
    if not record["valid"]:
        return {"valid": False, "reason": record["reason"]}
    rows = [row for row in record["samples"] if row.get("warmup") == "0"]
    return {
        "valid": True,
        "samples": len(rows),
        "solve_median_ms": median(float(row["solve_ms"]) for row in rows),
        "query_wall_median_ms": median(float(row["query_wall_ms"]) for row in rows),
        "solve_samples_ms": [float(row["solve_ms"]) for row in rows],
        "query_wall_samples_ms": [float(row["query_wall_ms"]) for row in rows],
    }


def read_reorder_summary(directory, graph_names):
    if directory is None:
        return {}
    summaries = {}
    pattern = re.compile(
        r"REORDER .*?read_ms=([0-9.]+) reorder_ms=([0-9.]+) "
        r"output_ms=([0-9.]+) total_ms=([0-9.]+)")
    for name in graph_names:
        path = directory / f"{name}.log"
        if not path.is_file():
            summaries[name] = {"valid": False, "reason": f"missing {path}"}
            continue
        match = pattern.search(path.read_text())
        if not match:
            summaries[name] = {"valid": False, "reason": f"missing REORDER line in {path}"}
            continue
        values = [float(value) for value in match.groups()]
        summaries[name] = {
            "valid": True,
            "read_ms": values[0],
            "reorder_ms": values[1],
            "output_ms": values[2],
            "total_ms": values[3],
            "log": str(path),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l3", required=True, type=Path)
    parser.add_argument("--adds", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--graph-template", default="USA-road-d.{name}.gr")
    parser.add_argument("--preprocess-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--delta", type=int,
                        help="explicit MLMQ delta for non-W graphs")
    parser.add_argument("--w-delta", type=int,
                        help="explicit MLMQ delta for W; falls back to --delta")
    parser.add_argument("--final-audit", choices=("failure", "all"),
                        default="failure")
    parser.add_argument("--queue", default="L1SLF_L2DQ",
                        choices=("L1SLF_L2DQ", "L1V_L2DQ"))
    parser.add_argument("--graphs", nargs="+", choices=GRAPH_NAMES,
                        default=list(GRAPH_NAMES))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU tests must run inside Slurm")
    if args.warmups < 0 or args.repeats < 1 or args.timeout < 1:
        parser.error("warmups/repeats/timeout must be positive as applicable")
    for name, value in (("delta", args.delta), ("w-delta", args.w_delta)):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    l3 = args.l3.resolve(strict=True)
    adds = args.adds.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    graphs = {name: (data_dir / args.graph_template.format(name=name)).resolve(strict=True)
              for name in args.graphs}
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)

    manifest = {
        "host": socket.gethostname(),
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "configuration": {
            "l3": str(l3), "adds": str(adds), "data_dir": str(data_dir),
            "graphs": list(args.graphs), "warmups": args.warmups,
            "repeats": args.repeats, "timeout": args.timeout,
            "source": args.source, "queue": args.queue,
            "graph_template": args.graph_template,
            "preprocess_dir": str(args.preprocess_dir.resolve()) if args.preprocess_dir else None,
            "l3_build": "L3_WINDOW_MODE=2, cooperative collect, direct RX, retained TX",
            "delta": {name: (args.w_delta if name == "W" and args.w_delta is not None
                              else args.delta)
                      for name in args.graphs},
            "final_audit": args.final_audit,
        },
        "inputs": {
            "l3": {"path": str(l3), "sha256": sha256(l3)},
            "adds": {"path": str(adds), "sha256": sha256(adds)},
            "graphs": {name: {"path": str(path), "sha256": sha256(path)}
                       for name, path in graphs.items()},
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "gpu_initial.log").write_text(gpu_snapshot()[0])
    preprocess = read_reorder_summary(
        args.preprocess_dir.resolve() if args.preprocess_dir else None, args.graphs)

    env = dict(os.environ)
    env.update({"MLMQ_BENCH": "1", "BENCH_WARMUPS": str(args.warmups),
                "BENCH_REPEATS": str(args.repeats),
                "BENCH_SOURCE": str(args.source), "BENCH_QUEUE": args.queue,
                "MLMQ_FINAL_AUDIT": args.final_audit})
    results = {}
    for graph_name in args.graphs:
        graph = graphs[graph_name]
        results[graph_name] = {}
        # Keep all three configurations in one process per configuration so
        # warmup/repeat handling is identical and easy to audit.
        configurations = (
            ("l3_n2", l3, "MLMQ", 2),
            ("l3_n1", l3, "MLMQ", 1),
            ("adds_n1", adds, "ADDS", 1),
        )
        for label, binary, algorithm, gpu_count in configurations:
            delta = args.w_delta if graph_name == "W" and args.w_delta is not None else args.delta
            record = run_one(
                binary, graph, algorithm, gpu_count,
                args.out / f"{graph_name}_{label}", env, args.source,
                args.queue, args.warmups, args.repeats, args.timeout, delta)
            results[graph_name][label] = config_summary(record, args.warmups)
            print(f"graph={graph_name} config={label} valid={record['valid']} "
                  f"rc={record['rc']} reason={record['reason']}", flush=True)

    summary = {"manifest": str(args.out / "manifest.json"), "graphs": {}}
    for graph_name, configs in results.items():
        row = dict(configs)
        row["preprocess"] = preprocess.get(graph_name, {})
        l3_2 = configs.get("l3_n2", {})
        l3_1 = configs.get("l3_n1", {})
        adds_1 = configs.get("adds_n1", {})
        for metric in ("solve_median_ms", "query_wall_median_ms"):
            if all(config.get("valid") for config in (l3_2, l3_1, adds_1)):
                row[f"l3_1_over_l3_2_{metric}"] = l3_1[metric] / l3_2[metric]
                row[f"adds_1_over_l3_2_{metric}"] = adds_1[metric] / l3_2[metric]
                row[f"adds_1_over_l3_1_{metric}"] = adds_1[metric] / l3_1[metric]
                if row["preprocess"].get("valid"):
                    prep = row["preprocess"]["total_ms"]
                    row[f"l3_1_plus_prep_over_l3_2_plus_prep_{metric}"] = ((l3_1[metric] + prep) /
                                                                              (l3_2[metric] + prep))
                    row[f"adds_1_plus_prep_over_l3_2_plus_prep_{metric}"] = ((adds_1[metric] + prep) /
                                                                               (l3_2[metric] + prep))
                    row[f"adds_1_plus_prep_over_l3_1_plus_prep_{metric}"] = ((adds_1[metric] + prep) /
                                                                               (l3_1[metric] + prep))
                else:
                    row[f"l3_1_plus_prep_over_l3_2_plus_prep_{metric}"] = None
                    row[f"adds_1_plus_prep_over_l3_2_plus_prep_{metric}"] = None
                    row[f"adds_1_plus_prep_over_l3_1_plus_prep_{metric}"] = None
            else:
                row[f"l3_1_over_l3_2_{metric}"] = None
                row[f"adds_1_over_l3_2_{metric}"] = None
                row[f"adds_1_over_l3_1_{metric}"] = None
                row[f"l3_1_plus_prep_over_l3_2_plus_prep_{metric}"] = None
                row[f"adds_1_plus_prep_over_l3_2_plus_prep_{metric}"] = None
                row[f"adds_1_plus_prep_over_l3_1_plus_prep_{metric}"] = None
        summary["graphs"][graph_name] = row
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fields = [
        "graph", "status", "L3_n2_solve_ms", "L3_n1_solve_ms",
        "ADDS_n1_solve_ms", "L3_n1/L3_n2_solve", "ADDS_n1/L3_n2_solve",
        "preprocess_total_ms", "L3_n1+prep/L3_n2+prep_solve",
        "ADDS_n1+prep/L3_n2+prep_solve",
        "L3_n2_query_ms", "L3_n1_query_ms", "ADDS_n1_query_ms",
        "L3_n1/L3_n2_query", "ADDS_n1/L3_n2_query",
        "L3_n1+prep/L3_n2+prep_query", "ADDS_n1+prep/L3_n2+prep_query",
    ]
    with (args.out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for graph_name, row in summary["graphs"].items():
            valid = all(row.get(label, {}).get("valid") for label in
                        ("l3_n2", "l3_n1", "adds_n1"))
            writer.writerow({
                "graph": graph_name, "status": "ok" if valid else "failed",
                "L3_n2_solve_ms": row.get("l3_n2", {}).get("solve_median_ms", ""),
                "L3_n1_solve_ms": row.get("l3_n1", {}).get("solve_median_ms", ""),
                "ADDS_n1_solve_ms": row.get("adds_n1", {}).get("solve_median_ms", ""),
                "L3_n1/L3_n2_solve": row.get("l3_1_over_l3_2_solve_median_ms", ""),
                "ADDS_n1/L3_n2_solve": row.get("adds_1_over_l3_2_solve_median_ms", ""),
                "preprocess_total_ms": row.get("preprocess", {}).get("total_ms", ""),
                "L3_n1+prep/L3_n2+prep_solve": row.get("l3_1_plus_prep_over_l3_2_plus_prep_solve_median_ms", ""),
                "ADDS_n1+prep/L3_n2+prep_solve": row.get("adds_1_plus_prep_over_l3_2_plus_prep_solve_median_ms", ""),
                "L3_n2_query_ms": row.get("l3_n2", {}).get("query_wall_median_ms", ""),
                "L3_n1_query_ms": row.get("l3_n1", {}).get("query_wall_median_ms", ""),
                "ADDS_n1_query_ms": row.get("adds_n1", {}).get("query_wall_median_ms", ""),
                "L3_n1/L3_n2_query": row.get("l3_1_over_l3_2_query_wall_median_ms", ""),
                "ADDS_n1/L3_n2_query": row.get("adds_1_over_l3_2_query_wall_median_ms", ""),
                "L3_n1+prep/L3_n2+prep_query": row.get("l3_1_plus_prep_over_l3_2_plus_prep_query_wall_median_ms", ""),
                "ADDS_n1+prep/L3_n2+prep_query": row.get("adds_1_plus_prep_over_l3_2_plus_prep_query_wall_median_ms", ""),
            })
    print(json.dumps(summary, indent=2))
    return 0 if all(
        all(row.get(label, {}).get("valid") for label in ("l3_n2", "l3_n1", "adds_n1"))
        for row in summary["graphs"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
