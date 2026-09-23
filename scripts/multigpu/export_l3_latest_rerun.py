#!/usr/bin/env python3
"""Audit and export the compact, raw evidence for Slurm job 38082.

The source directory is intentionally outside Git because it contains graphs
and binaries.  This exporter re-reads every per-process record and raw log,
checks the recorded input/binary hashes, then copies only compact evidence.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    "/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/"
    "tmp/l3_no_sync_rerun_20260923"
)
DEFAULT_OUTPUT = ROOT / "evidence/l3_latest_rerun_38082"
GRAPHS = ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA")
VARIANTS = ("M1-original", "M1-shortcut", "M2-original", "M2-shortcut")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fields(raw, tag):
    return [
        dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
        for line in raw.splitlines()
        if line.startswith(tag + " ")
    ]


def load(path):
    with path.open() as stream:
        return json.load(stream)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    matrix = source / "matrix"
    output = args.out.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load(matrix / "manifest.json")
    graphs = load(matrix / "graphs.json")
    records = load(matrix / "records.json")
    recorded_summary = load(matrix / "summary.json")
    complete = load(matrix / "complete.json")
    if manifest.get("job") != "38082":
        raise RuntimeError(f"unexpected job: {manifest.get('job')!r}")
    if manifest.get("host") != "ada-A100":
        raise RuntimeError(f"unexpected host: {manifest.get('host')!r}")
    if list(graphs) != list(GRAPHS) or manifest.get("graphs") != graphs:
        raise RuntimeError("graph manifest/order mismatch")
    if len(records) != 64:
        raise RuntimeError(f"expected 64 records, got {len(records)}")

    # Verify the exact binaries and graph/oracle bytes named by the manifest.
    verified_hashes = {}
    for kind in ("single", "dual"):
        item = manifest["binaries"][kind]
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"{kind} binary hash mismatch")
        build_hashes = load(source / f"{kind}_build/hashes.json")
        if actual != build_hashes["mlmq"]:
            raise RuntimeError(f"{kind} build hash mismatch")
        verified_hashes[str(path)] = actual
    for graph, item in graphs.items():
        for kind in ("original", "augmented", "oracle"):
            path = Path(item[kind])
            actual = sha256(path)
            if actual != item["hashes"][kind]:
                raise RuntimeError(f"{graph} {kind} hash mismatch")
            verified_hashes[str(path)] = actual

    seen = set()
    formal = 0
    total = 0
    wide_oracle_lines = 0
    l2_final_lines = 0
    values = {graph: {variant: [] for variant in VARIANTS} for graph in GRAPHS}
    round_values = {
        graph: {variant: {0: [], 1: []} for variant in VARIANTS}
        for graph in GRAPHS
    }
    dispersion = []
    for record in records:
        identity = (record["round"], record["graph"], record["variant"])
        if identity in seen:
            raise RuntimeError(f"duplicate record: {identity}")
        seen.add(identity)
        round_id, graph, variant = identity
        if round_id not in (0, 1) or graph not in GRAPHS or variant not in VARIANTS:
            raise RuntimeError(f"invalid record identity: {identity}")
        prefix = matrix / f"r{round_id}_{graph}_{variant}"
        per_run = load(prefix.with_suffix(".json"))
        expected_per_run = {
            key: value for key, value in record.items()
            if key not in ("round", "graph", "variant")
        }
        if per_run != expected_per_run:
            raise RuntimeError(f"per-run/records mismatch: {identity}")
        raw = prefix.with_suffix(".log").read_text()
        samples = fields(raw, "BENCH")
        if samples != record["samples"]:
            raise RuntimeError(f"raw/record BENCH mismatch: {identity}")
        if record["rc"] != 0 or not record["valid"] or record["reason"] != "ok":
            raise RuntimeError(f"failed record retained in complete run: {identity}")
        if "RUN_RC=0" not in raw or "Error at node" in raw:
            raise RuntimeError(f"raw log failure marker: {identity}")
        expected_gpus = "1" if variant.startswith("M1") else "2"
        expected_source = str(graphs[graph]["source"])
        if len(samples) != 6:
            raise RuntimeError(f"expected 6 samples: {identity}")
        for index, sample in enumerate(samples):
            expected = {
                "algorithm": "MLMQ",
                "gpu_count": expected_gpus,
                "source": expected_source,
                "repeat": str(index),
                "warmup": str(int(index == 0)),
                "queue": "L1SLF_L2DQ",
                "correct": "1",
            }
            for key, value in expected.items():
                if sample.get(key) != value:
                    raise RuntimeError(
                        f"{identity} sample {index} {key}={sample.get(key)!r}, "
                        f"expected {value!r}"
                    )
            solve_ms = float(sample["solve_ms"])
            if not math.isfinite(solve_ms) or solve_ms <= 0:
                raise RuntimeError(f"invalid solve time: {identity} sample {index}")
            total += 1
            if sample["warmup"] == "0":
                formal += 1
                values[graph][variant].append(solve_ms)
                round_values[graph][variant][round_id].append(solve_ms)
        wide_count = len(fields(raw, "WIDE_ORACLE"))
        if wide_count < 6:
            raise RuntimeError(f"missing wide oracle checks: {identity}")
        wide_oracle_lines += wide_count
        final_count = len(fields(raw, "L2_FINAL"))
        if expected_gpus == "2" and final_count != 12:
            raise RuntimeError(f"missing dual L2 conservation: {identity}")
        l2_final_lines += final_count

    if len(seen) != 64 or total != 384 or formal != 320:
        raise RuntimeError(
            f"coverage mismatch: processes={len(seen)} total={total} formal={formal}"
        )

    recomputed = []
    for graph in GRAPHS:
        row = {"graph": graph}
        for variant in VARIANTS:
            samples = values[graph][variant]
            if len(samples) != 10:
                raise RuntimeError(f"expected 10 formal samples: {graph} {variant}")
            med = statistics.median(samples)
            row[variant] = med
            for round_id in (0, 1):
                row[f"{variant}_r{round_id}"] = statistics.median(
                    round_values[graph][variant][round_id]
                )
            q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
            dispersion.append({
                "graph": graph,
                "variant": variant,
                "n": len(samples),
                "median_ms": med,
                "q25_ms": q1,
                "q75_ms": q3,
                "iqr_ms": q3 - q1,
                "mad_ms": statistics.median(abs(value - med) for value in samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            })
        row["same_original"] = row["M1-original"] / row["M2-original"]
        row["same_shortcut"] = row["M1-shortcut"] / row["M2-shortcut"]
        row["complete"] = row["M1-original"] / row["M2-shortcut"]
        row["single_shortcut_gain"] = row["M1-original"] / row["M1-shortcut"]
        recomputed.append(row)

    if len(recorded_summary) != len(recomputed):
        raise RuntimeError("summary length mismatch")
    for actual, recorded in zip(recomputed, recorded_summary):
        if actual.keys() != recorded.keys() or actual["graph"] != recorded["graph"]:
            raise RuntimeError("summary schema/order mismatch")
        for key in actual.keys() - {"graph"}:
            if not close(actual[key], recorded[key]):
                raise RuntimeError(f"summary mismatch: {actual['graph']} {key}")
    geomeans = {
        key: math.exp(sum(math.log(row[key]) for row in recomputed) / len(recomputed))
        for key in ("same_original", "same_shortcut", "complete", "single_shortcut_gain")
    }
    for key, value in geomeans.items():
        if not close(value, complete["geomeans"][key]):
            raise RuntimeError(f"geomean mismatch: {key}")
    if complete != {
        "processes": 64,
        "queries": 384,
        "formal": 320,
        "valid": True,
        "geomeans": complete["geomeans"],
    }:
        raise RuntimeError("unexpected complete.json metadata")

    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        shutil.copytree(matrix, temporary / "matrix")
        for kind in ("single", "dual"):
            destination = temporary / f"{kind}_build"
            destination.mkdir()
            for name in ("build.log", "command.json", "hashes.json", "source.tgz", "status.json"):
                shutil.copy2(source / f"{kind}_build" / name, destination / name)
        with (temporary / "timing_dispersion.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dispersion[0]))
            writer.writeheader()
            writer.writerows(dispersion)
        shutil.copy2(matrix / "summary.csv", temporary / "plot_source.csv")
        audit = {
            "source": str(source),
            "job": manifest["job"],
            "host": manifest["host"],
            "processes": len(seen),
            "queries": total,
            "formal_queries": formal,
            "all_records_rc0_valid_correct": True,
            "wide_oracle_lines": wide_oracle_lines,
            "dual_l2_final_conservation_lines": l2_final_lines,
            "verified_hashes": verified_hashes,
            "geomeans": geomeans,
            "scope_note": (
                "This verifies the retained job-38082 files and their internal consistency. "
                "It does not turn the reverted synchronization implementation into a formal "
                "cross-device memory-order proof."
            ),
        }
        (temporary / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        readme = """# Latest reverted L3 rerun: Slurm 38082

This directory is a compact export of the retained `tmp/l3_no_sync_rerun_20260923`
batch. `audit.json` was generated by re-reading all 64 raw process logs, all 384
query rows (320 formal), the per-process JSON records, the eight graph/oracle
hashes, and the two binary hashes.

The batch establishes reproducible output correctness for the recorded queries
and reproduces the published per-graph table. It does **not** prove that the
reverted cross-device synchronization protocol is safe. It is historical
evidence and must not be mixed with the final 30-hour candidate batch.

`matrix/summary.csv` is the exact retained table; `plot_source.csv` is an
identical plotting input. Raw per-process `.log`, `.json`, GPU snapshots,
manifest, records, runner, and build source tarballs are retained here. Graphs
and executable binaries are omitted because their verified paths and SHA256
values are recorded in `matrix/manifest.json` and `audit.json`.
"""
        (temporary / "README.md").write_text(readme)
        raw_hashes = {
            str(path.relative_to(temporary)): sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        (temporary / "exported_files_sha256.json").write_text(
            json.dumps(raw_hashes, indent=2) + "\n"
        )
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "status": "PASS",
        "output": str(output),
        "processes": len(seen),
        "queries": total,
        "formal": formal,
        "geomeans": geomeans,
    }, indent=2))


if __name__ == "__main__":
    main()
