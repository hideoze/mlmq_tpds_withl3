#!/usr/bin/env python3
"""Export one compact, immutable ledger from all Stage-1 sweep manifests."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def option(command, name):
    try:
        index = command.index(name)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def load_summary(root, record):
    output = record.get("output")
    if not output:
        return None, None
    directory = Path(output)
    if not directory.is_absolute():
        directory = root / directory
    path = directory / "summary.json"
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    stage = root / "evidence/l3_30h_20260923/stage1"
    rows = []
    seen = set()
    for manifest_path in sorted(stage.rglob("sweep_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        job = str(manifest.get("slurm_job_id", ""))
        cases = {case.get("name"): case for case in manifest.get("cases", [])}
        for record in manifest.get("records", []):
            command = [str(item) for item in record.get("command", [])]
            key = (job, record.get("name"), tuple(command))
            if key in seen:
                continue
            seen.add(key)
            case = cases.get(record.get("name"), {})
            summary, summary_hash = load_summary(root, record)
            combined = (summary or {}).get("combined", {})
            valid = record.get("measurement_valid")
            if valid is None:
                valid = combined.get("valid")
            target_met = record.get("target_met")
            if target_met is None:
                target_met = combined.get("target_met")
            rc = record.get("rc")
            if rc != 0:
                decision = "FAILED_RETAINED"
            elif valid is not True:
                decision = "INVALID_RETAINED"
            elif target_met is True:
                decision = "EXPLORATORY_TARGET_ONLY_NOT_FORMAL"
            else:
                decision = "VALID_BELOW_TARGET"
            t1 = combined.get("T1", {}).get("solve", {}).get("median_ms")
            t2 = combined.get("T2", {}).get("solve", {}).get("median_ms")
            row = {
                "job": job or None,
                "case": record.get("name"),
                "evidence_manifest": str(manifest_path.relative_to(root)),
                "manifest_sha256": sha256(manifest_path),
                "pair_build": manifest.get("pair_build"),
                "graph": case.get("graph", option(command, "--graph")),
                "oracle": case.get("oracle", option(command, "--oracle")),
                "source": case.get("source", option(command, "--source")),
                "delta": case.get("delta", option(command, "--delta")),
                "cut_percent": case.get(
                    "cut_percent", option(command, "--cut-percent")),
                "blocks": case.get("blocks", option(command, "--blocks")),
                "window_mode": case.get(
                    "expected_window_mode",
                    option(command, "--expected-window-mode")),
                "window_min": case.get(
                    "expected_window_min",
                    option(command, "--expected-window-min")),
                "window_max": case.get(
                    "expected_window_max",
                    option(command, "--expected-window-max")),
                "idle_backoff": case.get(
                    "expected_idle_backoff",
                    option(command, "--expected-idle-backoff")),
                "rc": rc,
                "measurement_valid": valid,
                "target_met": target_met,
                "T1_solve_median_ms": t1,
                "T2_solve_median_ms": t2,
                "S_solve_T1_over_T2": combined.get(
                    "S_solve_T1_over_T2"),
                "summary_sha256": summary_hash,
                "decision": decision,
                "evidence_class": "DIRTY_REVERTED_SYNC_EXPLORATORY",
            }
            rows.append(row)

    rows.sort(key=lambda row: (
        int(row["job"]) if row["job"] and row["job"].isdigit() else -1,
        row["case"] or "", row["pair_build"] or ""))
    counts = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    payload = {
        "schema": 1,
        "scope": "Stage-1 dirty reverted-sync exploratory sweeps only",
        "formal_acceptance": False,
        "count": len(rows),
        "decision_counts": counts,
        "records": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    fieldnames = list(rows[0]) if rows else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    print(f"CANDIDATE_LEDGER records={len(rows)} decisions={counts}")


if __name__ == "__main__":
    main()
