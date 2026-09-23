#!/usr/bin/env python3
"""Run an explicit list of L3 30-hour paired cases serially in one allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/multigpu/run_l3_30h.py"
EXPECTED_ARGUMENTS = (
    ("expected_window_mode", "--expected-window-mode"),
    ("expected_window_min", "--expected-window-min"),
    ("expected_window_max", "--expected-window-max"),
    ("expected_idle_backoff", "--expected-idle-backoff"),
)


def expected_runner_arguments(case):
    arguments = []
    for key, option in EXPECTED_ARGUMENTS:
        if key in case:
            value = case[key]
            if key == "expected_idle_backoff" and isinstance(value, bool):
                value = int(value)
            arguments.extend((option, str(value)))
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-build", type=Path,
        help="required for exploratory cases; formal cases build their own pair")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("sweep must run inside a Slurm allocation")
    cases_path = args.cases.resolve(strict=True)
    cases = json.loads(cases_path.read_text())
    if not isinstance(cases, list) or not cases:
        parser.error("cases JSON must be a nonempty list")
    names = [case.get("name") for case in cases]
    if any(not isinstance(name, str) or not name for name in names):
        parser.error("every case needs a nonempty string name")
    if len(names) != len(set(names)):
        parser.error("case names must be unique")
    needs_external_pair = any(
        case.get("sampling", "exploratory") == "exploratory" for case in cases)
    if needs_external_pair and args.pair_build is None:
        parser.error("exploratory cases require --pair-build")
    build = (args.pair_build.resolve(strict=True)
             if args.pair_build is not None else None)
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)

    manifest = {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "pair_build": None if build is None else str(build),
        "cases_file": str(cases_path),
        "cases": cases,
        "records": [],
    }
    (args.out / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for case in cases:
        required = ("graph", "oracle", "source", "delta", "cut_percent", "blocks")
        missing = [key for key in required if key not in case]
        if missing:
            parser.error(f"case {case['name']} missing fields: {missing}")
        case_out = args.out / case["name"]
        sampling = case.get("sampling", "exploratory")
        command = [
            sys.executable, str(RUNNER),
            "--sampling", sampling,
            "--graph", case["graph"],
            "--oracle", case["oracle"],
            "--source", str(case["source"]),
            "--delta", str(case["delta"]),
            "--cut-percent", str(case["cut_percent"]),
            "--blocks", str(case["blocks"]),
            "--warmups", str(case.get("warmups", 1)),
            "--repeats", str(case.get(
                "repeats", 5 if sampling == "formal" else 3)),
            "--rounds", str(case.get(
                "rounds", 2 if sampling == "formal" else 1)),
            "--timeout", str(case.get("timeout", 300)),
            "--out", str(case_out),
        ]
        if sampling == "exploratory":
            command[4:4] = [
                "--pair-build", str(build),
                "--dual-binary", str(build / "dual_build/mlmq"),
                "--single-binary", str(build / "single_build/mlmq"),
            ]
        elif sampling != "formal":
            parser.error(
                f"case {case['name']} has invalid sampling value {sampling!r}")
        if case.get("input_contract") is not None:
            command.extend(("--input-contract", str(case["input_contract"])))
        command.extend(expected_runner_arguments(case))
        log = args.out / f"{case['name']}.driver.log"
        with log.open("w") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        summary_path = case_out / "summary.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                summary = None
        measurement_valid = bool(
            completed.returncode == 0 and summary is not None and
            summary.get("measurement_valid") is True)
        target_met = None if summary is None else summary.get("target_met")
        acceptance_pass = measurement_valid and target_met is True
        record = {
            "name": case["name"],
            "rc": completed.returncode,
            "sampling": sampling,
            "measurement_valid": measurement_valid,
            "target_met": target_met,
            "acceptance_pass": acceptance_pass,
            "command": command,
            "output": str(case_out),
            "driver_log": str(log),
        }
        manifest["records"].append(record)
        (args.out / "sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(
            f"SWEEP_CASE name={case['name']} rc={completed.returncode} "
            f"measurement_valid={measurement_valid} target_met={target_met}",
            flush=True)
    success = all(record["measurement_valid"] for record in manifest["records"])
    manifest["measurement_valid"] = success
    manifest["acceptance_pass"] = all(
        record["acceptance_pass"] for record in manifest["records"])
    manifest["status"] = "MEASUREMENT_VALID" if success else "MEASUREMENT_INVALID"
    (args.out / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
