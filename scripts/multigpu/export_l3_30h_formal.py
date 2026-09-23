#!/usr/bin/env python3
"""Validate and compactly export the final L3 30-hour formal evidence.

The exporter deliberately separates execution validity from the 1.20
performance target.  A complete, correct batch whose measured speedup is below
the target is exported as ``VALID_BELOW_TARGET`` rather than as a failed run.

The initial export accepts the external Job A/Job B directories and creates an
immutable compact evidence tree.  ``--refresh-manifest`` is a safe second pass
for use after figures and prose have been added: it only rewrites
``archive_manifest.json`` and ``SHA256SUMS``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from typing import Any, Iterable


SCHEMA = 1
TARGET = 1.20
MAX_ARCHIVE_FILE_BYTES = 2 * 1024 * 1024
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
GRAPHS = ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA")
VIEWS = ("G", "G_plus")
PRIMARY_CONFIGS = ("single_no_l3", "dual_l3")
PRIMARY_LOGICAL_SOURCE = 0
PRIMARY_LAYOUT_SOURCE = 11973673
USA_SOURCE_BINDINGS = {
    "fixed_additional_source_1": (7982448, 18266241),
    "fixed_additional_source_2": (15964897, 6146689),
}
EXPANDED_TREE_COMPONENTS = {
    "single_source",
    "dual_source",
    "fixture_source",
    "guarded_core",
    "derived_sources",
    "formal_inputs",
    "sequential_fixture",
}
BINARY_TREE_COMPONENTS = {"binaries", "derived_binaries"}
ALLOWED_SUFFIXES = {".json", ".csv", ".log", ".txt", ".env", ".sha256"}


class EvidenceError(RuntimeError):
    """Raised when retained evidence violates a formal acceptance contract."""


HISTORICAL_ATTEMPTS = (
    {
        "job_id": "38227",
        "commit": "c0bc9ca8171ccb4012bf4bbbb1e6771f4161b4a6",
        "event_kind": "FORMAL_WORKFLOW_ATTEMPT",
        "role": "historical_job_a_attempt",
        "aggregate_allowed": False,
        "status": "INVALID_PRE_SAMPLING",
        "last_valid_stage": "build_pair",
        "reason": (
            "source provenance gate treated non-semantic tar mode differences "
            "as source changes; no GPU samples were taken"
        ),
    },
    {
        "job_id": "38238",
        "commit": "7cd563caaac7749521f833187ebd0b1f959b3388",
        "event_kind": "FORMAL_WORKFLOW_ATTEMPT",
        "role": "historical_job_a_attempt",
        "aggregate_allowed": False,
        "status": "INVALID_PARTIAL_PRIMARY",
        "last_valid_stage": "primary_round_0",
        "reason": (
            "BNUM=16 primary round 1 failed with CUDA 719 after warmup; the "
            "complete batch is invalid despite valid round-0 diagnostics"
        ),
    },
    {
        "job_id": "38251",
        "commit": None,
        "event_kind": "EXPLORATORY_PRELAUNCH",
        "role": "dirty_bnum8_prelaunch",
        "aggregate_allowed": False,
        "status": "EXPLORATORY_PRELAUNCH_FAILURE",
        "last_valid_stage": "none",
        "reason": (
            "compute node could not see a login-node /tmp binary; rc=127 and "
            "the algorithm did not execute"
        ),
    },
    {
        "job_id": "38254",
        "commit": None,
        "event_kind": "EXPLORATORY_PREFREEZE_PROBE",
        "role": "dirty_bnum8_recovery_probe",
        "aggregate_allowed": False,
        "status": "EXPLORATORY_RECOVERY_PASS",
        "last_valid_stage": "18_query_probe",
        "reason": (
            "three dirty-worktree dual-GPU processes completed 18 correct, "
            "conserving, no-wrap queries; engineering evidence only"
        ),
    },
    {
        "job_id": "38271",
        "commit": "ead2cdbac48baa3d1dc807a5a6647e3cbe68bc46",
        "event_kind": "FORMAL_WORKFLOW_ATTEMPT",
        "role": "historical_job_a_attempt",
        "aggregate_allowed": False,
        "status": "INVALID_AFTER_PRIMARY",
        "last_valid_stage": "01_primary",
        "reason": (
            "primary was valid below target, then final checks failed because "
            "the hard-coded /usr/local/cuda/bin/nvcc path was absent"
        ),
    },
    {
        "job_id": "38280",
        "commit": "a539b8a5c3b19b8bb11a5a7a7edcd1b2160680d9",
        "event_kind": "FORMAL_WORKFLOW_ATTEMPT",
        "role": "historical_job_a_attempt",
        "aggregate_allowed": False,
        "status": "INVALID_AFTER_PRIMARY",
        "last_valid_stage": "01_primary",
        "reason": (
            "primary was valid below target, then a mixed worktree/archive "
            "fixture compile included common.h twice"
        ),
    },
    {
        "job_id": "38292",
        "commit": "feb2d92b1e7d83f031c485795bde49693a9696b5",
        "event_kind": "FORMAL_WORKFLOW_ATTEMPT",
        "role": "historical_job_a_attempt",
        "aggregate_allowed": False,
        "status": "CANCELLED_DURING_NUMERIC",
        "last_valid_stage": "02_final_checks",
        "reason": (
            "numeric runner forced CUDA_LAUNCH_BLOCKING=1, the first single "
            "case timed out at 600 seconds, and the exact job was cancelled"
        ),
    },
)

HISTORICAL_ATTEMPT_DETAILS = {
    "38227": {
        "clean_worktree": True, "recorded_status": "FAILED",
        "effective_status": "FAILED_PRE_SAMPLING", "failure_step": "01_primary",
        "workflow_rc": 1, "child_rc": 1, "algorithm_launched": False,
        "primary_measurement_valid": None, "primary_target_met": None,
        "valid_subresult_scope": "pair_build_only",
        "downstream_not_run": "02-05;JobB",
        "formal_failure_evidence_allowed": True,
        "evidence_completeness": "full_formal_failure_directory",
    },
    "38238": {
        "clean_worktree": True, "recorded_status": "FAILED",
        "effective_status": "FAILED_PARTIAL_PRIMARY", "failure_step": "01_primary",
        "workflow_rc": 1, "child_rc": 2, "algorithm_launched": True,
        "primary_measurement_valid": False, "primary_target_met": None,
        "valid_subresult_scope": "round0_diagnostic_only",
        "downstream_not_run": "02-05;JobB",
        "formal_failure_evidence_allowed": True,
        "evidence_completeness": "full_formal_failure_directory",
    },
    "38251": {
        "base_head": "7cd563caaac7749521f833187ebd0b1f959b3388",
        "clean_worktree": False, "recorded_status": "FAILED",
        "effective_status": "DIRTY_EXPLORATORY_PRELAUNCH_FAILURE",
        "failure_step": "binary_visibility", "workflow_rc": 127,
        "child_rc": 127, "algorithm_launched": False,
        "primary_measurement_valid": None, "primary_target_met": None,
        "valid_subresult_scope": "none", "downstream_not_run": "all_queries",
        "formal_failure_evidence_allowed": False,
        "evidence_completeness": "summary_only",
    },
    "38254": {
        "base_head": "7cd563caaac7749521f833187ebd0b1f959b3388",
        "clean_worktree": False, "recorded_status": "COMPLETE",
        "effective_status": "EXPLORATORY_RECOVERY_PASS", "failure_step": None,
        "workflow_rc": 0, "child_rc": 0, "algorithm_launched": True,
        "primary_measurement_valid": None, "primary_target_met": None,
        "valid_subresult_scope": "18_query_bnum8_capacity_probe",
        "downstream_not_run": "formal_acceptance_not_applicable",
        "formal_failure_evidence_allowed": False,
        "evidence_completeness": "compact_probe_logs_and_hashes",
    },
    "38271": {
        "clean_worktree": True, "recorded_status": "FAILED",
        "effective_status": "FAILED_AFTER_VALID_PRIMARY",
        "failure_step": "02_final_checks", "workflow_rc": 1, "child_rc": 1,
        "algorithm_launched": True, "primary_measurement_valid": True,
        "primary_target_met": False, "valid_subresult_scope": "01_primary",
        "downstream_not_run": "03-05;JobB",
        "formal_failure_evidence_allowed": True,
        "evidence_completeness": "full_formal_failure_directory",
    },
    "38280": {
        "clean_worktree": True, "recorded_status": "FAILED",
        "effective_status": "FAILED_AFTER_VALID_PRIMARY",
        "failure_step": "02_final_checks", "workflow_rc": 1, "child_rc": 1,
        "algorithm_launched": True, "primary_measurement_valid": True,
        "primary_target_met": False, "valid_subresult_scope": "01_primary",
        "downstream_not_run": "03-05;JobB",
        "formal_failure_evidence_allowed": True,
        "evidence_completeness": "full_formal_failure_directory",
    },
    "38292": {
        "clean_worktree": True, "recorded_status": "RUNNING",
        "effective_status": "CANCELLED_INCOMPLETE_AFTER_NUMERIC_FAILURE",
        "failure_step": "03_numeric_add", "workflow_rc": None, "child_rc": 124,
        "algorithm_launched": True, "primary_measurement_valid": True,
        "primary_target_met": False,
        "valid_subresult_scope": "01_primary;02_final_checks",
        "downstream_not_run": "04-05;JobB",
        "formal_failure_evidence_allowed": True,
        "evidence_completeness": "cancelled_directory_with_stale_running_status",
        "termination_note": (
            "operator cancelled the exact job after the numeric timeout; an "
            "outer exit 143 was observed but is not present in the raw status file"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise EvidenceError(f"missing JSON evidence: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error


def parse_key_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise EvidenceError(f"missing key/value evidence: {path}")
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            raise EvidenceError(f"malformed {path}:{number}: {raw!r}")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise EvidenceError(f"duplicate/empty key in {path}:{number}: {key!r}")
        result[key] = value
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def close(actual: Any, expected: Any, label: str) -> None:
    try:
        good = math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        good = False
    if not good:
        raise EvidenceError(f"{label}: {actual!r} != recomputed {expected!r}")


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    require(numbers, "cannot summarize an empty sample set")
    require(
        all(math.isfinite(value) and value > 0 for value in numbers),
        "all timing samples must be finite and positive",
    )
    median = statistics.median(numbers)
    q1, _, q3 = statistics.quantiles(numbers, n=4, method="inclusive")
    return {
        "count": len(numbers),
        "median_ms": median,
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "mad_ms": statistics.median(abs(value - median) for value in numbers),
        "samples_ms": numbers,
    }


def compare_distribution(recorded: Any, actual: dict[str, Any], label: str) -> None:
    require(isinstance(recorded, dict), f"{label}: recorded distribution is absent")
    require(recorded.get("count") == actual["count"], f"{label}: count mismatch")
    for field in ("median_ms", "q1_ms", "q3_ms", "iqr_ms", "mad_ms"):
        close(recorded.get(field), actual[field], f"{label}.{field}")
    recorded_samples = recorded.get("samples_ms")
    require(
        isinstance(recorded_samples, list)
        and len(recorded_samples) == len(actual["samples_ms"]),
        f"{label}: sample-list length mismatch",
    )
    for index, (left, right) in enumerate(zip(recorded_samples, actual["samples_ms"])):
        close(left, right, f"{label}.samples_ms[{index}]")


def _configuration_samples(samples: list[dict[str, Any]], configuration: str,
                           round_id: int | None = None) -> list[dict[str, Any]]:
    selected = [
        item for item in samples
        if item.get("configuration") == configuration
        and item.get("warmup") == "0"
        and (round_id is None or item.get("round") == round_id)
    ]
    return selected


def _compare_pair_section(section: Any, t1: list[dict[str, Any]],
                          t2: list[dict[str, Any]], label: str,
                          expected_count: int) -> dict[str, Any]:
    require(isinstance(section, dict) and section.get("valid") is True,
            f"{label}: paired section is not valid")
    result: dict[str, Any] = {}
    for role, selected in (("T1", t1), ("T2", t2)):
        require(len(selected) == expected_count,
                f"{label}: expected {expected_count} {role} retained non-warmup samples, "
                f"got {len(selected)}")
        recorded = section.get(role)
        require(isinstance(recorded, dict) and recorded.get("valid") is True,
                f"{label}.{role}: invalid summary")
        require(recorded.get("reason") == "ok", f"{label}.{role}: reason is not ok")
        require(recorded.get("formal_sample_count") == expected_count,
                f"{label}.{role}: formal sample count mismatch")
        solve = distribution(item["solve_ms"] for item in selected)
        wall = distribution(item["query_wall_ms"] for item in selected)
        compare_distribution(recorded.get("solve"), solve, f"{label}.{role}.solve")
        compare_distribution(recorded.get("query_wall"), wall,
                             f"{label}.{role}.query_wall")
        result[role] = {"solve": solve, "query_wall": wall}
    solve_speedup = result["T1"]["solve"]["median_ms"] / result["T2"]["solve"]["median_ms"]
    wall_speedup = (
        result["T1"]["query_wall"]["median_ms"]
        / result["T2"]["query_wall"]["median_ms"]
    )
    close(section.get("S_solve_T1_over_T2"), solve_speedup,
          f"{label}.S_solve_T1_over_T2")
    close(section.get("S_query_wall_T1_over_T2"), wall_speedup,
          f"{label}.S_query_wall_T1_over_T2")
    result["solve_speedup"] = solve_speedup
    result["query_wall_speedup"] = wall_speedup
    return result


def validate_paired_summary(samples_path: Path, summary_path: Path, *,
                            evidence_id: str, job: str, category: str,
                            graph: str | None = None,
                            view: str | None = None,
                            source_name: str | None = None,
                            outer_workflow: str = "formal JobA",
                            expected_inner_sampling: str = "formal",
                            evidence_classification: str = "formal_primary") -> dict[str, Any]:
    samples = load_json(samples_path)
    summary = load_json(summary_path)
    require(isinstance(samples, list) and len(samples) == 24,
            f"{evidence_id}: expected 24 raw query samples")
    seen: set[tuple[Any, ...]] = set()
    for index, sample in enumerate(samples):
        require(isinstance(sample, dict), f"{evidence_id}: sample {index} is not an object")
        identity = (sample.get("round"), sample.get("configuration"), sample.get("repeat"))
        require(identity not in seen, f"{evidence_id}: duplicate sample {identity}")
        seen.add(identity)
        require(sample.get("round") in (0, 1), f"{evidence_id}: bad round in {identity}")
        require(sample.get("configuration") in PRIMARY_CONFIGS,
                f"{evidence_id}: bad configuration in {identity}")
        require(sample.get("warmup") in ("0", "1"),
                f"{evidence_id}: bad warmup in {identity}")
        require(sample.get("run_valid") is True and sample.get("correct") == "1",
                f"{evidence_id}: invalid/correctness-failed sample {identity}")
    for configuration in PRIMARY_CONFIGS:
        selected = [item for item in samples if item["configuration"] == configuration]
        require(len(selected) == 12, f"{evidence_id}: expected 12 {configuration} samples")
        for round_id in (0, 1):
            per_round = [item for item in selected if item["round"] == round_id]
            require(len(per_round) == 6,
                    f"{evidence_id}: expected 6 {configuration} samples in round {round_id}")
            require(sum(item["warmup"] == "1" for item in per_round) == 1,
                    f"{evidence_id}: expected one warmup in round {round_id} {configuration}")

    require(summary.get("measurement_valid") is True,
            f"{evidence_id}: measurement_valid is not true")
    require(summary.get("formal_integrity_valid") is True,
            f"{evidence_id}: formal_integrity_valid is not true")
    require(summary.get("formal_integrity_error") is None,
            f"{evidence_id}: formal integrity error is present")
    require(summary.get("sampling") == expected_inner_sampling,
            f"{evidence_id}: inner sampling is {summary.get('sampling')!r}, "
            f"expected {expected_inner_sampling!r}")
    t1 = _configuration_samples(samples, "single_no_l3")
    t2 = _configuration_samples(samples, "dual_l3")
    combined = _compare_pair_section(summary.get("combined"), t1, t2,
                                     f"{evidence_id}.combined", 10)

    recorded_rounds = summary.get("rounds")
    require(isinstance(recorded_rounds, list) and len(recorded_rounds) == 2,
            f"{evidence_id}: expected two round summaries")
    by_round = {item.get("round"): item for item in recorded_rounds}
    require(set(by_round) == {0, 1}, f"{evidence_id}: round summary IDs mismatch")
    for round_id in (0, 1):
        _compare_pair_section(
            by_round[round_id],
            _configuration_samples(samples, "single_no_l3", round_id),
            _configuration_samples(samples, "dual_l3", round_id),
            f"{evidence_id}.round{round_id}",
            5,
        )

    target = float(summary["combined"].get("target", TARGET))
    close(target, TARGET, f"{evidence_id}.target")
    target_met = combined["solve_speedup"] >= target
    require(summary["combined"].get("target_met") is target_met,
            f"{evidence_id}: combined target result mismatch")
    require(summary.get("target_met") is target_met,
            f"{evidence_id}: top-level target result mismatch")
    require(summary.get("numeric_target_met") is target_met,
            f"{evidence_id}: numeric target result mismatch")
    return {
        "evidence_id": evidence_id,
        "job": job,
        "category": category,
        "graph": graph,
        "view": view,
        "source_name": source_name,
        "outer_workflow": outer_workflow,
        "inner_sampling": expected_inner_sampling,
        "evidence_classification": evidence_classification,
        "measurement_valid": True,
        "performance_target": target,
        "performance_target_met": target_met,
        "classification": "TARGET_MET" if target_met else "VALID_BELOW_TARGET",
        "samples_per_role": 10,
        "formal_samples_per_role": 10 if expected_inner_sampling == "formal" else None,
        "T1_solve_median_ms": combined["T1"]["solve"]["median_ms"],
        "T1_solve_iqr_ms": combined["T1"]["solve"]["iqr_ms"],
        "T1_solve_mad_ms": combined["T1"]["solve"]["mad_ms"],
        "T2_solve_median_ms": combined["T2"]["solve"]["median_ms"],
        "T2_solve_iqr_ms": combined["T2"]["solve"]["iqr_ms"],
        "T2_solve_mad_ms": combined["T2"]["solve"]["mad_ms"],
        "solve_speedup_T1_over_T2": combined["solve_speedup"],
        "T1_query_wall_median_ms": combined["T1"]["query_wall"]["median_ms"],
        "T1_query_wall_iqr_ms": combined["T1"]["query_wall"]["iqr_ms"],
        "T1_query_wall_mad_ms": combined["T1"]["query_wall"]["mad_ms"],
        "T2_query_wall_median_ms": combined["T2"]["query_wall"]["median_ms"],
        "T2_query_wall_iqr_ms": combined["T2"]["query_wall"]["iqr_ms"],
        "T2_query_wall_mad_ms": combined["T2"]["query_wall"]["mad_ms"],
        "query_wall_speedup_T1_over_T2": combined["query_wall_speedup"],
        "samples_sha256": sha256(samples_path),
        "recorded_summary_sha256": sha256(summary_path),
    }


def _validate_job_identity(root: Path, expected_workflow: str,
                           expected_sha: str | None) -> dict[str, str]:
    status = parse_key_values(root / "orchestration.status")
    require(status.get("workflow") == expected_workflow,
            f"{root}: workflow is not {expected_workflow}")
    require(status.get("status") == "COMPLETE", f"{root}: workflow is not COMPLETE")
    require(status.get("rc") == "0", f"{root}: orchestration rc is not zero")
    head = status.get("head", "")
    require(bool(FULL_SHA.fullmatch(head)), f"{root}: invalid full clean SHA {head!r}")
    if expected_sha is not None:
        require(head == expected_sha, f"{root}: head {head} != expected {expected_sha}")
    job_id = status.get("slurm_job_id", "")
    require(job_id.isdigit(), f"{root}: invalid Slurm job ID {job_id!r}")
    require((root / "metadata/head.txt").read_text(encoding="utf-8").strip() == head,
            f"{root}: metadata/head.txt mismatch")
    environment = parse_key_values(root / "metadata/slurm.env")
    require(environment.get("SLURM_JOB_ID") == job_id,
            f"{root}: metadata Slurm ID mismatch")
    require(environment.get("SLURM_GPUS_ON_NODE") == "2",
            f"{root}: formal workflow did not receive exactly two GPUs")
    return {"head": head, "slurm_job_id": job_id}


def _validate_primary(root: Path, directory: str, job_label: str,
                      identity: dict[str, str]) -> dict[str, Any]:
    base = root / directory
    driver = parse_key_values(root / "driver_logs" / f"{directory}.result.env")
    require(driver.get("rc") == "0", f"{job_label}: primary driver rc is not zero")
    manifest = load_json(base / "manifest.json")
    require(manifest.get("status") == "measurement_valid",
            f"{job_label}: primary manifest status is not measurement_valid")
    require(manifest.get("measurement_valid") is True
            and manifest.get("formal_integrity_valid") is True
            and manifest.get("all_processes_valid") is True,
            f"{job_label}: primary validity gates failed")
    require(manifest.get("process_count") == 4,
            f"{job_label}: primary must contain four fresh processes")
    require(str(manifest.get("slurm_job_id")) == identity["slurm_job_id"],
            f"{job_label}: primary Slurm ID mismatch")
    git = manifest.get("git", {})
    require(git.get("head", {}).get("rc") == 0
            and git.get("head", {}).get("output") == identity["head"],
            f"{job_label}: primary Git head mismatch")
    require(git.get("status", {}).get("rc") == 0
            and git.get("status", {}).get("output") == "",
            f"{job_label}: primary source tree was not clean")
    configuration = manifest.get("configuration", {})
    require(configuration.get("sampling") == "formal",
            f"{job_label}: primary sampling is not formal")
    require(configuration.get("source") == PRIMARY_LAYOUT_SOURCE,
            f"{job_label}: primary source differs from the frozen layout ID")
    row = validate_paired_summary(
        base / "samples.json", base / "summary.json",
        evidence_id=f"{job_label}_primary", job=job_label, category="primary",
        graph="USA", view="G+", source_name="primary",
        outer_workflow=f"formal {job_label}",
        evidence_classification=(
            "formal_primary_confirmation" if job_label == "JobB" else "formal_primary"
        ),
    )
    row.update({
        "logical_source": PRIMARY_LOGICAL_SOURCE,
        "layout_source": PRIMARY_LAYOUT_SOURCE,
    })
    return row


def _validate_driver(root: Path, step: str) -> None:
    result = parse_key_values(root / "driver_logs" / f"{step}.result.env")
    require(result.get("step") == step, f"JobA {step}: driver step mismatch")
    require(result.get("rc") == "0", f"JobA {step}: driver rc is not zero")


def _validate_final_checks(job_a: Path, job_id: str) -> None:
    _validate_driver(job_a, "02_final_checks")
    results = load_json(job_a / "02_final_checks/results.json")
    manifest = load_json(job_a / "02_final_checks/manifest.json")
    require(results.get("valid") is True and results.get("problems") == [],
            "JobA final checks are not valid/problem-free")
    require(results.get("sequential_reset_valid") is True,
            "JobA sequential reset check failed")
    require(results.get("eight_graphs_valid") is True
            and results.get("eight_graph_count") == 8,
            "JobA eight-graph correctness check failed")
    fixtures = results.get("small_fixtures", {})
    require(set(fixtures) == {"primitives", "dq_publication", "capacity_full", "capacity_overflow"},
            "JobA final-check fixture set mismatch")
    require(all(item.get("valid") is True and item.get("rc") == 0
                and item.get("timed_out") is False for item in fixtures.values()),
            "JobA small fixture failed")
    require(manifest.get("status") == "complete" and manifest.get("problems") == [],
            "JobA final-check manifest is incomplete")
    require(str(manifest.get("slurm_job_id")) == job_id,
            "JobA final-check Slurm ID mismatch")


def _validate_numeric(job_a: Path, job_id: str) -> None:
    _validate_driver(job_a, "03_numeric_add")
    results = load_json(job_a / "03_numeric_add/results.json")
    manifest = load_json(job_a / "03_numeric_add/manifest.json")
    require(results.get("valid") is True and results.get("problems") == [],
            "JobA numeric checked-add gate failed")
    require(results.get("completed_runs") == results.get("expected_runs") == 10,
            "JobA numeric checked-add coverage is not 10/10")
    runs = results.get("runs")
    require(isinstance(runs, list) and len(runs) == 10,
            "JobA numeric checked-add run list is not length 10")
    require(all(run.get("valid") is True and run.get("rc") == 0
                and run.get("timed_out") is False
                and run.get("contract", {}).get("valid") is True for run in runs),
            "JobA numeric checked-add contains an invalid run")
    require(manifest.get("status") == "complete" and manifest.get("problems") == [],
            "JobA numeric manifest is incomplete")
    require(manifest.get("performance_claim_allowed") is False,
            "numeric diagnostic timings must be forbidden as performance evidence")
    require(manifest.get("numeric_gate") == "PASS_FOR_FROZEN_QUERY_SET",
            "numeric diagnostic gate is not PASS_FOR_FROZEN_QUERY_SET")
    require(str(manifest.get("slurm_job_id")) == job_id,
            "JobA numeric Slurm ID mismatch")


def _validate_usa(job_a: Path) -> list[dict[str, Any]]:
    _validate_driver(job_a, "04_usa_sources")
    base = job_a / "04_usa_sources"
    complete = load_json(base / "complete.json")
    status = load_json(base / "status.json")
    summary = load_json(base / "summary.json")
    require(complete.get("measurement_valid") is True
            and complete.get("sources") == 2
            and complete.get("expected_processes") == 8
            and complete.get("expected_queries") == 48
            and complete.get("expected_formal_queries") == 40,
            "JobA fixed USA-source completion contract failed")
    require(status.get("measurement_valid") is True
            and status.get("completed_sources") == status.get("expected_sources") == 2,
            "JobA fixed USA-source status is incomplete")
    require(summary.get("measurement_valid") is True
            and summary.get("measurement_validity_is_independent_of_target") is True,
            "JobA fixed USA-source summary is invalid")
    outer = summary.get("sources")
    require(isinstance(outer, list) and len(outer) == 2,
            "JobA fixed USA-source outer summary count mismatch")
    by_name = {Path(item["case_output"]).name: item for item in outer}
    expected = tuple(USA_SOURCE_BINDINGS)
    require(set(by_name) == set(expected), "JobA fixed USA-source names mismatch")
    rows = []
    for name in expected:
        directory = base / "sources" / name
        row = validate_paired_summary(
            directory / "samples.json", directory / "summary.json",
            evidence_id=f"JobA_USA_{name}", job="JobA", category="usa_source",
            graph="USA", view="G+", source_name=name,
            expected_inner_sampling="exploratory",
            evidence_classification="full_sample_clean_sha_extension_regression",
        )
        item = by_name[name]
        require(item.get("measurement_valid") is True and item.get("errors") == [],
                f"JobA USA source {name}: invalid outer record")
        logical_source, layout_source = USA_SOURCE_BINDINGS[name]
        require(item.get("old_logical_id") == logical_source
                and item.get("source") == layout_source,
                f"JobA USA source {name}: source binding mismatch")
        row.update({
            "logical_source": logical_source,
            "layout_source": layout_source,
        })
        metrics = item.get("metrics", {})
        close(metrics.get("S_solve_T1_over_T2"), row["solve_speedup_T1_over_T2"],
              f"JobA USA source {name}: outer solve speedup")
        close(metrics.get("S_query_wall_T1_over_T2"),
              row["query_wall_speedup_T1_over_T2"],
              f"JobA USA source {name}: outer query-wall speedup")
        rows.append(row)
    return rows


def _geomean(values: Iterable[float]) -> float:
    numbers = list(values)
    require(numbers and all(value > 0 and math.isfinite(value) for value in numbers),
            "geomean inputs must be finite and positive")
    return math.exp(sum(math.log(value) for value in numbers) / len(numbers))


def _validate_eight_graph(job_a: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_driver(job_a, "05_eight_graph")
    base = job_a / "05_eight_graph"
    complete = load_json(base / "complete.json")
    status = load_json(base / "status.json")
    summary = load_json(base / "summary.json")
    require(complete.get("measurement_valid") is True
            and complete.get("cases") == 16
            and complete.get("expected_processes") == complete.get("validated_processes") == 64
            and complete.get("expected_queries") == complete.get("validated_queries") == 384
            and complete.get("expected_formal_queries") == complete.get("validated_formal_queries") == 320,
            "JobA eight-graph completion contract failed")
    require(status.get("measurement_valid") is True
            and status.get("completed_cases") == status.get("expected_cases") == 16,
            "JobA eight-graph status is incomplete")
    require(summary.get("measurement_valid") is True
            and summary.get("measurement_validity_is_independent_of_target") is True
            and summary.get("case_count") == summary.get("expected_case_count") == 16,
            "JobA eight-graph summary is invalid")
    outer = summary.get("cases")
    require(isinstance(outer, list) and len(outer) == 16,
            "JobA eight-graph outer case count mismatch")
    by_id = {item.get("case_id"): item for item in outer}
    expected_ids = {f"{graph}_{view}" for graph in GRAPHS for view in VIEWS}
    require(set(by_id) == expected_ids, "JobA eight-graph case IDs mismatch")
    case_index = load_json(base / "case_index.json")
    require(isinstance(case_index, list) and len(case_index) == 16,
            "JobA eight-graph case index count mismatch")
    indexed = {item.get("case_id"): item for item in case_index
               if isinstance(item, dict)}
    require(set(indexed) == expected_ids,
            "JobA eight-graph case index IDs mismatch")
    rows = []
    for graph in GRAPHS:
        for view in VIEWS:
            case_id = f"{graph}_{view}"
            directory = base / "cases" / graph / view
            row = validate_paired_summary(
                directory / "samples.json", directory / "summary.json",
                evidence_id=f"JobA_{case_id}", job="JobA", category="eight_graph",
                graph=graph, view="G+" if view == "G_plus" else "G",
                expected_inner_sampling="exploratory",
                evidence_classification="full_sample_clean_sha_extension_regression",
            )
            item = by_id[case_id]
            require(item.get("measurement_valid") is True and item.get("error_count") == 0,
                    f"JobA {case_id}: invalid outer case record")
            close(item.get("T1_single_no_l3_median_solve_ms"), row["T1_solve_median_ms"],
                  f"JobA {case_id}: outer T1 solve median")
            close(item.get("T2_dual_l3_median_solve_ms"), row["T2_solve_median_ms"],
                  f"JobA {case_id}: outer T2 solve median")
            close(item.get("speedup_T1_over_T2"), row["solve_speedup_T1_over_T2"],
                  f"JobA {case_id}: outer solve speedup")
            index_item = indexed[case_id]
            require(index_item.get("graph") == graph
                    and index_item.get("view") == ("G+" if view == "G_plus" else "G")
                    and isinstance(index_item.get("source"), int),
                    f"JobA {case_id}: invalid case-index source binding")
            row["layout_source"] = index_item["source"]
            row["logical_source"] = index_item["source"]
            rows.append(row)
    geomeans: dict[str, Any] = {}
    for view in ("G", "G+"):
        selected = [row for row in rows if row["view"] == view]
        solve = _geomean(row["solve_speedup_T1_over_T2"] for row in selected)
        wall = _geomean(row["query_wall_speedup_T1_over_T2"] for row in selected)
        recorded = summary.get("geomeans", {}).get(view, {})
        complete_recorded = complete.get("geomeans", {}).get(view, {})
        require(recorded.get("valid") is True and recorded.get("graph_count") == 8,
                f"JobA {view}: geomean record invalid")
        close(recorded.get("solve_speedup_geomean_T1_over_T2"), solve,
              f"JobA {view}: solve geomean")
        close(recorded.get("query_wall_speedup_geomean_T1_over_T2"), wall,
              f"JobA {view}: query-wall geomean")
        close(complete_recorded.get("solve_speedup_geomean_T1_over_T2"), solve,
              f"JobA {view}: complete solve geomean")
        close(complete_recorded.get("query_wall_speedup_geomean_T1_over_T2"), wall,
              f"JobA {view}: complete query-wall geomean")
        geomeans[view] = {
            "graph_count": 8,
            "solve_speedup_geomean_T1_over_T2": solve,
            "query_wall_speedup_geomean_T1_over_T2": wall,
        }
    all_solve = _geomean(row["solve_speedup_T1_over_T2"] for row in rows)
    close(summary.get("all_16_solve_speedup_geomean_T1_over_T2"), all_solve,
          "JobA all-16 solve geomean")
    close(complete.get("all_16_solve_speedup_geomean_T1_over_T2"), all_solve,
          "JobA complete all-16 solve geomean")
    return rows, {"views": geomeans, "all_16_solve_speedup_geomean_T1_over_T2": all_solve}


def validate_formal_inputs(job_a: Path, job_b: Path,
                           expected_sha: str | None = None) -> dict[str, Any]:
    job_a = job_a.resolve(strict=True)
    job_b = job_b.resolve(strict=True)
    require(job_a.is_dir() and job_b.is_dir(), "Job A and Job B must be directories")
    require(job_a != job_b, "Job A and Job B must be different directories")
    if expected_sha is not None:
        require(bool(FULL_SHA.fullmatch(expected_sha)), "--expected-sha must be a full SHA-1")
    identity_a = _validate_job_identity(job_a, "job_a", expected_sha)
    identity_b = _validate_job_identity(job_b, "job_b", expected_sha)
    require(identity_a["head"] == identity_b["head"], "Job A and Job B use different SHAs")
    require(identity_a["slurm_job_id"] != identity_b["slurm_job_id"],
            "Job A and Job B must use different Slurm allocations")
    status_b = parse_key_values(job_b / "orchestration.status")
    require(status_b.get("job_a_slurm_job_id") == identity_a["slurm_job_id"],
            "Job B does not bind to the validated Job A Slurm ID")

    primary_a = _validate_primary(job_a, "01_primary", "JobA", identity_a)
    primary_b = _validate_primary(
        job_b, "01_primary_confirmation", "JobB", identity_b
    )
    _validate_final_checks(job_a, identity_a["slurm_job_id"])
    _validate_numeric(job_a, identity_a["slurm_job_id"])
    usa_rows = _validate_usa(job_a)
    eight_rows, geomeans = _validate_eight_graph(job_a)
    rows = [primary_a, primary_b, *usa_rows, *eight_rows]
    require(len(rows) == 20, "formal summary must contain exactly 20 paired measurements")

    primary_target_met = primary_a["performance_target_met"] and primary_b["performance_target_met"]
    classification = "TARGET_MET" if primary_target_met else "VALID_BELOW_TARGET"
    return {
        "schema": SCHEMA,
        "formal_sha": identity_a["head"],
        "execution_status": "PASS",
        "measurement_valid": True,
        "performance_target": TARGET,
        "performance_target_met": primary_target_met,
        "classification": classification,
        "jobs": {
            "JobA": {"slurm_job_id": identity_a["slurm_job_id"], "path": str(job_a)},
            "JobB": {"slurm_job_id": identity_b["slurm_job_id"], "path": str(job_b)},
        },
        "coverage": {
            "paired_measurements": 20,
            "primary_batches": 2,
            "fixed_usa_sources": 2,
            "eight_graph_cases": 16,
            "numeric_checked_add_runs": 10,
            "final_small_fixtures": 4,
        },
        "sampling_semantics": {
            "01_primary": {
                "outer_workflow": "formal JobA",
                "inner_sampling": "formal",
                "classification": "formal_primary",
            },
            "JobB_01_primary_confirmation": {
                "outer_workflow": "formal JobB",
                "inner_sampling": "formal",
                "classification": "formal_primary_confirmation",
            },
            "04_usa_sources": {
                "outer_workflow": "formal JobA",
                "inner_sampling": "exploratory",
                "classification": "full_sample_clean_sha_extension_regression",
            },
            "05_eight_graph": {
                "outer_workflow": "formal JobA",
                "inner_sampling": "exploratory",
                "classification": "full_sample_clean_sha_extension_regression",
            },
        },
        "eight_graph_geomeans": geomeans,
        "records": rows,
    }


def completion_matrix(summary: dict[str, Any]) -> dict[str, Any]:
    a = summary["jobs"]["JobA"]["slurm_job_id"]
    b = summary["jobs"]["JobB"]["slurm_job_id"]
    target_met = summary["performance_target_met"]
    records = [
        {"check": "same_clean_sha", "required": True, "status": "PASS",
         "target_met": None, "evidence": summary["formal_sha"]},
        {"check": "distinct_slurm_allocations", "required": True, "status": "PASS",
         "target_met": None, "evidence": f"JobA={a}; JobB={b}"},
        {"check": "job_a_primary", "required": True, "status": "PASS",
         "target_met": summary["records"][0]["performance_target_met"],
         "outer_workflow": "formal JobA", "inner_sampling": "formal",
         "classification": "formal_primary",
         "evidence": "01_primary: rc0, 10+10 formal samples, all validity gates"},
        {"check": "job_a_final_checks", "required": True, "status": "PASS",
         "target_met": None, "evidence": "4/4 fixtures, reset, and 8/8 graph correctness"},
        {"check": "job_a_numeric_checked_add", "required": True, "status": "PASS",
         "target_met": None, "evidence": "10/10 diagnostic-only checked-add runs"},
        {"check": "job_a_fixed_usa_sources", "required": True, "status": "PASS",
         "target_met": all(row["performance_target_met"] for row in summary["records"]
                           if row["category"] == "usa_source"),
         "outer_workflow": "formal JobA", "inner_sampling": "exploratory",
         "classification": "full_sample_clean_sha_extension_regression",
         "evidence": "2 sources, 8 processes, 48 queries (40 retained non-warmup)"},
        {"check": "job_a_eight_graph_regression", "required": True, "status": "PASS",
         "target_met": all(row["performance_target_met"] for row in summary["records"]
                           if row["category"] == "eight_graph"),
         "outer_workflow": "formal JobA", "inner_sampling": "exploratory",
         "classification": "full_sample_clean_sha_extension_regression",
         "evidence": "16 cases, 64 processes, 384 queries (320 retained non-warmup)"},
        {"check": "job_b_primary_confirmation", "required": True, "status": "PASS",
         "target_met": summary["records"][1]["performance_target_met"],
         "outer_workflow": "formal JobB", "inner_sampling": "formal",
         "classification": "formal_primary_confirmation",
         "evidence": "independent allocation and fresh 10+10 formal samples"},
        {"check": "performance_target_1.20", "required": False,
         "status": "TARGET_MET" if target_met else "VALID_BELOW_TARGET",
         "target_met": target_met,
         "evidence": "target outcome does not alter execution validity"},
        {"check": "formal_workflow", "required": True, "status": "PASS",
         "target_met": target_met,
         "evidence": summary["classification"]},
    ]
    return {"schema": SCHEMA, "overall_status": "PASS", "records": records}


def attempt_ledger(summary: dict[str, Any]) -> dict[str, Any]:
    records = []
    for item in HISTORICAL_ATTEMPTS:
        record = dict(item)
        record.update(HISTORICAL_ATTEMPT_DETAILS[record["job_id"]])
        record["final_performance_aggregate_allowed"] = False
        records.append(record)
    for label, role in (("JobA", "accepted_formal_job_a"),
                        ("JobB", "accepted_independent_confirmation")):
        records.append({
            "job_id": summary["jobs"][label]["slurm_job_id"],
            "commit": summary["formal_sha"],
            "event_kind": "FORMAL_WORKFLOW_SUCCESS",
            "role": role,
            "aggregate_allowed": True,
            "final_performance_aggregate_allowed": True,
            "status": "FORMAL_COMPLETE",
            "clean_worktree": True,
            "recorded_status": "COMPLETE",
            "effective_status": "FORMAL_COMPLETE",
            "failure_step": None,
            "workflow_rc": 0,
            "child_rc": 0,
            "algorithm_launched": True,
            "primary_measurement_valid": True,
            "primary_target_met": summary["performance_target_met"],
            "formal_failure_evidence_allowed": False,
            "evidence_completeness": "accepted_compact_formal_evidence",
            "last_valid_stage": "all_required_stages" if label == "JobA" else "primary_confirmation",
            "reason": (
                "all required execution and measurement-validity gates passed; "
                f"performance classification is {summary['classification']}"
            ),
        })
    return {
        "schema": SCHEMA,
        "scope": "retained L3 30-hour formal and pre-freeze attempts",
        "record_count": len(records),
        "records": records,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def _excluded_reason(path: Path, relative: Path, max_bytes: int) -> str | None:
    parts = set(relative.parts)
    suffix = path.suffix.lower()
    if path.is_symlink():
        return "symlink"
    if parts & BINARY_TREE_COMPONENTS:
        return "binary_tree"
    if parts & EXPANDED_TREE_COMPONENTS:
        return "expanded_source_or_fixture_tree"
    if suffix == ".gr":
        return "graph_input"
    if suffix in {".i32", ".oracle"}:
        return "oracle_input"
    if suffix in {".tgz", ".tar", ".gz", ".bz2", ".xz"}:
        return "source_or_bulk_archive"
    if path.stat().st_size > max_bytes:
        return "oversized_file"
    if path.name == "orchestration.status":
        return None
    if suffix in ALLOWED_SUFFIXES:
        return None
    if suffix == ".sh" and any(token in path.name for token in ("command", "invocation")):
        return None
    return "not_in_compact_allowlist"


def _record_exclusion(groups: dict[str, dict[str, Any]], reason: str,
                      relative: Path, size: int) -> None:
    group = groups.setdefault(reason, {"count": 0, "bytes": 0, "examples": []})
    group["count"] += 1
    group["bytes"] += size
    if len(group["examples"]) < 20:
        group["examples"].append(relative.as_posix())


def _gzip_n(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_stream, mtime=0) as archive:
            shutil.copyfileobj(input_stream, archive)


def compact_copy(source_root: Path, destination_root: Path, root_id: str,
                 max_bytes: int = MAX_ARCHIVE_FILE_BYTES) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    exclusions: dict[str, dict[str, Any]] = {}
    for source in sorted(source_root.rglob("*")):
        if source.is_dir() and not source.is_symlink():
            continue
        relative = source.relative_to(source_root)
        if source.is_symlink():
            _record_exclusion(exclusions, "symlink", relative, 0)
            continue
        require(source.is_file(), f"non-regular evidence entry: {source}")
        size = source.stat().st_size
        reason = _excluded_reason(source, relative, max_bytes)
        if reason is not None:
            _record_exclusion(exclusions, reason, relative, size)
            continue
        archive_relative = Path(root_id) / relative
        compressed = source.suffix.lower() == ".log"
        if compressed:
            archive_relative = archive_relative.with_name(archive_relative.name + ".gz")
        destination = destination_root / archive_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compressed:
            _gzip_n(source, destination)
            encoding = "gzip-n"
        else:
            shutil.copyfile(source, destination)
            encoding = "identity"
        copied.append({
            "source_root": root_id,
            "source_relative": relative.as_posix(),
            "archived_path": archive_relative.as_posix(),
            "encoding": encoding,
            "source_bytes": size,
            "source_sha256": sha256(source),
            "archived_bytes": destination.stat().st_size,
            "archived_sha256": sha256(destination),
        })
    return {
        "source_root": root_id,
        "copied_count": len(copied),
        "copied": copied,
        "excluded": dict(sorted(exclusions.items())),
        "policy": {
            "max_file_bytes": max_bytes,
            "allowed": "small JSON, CSV, command, environment, status, hash, text, and log evidence",
            "logs": "deterministic gzip with empty original filename and mtime=0",
            "forbidden": "binaries, graphs, oracles, expanded source/fixture trees, source archives, oversized files",
        },
    }


def _deliverables(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"delivery tree contains a symlink: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in {"archive_manifest.json", "SHA256SUMS"}:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return records


def refresh_manifests(output: Path, base_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    output = output.resolve(strict=True)
    require(output.is_dir(), f"delivery root is not a directory: {output}")
    manifest_path = output / "archive_manifest.json"
    if base_manifest is None:
        base_manifest = load_json(manifest_path)
    else:
        base_manifest = dict(base_manifest)
    for key in ("deliverables", "deliverable_count", "deliverable_bytes", "coverage_exclusions"):
        base_manifest.pop(key, None)
    deliverables = _deliverables(output)
    base_manifest.update({
        "schema": SCHEMA,
        "deliverable_count": len(deliverables),
        "deliverable_bytes": sum(item["bytes"] for item in deliverables),
        "deliverables": deliverables,
        "coverage_exclusions": {
            "archive_manifest.json": "self-reference; covered by SHA256SUMS",
            "SHA256SUMS": "checksum file cannot cover itself",
        },
    })
    temporary_manifest = manifest_path.with_name(".archive_manifest.json.tmp")
    write_json(temporary_manifest, base_manifest)
    os.replace(temporary_manifest, manifest_path)

    hash_records = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"delivery tree contains a symlink: {path}")
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        hash_records.append((path.relative_to(output).as_posix(), sha256(path)))
    temporary_sums = output / ".SHA256SUMS.tmp"
    temporary_sums.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in hash_records),
        encoding="utf-8",
    )
    os.replace(temporary_sums, output / "SHA256SUMS")
    return base_manifest


def export_formal(job_a: Path, job_b: Path, output: Path, *,
                  expected_sha: str | None = None,
                  max_bytes: int = MAX_ARCHIVE_FILE_BYTES) -> dict[str, Any]:
    job_a = job_a.resolve(strict=True)
    job_b = job_b.resolve(strict=True)
    output = output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    require(not output.is_relative_to(job_a) and not output.is_relative_to(job_b),
            "output must not be inside either source job directory")
    summary = validate_formal_inputs(job_a, job_b, expected_sha)
    matrix = completion_matrix(summary)
    ledger = attempt_ledger(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(temporary / "formal_summary.json", summary)
        write_csv(temporary / "formal_summary.csv", summary["records"])
        write_json(temporary / "completion_matrix.json", matrix)
        write_csv(temporary / "completion_matrix.csv", matrix["records"])
        write_json(temporary / "formal_attempt_ledger.json", ledger)
        write_csv(temporary / "formal_attempt_ledger.csv", ledger["records"])
        archives = [
            compact_copy(job_a, temporary, "jobA", max_bytes),
            compact_copy(job_b, temporary, "jobB", max_bytes),
        ]
        base_manifest = {
            "schema": SCHEMA,
            "formal_sha": summary["formal_sha"],
            "execution_status": "PASS",
            "performance_classification": summary["classification"],
            "source_jobs": summary["jobs"],
            "source_archive": archives,
            "refresh_contract": (
                "--refresh-manifest rewrites only archive_manifest.json and "
                "SHA256SUMS; it never overwrites raw evidence or other deliverables"
            ),
        }
        refresh_manifests(temporary, base_manifest)
        os.rename(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-a", type=Path, help="external completed Job A directory")
    parser.add_argument("--job-b", type=Path, help="external completed Job B directory")
    parser.add_argument("--out", type=Path, help="new repository evidence directory")
    parser.add_argument("--expected-sha", help="optional required full clean SHA")
    parser.add_argument("--max-file-bytes", type=int, default=MAX_ARCHIVE_FILE_BYTES)
    parser.add_argument("--validate-only", action="store_true",
                        help="validate inputs and print the summary without writing")
    parser.add_argument("--refresh-manifest", action="store_true",
                        help="only rebuild archive_manifest.json and SHA256SUMS")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.refresh_manifest:
        if args.out is None:
            parser.error("--refresh-manifest requires --out")
        if args.job_a is not None or args.job_b is not None or args.validate_only:
            parser.error("--refresh-manifest accepts only --out")
        manifest = refresh_manifests(args.out)
        print(json.dumps({
            "status": "PASS",
            "mode": "refresh-manifest",
            "output": str(args.out.resolve()),
            "deliverables": manifest["deliverable_count"],
        }, indent=2))
        return
    if args.job_a is None or args.job_b is None:
        parser.error("initial validation/export requires --job-a and --job-b")
    if args.max_file_bytes <= 0:
        parser.error("--max-file-bytes must be positive")
    if args.validate_only:
        if args.out is not None:
            parser.error("--validate-only does not accept --out")
        summary = validate_formal_inputs(args.job_a, args.job_b, args.expected_sha)
    else:
        if args.out is None:
            parser.error("initial export requires --out")
        summary = export_formal(
            args.job_a, args.job_b, args.out,
            expected_sha=args.expected_sha, max_bytes=args.max_file_bytes,
        )
    print(json.dumps({
        "status": "PASS",
        "mode": "validate-only" if args.validate_only else "export",
        "formal_sha": summary["formal_sha"],
        "execution_status": summary["execution_status"],
        "classification": summary["classification"],
        "performance_target_met": summary["performance_target_met"],
        "jobs": summary["jobs"],
    }, indent=2))


if __name__ == "__main__":
    main()
