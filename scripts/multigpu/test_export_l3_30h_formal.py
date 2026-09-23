#!/usr/bin/env python3
"""CPU-only contract tests for the final L3 formal evidence exporter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_l3_30h_formal as exporter  # noqa: E402


SHA = "9" * 40


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _pair_section(samples, round_id=None):
    section = {"valid": True}
    for role, configuration in (
        ("T1", "single_no_l3"),
        ("T2", "dual_l3"),
    ):
        selected = [
            row for row in samples
            if row["configuration"] == configuration
            and row["warmup"] == "0"
            and (round_id is None or row["round"] == round_id)
        ]
        solve = exporter.distribution(row["solve_ms"] for row in selected)
        wall = exporter.distribution(row["query_wall_ms"] for row in selected)
        section[role] = {
            "expected_formal_samples": len(selected),
            "formal_sample_count": len(selected),
            "reason": "ok",
            "solve": solve,
            "query_wall": wall,
            "valid": True,
        }
    section["S_solve_T1_over_T2"] = (
        section["T1"]["solve"]["median_ms"]
        / section["T2"]["solve"]["median_ms"]
    )
    section["S_query_wall_T1_over_T2"] = (
        section["T1"]["query_wall"]["median_ms"]
        / section["T2"]["query_wall"]["median_ms"]
    )
    section["target"] = exporter.TARGET
    section["target_met"] = section["S_solve_T1_over_T2"] >= exporter.TARGET
    return section


def _make_pair(directory: Path):
    samples = []
    for round_id in (0, 1):
        for configuration, base in (("single_no_l3", 20.0), ("dual_l3", 18.0)):
            for repeat in range(6):
                solve = base + round_id + repeat / 10
                samples.append({
                    "algorithm": "MLMQ",
                    "configuration": configuration,
                    "correct": "1",
                    "query_wall_ms": str(solve + 10),
                    "repeat": str(repeat),
                    "round": round_id,
                    "run_valid": True,
                    "solve_ms": str(solve),
                    "warmup": "1" if repeat == 0 else "0",
                })
    combined = _pair_section(samples)
    summary = {
        "measurement_valid": True,
        "formal_integrity_valid": True,
        "formal_integrity_error": None,
        "numeric_target_met": combined["target_met"],
        "target_met": combined["target_met"],
        "sampling": "formal",
        "combined": combined,
        "rounds": [
            {"round": round_id, **_pair_section(samples, round_id)}
            for round_id in (0, 1)
        ],
    }
    _write_json(directory / "samples.json", samples)
    _write_json(directory / "summary.json", summary)
    return samples, summary


def _identity_root(root: Path, workflow: str, job_id: str, job_a_id=None):
    root.mkdir(parents=True)
    lines = [
        f"workflow={workflow}",
        "status=COMPLETE",
        "rc=0",
        f"head={SHA}",
        f"slurm_job_id={job_id}",
    ]
    if job_a_id is not None:
        lines.append(f"job_a_slurm_job_id={job_a_id}")
    (root / "orchestration.status").write_text("\n".join(lines) + "\n")
    metadata = root / "metadata"
    metadata.mkdir()
    (metadata / "head.txt").write_text(SHA + "\n")
    (metadata / "slurm.env").write_text(
        f"SLURM_JOB_ID={job_id}\nSLURM_GPUS_ON_NODE=2\n"
    )


class DistributionTests(unittest.TestCase):
    def test_inclusive_quartiles_and_mad(self):
        result = exporter.distribution(range(1, 11))
        self.assertEqual(result["count"], 10)
        self.assertEqual(result["median_ms"], 5.5)
        self.assertEqual(result["q1_ms"], 3.25)
        self.assertEqual(result["q3_ms"], 7.75)
        self.assertEqual(result["iqr_ms"], 4.5)
        self.assertEqual(result["mad_ms"], 2.5)

    def test_nonpositive_or_nonfinite_sample_is_rejected(self):
        for values in ((1, 0), (1, float("nan")), (1, float("inf"))):
            with self.subTest(values=values):
                with self.assertRaises(exporter.EvidenceError):
                    exporter.distribution(values)


class PairedSummaryTests(unittest.TestCase):
    def test_raw_samples_are_independently_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_pair(root)
            row = exporter.validate_paired_summary(
                root / "samples.json", root / "summary.json",
                evidence_id="synthetic", job="JobA", category="primary",
            )
            self.assertTrue(row["measurement_valid"])
            self.assertEqual(row["samples_per_role"], 10)
            self.assertEqual(row["formal_samples_per_role"], 10)
            self.assertEqual(row["classification"], "VALID_BELOW_TARGET")
            self.assertFalse(row["performance_target_met"])

    def test_tampered_recorded_median_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, summary = _make_pair(root)
            summary["combined"]["T1"]["solve"]["median_ms"] += 1
            _write_json(root / "summary.json", summary)
            with self.assertRaisesRegex(exporter.EvidenceError, "median_ms"):
                exporter.validate_paired_summary(
                    root / "samples.json", root / "summary.json",
                    evidence_id="tampered", job="JobA", category="primary",
                )

    def test_incorrect_raw_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, _ = _make_pair(root)
            samples[7]["correct"] = "0"
            _write_json(root / "samples.json", samples)
            with self.assertRaisesRegex(exporter.EvidenceError, "correctness-failed"):
                exporter.validate_paired_summary(
                    root / "samples.json", root / "summary.json",
                    evidence_id="bad-correctness", job="JobA", category="primary",
                )


class WorkflowIdentityTests(unittest.TestCase):
    def test_same_slurm_allocation_is_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = root / "jobA"
            b = root / "jobB"
            _identity_root(a, "job_a", "123")
            _identity_root(b, "job_b", "123", "123")
            with self.assertRaisesRegex(exporter.EvidenceError, "different Slurm"):
                exporter.validate_formal_inputs(a, b, SHA)

    def test_mismatched_clean_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = root / "jobA"
            b = root / "jobB"
            _identity_root(a, "job_a", "123")
            _identity_root(b, "job_b", "124", "123")
            b_status = (b / "orchestration.status").read_text().replace(SHA, "8" * 40)
            (b / "orchestration.status").write_text(b_status)
            (b / "metadata/head.txt").write_text("8" * 40 + "\n")
            with self.assertRaisesRegex(exporter.EvidenceError, "expected"):
                exporter.validate_formal_inputs(a, b, SHA)


class ArchiveTests(unittest.TestCase):
    def test_generated_csv_uses_lf_line_endings(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.csv"
            exporter.write_csv(output, [{"field": "value"}])
            self.assertNotIn(b"\r", output.read_bytes())

    def test_allowlist_deterministic_gzip_and_safe_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "orchestration.status").write_text("status=COMPLETE\n")
            (source / "small.json").write_text("{}\n")
            (source / "run.log").write_text("evidence log\n")
            (source / "driver.command.sh").write_text("true\n")
            (source / "graph.gr").write_bytes(b"graph")
            (source / "oracle.i32").write_bytes(b"oracle")
            (source / "program").write_bytes(b"binary")
            (source / "huge.json").write_text("x" * 128)
            (source / "binaries").mkdir()
            (source / "binaries/program.json").write_text("{}")
            (source / "single_source").mkdir()
            (source / "single_source/metadata.json").write_text("{}")

            archived = exporter.compact_copy(source, output, "jobA", max_bytes=64)
            self.assertTrue((output / "jobA/small.json").is_file())
            compressed = output / "jobA/run.log.gz"
            self.assertTrue(compressed.is_file())
            with gzip.open(compressed, "rt", encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "evidence log\n")
            self.assertEqual(compressed.read_bytes()[4:8], b"\0\0\0\0")
            reasons = archived["excluded"]
            self.assertEqual(reasons["graph_input"]["count"], 1)
            self.assertEqual(reasons["oracle_input"]["count"], 1)
            self.assertEqual(reasons["binary_tree"]["count"], 1)
            self.assertEqual(reasons["expanded_source_or_fixture_tree"]["count"], 1)
            self.assertEqual(reasons["oversized_file"]["count"], 1)

            _write_json(output / "formal_summary.json", {"status": "PASS"})
            manifest = exporter.refresh_manifests(
                output, {"formal_sha": SHA, "source_archive": [archived]}
            )
            raw_hash = exporter.sha256(output / "jobA/small.json")
            (output / "figures").mkdir()
            (output / "figures/plot.png").write_bytes(b"png")
            (output / "README.md").write_text("delivery\n")
            exporter.refresh_manifests(output)
            self.assertEqual(exporter.sha256(output / "jobA/small.json"), raw_hash)

            refreshed = json.loads((output / "archive_manifest.json").read_text())
            paths = {item["path"] for item in refreshed["deliverables"]}
            self.assertIn("figures/plot.png", paths)
            self.assertIn("README.md", paths)
            self.assertNotIn("archive_manifest.json", paths)
            self.assertNotIn("SHA256SUMS", paths)
            sums = (output / "SHA256SUMS").read_text()
            self.assertIn("  archive_manifest.json\n", sums)
            self.assertIn("  figures/plot.png\n", sums)
            self.assertNotIn("  SHA256SUMS\n", sums)
            self.assertEqual(manifest["schema"], exporter.SCHEMA)


class DeliverySemanticsTests(unittest.TestCase):
    def _summary(self):
        rows = [
            {"category": "primary", "performance_target_met": False},
            {"category": "primary", "performance_target_met": False},
            *({"category": "usa_source", "performance_target_met": False}
              for _ in range(2)),
            *({"category": "eight_graph", "performance_target_met": False}
              for _ in range(16)),
        ]
        return {
            "formal_sha": SHA,
            "classification": "VALID_BELOW_TARGET",
            "performance_target_met": False,
            "jobs": {
                "JobA": {"slurm_job_id": "38304"},
                "JobB": {"slurm_job_id": "38305"},
            },
            "records": rows,
        }

    def test_target_miss_does_not_fail_completion_matrix(self):
        matrix = exporter.completion_matrix(self._summary())
        self.assertEqual(matrix["overall_status"], "PASS")
        target = next(row for row in matrix["records"]
                      if row["check"] == "performance_target_1.20")
        self.assertEqual(target["status"], "VALID_BELOW_TARGET")
        self.assertFalse(target["required"])

    def test_attempt_ledger_has_seven_history_and_two_successes(self):
        ledger = exporter.attempt_ledger(self._summary())
        self.assertEqual(ledger["record_count"], 9)
        by_job = {row["job_id"]: row for row in ledger["records"]}
        self.assertEqual(set(by_job), {
            "38227", "38238", "38251", "38254", "38271", "38280", "38292",
            "38304", "38305",
        })
        self.assertEqual(by_job["38254"]["status"], "EXPLORATORY_RECOVERY_PASS")
        self.assertEqual(by_job["38251"]["event_kind"], "EXPLORATORY_PRELAUNCH")
        self.assertEqual(by_job["38251"]["base_head"],
                         "7cd563caaac7749521f833187ebd0b1f959b3388")
        self.assertFalse(by_job["38251"]["clean_worktree"])
        self.assertFalse(by_job["38254"]["clean_worktree"])
        self.assertEqual(by_job["38292"]["recorded_status"], "RUNNING")
        self.assertEqual(
            by_job["38292"]["effective_status"],
            "CANCELLED_INCOMPLETE_AFTER_NUMERIC_FAILURE")
        self.assertEqual(by_job["38292"]["valid_subresult_scope"],
                         "01_primary;02_final_checks")
        self.assertFalse(by_job["38254"]["aggregate_allowed"])
        self.assertFalse(by_job["38254"]["final_performance_aggregate_allowed"])
        self.assertTrue(by_job["38304"]["aggregate_allowed"])
        self.assertTrue(by_job["38305"]["aggregate_allowed"])
        self.assertTrue(by_job["38304"]["clean_worktree"])
        self.assertEqual(by_job["38305"]["workflow_rc"], 0)


if __name__ == "__main__":
    unittest.main()
