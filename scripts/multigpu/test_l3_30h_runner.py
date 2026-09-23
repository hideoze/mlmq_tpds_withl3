#!/usr/bin/env python3
"""CPU-only contract tests for the L3 30-hour paired runner."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_l3_30h as runner  # noqa: E402
import run_l3_30h_final_checks as final_checks  # noqa: E402


def _rewrite_archive_modes(archive, transform):
    rewritten = archive.with_suffix(".mode-rewrite.tgz")
    with tarfile.open(archive, "r:*") as source:
        with tarfile.open(rewritten, "w:gz") as output:
            for original in source.getmembers():
                member = copy.copy(original)
                member.mode = transform(member)
                contents = source.extractfile(original) if original.isreg() else None
                output.addfile(member, contents)
    rewritten.replace(archive)


def _integer_line(prefix, fields, values):
    return prefix + " " + " ".join(f"{field}={values[field]}" for field in fields)


def _l3_config(gpu, queue_type=runner.QUEUE_TYPE_IDS["L1SLF_L2DQ"]):
    values = {
        "gpu": gpu,
        "work_blocks": 107,
        "delta": 400000,
        "queue_type": queue_type,
        "window_mode": 2,
        "window_min": 25000,
        "window_max": 25000,
        "idle_backoff": 0,
        "worker_recovery": 1,
        "term_wait_ack": 1,
        "rx_priority_bootstrap": 1,
        "rx_express": 1,
        "rx_express_enabled": 1,
        "rx_express_slots": 8,
        "rx_express_batch": 32,
        "rx_l2_pull": 1,
        "rx_l2_pull_claim": 1,
        "rx_l2_pull_enabled": 1,
    }
    return _integer_line("L3_CONFIG", runner.L3_CONFIG_FIELDS, values)


def _worker_ack(gpu):
    values = {
        "gpu": gpu,
        "active_slots": 1712,
        "capacity": 1712,
        "work_blocks": 107,
        "warps_per_block": 16,
    }
    return _integer_line("L3_WORKER_ACK", runner.L3_WORKER_ACK_FIELDS, values)


def _l2_final(gpu, reads=11, writes=11, completed=11,
              max_bucket_writes=3,
              per_bucket_capacity=runner.FORMAL_L2_PER_BUCKET_CAPACITY,
              counter_bits=32, overflow_guard=1, overflow_detected=0,
              no_wrap=1, buckets=runner.FORMAL_L2_BUCKETS):
    values = {
        "gpu": gpu,
        "buckets": buckets,
        "reads": reads,
        "writes": writes,
        "completed": completed,
        "guarded_writes": writes,
        "max_bucket_writes": max_bucket_writes,
        "per_bucket_capacity": per_bucket_capacity,
        "total_capacity": buckets * per_bucket_capacity,
        "counter_bits": counter_bits,
        "overflow_guard": overflow_guard,
        "overflow_detected": overflow_detected,
        "no_wrap": no_wrap,
    }
    return _integer_line("L2_FINAL", runner.L2_FINAL_FIELDS, values)


def _l2_capacity(buckets=runner.FORMAL_L2_BUCKETS):
    allocated_records = runner.FORMAL_L2_ALLOCATED_RECORDS
    values = {
        "budget": runner.FORMAL_L2_BUDGET_BYTES,
        "record_bytes": runner.FORMAL_L2_RECORD_BYTES,
        "buckets": buckets,
        "per_bucket": allocated_records // buckets // 512 * 512,
        "allocated_records": allocated_records,
        "counter_bits": runner.FORMAL_L2_COUNTER_BITS,
    }
    return _integer_line("L2_CAPACITY", runner.L2_CAPACITY_FIELDS, values)


def _dual_log(*, include_l2=True, queue_type=19, bad_l2=False):
    rows = [_l2_capacity(), _l2_capacity()]
    for gpu in (0, 1):
        rows.append(_l3_config(gpu, queue_type=queue_type))
        rows.append(_worker_ack(gpu))
        if include_l2:
            if bad_l2 and gpu == 1:
                rows.append(_l2_final(gpu, reads=11, writes=12, completed=12))
            else:
                rows.append(_l2_final(gpu))
    rows.append("BENCH algorithm=MLMQ solve_ms=1 query_wall_ms=2 warmup=0")
    return "\n".join(rows) + "\n"


def _parse_dual(raw, *, require_l2_final=True):
    return runner.parse_dual_contract(
        raw,
        expected_samples=1,
        blocks=107,
        delta=400000,
        queue_type=runner.QUEUE_TYPE_IDS["L1SLF_L2DQ"],
        expected_window_mode=2,
        expected_window_min=25000,
        expected_window_max=25000,
        expected_idle_backoff=0,
        require_l2_final=require_l2_final,
    )


class DualContractTests(unittest.TestCase):
    def test_valid_formal_contract(self):
        result = _parse_dual(_dual_log())
        self.assertTrue(result["valid"], result["errors"])
        self.assertTrue(result["l2_final_required"])
        self.assertTrue(result["l2_final_present"])

    def test_formal_missing_l2_final_is_rejected(self):
        result = _parse_dual(_dual_log(include_l2=False))
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("expected two l2_final lines, got 0" in error
                for error in result["errors"]),
            result["errors"],
        )

    def test_l2_conservation_failure_is_rejected(self):
        result = _parse_dual(_dual_log(bad_l2=True))
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("L2 final gate failed" in error for error in result["errors"]),
            result["errors"],
        )

    def test_l2_no_wrap_failure_is_rejected(self):
        rows = [_l2_capacity(), _l2_capacity()]
        for gpu in (0, 1):
            rows.append(_l3_config(gpu))
            rows.append(_worker_ack(gpu))
            rows.append(_l2_final(
                gpu,
                max_bucket_writes=runner.FORMAL_L2_PER_BUCKET_CAPACITY + 1,
                no_wrap=0))
        rows.append("BENCH algorithm=MLMQ solve_ms=1 query_wall_ms=2 warmup=0")
        result = _parse_dual("\n".join(rows) + "\n")
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("no_wrap=0" in error for error in result["errors"]),
            result["errors"],
        )

    def test_wrong_queue_type_is_rejected(self):
        result = _parse_dual(_dual_log(queue_type=0))
        self.assertFalse(result["valid"])
        queue_errors = [error for error in result["errors"]
                        if "queue_type=0, expected 19" in error]
        self.assertEqual(len(queue_errors), 2, result["errors"])

    def test_missing_capacity_is_rejected(self):
        raw = "\n".join(_dual_log().splitlines()[2:]) + "\n"
        result = _parse_dual(raw)
        self.assertFalse(result["valid"])
        self.assertTrue(any("L2_CAPACITY" in error for error in result["errors"]),
                        result["errors"])

    def test_formal_wrong_capacity_bucket_count_is_rejected(self):
        wrong_buckets = runner.FORMAL_L2_BUCKETS * 2
        per_bucket_capacity = (
            runner.FORMAL_L2_ALLOCATED_RECORDS // wrong_buckets // 512 * 512)
        rows = [_l2_capacity(wrong_buckets), _l2_capacity(wrong_buckets)]
        for gpu in (0, 1):
            rows.append(_l3_config(gpu))
            rows.append(_worker_ack(gpu))
            rows.append(_l2_final(
                gpu, buckets=wrong_buckets,
                per_bucket_capacity=per_bucket_capacity))
        rows.append("BENCH algorithm=MLMQ solve_ms=1 query_wall_ms=2 warmup=0")
        result = _parse_dual("\n".join(rows) + "\n")
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                "formal L2_CAPACITY differs from the frozen exact "
                "configuration" in error
                for error in result["errors"]
            ),
            result["errors"],
        )

    def test_overflow_guard_failure_is_rejected(self):
        raw = _dual_log().replace("overflow_guard=1", "overflow_guard=0", 1)
        result = _parse_dual(raw)
        self.assertFalse(result["valid"])
        self.assertTrue(any("overflow_guard=0" in error
                            for error in result["errors"]), result["errors"])

    def test_total_capacity_must_match_allocation(self):
        total_capacity = runner.FORMAL_L2_TOTAL_CAPACITY
        raw = _dual_log().replace(
            f"total_capacity={total_capacity}",
            f"total_capacity={total_capacity - 1}", 1)
        result = _parse_dual(raw)
        self.assertFalse(result["valid"])
        self.assertTrue(any(f"total_capacity={total_capacity - 1}" in error
                            for error in result["errors"]), result["errors"])

    def test_formal_self_consistent_small_capacity_is_rejected(self):
        small_budget = 1 << 28
        allocated = small_budget // runner.FORMAL_L2_RECORD_BYTES
        per_bucket = allocated // runner.FORMAL_L2_BUCKETS // 512 * 512
        values = {
            "budget": small_budget,
            "record_bytes": runner.FORMAL_L2_RECORD_BYTES,
            "buckets": runner.FORMAL_L2_BUCKETS,
            "per_bucket": per_bucket,
            "allocated_records": allocated,
            "counter_bits": runner.FORMAL_L2_COUNTER_BITS,
        }
        capacity = _integer_line(
            "L2_CAPACITY", runner.L2_CAPACITY_FIELDS, values)
        rows = [capacity, capacity]
        for gpu in (0, 1):
            rows.extend((
                _l3_config(gpu),
                _worker_ack(gpu),
                _l2_final(gpu, per_bucket_capacity=per_bucket),
            ))
        rows.append("BENCH algorithm=MLMQ solve_ms=1 query_wall_ms=2 warmup=0")
        result = _parse_dual("\n".join(rows) + "\n")
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            "formal L2_CAPACITY differs from the frozen exact configuration"
            in error for error in result["errors"]), result["errors"])

    def test_formal_single_capacity_requires_one_frozen_row(self):
        good = runner.parse_l2_capacity_contract(
            _l2_capacity() + "\n", 1, require_frozen=True)
        self.assertTrue(good["valid"], good["errors"])
        missing = runner.parse_l2_capacity_contract(
            "", 1, require_frozen=True)
        self.assertFalse(missing["valid"])


class OracleContractTests(unittest.TestCase):
    def test_oracle_is_grouped_once_per_bench_with_one_final_audit(self):
        raw = "\n".join((
            "WIDE_ORACLE vertices=8 correct=1",
            "BENCH algorithm=MLMQ solve_ms=1",
            "WIDE_ORACLE vertices=8 correct=1",
            "BENCH algorithm=MLMQ solve_ms=2",
            "WIDE_ORACLE vertices=8 correct=1",
        ))
        result = runner.parse_oracle_contract(
            raw, expected_samples=2, expected_vertices=8)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["observed_samples"], 2)
        self.assertEqual(result["trailing_final_audit"], 1)

    def test_duplicate_and_missing_grouped_oracles_are_rejected(self):
        raw = "\n".join((
            "WIDE_ORACLE vertices=8 correct=1",
            "WIDE_ORACLE vertices=8 correct=1",
            "BENCH algorithm=MLMQ solve_ms=1",
            "BENCH algorithm=MLMQ solve_ms=2",
        ))
        result = runner.parse_oracle_contract(
            raw, expected_samples=2, expected_vertices=8)
        self.assertFalse(result["valid"])
        self.assertTrue(any("sample 0 expected one" in error
                            for error in result["errors"]), result["errors"])
        self.assertTrue(any("sample 1 expected one" in error
                            for error in result["errors"]), result["errors"])

    def test_wrong_oracle_vertex_count_or_result_is_rejected(self):
        for oracle in (
                "WIDE_ORACLE vertices=7 correct=1",
                "WIDE_ORACLE vertices=8 correct=0"):
            with self.subTest(oracle=oracle):
                result = runner.parse_oracle_contract(
                    oracle + "\nBENCH algorithm=MLMQ solve_ms=1\n",
                    expected_samples=1,
                    expected_vertices=8,
                )
                self.assertFalse(result["valid"])


class FormalPairFixture:
    """Materialize the artifact shape consumed by validate_formal_pair_build."""

    REQUIRED = (
        "mlmq", "source.tgz", "command.json", "build.log", "status.json",
        "version.json", "provenance.json",
    )

    def __init__(self, root):
        self.repo_root = Path(runner.__file__).resolve().parents[2]
        self.root = (Path(root) / "pair").resolve()
        self.root.mkdir()
        self.head = subprocess.run(
            ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        self.builder_rel = "scripts/multigpu/build_l3_paper_config.sh"
        self.builder = self.repo_root / self.builder_rel
        self.compiler = Path(sys.executable).resolve()
        self._real_git_head_file = runner._git_head_file
        self.version = {
            "repository_root": str(self.repo_root),
            "head": self.head,
            "git_status_porcelain": [],
            "head_after_build": self.head,
            "git_status_porcelain_after_build": [],
            "dual_source_post_build_match": True,
            "dual_builder": self.builder_rel,
            "dual_builder_sha256": runner.sha256(self.builder),
            "pair_builder": "scripts/multigpu/build_l3_30h_pair.sh",
            "pair_builder_sha256": runner.sha256(
                self.repo_root / "scripts/multigpu/build_l3_30h_pair.sh"),
            "single_adapter": "scripts/multigpu/prepare_l3_30h_single.py",
            "single_adapter_sha256": runner.sha256(
                self.repo_root / "scripts/multigpu/prepare_l3_30h_single.py"),
            "nvcc": str(self.compiler),
            "nvcc_resolved": str(self.compiler),
            "nvcc_sha256": runner.sha256(self.compiler),
            "boost_include_dir": str(self.repo_root),
            "build_environment": {
                "PATH": runner.FORMAL_TOOL_PATH,
                "LANG": "C",
                "LC_ALL": "C",
                "NVCC": str(self.compiler),
                "BOOST_INCLUDE_DIR": str(self.repo_root),
            },
            "blocked_build_environment_variables": list(
                runner.FORMAL_BLOCKED_BUILD_ENV),
            "blocked_build_environment_present": [],
        }
        self._write_json(self.root / "status.json", {
            "rc": 0,
            "state": "complete",
            "dual_rc": 0,
            "single_rc": 0,
        })
        self._write_json(self.root / "version.json", self.version)
        with mock.patch.object(
                runner, "_git_head_file", side_effect=self._head_file):
            self._write_source_archives()
        commands = runner.expected_pair_commands(
            self.root, self.repo_root, self.version)
        for label, role in (
                ("dual", "dual_l3_paper"),
                ("single", "single_no_l3")):
            build = self.root / f"{label}_build"
            build.mkdir(exist_ok=True)
            (build / "mlmq").write_bytes((label + " binary").encode())
            (build / "mlmq").chmod(0o755)
            self._write_json(build / "command.json", commands[label])
            (build / "build.log").write_text("build passed\n")
            self._write_json(build / "status.json", {"rc": 0, "role": role})
            self._write_json(build / "version.json", self.version)
            self._write_json(build / "provenance.json", self.valid_provenance())
            self.rehash(label)

    def _write_source_archives(self):
        dual_build = self.root / "dual_build"
        single_build = self.root / "single_build"
        dual_build.mkdir()
        single_build.mkdir()
        runner._git_archive(
            self.repo_root, dual_build / "source.tgz", "SSSP", "core")
        source = self.root / "single_source"
        runner.materialize_expected_single_tree(self.repo_root, source)
        with tarfile.open(single_build / "source.tgz", "w:gz") as stream:
            stream.add(source / "SSSP", arcname="SSSP")
            stream.add(source / "core", arcname="core")

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def valid_provenance():
        return {
            "pair_contract": {
                "dual": "current dual-GPU L3 paper configuration",
                "single": "independent committed no-L3 single-GPU source",
                "single_is_dual_n1": False,
            },
            "dual": {
                "builder": "scripts/multigpu/build_l3_paper_config.sh",
                "macro_authority": "builder command.json",
            },
            "single": {
                "snapshot_materialization": "git archive HEAD (ignores working-tree edits)",
                "post_sync_check": "byte-for-byte recursive diff passed",
                "l3_compile_defines": [],
            },
        }

    def rehash(self, label):
        build = self.root / f"{label}_build"
        hashes = {name: runner.sha256(build / name) for name in self.REQUIRED}
        self._write_json(build / "hashes.json", hashes)

    def rewrite_pair_version(self):
        self._write_json(self.root / "version.json", self.version)

    @property
    def dual_binary(self):
        return (self.root / "dual_build/mlmq").resolve()

    @property
    def single_binary(self):
        return (self.root / "single_build/mlmq").resolve()

    def validate(self):
        with mock.patch.object(
                runner, "_git_head_file", side_effect=self._head_file):
            return runner.validate_formal_pair_build(
                self.root,
                self.dual_binary,
                self.single_binary,
                self.head,
                self.repo_root,
            )

    def _head_file(self, repo_root, relative):
        if relative == Path("scripts/multigpu/prepare_l3_30h_single.py"):
            return (self.repo_root / relative).read_bytes()
        return self._real_git_head_file(repo_root, relative)


class FormalPairBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = FormalPairFixture(self.temporary.name)

    def test_clean_head_provenance_and_hash_fixture_passes(self):
        result = self.fixture.validate()
        self.assertEqual(result["status"]["state"], "complete")
        self.assertEqual(result["version"]["head"], self.fixture.head)
        self.assertEqual(set(result["components"]), {"dual", "single"})
        for component in result["components"].values():
            self.assertEqual(
                set(component["verified_sha256"]),
                set(FormalPairFixture.REQUIRED),
            )

    def test_dirty_pair_build_snapshot_is_rejected(self):
        self.fixture.version["git_status_porcelain"] = [" M SSSP/sssp.cuh"]
        self.fixture.rewrite_pair_version()
        with self.assertRaisesRegex(SystemExit, "dirty worktree"):
            self.fixture.validate()

    def test_changed_pair_build_head_is_rejected(self):
        self.fixture.version["head_after_build"] = "0" * 40
        self.fixture.rewrite_pair_version()
        with self.assertRaisesRegex(SystemExit, "HEAD changed"):
            self.fixture.validate()

    def test_mutated_provenance_hash_is_rejected(self):
        provenance = self.fixture.root / "single_build/provenance.json"
        provenance.write_text("{}\n")
        with self.assertRaisesRegex(SystemExit, "hash mismatch for provenance.json"):
            self.fixture.validate()

    def test_semantically_invalid_but_rehashed_provenance_is_rejected(self):
        provenance = self.fixture.root / "single_build/provenance.json"
        provenance.write_text("{}\n")
        self.fixture.rehash("single")
        with self.assertRaisesRegex(SystemExit, "provenance"):
            self.fixture.validate()

    def test_implicit_window_defaults_are_rejected(self):
        command = self.fixture.root / "dual_build/command.json"
        values = json.loads(command.read_text())
        values.remove("-DL3_WINDOW_MIN_CYCLES=25000ull")
        self.fixture._write_json(command, values)
        self.fixture.rehash("dual")
        with self.assertRaisesRegex(SystemExit, "differs from canonical argv"):
            self.fixture.validate()

    def test_single_must_enable_the_shared_overflow_guard(self):
        command = self.fixture.root / "single_build/command.json"
        values = json.loads(command.read_text())
        values.remove("-DDQ_COUNTER_OVERFLOW_GUARD=true")
        self.fixture._write_json(command, values)
        self.fixture.rehash("single")
        with self.assertRaisesRegex(SystemExit, "differs from canonical argv"):
            self.fixture.validate()

    def test_single_and_dual_use_the_frozen_l2_geometry(self):
        commands = runner.expected_pair_commands(
            self.fixture.root, self.fixture.repo_root, self.fixture.version)
        for label in ("dual", "single"):
            with self.subTest(label=label):
                definitions = runner.parse_compile_definitions(
                    commands[label], label)
                self.assertEqual(definitions["BNUM"],
                                 str(runner.FORMAL_L2_BUCKETS))
                self.assertEqual(definitions["BUCKET_MAX"],
                                 str(runner.FORMAL_L2_BUCKET_MAX))
                self.assertEqual(definitions["l2_batch_size"],
                                 str(runner.FORMAL_L2_BATCH_SIZE))

    def test_compiler_influencing_environment_is_rejected(self):
        self.fixture.version["build_environment"]["CPATH"] = "/tmp/injected"
        self.fixture.rewrite_pair_version()
        with self.assertRaisesRegex(SystemExit, "environment keys differ"):
            self.fixture.validate()

    def test_component_version_must_equal_pair_version(self):
        path = self.fixture.root / "single_build/version.json"
        version = json.loads(path.read_text())
        version["head"] = "0" * 40
        self.fixture._write_json(path, version)
        self.fixture.rehash("single")
        with self.assertRaisesRegex(SystemExit, "component version differs"):
            self.fixture.validate()

    def test_rehashed_extra_dual_source_file_is_rejected(self):
        archive = self.fixture.root / "dual_build/source.tgz"
        expanded = Path(self.temporary.name) / "expanded"
        runner.safe_extract_regular_archive(archive, expanded)
        (expanded / "SSSP/untracked.cu").write_text("// injected\n")
        rewritten = archive.with_suffix(".new.tgz")
        with tarfile.open(rewritten, "w:gz") as stream:
            stream.add(expanded / "SSSP", arcname="SSSP")
            stream.add(expanded / "core", arcname="core")
        rewritten.replace(archive)
        self.fixture.rehash("dual")
        with self.assertRaisesRegex(SystemExit, "differs from clean HEAD"):
            self.fixture.validate()

    def test_non_git_permission_differences_are_accepted(self):
        archive = self.fixture.root / "dual_build/source.tgz"
        _rewrite_archive_modes(
            archive,
            lambda member: (0o700 if member.isdir() else
                            (0o700 if member.mode & 0o111 else 0o600)),
        )
        self.fixture.rehash("dual")
        result = self.fixture.validate()
        self.assertEqual(result["status"]["state"], "complete")

    def test_rehashed_executable_bit_change_is_rejected(self):
        archive = self.fixture.root / "dual_build/source.tgz"

        def change_makefile_mode(member):
            if member.name.rstrip("/") == "SSSP/Makefile":
                return member.mode | 0o100
            return member.mode

        _rewrite_archive_modes(archive, change_makefile_mode)
        self.fixture.rehash("dual")
        with self.assertRaisesRegex(SystemExit, "differs from clean HEAD"):
            self.fixture.validate()

    def test_alternate_or_duplicate_define_syntax_is_rejected(self):
        for command in (
                ["nvcc", "-D", "WORK_COUNT=false"],
                ["nvcc", "--define-macro=WORK_COUNT=false"],
                ["nvcc", "-DWORK_COUNT=false", "-DWORK_COUNT=true"]):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit):
                    runner.parse_compile_definitions(command, "test")


class SafeArchiveTests(unittest.TestCase):
    def test_git_semantic_tree_ignores_untracked_permission_bits(self):
        digest = "a" * 64
        expected = {
            "SSSP": {"type": "dir", "mode": 0o775},
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o664, "size": 7,
                "sha256": digest,
            },
            "SSSP/tool.sh": {
                "type": "file", "mode": 0o775, "size": 9,
                "sha256": digest,
            },
        }
        actual = {
            "SSSP": {"type": "dir", "mode": 0o700},
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o600, "size": 7,
                "sha256": digest,
            },
            "SSSP/tool.sh": {
                "type": "file", "mode": 0o755, "size": 9,
                "sha256": digest,
            },
        }
        self.assertEqual(
            runner.git_semantic_source_tree(actual),
            runner.git_semantic_source_tree(expected),
        )

    def test_git_semantic_tree_rejects_executable_bit_change(self):
        digest = "b" * 64
        expected = {
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o664, "size": 7,
                "sha256": digest,
            },
        }
        actual = {
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o755, "size": 7,
                "sha256": digest,
            },
        }
        self.assertNotEqual(
            runner.git_semantic_source_tree(actual),
            runner.git_semantic_source_tree(expected),
        )

    def test_git_semantic_tree_keeps_content_identity_strict(self):
        expected = {
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o664, "size": 7,
                "sha256": "a" * 64,
            },
        }
        actual = {
            "SSSP/plain.cu": {
                "type": "file", "mode": 0o600, "size": 7,
                "sha256": "b" * 64,
            },
        }
        self.assertNotEqual(
            runner.git_semantic_source_tree(actual),
            runner.git_semantic_source_tree(expected),
        )

    def test_symlink_traversal_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "attack.tar"
            outside = root / "outside"
            outside.mkdir()
            with tarfile.open(archive, "w") as stream:
                directory = tarfile.TarInfo("SSSP")
                directory.type = tarfile.DIRTYPE
                directory.mode = 0o755
                stream.addfile(directory)
                link = tarfile.TarInfo("SSSP/link")
                link.type = tarfile.SYMTYPE
                link.linkname = str(outside)
                stream.addfile(link)
            destination = root / "extract"
            with self.assertRaisesRegex(SystemExit, "forbidden non-regular"):
                runner.safe_extract_regular_archive(archive, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(outside.iterdir()), [])

    def test_final_checker_rejects_link_archive_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "final-check-attack.tar"
            with tarfile.open(archive, "w") as stream:
                directory = tarfile.TarInfo("SSSP")
                directory.type = tarfile.DIRTYPE
                stream.addfile(directory)
                link = tarfile.TarInfo("SSSP/link")
                link.type = tarfile.LNKTYPE
                link.linkname = "../../outside"
                stream.addfile(link)
            destination = root / "extract"
            with self.assertRaisesRegex(SystemExit, "forbidden non-regular"):
                final_checks.safe_extract(archive, destination)
            self.assertFalse(destination.exists())

    def test_final_checker_materializes_relative_fixture_in_archived_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            (snapshot / "SSSP").mkdir(parents=True)
            (snapshot / "core/include").mkdir(parents=True)
            (snapshot / "SSSP/sentinel").write_bytes(b"sssp")
            (snapshot / "core/include/sentinel").write_bytes(b"core")
            fixture_root, fixtures, tree = final_checks.materialize_fixture_sources(
                snapshot, root / "output", {
                    Path("scripts/multigpu/test_l3_supplement.cu"): b"fixture",
                })
            fixture = fixtures["scripts/multigpu/test_l3_supplement.cu"]
            self.assertEqual(fixture.read_bytes(), b"fixture")
            self.assertEqual((fixture_root / "SSSP/sentinel").read_bytes(), b"sssp")
            self.assertEqual(
                (fixture_root / "core/include/sentinel").read_bytes(), b"core")
            self.assertFalse((snapshot / "scripts").exists())
            self.assertEqual(tree["archived_path_count"], 5)

    def test_final_checker_rejects_unsafe_fixture_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            with self.assertRaisesRegex(ValueError, "unsafe fixture source path"):
                final_checks.materialize_fixture_sources(
                    snapshot, root / "output", {Path("../escape.cu"): b"fixture"})

    def test_path_traversal_and_duplicate_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, members, pattern in (
                    ("traversal.tar", ["../escape"], "unsafe archive member"),
                    ("duplicate.tar", ["SSSP", "SSSP"],
                     "duplicate source archive member")):
                archive = root / name
                with tarfile.open(archive, "w") as stream:
                    for member_name in members:
                        item = tarfile.TarInfo(member_name)
                        item.type = tarfile.DIRTYPE
                        item.mode = 0o755
                        stream.addfile(item)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(SystemExit, pattern):
                        runner.read_regular_archive_tree(archive)


class FormalGpuContractTests(unittest.TestCase):
    ROWS = (
        "0, GPU-a, NVIDIA A100 80GB PCIe, 8.0\n"
        "1, GPU-b, NVIDIA A100 80GB PCIe, 8.0\n"
    )

    def test_formal_gpu_contract_accepts_two_a100_cc80_on_a100_partition(self):
        with mock.patch.dict(
                runner.os.environ, {"SLURM_JOB_PARTITION": "a100"}, clear=True):
            rows = runner.parse_gpu_query(self.ROWS, require_a100=True)
        self.assertEqual([row["compute_capability"] for row in rows], ["8.0", "8.0"])

    def test_formal_gpu_contract_rejects_wrong_partition_or_architecture(self):
        with mock.patch.dict(
                runner.os.environ, {"SLURM_JOB_PARTITION": "debug"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_PARTITION=a100"):
                runner.parse_gpu_query(self.ROWS, require_a100=True)
        wrong_gpu = self.ROWS.replace("A100", "H100", 1)
        with mock.patch.dict(
                runner.os.environ, {"SLURM_JOB_PARTITION": "a100"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "A100 CC 8.0"):
                runner.parse_gpu_query(wrong_gpu, require_a100=True)

    def test_gpu_contract_rejects_duplicate_or_malformed_identity(self):
        duplicate_index = self.ROWS.replace("1, GPU-b", "0, GPU-b")
        with self.assertRaisesRegex(RuntimeError, "indices are not unique"):
            runner.parse_gpu_query(duplicate_index)
        duplicate_uuid = self.ROWS.replace("GPU-b", "GPU-a")
        with self.assertRaisesRegex(RuntimeError, "UUIDs are not unique"):
            runner.parse_gpu_query(duplicate_uuid)
        malformed_uuid = self.ROWS.replace("GPU-b", "MIG-b")
        with self.assertRaisesRegex(RuntimeError, "malformed GPU UUID"):
            runner.parse_gpu_query(malformed_uuid)


class SummaryContractTests(unittest.TestCase):
    @staticmethod
    def _entry(solve_ms):
        return {
            "valid": True,
            "solve": {"median_ms": solve_ms},
            "query_wall": {"median_ms": solve_ms + 1},
        }

    def test_measurement_valid_is_independent_of_target_met(self):
        paired = runner.paired_summary(self._entry(100), self._entry(90))
        target_met = paired["S_solve_T1_over_T2"] >= runner.TARGET_SPEEDUP
        self.assertTrue(runner.measurement_is_valid(
            [{"valid": True}, {"valid": True}], paired))
        self.assertFalse(target_met)

    def test_invalid_process_cannot_be_hidden_by_valid_combined_summary(self):
        paired = runner.paired_summary(self._entry(120), self._entry(90))
        self.assertGreaterEqual(
            paired["S_solve_T1_over_T2"], runner.TARGET_SPEEDUP)
        self.assertFalse(runner.measurement_is_valid(
            [{"valid": True}, {"valid": False}], paired))


class FormalOutputPathTests(unittest.TestCase):
    def test_formal_output_inside_repository_is_rejected_before_sampling(self):
        repo_root = Path(runner.__file__).resolve().parents[2]
        args = SimpleNamespace(
            sampling="formal",
            out=repo_root / "must-not-create-formal-output-here",
        )
        clean_git = {
            "head": {"rc": 0, "output": "a" * 40},
            "status": {"rc": 0, "output": ""},
        }
        with mock.patch.object(runner, "parse_args", return_value=args), \
                mock.patch.object(runner, "git_snapshot", return_value=clean_git):
            with self.assertRaisesRegex(
                    SystemExit, "formal output must be outside the repository"):
                runner.main()
        self.assertFalse(args.out.exists())


class FormalCliTests(unittest.TestCase):
    def test_formal_cli_rejects_all_external_pair_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            arguments = [
                "run_l3_30h.py", "--sampling", "formal",
                "--dual-binary", "/tmp/dual", "--single-binary", "/tmp/single",
                "--pair-build", "/tmp/pair", "--graph", "/tmp/graph",
                "--oracle", "/tmp/oracle", "--source", "0", "--delta", "1",
                "--cut-percent", "50", "--blocks", "1", "--warmups", "1",
                "--repeats", "5", "--rounds", "2", "--timeout", "1",
                "--out", str(Path(temporary) / "out"),
                "--expected-window-mode", "2", "--expected-window-min", "1",
                "--expected-window-max", "1", "--expected-idle-backoff", "0",
            ]
            with mock.patch.object(sys, "argv", arguments), \
                    mock.patch.dict(runner.os.environ, {"SLURM_JOB_ID": "1"},
                                    clear=True), \
                    self.assertRaises(SystemExit) as raised:
                runner.parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_pair_builder_rejects_even_empty_influencing_environment(self):
        builder = SCRIPT_DIR / "build_l3_30h_pair.sh"
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "PATH": runner.FORMAL_TOOL_PATH,
                "LANG": "C",
                "LC_ALL": "C",
                "CFLAGS": "",
            }
            process = subprocess.run(
                [str(builder), str(Path(temporary) / "pair")],
                env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=30)
            self.assertEqual(process.returncode, 2, process.stdout)
            self.assertIn(
                "compiler-influencing environment variable is forbidden: CFLAGS",
                process.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
