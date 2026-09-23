#!/usr/bin/env python3
"""CPU-only tests for the L3 checked candidate-add diagnostic runner."""

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_l3_30h_numeric_checks as numeric  # noqa: E402


def add_tar_member(stream, name, *, kind="file", data=b"x", linkname=""):
    info = tarfile.TarInfo(name)
    if kind == "file":
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
    elif kind == "dir":
        info.type = tarfile.DIRTYPE
        stream.addfile(info)
    elif kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = linkname
        stream.addfile(info)
    elif kind == "hardlink":
        info.type = tarfile.LNKTYPE
        info.linkname = linkname
        stream.addfile(info)
    elif kind == "fifo":
        info.type = tarfile.FIFOTYPE
        stream.addfile(info)
    elif kind == "char_device":
        info.type = tarfile.CHRTYPE
        info.devmajor = 1
        info.devminor = 3
        stream.addfile(info)
    else:
        raise ValueError(kind)


def minimal_source_tree(archive):
    with tarfile.open(archive, "w:gz") as stream:
        for directory in ("SSSP", "core", "core/include"):
            add_tar_member(stream, directory, kind="dir")
        for name in ("SSSP/main.cu", "SSSP/csr_graph.cu",
                     "SSSP/sssp_run.cu", "core/include/common.h"):
            add_tar_member(stream, name, data=name.encode())


def integer_line(prefix, fields, values):
    return prefix + " " + " ".join(
        f"{field}={values[field]}" for field in fields)


def dual_log(*, marker=None, gpu_count=2):
    lines = ["GPU0 partition: [0, 60)", "GPU1 partition: [60, 100)"]
    capacity = {
        "budget": 2147483647, "record_bytes": 8, "buckets": 16,
        "per_bucket": 16776704, "allocated_records": 268435455,
        "counter_bits": 32,
    }
    lines.extend(integer_line(
        "L2_CAPACITY", numeric.L2_CAPACITY_FIELDS, capacity) for _ in range(2))
    for gpu in (0, 1):
        config = {
            "gpu": gpu, "work_blocks": 107, "delta": 400000,
            "queue_type": 19, "window_mode": 2, "window_min": 25000,
            "window_max": 25000, "idle_backoff": 0,
            "worker_recovery": 1, "term_wait_ack": 1,
            "rx_priority_bootstrap": 0, "rx_express": 0,
            "rx_express_enabled": 0, "rx_express_slots": 64,
            "rx_express_batch": 32, "rx_l2_pull": 0,
            "rx_l2_pull_claim": 0, "rx_l2_pull_enabled": 0,
        }
        ack = {"gpu": gpu, "active_slots": 1712, "capacity": 1712,
               "work_blocks": 107, "warps_per_block": 16}
        final = {"gpu": gpu, "buckets": 16, "reads": 10,
                 "writes": 10, "completed": 10, "guarded_writes": 10,
                 "max_bucket_writes": 2,
                 "per_bucket_capacity": capacity["per_bucket"],
                 "total_capacity": (capacity["buckets"] *
                                    capacity["per_bucket"]),
                 "counter_bits": 32, "overflow_guard": 1,
                 "overflow_detected": 0, "no_wrap": 1}
        lines.append(integer_line("L3_CONFIG", numeric.L3_CONFIG_FIELDS, config))
        lines.append(integer_line("L3_WORKER_ACK", numeric.L3_WORKER_ACK_FIELDS, ack))
        lines.append(integer_line("L2_FINAL", numeric.L2_FINAL_FIELDS, final))
    if marker:
        lines.append(marker)
    lines.extend((
        "WIDE_ORACLE vertices=100 correct=1",
        f"BENCH algorithm=MLMQ gpu_count={gpu_count} source=7 repeat=0 "
        "warmup=0 queue=L1SLF_L2DQ correct=1 solve_ms=1 query_wall_ms=2",
    ))
    return "\n".join(lines) + "\n"


def single_log():
    return "\n".join((
        "NO_L3_CONFIG workers=512 delta=400000 queue=L1SLF_L2DQ",
        "NO_L3_LAUNCH work_blocks=107 delta=400000 repeat=0",
        "WIDE_ORACLE vertices=100 correct=1",
        "BENCH algorithm=MLMQ gpu_count=1 source=7 repeat=0 warmup=0 "
        "queue=L1SLF_L2DQ correct=1 solve_ms=1 query_wall_ms=2",
        "WIDE_ORACLE vertices=100 correct=1",
    )) + "\n"


class SafeExtractionTests(unittest.TestCase):
    def test_regular_tree_is_extracted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "source.tgz"
            minimal_source_tree(archive)
            result = numeric.safe_extract_source(archive, root / "source")
            self.assertEqual(result["members"], 7)
            self.assertTrue((root / "source/SSSP/main.cu").is_file())

    def test_malicious_tar_members_are_rejected(self):
        cases = (
            ("../escape", "file", ""),
            ("SSSP/link", "symlink", "../../escape"),
            ("SSSP/hard", "hardlink", "SSSP/main.cu"),
            ("SSSP/pipe", "fifo", ""),
            ("SSSP/device", "char_device", ""),
        )
        for name, kind, linkname in cases:
            with self.subTest(name=name, kind=kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    archive = root / "bad.tgz"
                    with tarfile.open(archive, "w:gz") as stream:
                        add_tar_member(
                            stream, name, kind=kind, linkname=linkname)
                    with self.assertRaises(RuntimeError):
                        numeric.safe_extract_source(archive, root / "out")
                    self.assertFalse((root / "escape").exists())


class SourcePatchTests(unittest.TestCase):
    def source_text(self, replacements):
        return (numeric.HELPER_ANCHOR + "\n" +
                "\n".join(old for old, _new, _description in replacements) +
                "\n")

    def test_dual_and_single_anchor_counts_are_exact(self):
        for role, replacements in (
                ("dual", numeric.DUAL_REPLACEMENTS),
                ("single", numeric.SINGLE_REPLACEMENTS)):
            with self.subTest(role=role):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / "source/SSSP").mkdir(parents=True)
                    path = root / "source/SSSP/sssp_run.cu"
                    path.write_text(self.source_text(replacements))
                    result = numeric.patch_numeric_source(
                        root / "source", role, root)
                    self.assertEqual(result["checked_add_call_count"], 2)
                    self.assertEqual(
                        path.read_text().count(
                            "diagnostic_checked_candidate_add("), 3)

    def test_duplicate_anchor_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source/SSSP").mkdir(parents=True)
            text = self.source_text(numeric.SINGLE_REPLACEMENTS)
            text += numeric.SINGLE_REPLACEMENTS[0][0] + "\n"
            (root / "source/SSSP/sssp_run.cu").write_text(text)
            with self.assertRaises(RuntimeError):
                numeric.patch_numeric_source(root / "source", "single", root)


class CommandTranslationTests(unittest.TestCase):
    def command(self):
        return [
            "nvcc", "/formal/SSSP/main.cu", "/formal/SSSP/csr_graph.cu",
            "/formal/SSSP/sssp_run.cu", "-o", "/formal/mlmq",
            "-DWORK_COUNT=false", "-O3", "-I/formal/core/include", "-lcusparse",
        ]

    def test_only_paths_output_and_one_macro_are_changed(self):
        command, evidence = numeric.derive_compile_command(
            self.command(), Path("/derived"), Path("/out/mlmq"), role="dual")
        self.assertEqual(command.count(numeric.DIAGNOSTIC_DEFINE), 1)
        self.assertEqual(evidence["diagnostic_define_count"], 1)
        self.assertIn("/derived/SSSP/sssp_run.cu", command)
        self.assertIn("-I/derived/core/include", command)
        self.assertNotIn("/formal/mlmq", command)
        self.assertIn("-DWORK_COUNT=false", command)

    def test_existing_diagnostic_macro_is_rejected(self):
        command = self.command() + [numeric.DIAGNOSTIC_DEFINE]
        with self.assertRaises(RuntimeError):
            numeric.derive_compile_command(
                command, Path("/derived"), Path("/out/mlmq"), role="dual")

    def test_default_scope_override_is_rejected(self):
        command = self.command() + ["-DGHOST_DEPTH=1"]
        with self.assertRaises(RuntimeError):
            numeric.derive_compile_command(
                command, Path("/derived"), Path("/out/mlmq"), role="dual")


class RunParserTests(unittest.TestCase):
    def test_valid_single_and_dual_logs(self):
        single = numeric.parse_numeric_run(
            single_log(), role="single", vertices=100, source=7,
            blocks=107, delta=400000, cut_percent=60)
        dual = numeric.parse_numeric_run(
            dual_log(), role="dual", vertices=100, source=7,
            blocks=107, delta=400000, cut_percent=60)
        self.assertTrue(single["valid"], single["errors"])
        self.assertTrue(dual["valid"], dual["errors"])

    def test_failure_marker_and_wrong_gpu_count_are_rejected(self):
        result = numeric.parse_numeric_run(
            dual_log(marker="NUMERIC_ADD_FAIL site=1 lhs=1 rhs=2 sum=3",
                     gpu_count=1),
            role="dual", vertices=100, source=7, blocks=107,
            delta=400000, cut_percent=60)
        self.assertFalse(result["valid"])
        self.assertTrue(result["failure_markers"])


if __name__ == "__main__":
    unittest.main()
