#!/usr/bin/env python3
"""CPU-only contract tests for the fixed additional-USA-source wrapper."""

from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_l3_30h as runner  # noqa: E402
import run_l3_30h_usa_sources as usa  # noqa: E402


def integer_line(prefix, fields, values):
    return prefix + " " + " ".join(f"{field}={values[field]}" for field in fields)


def capacity():
    return (
        f"L2_CAPACITY budget={runner.FORMAL_L2_BUDGET_BYTES} "
        f"record_bytes={runner.FORMAL_L2_RECORD_BYTES} "
        f"buckets={runner.FORMAL_L2_BUCKETS} "
        f"per_bucket={runner.FORMAL_L2_PER_BUCKET_CAPACITY} "
        f"allocated_records={runner.FORMAL_L2_ALLOCATED_RECORDS} "
        f"counter_bits={runner.FORMAL_L2_COUNTER_BITS}"
    )


def l3(gpu, *, rx_express=0):
    values = {
        "gpu": gpu, "work_blocks": 107, "delta": 400000, "queue_type": 19,
        "window_mode": 2, "window_min": 25000, "window_max": 25000,
        "idle_backoff": 0, "worker_recovery": 1, "term_wait_ack": 1,
        "rx_priority_bootstrap": 0, "rx_express": rx_express,
        "rx_express_enabled": 0, "rx_express_slots": 64,
        "rx_express_batch": 32, "rx_l2_pull": 0, "rx_l2_pull_claim": 0,
        "rx_l2_pull_enabled": 0,
    }
    return integer_line("L3_CONFIG", runner.L3_CONFIG_FIELDS, values)


def ack(gpu):
    values = {
        "gpu": gpu, "active_slots": 1712, "capacity": 1712,
        "work_blocks": 107, "warps_per_block": 16,
    }
    return integer_line("L3_WORKER_ACK", runner.L3_WORKER_ACK_FIELDS, values)


def l2(gpu, *, total_capacity=None):
    if total_capacity is None:
        total_capacity = runner.FORMAL_L2_TOTAL_CAPACITY
    values = {
        "gpu": gpu, "buckets": runner.FORMAL_L2_BUCKETS,
        "reads": 11, "writes": 11,
        "completed": 11, "guarded_writes": 11, "max_bucket_writes": 3,
        "per_bucket_capacity": runner.FORMAL_L2_PER_BUCKET_CAPACITY,
        "total_capacity": total_capacity,
        "counter_bits": 32, "overflow_guard": 1, "overflow_detected": 0,
        "no_wrap": 1,
    }
    return integer_line("L2_FINAL", runner.L2_FINAL_FIELDS, values)


def bench(gpus, source, repeat):
    return (
        f"BENCH algorithm=MLMQ gpu_count={gpus} source={source} repeat={repeat} "
        f"warmup={int(repeat == 0)} queue=L1SLF_L2DQ solve_ms={10 + repeat}.0 "
        f"query_wall_ms={20 + repeat}.0 correct=1"
    )


def valid_log(gpus, *, source=7, vertices=1000, bad_total=False,
              rx_express=0):
    total_capacity = runner.FORMAL_L2_TOTAL_CAPACITY
    rows = [capacity()] * gpus
    if gpus == 2:
        rows += ["GPU0 partition: [0, 600)", "GPU1 partition: [600, 1000)"]
    else:
        rows.append("NO_L3_CONFIG workers=512 delta=400000 queue=L1SLF_L2DQ")
    for repeat in range(usa.PROCESS_SAMPLES):
        if gpus == 2:
            for gpu in (0, 1):
                rows += [
                    l3(gpu, rx_express=rx_express), ack(gpu),
                    l2(gpu, total_capacity=(total_capacity - 1)
                       if bad_total and gpu == 1 else total_capacity),
                ]
            rows.append(
                "FINAL_AUDIT mismatches=0 residual_edges=0 cross_residual_edges=0"
            )
        else:
            rows.append(
                f"NO_L3_LAUNCH work_blocks=107 delta=400000 repeat={repeat}"
            )
        rows += [
            f"WIDE_ORACLE vertices={vertices} correct=1",
            bench(gpus, source, repeat),
        ]
    if gpus == 1:
        rows.append(f"WIDE_ORACLE vertices={vertices} correct=1")
    rows += ["", "RUN_RC=0", "HOST_ELAPSED_SECONDS=1.0"]
    return "\n".join(rows) + "\n"


class RawEvidenceTests(unittest.TestCase):
    def parse(self, raw, gpus):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(raw)
            return usa.validate_raw_log(
                path, gpu_count=gpus, source=7, vertices=1000)

    def test_valid_single_and_dual_logs(self):
        for gpus in (1, 2):
            with self.subTest(gpus=gpus):
                result = self.parse(valid_log(gpus), gpus)
                self.assertTrue(result["valid"], result["errors"])

    def test_total_capacity_mismatch_is_rejected(self):
        result = self.parse(valid_log(2, bad_total=True), 2)
        self.assertFalse(result["valid"])
        self.assertTrue(any("total_capacity" in error
                            for error in result["errors"]), result["errors"])

    def test_noncanonical_rx_mode_is_rejected(self):
        result = self.parse(valid_log(2, rx_express=1), 2)
        self.assertFalse(result["valid"])
        self.assertTrue(any("RX config" in error
                            for error in result["errors"]), result["errors"])

    def test_single_wrong_capacity_is_rejected(self):
        raw = valid_log(1).replace(
            f"buckets={runner.FORMAL_L2_BUCKETS}", "buckets=16", 1)
        result = self.parse(raw, 1)
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            "frozen exact configuration" in error
            for error in result["errors"]), result["errors"])


class CommandTests(unittest.TestCase):
    def test_fixed_sampling_shape_and_no_formal_rebuild_per_source(self):
        source = {
            "source": 7, "graph": Path("/graph"), "oracle": Path("/oracle"),
        }
        command = usa.source_command(
            source, Path("/pair"), Path("/outside/source"), 400)
        self.assertIn("exploratory", command)
        self.assertEqual(command[command.index("--rounds") + 1], "2")
        self.assertEqual(command[command.index("--warmups") + 1], "1")
        self.assertEqual(command[command.index("--repeats") + 1], "5")
        self.assertIn("--pair-build", command)
        self.assertNotIn("formal", command)


if __name__ == "__main__":
    unittest.main()
