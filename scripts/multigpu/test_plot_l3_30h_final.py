#!/usr/bin/env python3
"""CPU-only tests for the final L3 eight-graph plotting contract."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_l3_30h_final as plot  # noqa: E402


SOURCE_SHA = "a" * 40


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_rows(*, exported=True):
    rows = []
    for graph_index, graph in enumerate(plot.GRAPH_ORDER):
        for view_index, view in enumerate(plot.VIEW_ORDER):
            t2 = 10.0 + graph_index + view_index * 0.25
            speedup = 0.90 + graph_index * 0.025 + view_index * 0.04
            t1 = t2 * speedup
            wall_t2 = t2 + 3.0
            wall_speedup = speedup + 0.02
            wall_t1 = wall_t2 * wall_speedup
            row = {
                "case_id": f"{graph}_{'G' if view == 'G' else 'G_plus'}",
                "graph": graph,
                "view": view,
                "measurement_valid": "True",
                "T1_single_no_l3_median_solve_ms": repr(t1),
                "T2_dual_l3_median_solve_ms": repr(t2),
                "speedup_T1_over_T2": repr(speedup),
                "T1_single_no_l3_median_query_wall_ms": repr(wall_t1),
                "T2_dual_l3_median_query_wall_ms": repr(wall_t2),
                "query_wall_speedup_T1_over_T2": repr(wall_speedup),
                "target": "1.2",
                "target_met": str(speedup >= 1.2),
                "error_count": "0",
                "case_output": f"/formal/cases/{graph}/{view}",
            }
            if exported:
                row["source_sha"] = SOURCE_SHA
            rows.append(row)
    return rows


def write_csv(path, rows, *, exported=True):
    fields = list(plot.BASE_FIELDS)
    if exported:
        fields.append(plot.EXPORT_FIELD)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class MatrixContractTests(unittest.TestCase):
    def validated(self, rows):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "export.csv"
            write_csv(path, rows)
            return plot.load_validated_input(
                path, file_sha(path), SOURCE_SHA)

    def test_complete_export_is_typed_and_has_two_geomeans(self):
        bundle = self.validated(valid_rows())
        self.assertEqual(len(bundle["rows"]), 16)
        self.assertEqual(set(bundle["geomeans"]), {"G", "G+"})
        expected = math.exp(sum(math.log(0.90 + i * 0.025)
                                for i in range(8)) / 8)
        self.assertAlmostEqual(bundle["geomeans"]["G"], expected)
        self.assertEqual(bundle["source_binding_modes"],
                         ["csv_source_sha_column"])

    def test_partial_matrix_is_rejected_before_geomean(self):
        with self.assertRaisesRegex(plot.ContractError, "exactly 16"):
            self.validated(valid_rows()[:-1])

    def test_duplicate_view_is_rejected(self):
        rows = valid_rows()
        rows[-1] = dict(rows[-2])
        with self.assertRaisesRegex(plot.ContractError, "duplicates graph/view"):
            self.validated(rows)

    def test_invalid_measurement_is_rejected(self):
        rows = valid_rows()
        rows[3]["measurement_valid"] = "False"
        with self.assertRaisesRegex(plot.ContractError, "not measurement-valid"):
            self.validated(rows)

    def test_wrong_speedup_identity_is_rejected(self):
        rows = valid_rows()
        rows[4]["speedup_T1_over_T2"] = "9.0"
        rows[4]["target_met"] = "True"
        with self.assertRaisesRegex(plot.ContractError, "does not equal T1/T2"):
            self.validated(rows)

    def test_wrong_source_sha_is_rejected(self):
        rows = valid_rows()
        rows[7]["source_sha"] = "b" * 40
        with self.assertRaisesRegex(plot.ContractError, "differs from expected"):
            self.validated(rows)

    def test_wrong_input_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "export.csv"
            write_csv(path, valid_rows())
            with self.assertRaisesRegex(plot.ContractError, "SHA-256 differs"):
                plot.load_validated_input(path, "0" * 64, SOURCE_SHA)

    def test_unbound_raw_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            write_csv(path, valid_rows(exported=False), exported=False)
            with self.assertRaisesRegex(plot.ContractError,
                                        "source SHA is not bound"):
                plot.load_validated_input(path, file_sha(path), SOURCE_SHA)


class RenderingTests(unittest.TestCase):
    def test_outputs_have_single_column_geometry_and_auditable_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "export.csv"
            output = root / "figure"
            write_csv(source, valid_rows())
            provenance = plot.generate_outputs(
                source, output, file_sha(source), SOURCE_SHA)

            self.assertEqual(set(path.name for path in output.iterdir()),
                             set(plot.OUTPUT_NAMES))
            self.assertAlmostEqual(
                provenance["figure"]["pdf_geometry"]["width_inches"],
                3.5, places=3)
            self.assertGreaterEqual(
                provenance["figure"]["png_geometry"]["x_dpi"], 299.9)
            self.assertGreaterEqual(
                provenance["figure"]["png_geometry"]["width_pixels"], 1050)
            self.assertEqual(provenance["validation"]["case_count"], 16)
            self.assertTrue(provenance["input"]["hash_verified"])
            self.assertTrue(provenance["input"]["source_sha_verified"])

            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                metrics = list(csv.DictReader(stream))
            self.assertNotIn(b"\r", (output / "metrics.csv").read_bytes())
            self.assertEqual(len(metrics), 18)
            self.assertEqual(sum(row["row_type"] == "case" for row in metrics), 16)
            self.assertEqual(sum(row["row_type"] == "geomean" for row in metrics), 2)
            self.assertTrue(all(row["source_sha"] == SOURCE_SHA for row in metrics))

            recorded = json.loads((output / "provenance.json").read_text())
            for name, digest in recorded["output_sha256"].items():
                self.assertEqual(file_sha(output / name), digest)

    def test_validation_failure_creates_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bad.csv"
            output = root / "figure"
            rows = valid_rows()
            rows[0]["measurement_valid"] = "False"
            write_csv(source, rows)
            with self.assertRaises(plot.ContractError):
                plot.generate_outputs(
                    source, output, file_sha(source), SOURCE_SHA)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
