#!/usr/bin/env python3
"""Render the final 30-hour L3 eight-graph result from pinned evidence.

The input is either the original Job A ``05_eight_graph/summary.csv`` or an
equivalent exported CSV.  Original Job A input is cross-checked against its
evidence sidecars.  A standalone export must add one constant ``source_sha``
column (or be accompanied by ``--source-metadata``).  In both modes callers
must pin the exact CSV SHA-256 and the formal 40-hex source commit.

Only a complete, valid 8-graph x {G, G+} matrix is renderable.  This is
intentional: a failed or partial batch must not produce a paper figure.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
from typing import Any


GRAPH_ORDER = ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA")
VIEW_ORDER = ("G", "G+")
TARGET_SPEEDUP = 1.20
FIGURE_WIDTH_IN = 3.5
FIGURE_HEIGHT_IN = 2.55
PNG_DPI = 400

BASE_FIELDS = (
    "case_id",
    "graph",
    "view",
    "measurement_valid",
    "T1_single_no_l3_median_solve_ms",
    "T2_dual_l3_median_solve_ms",
    "speedup_T1_over_T2",
    "T1_single_no_l3_median_query_wall_ms",
    "T2_dual_l3_median_query_wall_ms",
    "query_wall_speedup_T1_over_T2",
    "target",
    "target_met",
    "error_count",
    "case_output",
)
EXPORT_FIELD = "source_sha"

OUTPUT_NAMES = (
    "l3_30h_final.pdf",
    "l3_30h_final.png",
    "metrics.csv",
    "provenance.json",
)

RAW_SIDECARS = (
    "artifact_sha256.json",
    "run_manifest.json",
    "summary.json",
    "complete.json",
    "status.json",
    "repository_after_sampling.json",
)


class ContractError(RuntimeError):
    """The input cannot be used as final formal plotting evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {description} {path}: {error}") from error


def require_sha256(value: str, description: str) -> str:
    require(
        re.fullmatch(r"[0-9a-f]{64}", value or "") is not None,
        f"{description} must be an exact lowercase SHA-256 digest",
    )
    return value


def require_source_sha(value: str, description: str) -> str:
    require(
        re.fullmatch(r"[0-9a-f]{40}", value or "") is not None,
        f"{description} must be an exact lowercase 40-hex Git source SHA",
    )
    return value


def parse_bool(value: str, description: str) -> bool:
    require(value in ("True", "False"), f"{description} must be True or False")
    return value == "True"


def parse_positive(value: str, description: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{description} is not numeric: {value!r}") from error
    require(math.isfinite(parsed) and parsed > 0.0,
            f"{description} must be finite and positive")
    return parsed


def parse_nonnegative_int(value: str, description: str) -> int:
    require(re.fullmatch(r"0|[1-9][0-9]*", value or "") is not None,
            f"{description} must be a canonical nonnegative integer")
    return int(value)


def geometric_mean(values: list[float]) -> float:
    require(values and all(math.isfinite(value) and value > 0 for value in values),
            "geometric mean inputs must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def read_and_validate_rows(
    path: Path, expected_source_sha: str
) -> tuple[list[dict[str, Any]], bool]:
    """Return typed rows and whether the CSV itself carries source provenance."""

    try:
        stream = path.open("r", newline="", encoding="utf-8")
    except OSError as error:
        raise ContractError(f"cannot open input CSV {path}: {error}") from error

    with stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields is not None, "input CSV has no header")
        require(len(fields) == len(set(fields)), "input CSV has duplicate columns")
        field_set = set(fields)
        base_set = set(BASE_FIELDS)
        require(
            field_set in (base_set, base_set | {EXPORT_FIELD}),
            "input CSV schema differs from Job A summary.csv: "
            f"missing={sorted(base_set - field_set)} "
            f"extra={sorted(field_set - base_set)}",
        )
        carries_source = EXPORT_FIELD in field_set
        raw_rows = list(reader)

    require(len(raw_rows) == 16,
            f"final plot requires exactly 16 cases, found {len(raw_rows)}")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        prefix = f"CSV row {index}"
        require(None not in raw, f"{prefix} contains surplus unheaded fields")
        graph = raw["graph"]
        view = raw["view"]
        require(graph in GRAPH_ORDER, f"{prefix} has unknown graph {graph!r}")
        require(view in VIEW_ORDER, f"{prefix} has unknown view {view!r}")
        identity = (graph, view)
        require(identity not in identities,
                f"{prefix} duplicates graph/view {graph}/{view}")
        identities.add(identity)

        expected_case_id = f"{graph}_{'G' if view == 'G' else 'G_plus'}"
        require(raw["case_id"] == expected_case_id,
                f"{prefix} case_id must be {expected_case_id!r}")
        require(raw["case_id"] not in case_ids,
                f"{prefix} duplicates case_id {raw['case_id']!r}")
        case_ids.add(raw["case_id"])

        valid = parse_bool(raw["measurement_valid"],
                           f"{prefix} measurement_valid")
        require(valid, f"{prefix} is not measurement-valid")
        error_count = parse_nonnegative_int(raw["error_count"],
                                            f"{prefix} error_count")
        require(error_count == 0, f"{prefix} has error_count={error_count}")

        t1 = parse_positive(raw["T1_single_no_l3_median_solve_ms"],
                            f"{prefix} T1 solve median")
        t2 = parse_positive(raw["T2_dual_l3_median_solve_ms"],
                            f"{prefix} T2 solve median")
        speedup = parse_positive(raw["speedup_T1_over_T2"],
                                 f"{prefix} solve speedup")
        require(math.isclose(speedup, t1 / t2, rel_tol=1e-12, abs_tol=1e-12),
                f"{prefix} solve speedup does not equal T1/T2")

        wall_t1 = parse_positive(raw["T1_single_no_l3_median_query_wall_ms"],
                                 f"{prefix} T1 query-wall median")
        wall_t2 = parse_positive(raw["T2_dual_l3_median_query_wall_ms"],
                                 f"{prefix} T2 query-wall median")
        wall_speedup = parse_positive(raw["query_wall_speedup_T1_over_T2"],
                                      f"{prefix} query-wall speedup")
        require(
            math.isclose(wall_speedup, wall_t1 / wall_t2,
                         rel_tol=1e-12, abs_tol=1e-12),
            f"{prefix} query-wall speedup does not equal T1/T2",
        )

        target = parse_positive(raw["target"], f"{prefix} target")
        require(math.isclose(target, TARGET_SPEEDUP, rel_tol=0.0, abs_tol=1e-12),
                f"{prefix} target is {target}, expected {TARGET_SPEEDUP}")
        target_met = parse_bool(raw["target_met"], f"{prefix} target_met")
        require(target_met == (speedup >= TARGET_SPEEDUP),
                f"{prefix} target_met disagrees with solve speedup")
        require(bool(raw["case_output"]), f"{prefix} has an empty case_output")

        if carries_source:
            row_source = require_source_sha(raw[EXPORT_FIELD],
                                             f"{prefix} source_sha")
            require(row_source == expected_source_sha,
                    f"{prefix} source_sha {row_source} differs from expected "
                    f"{expected_source_sha}")

        rows.append({
            "case_id": raw["case_id"],
            "graph": graph,
            "view": view,
            "measurement_valid": valid,
            "T1_single_no_l3_median_solve_ms": t1,
            "T2_dual_l3_median_solve_ms": t2,
            "speedup_T1_over_T2": speedup,
            "T1_single_no_l3_median_query_wall_ms": wall_t1,
            "T2_dual_l3_median_query_wall_ms": wall_t2,
            "query_wall_speedup_T1_over_T2": wall_speedup,
            "target": target,
            "target_met": target_met,
            "error_count": error_count,
            "case_output": raw["case_output"],
        })

    expected_identities = {
        (graph, view) for graph in GRAPH_ORDER for view in VIEW_ORDER
    }
    require(identities == expected_identities,
            "input does not contain exactly one G and one G+ case per graph")
    rows.sort(key=lambda row: (
        GRAPH_ORDER.index(row["graph"]), VIEW_ORDER.index(row["view"])
    ))
    return rows, carries_source


def _dig(document: Any, path: tuple[str, ...]) -> Any:
    value = document
    for component in path:
        if not isinstance(value, dict) or component not in value:
            return None
        value = value[component]
    return value


SOURCE_PATHS = (
    ("source_sha",),
    ("formal_source_sha",),
    ("repository", "head"),
    ("repository_after_sampling", "head"),
    ("input", "source_sha"),
    ("source", "sha"),
    ("formal_source", "sha"),
)


def validate_source_metadata(path: Path, expected_source_sha: str) -> dict[str, Any]:
    document = read_json(path, "source metadata")
    require(isinstance(document, dict), "source metadata must be a JSON object")
    found: dict[str, str] = {}
    for key_path in SOURCE_PATHS:
        value = _dig(document, key_path)
        if value is not None:
            label = ".".join(key_path)
            found[label] = require_source_sha(value, f"source metadata {label}")
    require(found, f"source metadata {path} contains no recognized source SHA")
    for label, value in found.items():
        require(value == expected_source_sha,
                f"source metadata {label}={value} differs from expected "
                f"{expected_source_sha}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "source_fields": found,
    }


def _require_close(actual: Any, expected: float, description: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool),
            f"{description} must be numeric")
    require(math.isfinite(float(actual)) and
            math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{description} differs from recomputed value")


def _validate_summary_document(
    document: Any, rows: list[dict[str, Any]], geomeans: dict[str, float]
) -> None:
    require(isinstance(document, dict), "raw summary.json must be an object")
    require(document.get("measurement_valid") is True,
            "raw summary.json is not measurement-valid")
    require(document.get("case_count") == 16,
            "raw summary.json case_count is not 16")
    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) == 16,
            "raw summary.json must contain 16 cases")
    by_case = {case.get("case_id"): case for case in cases
               if isinstance(case, dict)}
    require(len(by_case) == 16, "raw summary.json case IDs are incomplete/duplicate")
    for row in rows:
        case = by_case.get(row["case_id"])
        require(case is not None, f"raw summary.json lacks {row['case_id']}")
        require(case.get("graph") == row["graph"] and
                case.get("view") == row["view"],
                f"raw summary.json identity differs for {row['case_id']}")
        require(case.get("measurement_valid") is True,
                f"raw summary.json marks {row['case_id']} invalid")
        for field in (
            "T1_single_no_l3_median_solve_ms",
            "T2_dual_l3_median_solve_ms",
            "speedup_T1_over_T2",
            "T1_single_no_l3_median_query_wall_ms",
            "T2_dual_l3_median_query_wall_ms",
            "query_wall_speedup_T1_over_T2",
        ):
            _require_close(case.get(field), row[field],
                           f"raw summary.json {row['case_id']} {field}")
    summary_geomeans = document.get("geomeans")
    require(isinstance(summary_geomeans, dict),
            "raw summary.json geomeans is missing")
    for view in VIEW_ORDER:
        view_summary = summary_geomeans.get(view)
        require(isinstance(view_summary, dict) and view_summary.get("valid") is True,
                f"raw summary.json {view} geomean is invalid")
        require(view_summary.get("graph_count") == 8,
                f"raw summary.json {view} graph_count is not 8")
        _require_close(
            view_summary.get("solve_speedup_geomean_T1_over_T2"),
            geomeans[view], f"raw summary.json {view} solve geomean",
        )


def validate_raw_sidecars(
    directory: Path,
    input_name: str,
    input_sha256: str,
    expected_source_sha: str,
    rows: list[dict[str, Any]],
    geomeans: dict[str, float],
) -> dict[str, Any] | None:
    present = [name for name in RAW_SIDECARS if (directory / name).is_file()]
    if not present:
        return None
    missing = [name for name in RAW_SIDECARS if not (directory / name).is_file()]
    require(not missing,
            "raw Job A sidecar set is incomplete: missing=" + ",".join(missing))
    require(input_name == "summary.csv",
            "raw Job A evidence must use its canonical summary.csv filename")

    paths = {name: directory / name for name in RAW_SIDECARS}
    artifact = read_json(paths["artifact_sha256.json"], "artifact hash ledger")
    require(isinstance(artifact, dict), "artifact hash ledger must be an object")
    critical = [
        "summary.csv", "run_manifest.json", "summary.json", "complete.json",
        "status.json", "repository_after_sampling.json",
    ]
    for name in critical:
        recorded = require_sha256(artifact.get(name, ""),
                                  f"artifact hash for {name}")
        actual = input_sha256 if name == "summary.csv" else sha256(directory / name)
        require(recorded == actual,
                f"artifact hash mismatch for {name}: {actual} != {recorded}")

    run_manifest = read_json(paths["run_manifest.json"], "raw run manifest")
    source_record = validate_source_metadata(
        paths["run_manifest.json"], expected_source_sha)
    require(_dig(run_manifest, ("repository", "status_porcelain")) == [],
            "raw run manifest does not record a clean repository")
    configuration = run_manifest.get("configuration") if isinstance(run_manifest, dict) else None
    require(isinstance(configuration, dict),
            "raw run manifest configuration is missing")
    require(configuration.get("graphs") == list(GRAPH_ORDER),
            "raw run manifest graph list differs from the final matrix")
    require(configuration.get("physical_views") ==
            {"G": "original", "G+": "augmented"},
            "raw run manifest physical views differ")
    require(configuration.get("case_count") == 16,
            "raw run manifest case_count is not 16")

    repository_after = read_json(
        paths["repository_after_sampling.json"], "repository-after-sampling")
    require(repository_after == {
        "root": repository_after.get("root") if isinstance(repository_after, dict) else None,
        "head": expected_source_sha,
        "status_porcelain": [],
    }, "repository-after-sampling is not the expected clean source SHA")

    summary = read_json(paths["summary.json"], "raw summary")
    _validate_summary_document(summary, rows, geomeans)

    complete = read_json(paths["complete.json"], "raw completion record")
    require(isinstance(complete, dict) and
            complete.get("measurement_valid") is True and
            complete.get("cases") == 16,
            "raw complete.json is not a valid 16-case completion")
    complete_geomeans = complete.get("geomeans")
    require(isinstance(complete_geomeans, dict),
            "raw complete.json lacks geomeans")
    for view in VIEW_ORDER:
        view_complete = complete_geomeans.get(view)
        require(isinstance(view_complete, dict) and view_complete.get("valid") is True,
                f"raw complete.json {view} geomean is invalid")
        _require_close(
            view_complete.get("solve_speedup_geomean_T1_over_T2"),
            geomeans[view], f"raw complete.json {view} solve geomean",
        )

    status = read_json(paths["status.json"], "raw status")
    require(isinstance(status, dict) and
            status.get("status") == "measurement_valid" and
            status.get("measurement_valid") is True and
            status.get("completed_cases") == 16 and
            status.get("expected_cases") == 16,
            "raw status.json is not a valid completed 16-case batch")

    return {
        "mode": "raw_job_a_sidecars",
        "directory": str(directory.resolve()),
        "artifact_ledger_sha256": sha256(paths["artifact_sha256.json"]),
        "source_metadata": source_record,
        "validated_sidecars": {
            name: sha256(directory / name) for name in critical[1:]
        },
    }


def load_validated_input(
    input_csv: Path,
    expected_input_sha256: str,
    expected_source_sha: str,
    source_metadata: Path | None = None,
) -> dict[str, Any]:
    expected_input_sha256 = require_sha256(
        expected_input_sha256, "--expected-input-sha256")
    expected_source_sha = require_source_sha(
        expected_source_sha, "--expected-source-sha")
    require(input_csv.is_file() and not input_csv.is_symlink(),
            f"input must be a regular non-symlink file: {input_csv}")
    actual_input_sha256 = sha256(input_csv)
    require(actual_input_sha256 == expected_input_sha256,
            "input CSV SHA-256 differs from --expected-input-sha256: "
            f"{actual_input_sha256} != {expected_input_sha256}")

    rows, csv_carries_source = read_and_validate_rows(
        input_csv, expected_source_sha)
    geomeans = {
        view: geometric_mean([
            row["speedup_T1_over_T2"] for row in rows if row["view"] == view
        ])
        for view in VIEW_ORDER
    }
    raw_evidence = validate_raw_sidecars(
        input_csv.parent, input_csv.name, actual_input_sha256,
        expected_source_sha, rows, geomeans,
    )
    explicit_evidence = None
    if source_metadata is not None:
        require(source_metadata.is_file() and not source_metadata.is_symlink(),
                f"source metadata must be a regular non-symlink file: "
                f"{source_metadata}")
        explicit_evidence = validate_source_metadata(
            source_metadata, expected_source_sha)

    require(raw_evidence is not None or csv_carries_source or
            explicit_evidence is not None,
            "source SHA is not bound to the CSV: use original Job A sidecars, "
            "an exported source_sha column, or --source-metadata")
    source_modes = []
    if raw_evidence is not None:
        source_modes.append("raw_job_a_sidecars")
    if csv_carries_source:
        source_modes.append("csv_source_sha_column")
    if explicit_evidence is not None:
        source_modes.append("explicit_source_metadata")

    return {
        "rows": rows,
        "geomeans": geomeans,
        "input_sha256": actual_input_sha256,
        "source_sha": expected_source_sha,
        "source_binding_modes": source_modes,
        "raw_evidence": raw_evidence,
        "explicit_source_evidence": explicit_evidence,
    }


def _format_number(value: float) -> str:
    return format(value, ".17g")


def write_metrics(path: Path, bundle: dict[str, Any]) -> None:
    fields = (
        "row_type", "graph", "view", "measurement_valid",
        "T1_single_no_l3_median_solve_ms",
        "T2_dual_l3_median_solve_ms", "solve_speedup_T1_over_T2",
        "target", "target_met", "source_sha", "input_sha256",
    )
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in bundle["rows"]:
            writer.writerow({
                "row_type": "case",
                "graph": row["graph"],
                "view": row["view"],
                "measurement_valid": "True",
                "T1_single_no_l3_median_solve_ms": _format_number(
                    row["T1_single_no_l3_median_solve_ms"]),
                "T2_dual_l3_median_solve_ms": _format_number(
                    row["T2_dual_l3_median_solve_ms"]),
                "solve_speedup_T1_over_T2": _format_number(
                    row["speedup_T1_over_T2"]),
                "target": _format_number(TARGET_SPEEDUP),
                "target_met": str(row["target_met"]),
                "source_sha": bundle["source_sha"],
                "input_sha256": bundle["input_sha256"],
            })
        for view in VIEW_ORDER:
            writer.writerow({
                "row_type": "geomean",
                "graph": "ALL_8",
                "view": view,
                "measurement_valid": "True",
                "T1_single_no_l3_median_solve_ms": "",
                "T2_dual_l3_median_solve_ms": "",
                "solve_speedup_T1_over_T2": _format_number(
                    bundle["geomeans"][view]),
                "target": _format_number(TARGET_SPEEDUP),
                "target_met": str(bundle["geomeans"][view] >= TARGET_SPEEDUP),
                "source_sha": bundle["source_sha"],
                "input_sha256": bundle["input_sha256"],
            })


def render_figure(directory: Path, bundle: dict[str, Any]) -> str:
    # Keep Matplotlib from attempting to write into a possibly read-only home.
    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    by_identity = {
        (row["graph"], row["view"]): row for row in bundle["rows"]
    }
    x = list(range(len(GRAPH_ORDER)))
    g_values = [by_identity[(graph, "G")]["speedup_T1_over_T2"]
                for graph in GRAPH_ORDER]
    gp_values = [by_identity[(graph, "G+")]["speedup_T1_over_T2"]
                 for graph in GRAPH_ORDER]

    colors = {"G": "#F0A780", "G+": "#9392BE"}
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, axis = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
        width = 0.36
        axis.bar([value - width / 2 for value in x], g_values, width=width,
                 color=colors["G"], edgecolor=colors["G"],
                 label=f"G (GM {bundle['geomeans']['G']:.3f}x)")
        axis.bar([value + width / 2 for value in x], gp_values, width=width,
                 color=colors["G+"], edgecolor=colors["G+"],
                 label=f"G+ (GM {bundle['geomeans']['G+']:.3f}x)")
        axis.axhline(1.0, color="#404040", linewidth=1.0, linestyle="--",
                     label="1.00x reference")
        axis.axhline(TARGET_SPEEDUP, color="#547AA5", linewidth=1.0,
                     linestyle=":", label="1.20x target")
        axis.set_xticks(x, GRAPH_ORDER)
        axis.set_xlim(-0.55, len(GRAPH_ORDER) - 0.45)
        maximum = max(max(g_values), max(gp_values), TARGET_SPEEDUP)
        axis.set_ylim(0.0, maximum * 1.16)
        axis.set_ylabel("Solve speedup (T1 / T2)")
        axis.set_xlabel("Road-network graph")
        axis.tick_params(direction="out", width=0.7, labelsize=6.7)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#dddddd", linewidth=0.45, linestyle="-")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                    fontsize=6.2, frameon=True, framealpha=0.95,
                    columnspacing=0.9, handlelength=1.5, borderpad=0.35)
        fig.subplots_adjust(left=0.17, right=0.985, bottom=0.20, top=0.97)
        fig.savefig(
            directory / "l3_30h_final.pdf",
            metadata={
                "Title": "Final L3 G/G+ solve speedup",
                "Subject": "T1 single-GPU no-L3 over T2 dual-GPU L3",
                "Creator": "plot_l3_30h_final.py",
            },
        )
        fig.savefig(
            directory / "l3_30h_final.png", dpi=PNG_DPI,
            metadata={"Software": "plot_l3_30h_final.py"},
        )
        plt.close(fig)
    return matplotlib.__version__


def png_geometry(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    require(raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 33,
            "generated PNG has an invalid signature")
    width, height = struct.unpack(">II", raw[16:24])
    offset = 8
    x_pixels_per_metre = None
    y_pixels_per_metre = None
    while offset + 12 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunk_type = raw[offset + 4:offset + 8]
        data = raw[offset + 8:offset + 8 + length]
        require(offset + 12 + length <= len(raw), "generated PNG is truncated")
        if chunk_type == b"pHYs":
            require(length == 9, "generated PNG has malformed pHYs metadata")
            x_pixels_per_metre, y_pixels_per_metre, unit = struct.unpack(">IIB", data)
            require(unit == 1, "generated PNG density is not in physical units")
            break
        offset += 12 + length
    require(x_pixels_per_metre is not None and y_pixels_per_metre is not None,
            "generated PNG lacks physical-density metadata")
    x_dpi = x_pixels_per_metre * 0.0254
    y_dpi = y_pixels_per_metre * 0.0254
    require(x_dpi >= 300.0 - 0.1 and y_dpi >= 300.0 - 0.1,
            f"generated PNG density is below 300 dpi: {x_dpi}, {y_dpi}")
    require(width >= math.ceil(FIGURE_WIDTH_IN * 300),
            "generated PNG is too narrow for a 3.5-inch 300-dpi figure")
    return {
        "width_pixels": width,
        "height_pixels": height,
        "x_dpi": x_dpi,
        "y_dpi": y_dpi,
    }


def pdf_geometry(path: Path) -> dict[str, float]:
    raw = path.read_bytes()
    match = re.search(
        rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+"
        rb"([-+0-9.]+)\s+([-+0-9.]+)\s*\]", raw,
    )
    require(match is not None, "generated PDF has no readable MediaBox")
    x0, y0, x1, y1 = (float(value) for value in match.groups())
    width_inches = (x1 - x0) / 72.0
    height_inches = (y1 - y0) / 72.0
    require(math.isclose(width_inches, FIGURE_WIDTH_IN, abs_tol=0.002),
            f"generated PDF width is {width_inches}, expected 3.5 inches")
    return {"width_inches": width_inches, "height_inches": height_inches}


def generate_outputs(
    input_csv: Path,
    output: Path,
    expected_input_sha256: str,
    expected_source_sha: str,
    source_metadata: Path | None = None,
) -> dict[str, Any]:
    bundle = load_validated_input(
        input_csv, expected_input_sha256, expected_source_sha, source_metadata)
    require(not output.exists() and not output.is_symlink(),
            f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp.", dir=str(output.parent)))
    try:
        write_metrics(temporary / "metrics.csv", bundle)
        matplotlib_version = render_figure(temporary, bundle)
        shutil.rmtree(temporary / ".matplotlib", ignore_errors=True)
        png = png_geometry(temporary / "l3_30h_final.png")
        pdf = pdf_geometry(temporary / "l3_30h_final.pdf")
        output_hashes = {
            name: sha256(temporary / name)
            for name in OUTPUT_NAMES if name != "provenance.json"
        }
        provenance = {
            "schema": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "final clean-SHA eight-graph G/G+ solve-speedup figure",
            "input": {
                "path": str(input_csv.resolve()),
                "sha256": bundle["input_sha256"],
                "expected_sha256": expected_input_sha256,
                "hash_verified": True,
                "source_sha": bundle["source_sha"],
                "expected_source_sha": expected_source_sha,
                "source_sha_verified": True,
                "source_binding_modes": bundle["source_binding_modes"],
                "raw_evidence": bundle["raw_evidence"],
                "explicit_source_evidence": bundle[
                    "explicit_source_evidence"],
            },
            "validation": {
                "measurement_valid": True,
                "case_count": 16,
                "expected_case_count": 16,
                "graphs": list(GRAPH_ORDER),
                "views": list(VIEW_ORDER),
                "cases_per_view": 8,
                "all_error_counts_zero": True,
                "solve_and_query_wall_speedup_identities_recomputed": True,
                "failed_or_partial_batches_allowed": False,
            },
            "metric": {
                "definition": (
                    "per-case median T1 single-GPU no-L3 solve_ms divided by "
                    "median T2 dual-GPU L3 solve_ms"
                ),
                "geomean_formula": "exp(mean(log(per-case speedup)))",
                "geomean_requires_all_16_valid_cases": True,
                "geomeans": bundle["geomeans"],
                "reference": 1.0,
                "target": TARGET_SPEEDUP,
            },
            "figure": {
                "pdf": "l3_30h_final.pdf",
                "png": "l3_30h_final.png",
                "width_inches": FIGURE_WIDTH_IN,
                "height_inches": FIGURE_HEIGHT_IN,
                "requested_png_dpi": PNG_DPI,
                "png_geometry": png,
                "pdf_geometry": pdf,
                "single_column": True,
            },
            "metrics_csv": {
                "path": "metrics.csv",
                "case_rows": 16,
                "geomean_rows": 2,
            },
            "software": {
                "python": sys.version.split()[0],
                "matplotlib": matplotlib_version,
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256(Path(__file__)),
            },
            "output_sha256": output_hashes,
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "--summary", dest="input_csv", required=True, type=Path,
        help="Job A 05_eight_graph/summary.csv or an equivalent export",
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="new output directory")
    parser.add_argument("--expected-input-sha256", required=True,
                        help="exact SHA-256 of the accepted input CSV")
    parser.add_argument("--expected-source-sha", required=True,
                        help="exact 40-hex formal source commit")
    parser.add_argument(
        "--source-metadata", type=Path,
        help=("JSON binding the exported CSV to the formal source SHA; not "
              "needed for original Job A sidecars or a source_sha CSV column"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    try:
        provenance = generate_outputs(
            args.input_csv, args.out, args.expected_input_sha256,
            args.expected_source_sha, args.source_metadata,
        )
    except ContractError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "complete",
        "output": str(args.out),
        "source_sha": args.expected_source_sha,
        "input_sha256": args.expected_input_sha256,
        "geomeans": provenance["metric"]["geomeans"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
