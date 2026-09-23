#!/usr/bin/env python3
"""Stream-audit Galois v1 GR files without loading the graph into GPU memory."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import numpy as np


INT_MAX = (1 << 31) - 1
CHUNK = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def source_ids(rows, count):
    """Pick first, 1/3, and 2/3 ranks from vertices with nonzero outdegree."""
    if count == 0:
        return []
    ranks = sorted({0, (count - 1) // 3, (2 * (count - 1)) // 3})
    result = []
    rank_index = 0
    positive_seen = 0
    previous = 0
    for begin in range(0, len(rows), CHUNK // 8):
        ends = np.asarray(rows[begin:begin + CHUNK // 8], dtype=np.uint64)
        starts = np.empty_like(ends)
        starts[0] = previous
        if len(ends) > 1:
            starts[1:] = ends[:-1]
        local = np.flatnonzero(ends > starts)
        for offset in local:
            while rank_index < len(ranks) and positive_seen == ranks[rank_index]:
                result.append(int(begin + offset))
                rank_index += 1
            positive_seen += 1
        previous = int(ends[-1]) if len(ends) else previous
    if rank_index != len(ranks):
        raise RuntimeError("source rank selection failed")
    return result


def audit(path):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size < 32:
        raise RuntimeError(f"not a nonempty GR file: {resolved}")
    with resolved.open("rb") as stream:
        header_bytes = stream.read(32)
    version, edge_bytes, nodes, edges = struct.unpack("<QQQQ", header_bytes)
    destination_offset = 32 + 8 * nodes
    destination_bytes = 4 * edges
    destination_padding = 4 if edges & 1 else 0
    weight_offset = destination_offset + destination_bytes + destination_padding
    expected_bytes = weight_offset + edge_bytes * edges
    trailing_bytes = stat.st_size - expected_bytes
    if expected_bytes > stat.st_size:
        raise RuntimeError(
            f"truncated file {resolved}: expected at least {expected_bytes}, got {stat.st_size}"
        )
    if nodes == 0:
        raise RuntimeError(f"zero-node graph: {resolved}")

    rows = np.memmap(resolved, mode="r", dtype="<u8", offset=32, shape=(nodes,))
    destinations = np.memmap(
        resolved, mode="r", dtype="<u4", offset=destination_offset, shape=(edges,)
    )
    row_nonmonotonic = 0
    row_out_of_range = 0
    positive_degree_vertices = 0
    maximum_degree = 0
    previous = 0
    for begin in range(0, nodes, CHUNK // 8):
        ends = np.asarray(rows[begin:begin + CHUNK // 8], dtype=np.uint64)
        if len(ends) == 0:
            continue
        starts = np.empty_like(ends)
        starts[0] = previous
        if len(ends) > 1:
            starts[1:] = ends[:-1]
        row_nonmonotonic += int(np.count_nonzero(ends < starts))
        row_out_of_range += int(np.count_nonzero(ends > edges))
        valid = ends >= starts
        degrees = np.where(valid, ends - starts, 0)
        positive_degree_vertices += int(np.count_nonzero(degrees))
        maximum_degree = max(maximum_degree, int(degrees.max(initial=0)))
        previous = int(ends[-1])
    last_row_end = int(rows[-1])

    invalid_targets = 0
    target_min = None
    target_max = None
    for begin in range(0, edges, CHUNK // 4):
        chunk = np.asarray(destinations[begin:begin + CHUNK // 4], dtype=np.uint64)
        if len(chunk) == 0:
            continue
        invalid_targets += int(np.count_nonzero(chunk >= nodes))
        local_min = int(chunk.min())
        local_max = int(chunk.max())
        target_min = local_min if target_min is None else min(target_min, local_min)
        target_max = local_max if target_max is None else max(target_max, local_max)

    weight_summary = None
    encoding_diagnostic = "unweighted_no_edge_payload"
    if edge_bytes == 4:
        weights = np.memmap(
            resolved, mode="r", dtype="<i4", offset=weight_offset, shape=(edges,)
        )
        minimum = None
        maximum = None
        negative = zero = one = 0
        total_weight = 0
        histogram = np.zeros(31, dtype=np.uint64)
        float_finite = float_nan = float_inf = float_negative = float_zero = 0
        float_min = None
        float_max = None
        for begin in range(0, edges, CHUNK // 4):
            chunk = np.asarray(weights[begin:begin + CHUNK // 4], dtype=np.int32)
            if len(chunk) == 0:
                continue
            local_min = int(chunk.min())
            local_max = int(chunk.max())
            minimum = local_min if minimum is None else min(minimum, local_min)
            maximum = local_max if maximum is None else max(maximum, local_max)
            negative += int(np.count_nonzero(chunk < 0))
            zero += int(np.count_nonzero(chunk == 0))
            one += int(np.count_nonzero(chunk == 1))
            total_weight += int(chunk.astype(np.int64).sum(dtype=np.int64))
            positive = chunk[chunk > 0].astype(np.uint64)
            if len(positive):
                buckets = np.floor(np.log2(positive)).astype(np.int64)
                histogram += np.bincount(buckets, minlength=31)[:31].astype(np.uint64)
            floats = chunk.view("<f4")
            finite = np.isfinite(floats)
            float_finite += int(np.count_nonzero(finite))
            float_nan += int(np.count_nonzero(np.isnan(floats)))
            float_inf += int(np.count_nonzero(np.isinf(floats)))
            float_negative += int(np.count_nonzero(floats < 0))
            float_zero += int(np.count_nonzero(floats == 0))
            if np.any(finite):
                local_float_min = float(floats[finite].min())
                local_float_max = float(floats[finite].max())
                float_min = local_float_min if float_min is None else min(float_min, local_float_min)
                float_max = local_float_max if float_max is None else max(float_max, local_float_max)
        labels = {
            "1" if power == 0 else f"{1 << power}-{(1 << (power + 1)) - 1}": int(count)
            for power, count in enumerate(histogram)
            if count
        }
        weight_summary = {
            "interpretation": "little_endian_signed_int32",
            "minimum": minimum,
            "maximum": maximum,
            "negative_count": negative,
            "zero_count": zero,
            "one_count": one,
            "mean": total_weight / edges if edges else None,
            "full_log2_histogram": labels,
            "same_bytes_as_float32": {
                "finite_count": float_finite,
                "nan_count": float_nan,
                "infinite_count": float_inf,
                "negative_count": float_negative,
                "zero_count": float_zero,
                "finite_minimum": float_min,
                "finite_maximum": float_max,
            },
        }
        if negative == 0 and maximum is not None:
            if float_nan or float_inf or (float_max is not None and 0 < float_max < 1e-30):
                encoding_diagnostic = "bytes_are_consistent_with_nonnegative_int32_not_float32"
            else:
                encoding_diagnostic = "bytes_need_external_weight_semantics_confirmation"
        else:
            encoding_diagnostic = "not_admissible_as_nonnegative_int32"

    padding_hex = ""
    if trailing_bytes > 0:
        with resolved.open("rb") as stream:
            stream.seek(expected_bytes)
            padding_hex = stream.read(min(trailing_bytes, 64)).hex()
    selected_sources = source_ids(rows, positive_degree_vertices)
    maximum_weight = weight_summary["maximum"] if weight_summary else None
    coarse_bound = None
    coarse_sufficient = False
    if maximum_weight is not None and maximum_weight >= 0:
        coarse_bound = (nodes - 1) * maximum_weight
        coarse_sufficient = coarse_bound < INT_MAX
    format_valid = (
        version == 1
        and edge_bytes in (0, 4)
        and nodes < INT_MAX
        and edges < INT_MAX
        and trailing_bytes >= 0
        and row_nonmonotonic == 0
        and row_out_of_range == 0
        and last_row_end == edges
        and invalid_targets == 0
    )
    weighted_admissible = (
        format_valid
        and edge_bytes == 4
        and weight_summary is not None
        and weight_summary["negative_count"] == 0
    )
    return {
        "requested_path": str(path),
        "realpath": str(resolved),
        "sha256": sha256(resolved),
        "file_bytes": stat.st_size,
        "header": {
            "version": version,
            "edge_payload_bytes": edge_bytes,
            "vertices": nodes,
            "directed_edges": edges,
        },
        "layout": {
            "row_end_offset": 32,
            "destination_offset": destination_offset,
            "destination_padding_bytes": destination_padding,
            "weight_offset": weight_offset,
            "expected_payload_end": expected_bytes,
            "trailing_bytes": trailing_bytes,
            "trailing_prefix_hex": padding_hex,
        },
        "csr_checks": {
            "row_nonmonotonic_count": row_nonmonotonic,
            "row_out_of_range_count": row_out_of_range,
            "last_row_end": last_row_end,
            "invalid_target_count": invalid_targets,
            "target_minimum": target_min,
            "target_maximum": target_max,
            "positive_outdegree_vertices": positive_degree_vertices,
            "maximum_outdegree": maximum_degree,
        },
        "weights": weight_summary,
        "weight_encoding_diagnostic": encoding_diagnostic,
        "weight_provenance": "not_provable_from_GR_bytes_requires_external_confirmation",
        "fixed_source_rule": (
            "rank 0, floor((k-1)/3), floor(2*(k-1)/3) in ascending vertices "
            "with positive outdegree"
        ),
        "fixed_sources": selected_sources,
        "int32_path_diagnostic": {
            "coarse_simple_path_bound": coarse_bound,
            "coarse_bound_is_sufficient": coarse_sufficient,
            "note": (
                "A false sufficient check does not reject the graph; run an int64 oracle "
                "and checked-add GPU validation for each selected source."
            ),
        },
        "minimum_graph_storage_bytes": {
            "csr_row_int32": 4 * (nodes + 1),
            "csr_destination_int32": 4 * edges,
            "csr_weight_int32": 4 * edges if edge_bytes == 4 else 0,
            "distance_int32": 4 * nodes,
            "total": 4 * (nodes + 1) + 4 * edges + (4 * edges if edge_bytes == 4 else 0) + 4 * nodes,
            "note": "Excludes L2, candidate, inbox, boundary index, retained TX, and workspace allocations.",
        },
        "format_valid": format_valid,
        "weighted_nonnegative_int32_admissible": weighted_admissible,
        "oracle_status": "NOT_RUN",
        "gpu_capacity_status": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.paths:
        print(f"AUDIT_GR begin path={path}", flush=True)
        records.append(audit(path))
        print(
            f"AUDIT_GR done path={path} admissible="
            f"{int(records[-1]['weighted_nonnegative_int32_admissible'])}",
            flush=True,
        )
    result = {
        "schema": 1,
        "scan": "full rows, destinations, and edge payload; streaming SHA256",
        "records": records,
        "all_format_valid": all(record["format_valid"] for record in records),
        "all_weighted_nonnegative_int32_admissible": all(
            record["weighted_nonnegative_int32_admissible"] for record in records
        ),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        if args.out.exists():
            parser.error(f"output already exists: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
