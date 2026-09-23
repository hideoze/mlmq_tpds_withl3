#!/usr/bin/env python3
"""Run paired no-L3 single-GPU/L3 dual-GPU samples for the 30 h plan.

This runner is intentionally Slurm-only and never substitutes an L3 ``-n 1``
run for the independent no-L3 binary.  Every process gets its own warmups and
formal samples; configurations are run in the opposite order in adjacent
rounds to reduce order bias.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import socket
import struct
import subprocess
import sys
from statistics import median
import tarfile
import tempfile

from run_usa_road_matrix import run_one, sha256


TARGET_SPEEDUP = 1.20
CONFIGURATIONS = {
    "single_no_l3": {"gpu_count": 1},
    "dual_l3": {"gpu_count": 2},
}
L3_CONFIG_FIELDS = (
    "gpu", "work_blocks", "delta", "queue_type", "window_mode", "window_min",
    "window_max", "idle_backoff", "worker_recovery", "term_wait_ack",
    "rx_priority_bootstrap", "rx_express", "rx_express_enabled",
    "rx_express_slots", "rx_express_batch", "rx_l2_pull",
    "rx_l2_pull_claim", "rx_l2_pull_enabled",
)
L3_WORKER_ACK_FIELDS = (
    "gpu", "active_slots", "capacity", "work_blocks", "warps_per_block",
)
L2_FINAL_FIELDS = (
    "gpu", "buckets", "reads", "writes", "completed",
    "guarded_writes", "max_bucket_writes", "per_bucket_capacity",
    "total_capacity", "counter_bits", "overflow_guard", "overflow_detected",
    "no_wrap",
)
L2_CAPACITY_FIELDS = (
    "budget", "record_bytes", "buckets", "per_bucket", "allocated_records",
    "counter_bits",
)
QUEUE_TYPE_IDS = {"L1V_L2DQ": 0, "L1SLF_L2DQ": 19}
FORMAL_INPUT_CONTRACT = Path(
    "evidence/l3_30h_20260923/final/formal_input_contract.json")
FORMAL_BUILD_TIMEOUT_SECONDS = 3600
FORMAL_L2_BUCKETS = 8
FORMAL_L2_BUCKET_MAX = 4
FORMAL_L2_BATCH_SIZE = 8
FORMAL_L2_BUDGET_BYTES = 2147483647
FORMAL_L2_RECORD_BYTES = 8
FORMAL_L2_ALLOCATED_RECORDS = 268435455
FORMAL_L2_PER_BUCKET_CAPACITY = 33553920
FORMAL_L2_TOTAL_CAPACITY = 268431360
FORMAL_L2_COUNTER_BITS = 32
if (FORMAL_L2_ALLOCATED_RECORDS !=
        FORMAL_L2_BUDGET_BYTES // FORMAL_L2_RECORD_BYTES or
        FORMAL_L2_PER_BUCKET_CAPACITY !=
        FORMAL_L2_ALLOCATED_RECORDS // FORMAL_L2_BUCKETS // 512 * 512 or
        FORMAL_L2_TOTAL_CAPACITY !=
        FORMAL_L2_BUCKETS * FORMAL_L2_PER_BUCKET_CAPACITY):
    raise RuntimeError("frozen L2 capacity constants are internally inconsistent")
FORMAL_TOOL_PATH = "/usr/local/cuda/bin:/usr/bin:/bin"
GIT_EXECUTABLE = "/usr/bin/git"
NVIDIA_SMI_EXECUTABLE = "/usr/bin/nvidia-smi"
FORMAL_BLOCKED_BUILD_ENV = (
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "LD_PRELOAD",
    "LD_LIBRARY_PATH", "CUDAFLAGS", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
    "LDFLAGS", "LIBS", "COMPILER_PATH", "GCC_EXEC_PREFIX", "CUDA_HOME",
    "CUDA_PATH", "CUDAHOSTCXX", "NVCC_CCBIN",
)
FORMAL_RUNTIME_ENV_KEYS = (
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _archive_member_name(raw_name):
    """Return a safe normalized POSIX archive name or raise ValueError."""
    if not isinstance(raw_name, str) or not raw_name or "\0" in raw_name:
        raise ValueError(f"invalid empty/NUL archive member name: {raw_name!r}")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe archive member path: {raw_name!r}")
    normalized = path.as_posix().rstrip("/")
    if not normalized:
        raise ValueError(f"invalid archive member path: {raw_name!r}")
    return normalized


def read_regular_archive_tree(archive):
    """Read an archive as an exact regular-file/directory tree.

    Links, devices, FIFOs, duplicate normalized names and traversal paths are
    rejected.  The returned mapping retains exact archive permission bits for
    safe extraction and byte-for-byte post-extraction validation.
    """
    archive = Path(archive)
    tree = {}
    try:
        stream = tarfile.open(archive, "r:*")
    except (OSError, tarfile.TarError) as error:
        raise SystemExit(f"cannot open source archive {archive}: {error}") from error
    with stream:
        for member in stream.getmembers():
            try:
                name = _archive_member_name(member.name)
            except ValueError as error:
                raise SystemExit(str(error)) from error
            if name in tree:
                raise SystemExit(f"duplicate source archive member: {name}")
            mode = member.mode & 0o777
            if member.isdir():
                tree[name] = {"type": "dir", "mode": mode}
            elif member.isreg():
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise SystemExit(f"cannot read source archive member: {name}")
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                if size != member.size:
                    raise SystemExit(
                        f"source archive member size changed for {name}: "
                        f"header={member.size} read={size}")
                tree[name] = {
                    "type": "file", "mode": mode, "size": size,
                    "sha256": digest.hexdigest(),
                }
            else:
                raise SystemExit(
                    f"source archive contains forbidden non-regular member "
                    f"{name}: type={member.type!r}")
    for name in tree:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            parent_name = parent.as_posix()
            if parent_name not in tree:
                raise SystemExit(
                    f"source archive omits parent directory {parent_name} "
                    f"for {name}")
            if tree[parent_name]["type"] != "dir":
                raise SystemExit(
                    f"source archive path prefix is not a directory: "
                    f"{parent_name}")
            parent = parent.parent
    return tree


def git_semantic_source_tree(tree):
    """Normalize a source manifest to permissions representable by Git.

    Git records the executable bit for regular files, but not the remaining
    permission bits, and it does not record directory permission bits.  Keep
    every path, type, size and digest while reducing permissions to precisely
    that source-control semantic boundary.
    """
    normalized = {}
    for name, row in tree.items():
        normalized_row = dict(row)
        mode = normalized_row.pop("mode")
        if row["type"] == "file":
            normalized_row["git_executable"] = bool(mode & 0o111)
        normalized[name] = normalized_row
    return normalized


def safe_extract_regular_archive(archive, destination):
    """Extract only validated regular files/directories without tarfile links."""
    archive = Path(archive)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"archive destination already exists: {destination}")
    # Validate the complete archive before creating any output.
    expected = read_regular_archive_tree(archive)
    destination.mkdir(parents=False)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as stream:
        members = {}
        for member in stream.getmembers():
            members[_archive_member_name(member.name)] = member
        for name, row in sorted(expected.items(), key=lambda item: (item[0].count("/"), item[0])):
            if row["type"] != "dir":
                continue
            target = destination.joinpath(*PurePosixPath(name).parts)
            resolved_parent = target.parent.resolve()
            if resolved_parent != root and root not in resolved_parent.parents:
                raise RuntimeError(f"archive directory escapes destination: {name}")
            target.mkdir(exist_ok=False)
            target.chmod(row["mode"])
        for name, row in sorted(expected.items()):
            if row["type"] != "file":
                continue
            target = destination.joinpath(*PurePosixPath(name).parts)
            resolved_parent = target.parent.resolve()
            if resolved_parent != root and root not in resolved_parent.parents:
                raise RuntimeError(f"archive file escapes destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = stream.extractfile(members[name])
            if extracted is None:
                raise RuntimeError(f"cannot extract source archive member: {name}")
            with target.open("xb") as output:
                shutil.copyfileobj(extracted, output)
            target.chmod(row["mode"])
    actual = filesystem_regular_tree(destination)
    if actual != expected:
        raise RuntimeError("extracted source tree differs from validated archive tree")
    return expected


def filesystem_regular_tree(root):
    """Return a regular-file/directory manifest, rejecting every link/special."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"expected a real source directory: {root}")
    tree = {}
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(dirs):
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise SystemExit(f"source tree contains forbidden directory/link: {path}")
            relative = path.relative_to(root).as_posix()
            tree[relative] = {
                "type": "dir", "mode": path.stat(follow_symlinks=False).st_mode & 0o777,
            }
        for name in sorted(files):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise SystemExit(f"source tree contains forbidden file/link: {path}")
            relative = path.relative_to(root).as_posix()
            stat = path.stat(follow_symlinks=False)
            tree[relative] = {
                "type": "file", "mode": stat.st_mode & 0o777,
                "size": stat.st_size, "sha256": sha256(path),
            }
    return tree


def _git_archive(repo_root, output, *paths):
    with output.open("wb") as stream:
        process = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), "archive", "--format=tar",
             "HEAD", *paths],
            stdout=stream, stderr=subprocess.PIPE)
    if process.returncode:
        raise SystemExit(
            "cannot materialize clean HEAD source tree: " +
            process.stderr.decode(errors="replace").strip())


def expected_head_tree(repo_root, paths):
    with tempfile.TemporaryDirectory(prefix="l3-head-tree-") as temporary:
        archive = Path(temporary) / "head.tar"
        _git_archive(repo_root, archive, *paths)
        return read_regular_archive_tree(archive)


def materialize_expected_single_tree(repo_root, destination):
    """Build the exact independent-single source tree expected from HEAD."""
    repo_root = Path(repo_root)
    destination = Path(destination)
    if destination.exists():
        raise SystemExit(f"expected-single destination exists: {destination}")
    single_rel = Path("evidence/l3_supplement_20260921_v2/single_source")
    with tempfile.TemporaryDirectory(prefix="l3-single-head-") as temporary:
        temporary = Path(temporary)
        snapshot_archive = temporary / "single.tar"
        core_archive = temporary / "core.tar"
        _git_archive(repo_root, snapshot_archive, single_rel.as_posix())
        _git_archive(repo_root, core_archive, "core")
        snapshot = temporary / "snapshot"
        clean_core = temporary / "clean_core"
        safe_extract_regular_archive(snapshot_archive, snapshot)
        safe_extract_regular_archive(core_archive, clean_core)
        source = snapshot / single_rel
        destination.mkdir()
        shutil.copytree(source / "SSSP", destination / "SSSP")
        shutil.copytree(clean_core / "core", destination / "core")
    adapter_relative = Path("scripts/multigpu/prepare_l3_30h_single.py")
    adapter_bytes = _git_head_file(repo_root, adapter_relative)
    with tempfile.TemporaryDirectory(prefix="l3-single-adapter-") as temporary:
        adapter = Path(temporary) / "prepare_l3_30h_single.py"
        adapter.write_bytes(adapter_bytes)
        process = subprocess.run(
            [sys.executable, str(adapter), str(destination)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
            timeout=60)
    if process.returncode:
        raise SystemExit(
            f"cannot deterministically apply the single adapter: {process.stdout}")
    return filesystem_regular_tree(destination)


def validate_pair_source_trees(pair, repo_root):
    expected_dual = expected_head_tree(repo_root, ("SSSP", "core"))
    actual_dual = read_regular_archive_tree(pair / "dual_build/source.tgz")
    expected_dual_git = git_semantic_source_tree(expected_dual)
    actual_dual_git = git_semantic_source_tree(actual_dual)
    if actual_dual_git != expected_dual_git:
        missing = sorted(set(expected_dual) - set(actual_dual))[:8]
        extra = sorted(set(actual_dual) - set(expected_dual))[:8]
        changed = sorted(
            name for name in set(actual_dual) & set(expected_dual)
            if actual_dual_git[name] != expected_dual_git[name])[:8]
        raise SystemExit(
            "formal dual source archive differs from clean HEAD "
            "SSSP/core under Git path/type/content/executable semantics: "
            f"missing={missing} extra={extra} changed={changed}")
    with tempfile.TemporaryDirectory(prefix="l3-single-expected-") as temporary:
        expected_root = Path(temporary) / "source"
        expected_single = materialize_expected_single_tree(repo_root, expected_root)
    actual_single = read_regular_archive_tree(pair / "single_build/source.tgz")
    expected_single_git = git_semantic_source_tree(expected_single)
    actual_single_git = git_semantic_source_tree(actual_single)
    if actual_single_git != expected_single_git:
        missing = sorted(set(expected_single) - set(actual_single))[:8]
        extra = sorted(set(actual_single) - set(expected_single))[:8]
        changed = sorted(
            name for name in set(actual_single) & set(expected_single)
            if actual_single_git[name] != expected_single_git[name])[:8]
        raise SystemExit(
            "formal single source archive differs from deterministic HEAD "
            "snapshot+core+adapter under Git path/type/content/executable "
            f"semantics: missing={missing} extra={extra} changed={changed}")
    return {
        "dual": {"entries": len(actual_dual), "archive_sha256": sha256(
            pair / "dual_build/source.tgz")},
        "single": {"entries": len(actual_single), "archive_sha256": sha256(
            pair / "single_build/source.tgz")},
    }


def run_capture(command, path, *, environment=None):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=30, env=environment)
    path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"command failed with rc={result.returncode}: {' '.join(command)}")
    return result.stdout


def csv_items(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def slurm_gpu_evidence(environment):
    """Return allocation evidence, rejecting anything other than two GPUs."""
    evidence = {"SLURM_JOB_PARTITION": environment.get("SLURM_JOB_PARTITION")}
    allocation_counts = []
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = environment.get(key)
        if value:
            count = len(csv_items(value))
            evidence[key] = {"value": value, "count": count}
            allocation_counts.append((key, count))

    on_node = environment.get("SLURM_GPUS_ON_NODE")
    if on_node:
        count = None
        if on_node.isdigit():
            count = int(on_node)
        else:
            match = re.search(r":(\d+)(?:\(|$)", on_node)
            if match:
                count = int(match.group(1))
        evidence["SLURM_GPUS_ON_NODE"] = {"value": on_node, "count": count}
        if count is not None:
            allocation_counts.append(("SLURM_GPUS_ON_NODE", count))

    if not allocation_counts:
        raise RuntimeError(
            "cannot prove a two-GPU Slurm allocation: no count-bearing "
            "SLURM_STEP_GPUS, SLURM_JOB_GPUS, or SLURM_GPUS_ON_NODE")
    wrong = [(key, count) for key, count in allocation_counts if count != 2]
    if wrong:
        raise RuntimeError(f"Slurm allocation is not exactly two GPUs: {wrong}")

    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible_count = len(csv_items(visible))
        evidence["CUDA_VISIBLE_DEVICES"] = {
            "value": visible, "count": visible_count}
        if visible_count != 2:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES does not expose exactly two GPUs: "
                f"{visible!r}")
    return evidence


def parse_gpu_query(raw, *, require_a100=False):
    rows = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            raise RuntimeError(f"malformed nvidia-smi GPU row: {row!r}")
        index, uuid, name, compute_capability = (item.strip() for item in row)
        rows.append({
            "index": index, "uuid": uuid, "name": name,
            "compute_capability": compute_capability,
        })
    if len(rows) != 2:
        raise RuntimeError(
            f"nvidia-smi exposes {len(rows)} GPUs; exactly two are required")
    indices = [row["index"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    if any(not index.isdigit() for index in indices):
        raise RuntimeError(f"nvidia-smi returned a non-numeric GPU index: {indices}")
    if len(set(indices)) != 2:
        raise RuntimeError(f"nvidia-smi GPU indices are not unique: {indices}")
    if any(not uuid.startswith("GPU-") for uuid in uuids):
        raise RuntimeError(f"nvidia-smi returned a malformed GPU UUID: {uuids}")
    if len(set(uuids)) != 2:
        raise RuntimeError(f"nvidia-smi GPU UUIDs are not unique: {uuids}")
    if require_a100:
        if os.environ.get("SLURM_JOB_PARTITION") != "a100":
            raise RuntimeError(
                "formal evidence requires SLURM_JOB_PARTITION=a100, got "
                f"{os.environ.get('SLURM_JOB_PARTITION')!r}")
        for row in rows:
            if "A100" not in row["name"] or row["compute_capability"] != "8.0":
                raise RuntimeError(
                    "formal evidence requires two A100 CC 8.0 GPUs, got "
                    f"{row['name']!r} CC {row['compute_capability']!r}")
    return rows


def gpu_preflight(out_dir, *, require_a100=False):
    evidence = slurm_gpu_evidence(os.environ)
    nvidia_smi = Path(NVIDIA_SMI_EXECUTABLE)
    if (not nvidia_smi.is_file() or nvidia_smi.is_symlink() or
            not os.access(nvidia_smi, os.X_OK)):
        raise RuntimeError(
            f"canonical nvidia-smi is not a regular executable: {nvidia_smi}")
    runtime_environment, _ = formal_runtime_environment()
    full = run_capture(
        [str(nvidia_smi)], out_dir / "gpu_initial.log",
        environment=runtime_environment)
    query = run_capture(
        [str(nvidia_smi), "--query-gpu=index,uuid,name,compute_cap",
         "--format=csv,noheader,nounits"],
        out_dir / "gpu_query.csv", environment=runtime_environment)
    gpu_rows = parse_gpu_query(query, require_a100=require_a100)
    topology = run_capture(
        [str(nvidia_smi), "topo", "-m"], out_dir / "gpu_topology.log",
        environment=runtime_environment)
    apps = run_capture(
        [str(nvidia_smi), "--query-compute-apps=pid", "--format=csv,noheader"],
        out_dir / "gpu_apps_initial.log",
        environment=runtime_environment).strip()
    if apps:
        raise RuntimeError("GPU compute applications present before sampling: " + apps)
    return {
        "allocation": evidence,
        "visible_gpu_count": len(gpu_rows),
        "visible_gpus": gpu_rows,
        "formal_a100_cc80_required": require_a100,
        "nvidia_smi": {
            "path": str(nvidia_smi), "sha256": sha256(nvidia_smi)},
        "compute_apps_before": [],
        "snapshot_bytes": len(full.encode()),
        "topology_bytes": len(topology.encode()),
    }


def run_one_with_canonical_tools(*args, **kwargs):
    """Run the shared harness while pinning its internal GPU snapshots.

    ``run_one`` passes the requested environment to the benchmark itself but
    its before/after ``nvidia-smi`` snapshots inherit this Python process's
    PATH.  Formal evidence must not resolve those snapshots through a
    caller-controlled PATH.
    """
    previous_path = os.environ.get("PATH")
    os.environ["PATH"] = FORMAL_TOOL_PATH
    try:
        return run_one(*args, **kwargs)
    finally:
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path


def read_gr_header(path):
    with path.open("rb") as stream:
        header = stream.read(32)
    if len(header) != 32:
        raise ValueError(f"graph is shorter than the 32-byte GR header: {path}")
    version, edge_size, vertices, edges = struct.unpack("<4Q", header)
    if version != 1 or edge_size != 4:
        raise ValueError(
            f"unsupported GR header version={version} edge_size={edge_size}: {path}")
    if vertices == 0:
        raise ValueError(f"graph has no vertices: {path}")
    return {
        "version": version, "edge_size": edge_size,
        "vertices": vertices, "edges": edges,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_statistics(rows, key):
    values = [float(row[key]) for row in rows]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"invalid or missing positive {key} samples")
    center = median(values)
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "samples_ms": values,
        "median_ms": center,
        "q1_ms": q1,
        "q3_ms": q3,
        "iqr_ms": q3 - q1,
        "mad_ms": median(abs(value - center) for value in values),
    }


def formal_rows(record):
    return [row for row in record.get("samples", [])
            if row.get("warmup") == "0"]


def summarize_record(record, expected_repeats):
    rows = formal_rows(record)
    result = {
        "valid": record.get("valid", False),
        "reason": record.get("reason"),
        "formal_sample_count": len(rows),
        "expected_formal_samples": expected_repeats,
    }
    if result["valid"] and len(rows) == expected_repeats:
        try:
            result["solve"] = sample_statistics(rows, "solve_ms")
            result["query_wall"] = sample_statistics(rows, "query_wall_ms")
        except (KeyError, ValueError) as error:
            result.update(valid=False, reason=str(error))
    return result


def paired_summary(single, dual):
    valid = single.get("valid", False) and dual.get("valid", False)
    result = {"valid": valid, "T1": single, "T2": dual}
    if valid:
        result["S_solve_T1_over_T2"] = (
            single["solve"]["median_ms"] / dual["solve"]["median_ms"])
        result["S_query_wall_T1_over_T2"] = (
            single["query_wall"]["median_ms"] /
            dual["query_wall"]["median_ms"])
    else:
        result["S_solve_T1_over_T2"] = None
        result["S_query_wall_T1_over_T2"] = None
    return result


def measurement_is_valid(records, combined_summary):
    return (all(record.get("valid", False) for record in records) and
            combined_summary.get("valid", False))


def parse_integer_log_line(line, prefix, expected_fields):
    """Parse one strict ``PREFIX key=int ...`` line without I/O."""
    tokens = line.split()
    errors = []
    if not tokens or tokens[0] != prefix:
        return None, [f"line does not start with {prefix}: {line!r}"]
    values = {}
    for token in tokens[1:]:
        if "=" not in token:
            errors.append(f"{prefix} token lacks '=': {token!r}")
            continue
        key, raw_value = token.split("=", 1)
        if key in values:
            errors.append(f"{prefix} duplicate field {key}")
            continue
        try:
            values[key] = int(raw_value)
        except ValueError:
            errors.append(f"{prefix} non-integer {key}={raw_value!r}")
    expected = set(expected_fields)
    actual = set(values)
    if actual != expected:
        errors.append(
            f"{prefix} fields differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}")
    return values, errors


def frozen_l2_capacity():
    return {
        "budget": FORMAL_L2_BUDGET_BYTES,
        "record_bytes": FORMAL_L2_RECORD_BYTES,
        "buckets": FORMAL_L2_BUCKETS,
        "per_bucket": FORMAL_L2_PER_BUCKET_CAPACITY,
        "allocated_records": FORMAL_L2_ALLOCATED_RECORDS,
        "counter_bits": FORMAL_L2_COUNTER_BITS,
    }


def parse_l2_capacity_contract(raw, expected_rows, *, require_frozen):
    """Validate process-level L2 allocation logs, including formal geometry."""
    errors = []
    rows = []
    for line in raw.splitlines():
        if not line.startswith("L2_CAPACITY "):
            continue
        values, line_errors = parse_integer_log_line(
            line, "L2_CAPACITY", L2_CAPACITY_FIELDS)
        errors.extend(line_errors)
        rows.append(None if line_errors else values)

    if len(rows) != expected_rows:
        errors.append(
            f"expected {expected_rows} process-level L2_CAPACITY lines, "
            f"got {len(rows)}")
    valid_rows = [row for row in rows if row is not None]
    if len(valid_rows) > 1 and any(
            row != valid_rows[0] for row in valid_rows[1:]):
        errors.append("the GPUs reported different L2_CAPACITY values")

    for row in valid_rows:
        expected_records = (row["budget"] // row["record_bytes"]
                            if row["record_bytes"] > 0 else -1)
        expected_per_bucket = (
            expected_records // row["buckets"] // 512 * 512
            if row["buckets"] > 0 else -1)
        if (row["budget"] <= 0 or row["record_bytes"] <= 0 or
                row["buckets"] <= 0 or row["per_bucket"] <= 0 or
                row["allocated_records"] != expected_records or
                row["per_bucket"] != expected_per_bucket or
                row["counter_bits"] != 32):
            errors.append(
                "L2_CAPACITY differs from the exact allocation formula "
                f"budget/record/buckets with 512-record blocks: {row}")
        if require_frozen and row != frozen_l2_capacity():
            errors.append(
                "formal L2_CAPACITY differs from the frozen exact "
                f"configuration: actual={row} "
                f"expected={frozen_l2_capacity()}")

    return {"valid": not errors, "rows": rows, "errors": errors}


def parse_oracle_contract(raw, expected_samples, expected_vertices):
    """Require one successful WIDE_ORACLE immediately before every BENCH."""
    pattern = re.compile(r"^WIDE_ORACLE vertices=(\d+) correct=(\d+)$")
    pending = []
    samples = []
    errors = []
    for line in raw.splitlines():
        if line.startswith("WIDE_ORACLE "):
            match = pattern.fullmatch(line)
            if not match:
                errors.append(f"malformed WIDE_ORACLE line: {line!r}")
                pending.append(None)
            else:
                pending.append(tuple(map(int, match.groups())))
        elif line.startswith("BENCH ") and "algorithm=MLMQ" in line:
            samples.append(pending)
            pending = []
    if len(samples) != expected_samples:
        errors.append(
            f"expected {expected_samples} BENCH oracle groups, got {len(samples)}")
    for index, rows in enumerate(samples):
        if len(rows) != 1:
            errors.append(
                f"sample {index} expected one WIDE_ORACLE line, got {len(rows)}")
        elif rows[0] is not None and rows[0] != (expected_vertices, 1):
            errors.append(
                f"sample {index} WIDE_ORACLE is vertices={rows[0][0]} "
                f"correct={rows[0][1]}, expected vertices={expected_vertices} "
                "correct=1")
    # The independent single implementation prints one final full audit after
    # its per-query checks.  It is evidence, but it must not be mistaken for a
    # query sample or allowed to accumulate without bound.
    if len(pending) > 1:
        errors.append(f"expected at most one trailing final oracle, got {len(pending)}")
    elif pending and pending[0] != (expected_vertices, 1):
        errors.append("trailing final WIDE_ORACLE is malformed or unsuccessful")
    return {
        "valid": not errors,
        "expected_samples": expected_samples,
        "observed_samples": len(samples),
        "trailing_final_audit": len(pending),
        "errors": errors,
    }


def parse_dual_contract(raw, expected_samples, blocks, delta, queue_type,
                        expected_window_mode=None, expected_window_min=None,
                        expected_window_max=None, expected_idle_backoff=None,
                        require_l2_final=False):
    """Parse and validate per-query dual configuration/protocol log fixtures."""
    errors = []
    samples = []
    pending = {"l3_config": [], "worker_ack": [], "l2_final": []}
    l2_present = any(line.startswith("L2_FINAL ")
                     for line in raw.splitlines())
    capacity_present = any(line.startswith("L2_CAPACITY ")
                           for line in raw.splitlines())

    def add_line(line, prefix, fields, key):
        values, line_errors = parse_integer_log_line(line, prefix, fields)
        errors.extend(line_errors)
        pending[key].append(None if line_errors else values)

    for line in raw.splitlines():
        if line.startswith("L3_CONFIG "):
            add_line(line, "L3_CONFIG", L3_CONFIG_FIELDS, "l3_config")
        elif line.startswith("L3_WORKER_ACK "):
            add_line(line, "L3_WORKER_ACK", L3_WORKER_ACK_FIELDS, "worker_ack")
        elif line.startswith("L2_FINAL "):
            add_line(line, "L2_FINAL", L2_FINAL_FIELDS, "l2_final")
        elif line.startswith("BENCH ") and "algorithm=MLMQ" in line:
            samples.append(pending)
            pending = {"l3_config": [], "worker_ack": [], "l2_final": []}

    if any(pending.values()):
        errors.append("dual contract lines remain after the final BENCH row")
    if len(samples) != expected_samples:
        errors.append(
            f"expected {expected_samples} dual BENCH contract groups, "
            f"got {len(samples)}")

    check_capacity = require_l2_final or l2_present or capacity_present
    capacity_contract = parse_l2_capacity_contract(
        raw, 2 if check_capacity else 0, require_frozen=require_l2_final)
    errors.extend(capacity_contract["errors"])
    capacity_rows = capacity_contract["rows"]
    valid_capacity = [row for row in capacity_rows if row is not None]

    expected_window = {
        "window_mode": expected_window_mode,
        "window_min": expected_window_min,
        "window_max": expected_window_max,
        "idle_backoff": expected_idle_backoff,
    }
    for sample_index, sample in enumerate(samples):
        for key in ("l3_config", "worker_ack"):
            if len(sample[key]) != 2:
                errors.append(
                    f"sample {sample_index} expected two {key} lines, "
                    f"got {len(sample[key])}")
        check_l2_final = require_l2_final or l2_present
        if check_l2_final and len(sample["l2_final"]) != 2:
            errors.append(
                f"sample {sample_index} expected two l2_final lines, "
                f"got {len(sample['l2_final'])}")

        for key in ("l3_config", "worker_ack"):
            valid_rows = [row for row in sample[key] if row is not None]
            if len(valid_rows) == 2 and sorted(row["gpu"] for row in valid_rows) != [0, 1]:
                errors.append(
                    f"sample {sample_index} {key} GPUs are not exactly 0 and 1")
        if check_l2_final:
            valid_l2 = [row for row in sample["l2_final"] if row is not None]
            if len(valid_l2) == 2 and sorted(row["gpu"] for row in valid_l2) != [0, 1]:
                errors.append(
                    f"sample {sample_index} l2_final GPUs are not exactly 0 and 1")

        for row in sample["l3_config"]:
            if row is None:
                continue
            if (row["work_blocks"] <= 0 or row["delta"] <= 0 or
                    row["work_blocks"] != blocks or row["delta"] != delta):
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} L3_CONFIG "
                    "blocks/delta differs from requested values")
            if row["queue_type"] != queue_type:
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} queue_type="
                    f"{row['queue_type']}, expected {queue_type}")
            if row["worker_recovery"] != 1 or row["term_wait_ack"] != 1:
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} safety flags "
                    f"worker_recovery={row['worker_recovery']} "
                    f"term_wait_ack={row['term_wait_ack']}, expected 1/1")
            if (row["window_mode"] not in (0, 1, 2) or
                    row["window_min"] <= 0 or
                    row["window_max"] < row["window_min"] or
                    row["idle_backoff"] not in (0, 1)):
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} invalid window "
                    f"mode={row['window_mode']} min={row['window_min']} "
                    f"max={row['window_max']} idle={row['idle_backoff']}")
            for field, expected in expected_window.items():
                if expected is not None and row[field] != expected:
                    errors.append(
                        f"sample {sample_index} gpu {row['gpu']} {field}="
                        f"{row[field]}, expected {expected}")

        for row in sample["worker_ack"]:
            if row is None:
                continue
            if row["work_blocks"] != blocks:
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} ACK blocks="
                    f"{row['work_blocks']}, expected {blocks}")
            expected_active = row["work_blocks"] * row["warps_per_block"]
            if (row["work_blocks"] <= 0 or row["warps_per_block"] <= 0 or
                    row["warps_per_block"] != 16 or
                    row["active_slots"] != expected_active or
                    row["active_slots"] != row["capacity"]):
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} invalid ACK geometry "
                    f"active={row['active_slots']} expected={expected_active} "
                    f"capacity={row['capacity']}")

        for row in sample["l2_final"]:
            if row is None:
                continue
            if (row["buckets"] <= 0 or row["reads"] < 0 or
                    row["reads"] != row["writes"] or
                    row["writes"] != row["completed"] or
                    row["guarded_writes"] != row["writes"] or
                    row["max_bucket_writes"] < 0 or
                    row["max_bucket_writes"] > row["writes"] or
                    row["per_bucket_capacity"] <= 0 or
                    row["max_bucket_writes"] > row["per_bucket_capacity"] or
                    row["total_capacity"] !=
                    row["buckets"] * row["per_bucket_capacity"] or
                    row["total_capacity"] <= 0 or
                    row["total_capacity"] > 2147483647 or
                    row["counter_bits"] != 32 or
                    row["overflow_guard"] != 1 or
                    row["overflow_detected"] != 0 or row["no_wrap"] != 1):
                errors.append(
                    f"sample {sample_index} gpu {row['gpu']} L2 final gate "
                    f"failed reads={row['reads']} writes={row['writes']} "
                    f"completed={row['completed']} guarded_writes="
                    f"{row['guarded_writes']} max_bucket_writes="
                    f"{row['max_bucket_writes']} per_bucket_capacity="
                    f"{row['per_bucket_capacity']} total_capacity="
                    f"{row['total_capacity']} counter_bits="
                    f"{row['counter_bits']} overflow_guard="
                    f"{row['overflow_guard']} overflow_detected="
                    f"{row['overflow_detected']} no_wrap={row['no_wrap']}")
            if len(valid_capacity) == 2:
                capacity = valid_capacity[0]
                if (row["buckets"] != capacity["buckets"] or
                        row["per_bucket_capacity"] != capacity["per_bucket"] or
                        row["total_capacity"] !=
                        capacity["buckets"] * capacity["per_bucket"] or
                        row["total_capacity"] > capacity["allocated_records"] or
                        row["counter_bits"] != capacity["counter_bits"]):
                    errors.append(
                        f"sample {sample_index} gpu {row['gpu']} L2_FINAL "
                        "does not match this process's L2_CAPACITY")

    return {
        "valid": not errors,
        "expected_samples": expected_samples,
        "observed_samples": len(samples),
        "l2_final_required": require_l2_final,
        "l2_final_present": l2_present,
        "l2_capacity_present": capacity_present,
        "l2_capacity_rows": capacity_rows,
        "samples": samples,
        "errors": errors,
    }


def verify_effective_contract(record, log_path, gpu_count, args, graph_header):
    """Make logged oracle and effective runtime parameters part of validity."""
    raw = log_path.read_text(errors="replace")
    expected_process_samples = args.warmups + args.repeats
    errors = []

    oracle_contract = parse_oracle_contract(
        raw, expected_process_samples, graph_header["vertices"])
    oracle_lines = raw.count("WIDE_ORACLE ")
    errors.extend(oracle_contract["errors"])
    for marker in ("ORACLE_MISMATCH", "ORACLE_SIZE_ERROR"):
        if marker in raw:
            errors.append(f"solver reported {marker}")

    l3_config_lines = [line for line in raw.splitlines()
                       if line.startswith("L3_CONFIG ")]
    no_l3_pattern = re.compile(
        r"NO_L3_CONFIG workers=(\d+) delta=(\d+) queue=(\S+)")
    no_l3_configs = [match.groups() for match in no_l3_pattern.finditer(raw)]
    no_l3_launch_pattern = re.compile(
        r"NO_L3_LAUNCH work_blocks=(\d+) delta=(\d+) repeat=(\d+)")
    no_l3_launches = [tuple(map(int, match.groups()))
                      for match in no_l3_launch_pattern.finditer(raw)]
    dual_contract = None
    single_capacity_contract = None
    if gpu_count == 2:
        expected_configs = gpu_count * expected_process_samples
        dual_contract = parse_dual_contract(
            raw, expected_process_samples, args.blocks, args.delta,
            QUEUE_TYPE_IDS[args.queue],
            args.expected_window_mode, args.expected_window_min,
            args.expected_window_max, args.expected_idle_backoff,
            require_l2_final=args.sampling == "formal")
        errors.extend(dual_contract["errors"])
    else:
        expected_configs = 0
        if len(no_l3_configs) != 1:
            errors.append(
                f"expected one NO_L3_CONFIG line, got {len(no_l3_configs)}")
        elif (int(no_l3_configs[0][1]) != args.delta or
              no_l3_configs[0][2] != args.queue):
            errors.append("logged no-L3 delta/queue differs from requested values")
        if l3_config_lines:
            errors.append("independent no-L3 binary unexpectedly logged L3_CONFIG")
        if len(no_l3_launches) != expected_process_samples:
            errors.append(
                f"expected {expected_process_samples} NO_L3_LAUNCH lines, "
                f"got {len(no_l3_launches)}")
        elif any(blocks != args.blocks or delta != args.delta or repeat != index
                 for index, (blocks, delta, repeat) in enumerate(no_l3_launches)):
            errors.append("logged no-L3 blocks/delta/repeat differs from requested values")
        capacity_present = any(
            line.startswith("L2_CAPACITY ") for line in raw.splitlines())
        if args.sampling == "formal" or capacity_present:
            single_capacity_contract = parse_l2_capacity_contract(
                raw, 1, require_frozen=args.sampling == "formal")
            errors.extend(single_capacity_contract["errors"])

    partition_pattern = re.compile(
        r"GPU(\d+) partition: \[(\d+), (\d+)\)")
    partitions = [tuple(map(int, match.groups()))
                  for match in partition_pattern.finditer(raw)]
    vertices = graph_header["vertices"]
    if gpu_count == 1:
        expected_partitions = []
        if partitions:
            errors.append("independent no-L3 binary unexpectedly logged GPU partitions")
    else:
        cut = vertices * args.cut_percent // 100
        expected_partitions = [(0, 0, cut), (1, cut, vertices)]
        for expected in expected_partitions:
            if expected not in partitions:
                errors.append(f"missing effective partition {expected}")

    record["effective_contract"] = {
        "oracle_lines": oracle_lines,
        "expected_oracle_lines": expected_process_samples,
        "oracle_contract": oracle_contract,
        "l3_config_lines": len(l3_config_lines),
        "no_l3_config_lines": len(no_l3_configs),
        "no_l3_launch_lines": len(no_l3_launches),
        "expected_l3_config_lines": expected_configs,
        "dual_contract": dual_contract,
        "single_capacity_contract": single_capacity_contract,
        "partitions": partitions,
        "expected_partitions": expected_partitions,
        "errors": errors,
    }
    if errors:
        previous = record.get("reason", "unknown failure")
        record["valid"] = False
        record["reason"] = previous + "; contract: " + "; ".join(errors)
    return record


def git_snapshot(repo_root):
    def git(*arguments):
        result = subprocess.run(
            [GIT_EXECUTABLE, "-C", str(repo_root), *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=30)
        return {"rc": result.returncode, "output": result.stdout.rstrip()}
    return {"head": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def read_json_object(path, description):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{description} is not a JSON object: {path}")
    return value


def read_json_array(path, description):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{description} is not a JSON string array: {path}")
    return value


def validate_pair_provenance(provenance):
    pair = provenance.get("pair_contract")
    dual = provenance.get("dual")
    single = provenance.get("single")
    if not isinstance(pair, dict) or not isinstance(dual, dict) or not isinstance(single, dict):
        raise SystemExit("formal provenance lacks pair_contract/dual/single objects")
    if (pair.get("dual") != "current dual-GPU L3 paper configuration" or
            pair.get("single") != "independent committed no-L3 single-GPU source" or
            pair.get("single_is_dual_n1") is not False):
        raise SystemExit("formal provenance has an invalid pair contract")
    if (dual.get("builder") != "scripts/multigpu/build_l3_paper_config.sh" or
            dual.get("macro_authority") != "builder command.json"):
        raise SystemExit("formal provenance has an invalid dual builder contract")
    if (single.get("snapshot_materialization") !=
            "git archive HEAD (ignores working-tree edits)" or
            single.get("post_sync_check") !=
            "byte-for-byte recursive diff passed" or
            single.get("l3_compile_defines") != []):
        raise SystemExit("formal provenance has an invalid independent-single contract")


def parse_compile_definitions(command, label):
    definitions = {}
    for argument in command:
        if argument == "-D" or argument.startswith("--define-macro"):
            raise SystemExit(
                f"formal {label} command uses forbidden alternate define syntax: "
                f"{argument!r}")
        if not argument.startswith("-D"):
            continue
        definition = argument[2:]
        if not definition or "=" not in definition:
            raise SystemExit(
                f"formal {label} command has non-canonical define: {argument!r}")
        name, value = definition.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or not value:
            raise SystemExit(
                f"formal {label} command has malformed define: {argument!r}")
        if name in definitions:
            raise SystemExit(
                f"formal {label} command defines {name} more than once")
        definitions[name] = value
    return definitions


def expected_pair_commands(pair, repo_root, pair_version):
    compiler = str(Path(pair_version.get("nvcc", "")).resolve(strict=True))
    boost = str(Path(pair_version.get("boost_include_dir", "")).resolve(strict=True))
    dual = [
        compiler,
        str(repo_root / "SSSP/main.cu"),
        str(repo_root / "SSSP/csr_graph.cu"),
        str(repo_root / "SSSP/sssp_run.cu"),
        "-o", str(pair / "dual_build/mlmq"),
        "-DWORK_COUNT=false",
        "-DMLMQ_WORKER_THREADS=512",
        f"-DBNUM={FORMAL_L2_BUCKETS}",
        f"-DBUCKET_MAX={FORMAL_L2_BUCKET_MAX}",
        f"-Dl2_batch_size={FORMAL_L2_BATCH_SIZE}",
        "-DL3_COOPERATIVE_COLLECT=true",
        "-DL3_DIRECT_RX=true",
        "-DL3_RETAIN_TX=true",
        "-DL3_WINDOW_MODE=2",
        "-DL3_WINDOW_MIN_CYCLES=25000ull",
        "-DL3_WINDOW_MAX_CYCLES=25000ull",
        "-DL3_IDLE_BACKOFF=false",
        "-DL3_WORKER_RECOVERY=true",
        "-DL3_TERM_WAIT_ACK=true",
        "-DL3_ACK_SCAN=true",
        "-DL3_ACK_WIDE_SCAN=true",
        "-DL3_BOUNDARY_INDEX=true",
        "-DL3_L2_FINAL_COUNTS=true",
        "-DDQ_COUNTER_OVERFLOW_GUARD=true",
        "-DL3_CHAIN_SHORTCUTS=false",
        "-DL3_IDLE_TOKEN_PROBE=true",
        "-O3", "-m64", "-gencode=arch=compute_80,code=sm_80", "-rdc=true",
        "-lcuda", "-lcudart", "-w",
        "-I" + str(repo_root / "core/include"), "-I" + boost, "-lcusparse",
    ]
    single_source = pair / "single_source"
    single = [
        compiler,
        str(single_source / "SSSP/main.cu"),
        str(single_source / "SSSP/csr_graph.cu"),
        str(single_source / "SSSP/sssp_run.cu"),
        "-o", str(pair / "single_build/mlmq"),
        "-DWORK_COUNT=false", "-DMLMQ_WORKER_THREADS=512",
        f"-DBNUM={FORMAL_L2_BUCKETS}",
        f"-DBUCKET_MAX={FORMAL_L2_BUCKET_MAX}",
        f"-Dl2_batch_size={FORMAL_L2_BATCH_SIZE}",
        "-DDQ_COUNTER_OVERFLOW_GUARD=true",
        "-O3", "-m64", "-gencode=arch=compute_80,code=sm_80", "-rdc=true",
        "-lcuda", "-lcudart", "-w",
        "-I" + str(single_source / "core/include"), "-I" + boost, "-lcusparse",
    ]
    return {"dual": dual, "single": single}


def validate_pair_commands(dual_command, single_command, *, pair, repo_root,
                           pair_version):
    definitions = {
        "dual": parse_compile_definitions(dual_command, "dual"),
        "single": parse_compile_definitions(single_command, "single"),
    }
    expected = expected_pair_commands(pair, repo_root, pair_version)
    actual = {"dual": dual_command, "single": single_command}
    for label in ("dual", "single"):
        if actual[label] != expected[label]:
            mismatch = next(
                (index for index, values in enumerate(zip(actual[label], expected[label]))
                 if values[0] != values[1]),
                min(len(actual[label]), len(expected[label])))
            raise SystemExit(
                f"formal {label} command differs from canonical argv at index "
                f"{mismatch}: actual={actual[label]} expected={expected[label]}")
    expected_definitions = {
        label: parse_compile_definitions(command, "expected-" + label)
        for label, command in expected.items()
    }
    if definitions != expected_definitions:
        raise SystemExit(
            f"formal compile definitions differ: actual={definitions} "
            f"expected={expected_definitions}")


def validate_formal_pair_build(pair_build, dual_binary, single_binary,
                               git_head, repo_root):
    """Bind formal binaries to one successful clean-HEAD pair build."""
    repo_root = Path(repo_root).resolve(strict=True)
    pair = pair_build.resolve(strict=True)
    if not pair.is_dir():
        raise SystemExit(f"formal pair-build is not a directory: {pair}")
    expected_paths = {
        "dual": (pair / "dual_build/mlmq").resolve(strict=True),
        "single": (pair / "single_build/mlmq").resolve(strict=True),
    }
    actual_paths = {"dual": dual_binary, "single": single_binary}
    if actual_paths != expected_paths:
        raise SystemExit(
            "formal binaries must be the exact pair-build outputs: "
            f"expected={expected_paths} actual={actual_paths}")

    pair_status = read_json_object(pair / "status.json", "pair status")
    if (pair_status.get("rc") != 0 or pair_status.get("state") != "complete" or
            pair_status.get("dual_rc") != 0 or
            pair_status.get("single_rc") != 0):
        raise SystemExit(f"formal pair-build is incomplete: {pair_status}")

    pair_version = read_json_object(pair / "version.json", "pair version")
    if pair_version.get("head") != git_head:
        raise SystemExit(
            f"formal pair-build HEAD {pair_version.get('head')!r} "
            f"differs from current HEAD {git_head!r}")
    if pair_version.get("git_status_porcelain") != []:
        raise SystemExit("formal pair-build was created from a dirty worktree")
    if pair_version.get("head_after_build") != git_head:
        raise SystemExit("formal pair-build HEAD changed while compiling")
    if pair_version.get("git_status_porcelain_after_build") != []:
        raise SystemExit("formal pair-build worktree became dirty while compiling")
    if pair_version.get("dual_source_post_build_match") is not True:
        raise SystemExit("formal pair-build source archive differs after compilation")
    if pair_version.get("dual_builder") != \
            "scripts/multigpu/build_l3_paper_config.sh":
        raise SystemExit("formal pair-build names a non-canonical dual builder")
    builder = repo_root / pair_version["dual_builder"]
    if (not builder.is_file() or builder.is_symlink() or
            pair_version.get("dual_builder_sha256") != sha256(builder)):
        raise SystemExit("formal pair-build dual builder hash is stale or missing")
    expected_script_hashes = {
        "pair_builder": "scripts/multigpu/build_l3_30h_pair.sh",
        "single_adapter": "scripts/multigpu/prepare_l3_30h_single.py",
    }
    for field, expected_relative in expected_script_hashes.items():
        if pair_version.get(field) != expected_relative:
            raise SystemExit(f"formal pair-build names a non-canonical {field}")
        script = repo_root / expected_relative
        if (not script.is_file() or script.is_symlink() or
                pair_version.get(field + "_sha256") != sha256(script)):
            raise SystemExit(f"formal pair-build {field} hash is stale or missing")
    try:
        compiler = Path(pair_version["nvcc"]).resolve(strict=True)
        boost = Path(pair_version["boost_include_dir"]).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise SystemExit(f"formal pair-build compiler/include path is invalid: {error}") from error
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"formal compiler is not an executable regular file: {compiler}")
    if not boost.is_dir():
        raise SystemExit(f"formal Boost include path is not a directory: {boost}")
    if pair_version.get("nvcc_resolved") != str(compiler):
        raise SystemExit("formal pair-build resolved compiler path differs")
    if pair_version.get("nvcc_sha256") != sha256(compiler):
        raise SystemExit("formal pair-build compiler hash is stale or missing")
    if pair_version.get("nvcc") != str(compiler) or not compiler.is_absolute():
        raise SystemExit("formal pair-build must invoke the resolved absolute compiler")
    if pair_version.get("repository_root") != str(repo_root):
        raise SystemExit("formal pair-build repository root differs from clean HEAD")
    build_environment = pair_version.get("build_environment")
    if not isinstance(build_environment, dict):
        raise SystemExit("formal pair-build lacks its selected build environment")
    allowed_build_environment = {
        "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "NVCC",
        "BOOST_INCLUDE_DIR",
    }
    required_build_environment = {
        "PATH", "LANG", "LC_ALL", "NVCC", "BOOST_INCLUDE_DIR",
    }
    if (set(build_environment) - allowed_build_environment or
            not required_build_environment.issubset(build_environment)):
        raise SystemExit(
            "formal pair-build environment keys differ from the allowlist: "
            f"{sorted(build_environment)}")
    if (build_environment.get("PATH") != FORMAL_TOOL_PATH or
            build_environment.get("LANG") != "C" or
            build_environment.get("LC_ALL") != "C" or
            build_environment.get("NVCC") != str(compiler) or
            build_environment.get("BOOST_INCLUDE_DIR") !=
            pair_version.get("boost_include_dir")):
        raise SystemExit("formal pair-build environment values are not canonical")
    if pair_version.get("blocked_build_environment_variables") != \
            list(FORMAL_BLOCKED_BUILD_ENV):
        raise SystemExit("formal pair-build blocked-environment contract differs")
    if pair_version.get("blocked_build_environment_present") != []:
        raise SystemExit("formal pair-build inherited a compiler-influencing variable")

    components = {}
    provenances = {}
    commands = {}
    expected_roles = {"dual": "dual_l3_paper", "single": "single_no_l3"}
    for label, role in expected_roles.items():
        build_dir = pair / f"{label}_build"
        status = read_json_object(build_dir / "status.json", f"{label} status")
        hashes = read_json_object(build_dir / "hashes.json", f"{label} hashes")
        if status.get("rc") != 0 or status.get("role") != role:
            raise SystemExit(f"formal {label} build status is invalid: {status}")
        required = (
            "mlmq", "source.tgz", "command.json", "build.log",
            "status.json", "version.json", "provenance.json",
        )
        verified = {}
        for name in required:
            artifact = build_dir / name
            expected_hash = hashes.get(name)
            if (artifact.is_symlink() or not artifact.is_file() or
                    not isinstance(expected_hash, str)):
                raise SystemExit(
                    f"formal {label} build lacks hashed artifact {name}")
            actual_hash = sha256(artifact)
            if actual_hash != expected_hash:
                raise SystemExit(
                    f"formal {label} artifact hash mismatch for {name}")
            verified[name] = actual_hash
        provenance = read_json_object(
            build_dir / "provenance.json", f"{label} provenance")
        validate_pair_provenance(provenance)
        provenances[label] = provenance
        component_version = read_json_object(
            build_dir / "version.json", f"{label} component version")
        if component_version != pair_version:
            raise SystemExit(
                f"formal {label} component version differs from pair version")
        commands[label] = read_json_array(
            build_dir / "command.json", f"{label} command")
        components[label] = {
            "directory": str(build_dir),
            "role": role,
            "verified_sha256": verified,
        }
    if provenances["dual"] != provenances["single"]:
        raise SystemExit("formal dual and single provenance records differ")
    validate_pair_commands(
        commands["dual"], commands["single"], pair=pair,
        repo_root=repo_root, pair_version=pair_version)
    source_binding = validate_pair_source_trees(pair, repo_root)
    return {
        "directory": str(pair),
        "status": pair_status,
        "version": pair_version,
        "components": components,
        "compiler": {"path": str(compiler), "sha256": sha256(compiler)},
        "source_binding": source_binding,
    }


def formal_build_environment():
    compiler = shutil.which("nvcc", path=FORMAL_TOOL_PATH)
    if compiler is None:
        raise SystemExit("formal build requires nvcc on PATH")
    compiler = str(Path(compiler).resolve(strict=True))
    if not Path(compiler).is_file() or not os.access(compiler, os.X_OK):
        raise SystemExit(f"formal nvcc is not an executable regular file: {compiler}")
    boost = Path("/a100-data/wyh/boost_1_87_0").resolve(strict=True)
    if not boost.is_dir():
        raise SystemExit(f"formal build Boost include is not a directory: {boost}")
    environment = {
        "PATH": FORMAL_TOOL_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "NVCC": compiler,
        "BOOST_INCLUDE_DIR": str(boost),
    }
    for key in ("TZ", "TMPDIR"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    cleared = sorted(
        key for key in os.environ
        if key in FORMAL_BLOCKED_BUILD_ENV or
        key.startswith(("CMAKE_", "CUDAFLAGS_")))
    return environment, cleared


def build_formal_pair(repo_root, output):
    builder = repo_root / "scripts/multigpu/build_l3_30h_pair.sh"
    if not builder.is_file() or builder.is_symlink() or not os.access(builder, os.X_OK):
        raise SystemExit(f"formal pair builder is not a trusted executable file: {builder}")
    pair = output / "pair_build"
    environment, cleared = formal_build_environment()
    command = [str(builder), str(pair)]
    record = {
        "command": command,
        "builder_sha256": sha256(builder),
        "environment": environment,
        "cleared_environment_variables": cleared,
        "started_utc": now_utc(),
        "timeout_seconds": FORMAL_BUILD_TIMEOUT_SECONDS,
    }
    log = output / "formal_pair_build_driver.log"
    try:
        process = subprocess.run(
            command, cwd=repo_root, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=FORMAL_BUILD_TIMEOUT_SECONDS)
        record.update(rc=process.returncode, timed_out=False)
        raw = process.stdout
    except subprocess.TimeoutExpired as error:
        raw = error.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        raw += f"\nTIMEOUT_SECONDS={FORMAL_BUILD_TIMEOUT_SECONDS}\n"
        record.update(rc=124, timed_out=True)
    log.write_text(raw)
    record.update(
        finished_utc=now_utc(), log=str(log), log_sha256=sha256(log))
    write_json(output / "formal_pair_build_record.json", record)
    if record["rc"] != 0:
        raise SystemExit(
            f"fresh formal pair build failed with rc={record['rc']}; see {log}")
    return pair, record


def formal_pair_integrity_snapshot(pair, repo_root):
    names = (
        "status.json", "version.json",
        "dual_build/mlmq", "dual_build/source.tgz", "dual_build/command.json",
        "dual_build/build.log", "dual_build/status.json",
        "dual_build/hashes.json", "dual_build/version.json",
        "dual_build/provenance.json", "single_build/mlmq",
        "single_build/source.tgz", "single_build/command.json",
        "single_build/build.log", "single_build/status.json",
        "single_build/hashes.json", "single_build/version.json",
        "single_build/provenance.json",
    )
    hashes = {}
    for name in names:
        path = pair / name
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"formal pair integrity file is missing/not regular: {path}")
        hashes[name] = sha256(path)
    return {"git": git_snapshot(repo_root), "sha256": hashes}


def data_integrity_snapshot(paths):
    snapshot = {}
    for name in ("graph", "oracle"):
        path = paths[name]
        stat = path.stat()
        snapshot[name] = {
            "path": str(path), "size": stat.st_size, "sha256": sha256(path),
        }
    return snapshot


def _git_head_file(repo_root, relative):
    process = subprocess.run(
        [GIT_EXECUTABLE, "-C", str(repo_root), "show",
         f"HEAD:{relative.as_posix()}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise SystemExit(
            f"formal input contract is not committed at HEAD: {relative}")
    return process.stdout


def validate_formal_input_contract(repo_root, contract_path, args, paths,
                                   graph_header):
    canonical = (repo_root / FORMAL_INPUT_CONTRACT).resolve(strict=True)
    contract_path = Path(contract_path)
    if not contract_path.is_absolute():
        contract_path = repo_root / contract_path
    contract_path = contract_path.resolve(strict=True)
    if contract_path != canonical:
        raise SystemExit(
            f"formal input contract must be the canonical committed file {canonical}")
    relative = canonical.relative_to(repo_root)
    raw = canonical.read_bytes()
    if raw != _git_head_file(repo_root, relative):
        raise SystemExit("formal input contract differs from clean HEAD")
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"formal input contract is invalid JSON: {error}") from error
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        raise SystemExit("formal input contract must be a schema-1 object")
    expected = {
        "graph_path": str(paths["graph"]),
        "graph_sha256": sha256(paths["graph"]),
        "oracle_path": str(paths["oracle"]),
        "oracle_sha256": sha256(paths["oracle"]),
        "source": args.source,
        "delta": args.delta,
        "cut_percent": args.cut_percent,
        "blocks": args.blocks,
        "l2_buckets": FORMAL_L2_BUCKETS,
        "l2_bucket_max": FORMAL_L2_BUCKET_MAX,
        "l2_batch_size": FORMAL_L2_BATCH_SIZE,
        "l2_budget_bytes": FORMAL_L2_BUDGET_BYTES,
        "l2_record_bytes": FORMAL_L2_RECORD_BYTES,
        "l2_allocated_records": FORMAL_L2_ALLOCATED_RECORDS,
        "l2_per_bucket_capacity": FORMAL_L2_PER_BUCKET_CAPACITY,
        "l2_total_capacity": FORMAL_L2_TOTAL_CAPACITY,
        "l2_counter_bits": FORMAL_L2_COUNTER_BITS,
        "queue": args.queue,
        "final_audit": args.final_audit,
        "warmups_per_process": args.warmups,
        "repeats_per_process": args.repeats,
        "rounds": args.rounds,
        "window_mode": args.expected_window_mode,
        "window_min": args.expected_window_min,
        "window_max": args.expected_window_max,
        "idle_backoff": args.expected_idle_backoff,
        "vertices": graph_header["vertices"],
        "edges": graph_header["edges"],
    }
    expected_keys = {"schema", *expected}
    if set(contract) != expected_keys:
        raise SystemExit(
            "formal input contract fields differ: "
            f"missing={sorted(expected_keys - set(contract))} "
            f"extra={sorted(set(contract) - expected_keys)}")
    observed = {key: contract.get(key) for key in expected}
    if observed != expected:
        differences = {
            key: {"contract": observed[key], "requested": expected[key]}
            for key in expected if observed[key] != expected[key]
        }
        raise SystemExit(
            f"formal request differs from committed input contract: {differences}")
    return {
        "path": str(canonical), "sha256": hashlib.sha256(raw).hexdigest(),
        "head_path": relative.as_posix(), "contract": contract,
    }


def formal_runtime_environment():
    environment = {
        key: value for key, value in os.environ.items()
        if key in FORMAL_RUNTIME_ENV_KEYS or key.startswith("SLURM_")
    }
    environment["LANG"] = "C"
    environment["LC_ALL"] = "C"
    environment["PATH"] = FORMAL_TOOL_PATH
    cleared = sorted(key for key in os.environ if key not in environment)
    return environment, cleared


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", required=True,
                        choices=("exploratory", "formal"),
                        help="evidence label; formal enforces the plan's minimum sample counts")
    parser.add_argument("--dual-binary", type=Path,
                        help="exploratory-only L3 MLMQ binary, invoked with -n 2")
    parser.add_argument("--single-binary", type=Path,
                        help="exploratory-only independent no-L3 binary, invoked with -n 1")
    parser.add_argument("--pair-build", type=Path,
                        help="exploratory metadata only; formal always builds a fresh pair")
    parser.add_argument(
        "--input-contract", type=Path, default=FORMAL_INPUT_CONTRACT,
        help="formal-only canonical committed graph/oracle/configuration contract")
    parser.add_argument("--graph", required=True, type=Path,
                        help="the identical physical GR input passed to both binaries")
    parser.add_argument("--oracle", required=True, type=Path,
                        help="oracle consumed through L3_SUPPLEMENT_ORACLE by both binaries")
    parser.add_argument("--source", required=True, type=int)
    parser.add_argument("--delta", required=True, type=int)
    parser.add_argument("--cut-percent", required=True, type=int)
    parser.add_argument("--blocks", required=True, type=int,
                        help="MLMQ_WORK_BLOCKS for both independently built configurations")
    parser.add_argument("--warmups", required=True, type=int,
                        help="warmups in every configuration process")
    parser.add_argument("--repeats", required=True, type=int,
                        help="formal samples in every configuration process")
    parser.add_argument("--rounds", required=True, type=int,
                        help="paired rounds; adjacent rounds reverse configuration order")
    parser.add_argument("--timeout", required=True, type=int,
                        help="per-configuration process timeout in seconds")
    parser.add_argument("--out", required=True, type=Path,
                        help="new output directory; an existing path is rejected")
    parser.add_argument("--queue", choices=("L1SLF_L2DQ", "L1V_L2DQ"),
                        default="L1SLF_L2DQ")
    parser.add_argument("--final-audit", choices=("failure", "all"),
                        default="failure")
    parser.add_argument("--expected-window-mode", type=int, choices=(0, 1, 2),
                        help="expected dual L3_CONFIG window_mode")
    parser.add_argument("--expected-window-min", type=int,
                        help="expected dual L3_CONFIG window_min cycles")
    parser.add_argument("--expected-window-max", type=int,
                        help="expected dual L3_CONFIG window_max cycles")
    parser.add_argument("--expected-idle-backoff", type=int, choices=(0, 1),
                        help="expected dual L3_CONFIG idle_backoff (0 or 1)")
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("sampling must run inside a Slurm allocation")
    if args.source < 0:
        parser.error("source must be non-negative")
    if args.delta <= 0 or args.blocks <= 0 or args.timeout <= 0:
        parser.error("delta, blocks, and timeout must be positive")
    if not 1 <= args.cut_percent <= 99:
        parser.error("cut-percent must be in [1, 99]")
    if args.warmups < 0 or args.repeats < 1 or args.rounds < 1:
        parser.error("warmups must be non-negative; repeats/rounds must be positive")
    if ((args.expected_window_min is not None and args.expected_window_min <= 0) or
            (args.expected_window_max is not None and args.expected_window_max <= 0)):
        parser.error("expected window min/max must be positive when provided")
    if (args.expected_window_min is not None and
            args.expected_window_max is not None and
            args.expected_window_max < args.expected_window_min):
        parser.error("expected-window-max must be >= expected-window-min")
    if args.sampling == "formal" and (
            args.warmups != 1 or args.repeats != 5 or args.rounds != 2):
        parser.error(
            "formal sampling requires exactly 1 warmup, 5 repeats, and 2 "
            "reverse-order rounds (10 formal samples per configuration)")
    expected_window = (
        args.expected_window_mode, args.expected_window_min,
        args.expected_window_max, args.expected_idle_backoff,
    )
    if args.sampling == "formal" and any(value is None for value in expected_window):
        parser.error(
            "formal sampling requires all --expected-window-mode/min/max and "
            "--expected-idle-backoff values")
    if args.sampling == "formal" and any(
            value is not None for value in
            (args.dual_binary, args.single_binary, args.pair_build)):
        parser.error(
            "formal sampling forbids external --dual-binary/--single-binary/"
            "--pair-build; it always creates a fresh pair inside --out")
    if args.sampling == "exploratory" and (
            args.dual_binary is None or args.single_binary is None):
        parser.error(
            "exploratory sampling requires --dual-binary and --single-binary")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    return args


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2].resolve(strict=True)
    args.out = args.out.resolve(strict=False)
    git_info = git_snapshot(repo_root)
    if args.sampling == "formal":
        if git_info["head"]["rc"] != 0 or git_info["status"]["rc"] != 0:
            raise SystemExit("formal sampling requires readable git HEAD/status")
        if git_info["status"]["output"]:
            raise SystemExit(
                "formal sampling requires a clean worktree; git status is:\n" +
                git_info["status"]["output"])
        if args.out == repo_root or repo_root in args.out.parents:
            raise SystemExit(
                "formal output must be outside the repository so sampling "
                "cannot dirty the audited worktree")
    paths = {
        "graph": args.graph.resolve(strict=True),
        "oracle": args.oracle.resolve(strict=True),
    }
    for name in ("graph", "oracle"):
        if not paths[name].is_file():
            raise SystemExit(f"{name} is not a file: {paths[name]}")
    graph_header = read_gr_header(paths["graph"])
    if args.source >= graph_header["vertices"]:
        raise SystemExit(
            f"source {args.source} is outside graph with "
            f"{graph_header['vertices']} vertices")
    expected_oracle_bytes = graph_header["vertices"] * 4
    if paths["oracle"].stat().st_size != expected_oracle_bytes:
        raise SystemExit(
            f"oracle size is {paths['oracle'].stat().st_size}, expected "
            f"{expected_oracle_bytes} int32 bytes")

    args.out.mkdir(parents=True, exist_ok=False)
    formal_input = None
    pair_build_evidence = None
    pair_build_record = None
    pair_build = None
    integrity_before = None
    data_integrity_before = None
    if args.sampling == "formal":
        formal_input = validate_formal_input_contract(
            repo_root, args.input_contract, args, paths, graph_header)
        pair_build, pair_build_record = build_formal_pair(repo_root, args.out)
        post_build_git = git_snapshot(repo_root)
        if (post_build_git != git_info or post_build_git["status"]["output"] or
                post_build_git["head"]["output"] != git_info["head"]["output"]):
            raise SystemExit(
                "repository HEAD/status changed during the fresh formal build")
        paths.update({
            "dual_binary": (pair_build / "dual_build/mlmq").resolve(strict=True),
            "single_binary": (pair_build / "single_build/mlmq").resolve(strict=True),
        })
        pair_build_evidence = validate_formal_pair_build(
            pair_build, paths["dual_binary"], paths["single_binary"],
            git_info["head"]["output"], repo_root)
        integrity_before = formal_pair_integrity_snapshot(pair_build, repo_root)
        if integrity_before["git"] != git_info:
            raise SystemExit("repository changed before formal sampling")
        data_integrity_before = data_integrity_snapshot(paths)
    else:
        paths.update({
            "dual_binary": args.dual_binary.resolve(strict=True),
            "single_binary": args.single_binary.resolve(strict=True),
        })

    for name in ("dual_binary", "single_binary"):
        if not paths[name].is_file() or not os.access(paths[name], os.X_OK):
            raise SystemExit(f"{name} is not an executable file: {paths[name]}")
    if paths["dual_binary"] == paths["single_binary"]:
        raise SystemExit("dual and independent single binaries must be different files")
    binary_hashes = {name: sha256(paths[name])
                     for name in ("dual_binary", "single_binary")}
    if binary_hashes["dual_binary"] == binary_hashes["single_binary"]:
        raise SystemExit(
            "dual and independent single binaries have identical SHA256 hashes")

    manifest = {
        "created_utc": now_utc(),
        "status": "preflight_pending",
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "command": sys.argv,
        "configuration": {
            "sampling": args.sampling,
            "source": args.source,
            "delta": args.delta,
            "cut_percent": args.cut_percent,
            "blocks": args.blocks,
            "blocks_scope": "same requested value; independently logged by T1 and T2",
            "l2_geometry": {
                "buckets": FORMAL_L2_BUCKETS,
                "bucket_max": FORMAL_L2_BUCKET_MAX,
                "batch_size": FORMAL_L2_BATCH_SIZE,
                "budget_bytes": FORMAL_L2_BUDGET_BYTES,
                "record_bytes": FORMAL_L2_RECORD_BYTES,
                "allocated_records": FORMAL_L2_ALLOCATED_RECORDS,
                "per_bucket_capacity": FORMAL_L2_PER_BUCKET_CAPACITY,
                "total_capacity": FORMAL_L2_TOTAL_CAPACITY,
                "counter_bits": FORMAL_L2_COUNTER_BITS,
            },
            "warmups_per_process": args.warmups,
            "formal_repeats_per_process": args.repeats,
            "rounds": args.rounds,
            "timeout_seconds": args.timeout,
            "queue": args.queue,
            "final_audit": args.final_audit,
            "expected_window": {
                "mode": args.expected_window_mode,
                "min_cycles": args.expected_window_min,
                "max_cycles": args.expected_window_max,
                "idle_backoff": args.expected_idle_backoff,
            },
            "target_speedup": TARGET_SPEEDUP,
            "timing_formula": "S = median(all T1 solve_ms) / median(all T2 solve_ms)",
            "gpu_contract": {
                "single_no_l3": 1,
                "dual_l3": 2,
                "fallback_allowed": False,
            },
        },
        "inputs": {
            "dual_binary": {"path": str(paths["dual_binary"]),
                            "sha256": binary_hashes["dual_binary"]},
            "single_binary": {"path": str(paths["single_binary"]),
                              "sha256": binary_hashes["single_binary"]},
            "graph": {"path": str(paths["graph"]),
                      "sha256": sha256(paths["graph"]),
                      "header": graph_header},
            "oracle": {"path": str(paths["oracle"]),
                       "sha256": sha256(paths["oracle"]),
                       "bytes": paths["oracle"].stat().st_size},
            "runner": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256(Path(__file__).resolve())},
        },
        "git": git_info,
        "formal_input_contract": formal_input,
        "formal_pair_build": pair_build_evidence,
        "formal_pair_build_record": pair_build_record,
        "formal_integrity_before_sampling": integrity_before,
        "formal_data_integrity_before_sampling": data_integrity_before,
    }
    write_json(args.out / "manifest.json", manifest)

    try:
        manifest["preflight"] = gpu_preflight(
            args.out, require_a100=args.sampling == "formal")
    except Exception as error:
        manifest.update(status="preflight_failed", failure=repr(error))
        write_json(args.out / "manifest.json", manifest)
        write_json(args.out / "records.json", [])
        print(f"PRECHECK_FAILED {error}", file=sys.stderr)
        return 2
    manifest["status"] = "sampling"
    write_json(args.out / "manifest.json", manifest)

    if args.sampling == "formal":
        env, cleared_runtime = formal_runtime_environment()
    else:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("MLMQ_", "BENCH_", "L3_SUPPLEMENT_"))}
        cleared_runtime = []
    env.update({
        "MLMQ_BENCH": "1",
        "MLMQ_WORK_BLOCKS": str(args.blocks),
        "MLMQ_CUT_PERCENT": str(args.cut_percent),
        "MLMQ_FINAL_AUDIT": args.final_audit,
        "BENCH_DELTA": str(args.delta),
        "BENCH_SOURCE": str(args.source),
        "BENCH_WARMUPS": str(args.warmups),
        "BENCH_REPEATS": str(args.repeats),
        "BENCH_QUEUE": args.queue,
        "L3_SUPPLEMENT_ORACLE": str(paths["oracle"]),
    })
    manifest["runtime_environment"] = {
        "base_whitelist": sorted(
            key for key in env
            if not key.startswith(("MLMQ_", "BENCH_", "L3_SUPPLEMENT_"))),
        "cleared_environment_variables": cleared_runtime,
        "effective": env,
    }
    write_json(args.out / "manifest.json", manifest)

    records = []
    all_samples = []
    for round_id in range(args.rounds):
        order = (["single_no_l3", "dual_l3"] if round_id % 2 == 0
                 else ["dual_l3", "single_no_l3"])
        for sequence, label in enumerate(order):
            gpu_count = CONFIGURATIONS[label]["gpu_count"]
            binary = paths["single_binary" if label == "single_no_l3"
                           else "dual_binary"]
            prefix = args.out / f"round{round_id:03d}_{sequence}_{label}"
            started = now_utc()
            run_harness = (run_one_with_canonical_tools
                           if args.sampling == "formal" else run_one)
            record = run_harness(
                binary, paths["graph"], "MLMQ", gpu_count, prefix, env,
                args.source, args.queue, args.warmups, args.repeats,
                args.timeout, args.delta)
            record = verify_effective_contract(
                record, prefix.with_suffix(".log"), gpu_count, args,
                graph_header)
            record.update({
                "round": round_id,
                "sequence_in_round": sequence,
                "configuration": label,
                "requested_gpu_count": gpu_count,
                "started_utc": started,
                "finished_utc": now_utc(),
                "prefix": str(prefix),
            })
            write_json(prefix.with_suffix(".json"), record)
            records.append(record)
            for sample in record.get("samples", []):
                all_samples.append({
                    "round": round_id,
                    "sequence_in_round": sequence,
                    "configuration": label,
                    "run_valid": record["valid"],
                    **sample,
                })
            write_json(args.out / "records.json", records)
            write_json(args.out / "samples.json", all_samples)
            print(
                f"round={round_id} sequence={sequence} config={label} "
                f"gpu_count={gpu_count} valid={record['valid']} rc={record['rc']} "
                f"reason={record['reason']}", flush=True)

    rounds = []
    for round_id in range(args.rounds):
        by_label = {record["configuration"]: record for record in records
                    if record["round"] == round_id}
        single = summarize_record(by_label["single_no_l3"], args.repeats)
        dual = summarize_record(by_label["dual_l3"], args.repeats)
        rounds.append({"round": round_id, **paired_summary(single, dual)})

    combined = {}
    for label in CONFIGURATIONS:
        selected = [record for record in records
                    if record["configuration"] == label]
        rows = [row for record in selected for row in formal_rows(record)]
        valid = (len(selected) == args.rounds and
                 all(record["valid"] for record in selected) and
                 len(rows) == args.rounds * args.repeats)
        entry = {
            "valid": valid,
            "reason": "ok" if valid else "one or more runs/samples invalid",
            "formal_sample_count": len(rows),
            "expected_formal_samples": args.rounds * args.repeats,
        }
        if valid:
            try:
                entry["solve"] = sample_statistics(rows, "solve_ms")
                entry["query_wall"] = sample_statistics(rows, "query_wall_ms")
            except (KeyError, ValueError) as error:
                entry.update(valid=False, reason=str(error))
        combined[label] = entry

    paired = paired_summary(
        combined["single_no_l3"], combined["dual_l3"])
    speedup = paired["S_solve_T1_over_T2"]
    paired["target"] = TARGET_SPEEDUP
    paired["target_met"] = None if speedup is None else speedup >= TARGET_SPEEDUP
    all_processes_valid = all(record["valid"] for record in records)
    integrity_after = None
    data_integrity_after = None
    integrity_valid = True
    integrity_error = None
    if args.sampling == "formal":
        try:
            integrity_after = formal_pair_integrity_snapshot(pair_build, repo_root)
            if integrity_after != integrity_before:
                raise RuntimeError(
                    "formal pair artifacts or repository HEAD/status changed "
                    "during sampling")
            data_integrity_after = data_integrity_snapshot(paths)
            if data_integrity_after != data_integrity_before:
                raise RuntimeError(
                    "formal graph/oracle path, size, or hash changed during sampling")
            validate_formal_pair_build(
                pair_build, paths["dual_binary"], paths["single_binary"],
                git_info["head"]["output"], repo_root)
        except (OSError, RuntimeError, SystemExit) as error:
            integrity_valid = False
            integrity_error = str(error)
    measurement_valid = (
        measurement_is_valid(records, paired) and integrity_valid)
    numeric_target_met = paired["target_met"]
    accepted_target_met = numeric_target_met if measurement_valid else None
    summary = {
        "sampling": args.sampling,
        "formula": "S = median(T1) / median(T2)",
        "measurement_valid": measurement_valid,
        "target_met": accepted_target_met,
        "numeric_target_met": numeric_target_met,
        "formal_integrity_valid": integrity_valid,
        "formal_integrity_error": integrity_error,
        "rounds": rounds,
        "combined": paired,
        "all_samples_file": str(args.out / "samples.json"),
        "records_file": str(args.out / "records.json"),
    }
    write_json(args.out / "summary.json", summary)

    all_valid = measurement_valid
    manifest.update(
        status="measurement_valid" if all_valid else "measurement_invalid",
        completed_utc=now_utc(),
        process_count=len(records),
        all_processes_valid=all_processes_valid,
        measurement_valid=measurement_valid,
        target_met=accepted_target_met,
        numeric_target_met=numeric_target_met,
        formal_integrity_after_sampling=integrity_after,
        formal_data_integrity_after_sampling=data_integrity_after,
        formal_integrity_valid=integrity_valid,
        formal_integrity_error=integrity_error,
        summary=str(args.out / "summary.json"))
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
