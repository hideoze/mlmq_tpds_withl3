#!/usr/bin/env python3
"""Check transient int32 candidate additions on the frozen L3 query set.

This is a correctness-only diagnostic.  It validates a clean formal pair,
extracts each archived source tree without trusting tar metadata, derives two
new binaries whose candidate additions use checked int64 arithmetic, and runs
the independent single-GPU and dual-GPU binaries once for each frozen query.
The formal binaries and repository sources are never modified or executed as
numeric diagnostics, and no timings from this runner are performance evidence.
"""

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time

from run_l3_30h import (
    FORMAL_L2_ALLOCATED_RECORDS,
    FORMAL_L2_BATCH_SIZE,
    FORMAL_L2_BUDGET_BYTES,
    FORMAL_L2_BUCKET_MAX,
    FORMAL_L2_BUCKETS,
    FORMAL_L2_COUNTER_BITS,
    FORMAL_L2_PER_BUCKET_CAPACITY,
    FORMAL_L2_RECORD_BYTES,
    L2_CAPACITY_FIELDS,
    L2_FINAL_FIELDS,
    L3_CONFIG_FIELDS,
    L3_WORKER_ACK_FIELDS,
    QUEUE_TYPE_IDS,
    git_snapshot,
    parse_gpu_query,
    parse_dual_contract,
    parse_l2_capacity_contract,
    parse_oracle_contract,
    read_gr_header,
    slurm_gpu_evidence,
    validate_formal_pair_build,
)
from run_usa_road_matrix import parse_bench, sha256, validate


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_DEFINE = "-DMLMQ_CHECKED_ADD_DIAG=true"
DIAGNOSTIC_MAX_REGISTERS = 96
DIAGNOSTIC_REGISTER_FLAG = (
    f"--ptxas-options=-maxrregcount={DIAGNOSTIC_MAX_REGISTERS}")
QUEUE = "L1SLF_L2DQ"
WINDOW_MODE = 2
WINDOW_MIN = 25000
WINDOW_MAX = 25000
IDLE_BACKOFF = 0
SPEC_FIELDS = ("name", "graph", "oracle", "source", "delta", "cut_percent")
ENV_PREFIXES = ("MLMQ_", "BENCH_", "L3_SUPPLEMENT_")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# These hashes make "frozen query set" an exact, reviewable claim.  Paths may
# move, but inputs, source IDs, and the algorithmic configuration may not.
FROZEN_QUERIES = {
    "atmosmodm": {
        "graph_sha256": "c428695cedd4cdec1e889a73bb53cc445aae53e8d7a7635fcda206df944ae478",
        "oracle_sha256": "15e8a91b2db3f4b4c358b3aa23972c6dc843dbdd2dfc831bc75abf5c0cb2be8b",
        "vertices": 1489752, "source": 0, "delta": 200000, "cut_percent": 50,
    },
    "rmat22": {
        "graph_sha256": "164e61cce40603749d8df44fd5bcff9445a0b4bd83f0045be11670a63234b994",
        "oracle_sha256": "ef421f45303c022800cc317b672d1a7aeddb02df29d7450230e49a5113ccf60e",
        "vertices": 4194304, "source": 0, "delta": 200000, "cut_percent": 50,
    },
    "usa_primary": {
        "graph_sha256": "85c273900a89422369a06f5524f784b58ed91ceef36addb4ae3710dff5a1d6eb",
        "oracle_sha256": "19b674d79f5bc48d26107facb6327c8f8661d40a5bb80db67dd8feb260757a44",
        "vertices": 23947347, "source": 11973673, "delta": 400000, "cut_percent": 60,
    },
    "usa_additional_1": {
        "graph_sha256": "85c273900a89422369a06f5524f784b58ed91ceef36addb4ae3710dff5a1d6eb",
        "oracle_sha256": "f86e1b6509badf703eb50125156f3479938a928b2abbb2430910673bcf9ca9e3",
        "vertices": 23947347, "source": 18266241, "delta": 400000, "cut_percent": 60,
    },
    "usa_additional_2": {
        "graph_sha256": "85c273900a89422369a06f5524f784b58ed91ceef36addb4ae3710dff5a1d6eb",
        "oracle_sha256": "84664767003586c247a6657450796b0ee2ffbfde8ba84445f70ba05b9c008294",
        "vertices": 23947347, "source": 6146689, "delta": 400000, "cut_percent": 60,
    },
}

HELPER_ANCHOR = "// L0 vector queue\n"
HELPER = r'''#include <assert.h>
#include <limits.h>
#include <stdint.h>

#if !defined(MLMQ_CHECKED_ADD_DIAG) || (MLMQ_CHECKED_ADD_DIAG != true)
#error "derived numeric source requires MLMQ_CHECKED_ADD_DIAG=true"
#endif
#ifdef NDEBUG
#error "derived numeric source requires device assertions"
#endif
#ifndef TYPE_INT
#error "derived numeric source requires VALUE_TYPE=int"
#endif

__device__ __forceinline__ int diagnostic_checked_candidate_add(
    int lhs, int rhs, int site)
{
    const int64_t lhs64 = static_cast<int64_t>(lhs);
    const int64_t rhs64 = static_cast<int64_t>(rhs);
    const int64_t sum64 = lhs64 + rhs64;
    const bool ok = lhs64 >= 0 && lhs64 <= INT_MAX &&
                    rhs64 >= 0 && rhs64 <= INT_MAX &&
                    sum64 >= 0 && sum64 <= INT_MAX;
    if (!ok)
        printf("NUMERIC_ADD_FAIL site=%d lhs=%d rhs=%d sum=%lld\\n",
               site, lhs, rhs, static_cast<long long>(sum64));
    assert(ok);
    return static_cast<int>(sum64);
}

'''

DUAL_REPLACEMENTS = (
    (
        "new_dist = *((volatile VALUE_TYPE *)&node_data[large_v - v_begin]) + "
        "edge_data[large_st + coop_idx];",
        "new_dist = diagnostic_checked_candidate_add(\n"
        "                        *((volatile VALUE_TYPE *)&node_data[large_v - v_begin]),\n"
        "                        edge_data[large_st + coop_idx], 1);",
        "dual large-degree ordinary expansion",
    ),
    (
        "new_dist = *((volatile VALUE_TYPE *)&node_data[leader_vertex - v_begin]) + "
        "edge_data[global_idx];",
        "new_dist = diagnostic_checked_candidate_add(\n"
        "                    *((volatile VALUE_TYPE *)&node_data[leader_vertex - v_begin]),\n"
        "                    edge_data[global_idx], 2);",
        "dual small-degree ordinary expansion",
    ),
)

SINGLE_REPLACEMENTS = (
    (
        "new_dist = node_data[large_v] + edge_data[large_st + coop_idx];",
        "new_dist = diagnostic_checked_candidate_add(\n"
        "                        node_data[large_v],\n"
        "                        edge_data[large_st + coop_idx], 1);",
        "single large-degree ordinary expansion",
    ),
    (
        "new_dist = node_data[leader_vertex] + edge_data[global_idx];",
        "new_dist = diagnostic_checked_candidate_add(\n"
        "                    node_data[leader_vertex], edge_data[global_idx], 2);",
        "single small-degree ordinary expansion",
    ),
)

DUAL_DEFAULTS = {
    "core/include/common.h": {
        "MLMQ_TYPE": "L1SLF_L2DQ",
        "TYPE_INT": None,
    },
    "SSSP/sssp.cuh": {
        "L0_SOURCE_SNAPSHOT": "false",
        "L0_DIRECT_SMALL": "false",
        "L3_REGION_RELAX": "false",
        "L3_CHAIN_PARTITION": "false",
        "L3_TILE_LOAN": "false",
        "L3_CONTINUATION": "false",
        "L3_OWNER_COMMIT": "false",
        "L3_RX_PRIORITY_BOOTSTRAP": "false",
        "L3_RX_EXPRESS": "false",
        "SP_ASYNC_BF": "false",
        "BULK_ROUND": "true",
        "GHOST_DEPTH": "0",
    },
}

DEFAULT_ONLY_MACROS = frozenset(
    name for definitions in DUAL_DEFAULTS.values() for name in definitions)

FAILURE_MARKERS = (
    "NUMERIC_ADD_FAIL",
    "device-side assert",
    "Assertion `",
    "assertion failed",
    "Error at node",
    "ORACLE_MISMATCH",
    "ORACLE_SIZE_ERROR",
)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def selected_environment(environment):
    keys = sorted(key for key in environment
                  if key.startswith(("SLURM_", "CUDA_", *ENV_PREFIXES)))
    return {key: environment[key] for key in keys}


def safe_extract_source(archive, destination):
    """Extract only regular files/directories, without tarfile.extract*."""
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"source destination already exists: {destination}")
    destination.mkdir(parents=True)
    root = destination.resolve()
    seen = set()
    total_bytes = 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream.getmembers():
            raw_name = member.name
            name = PurePosixPath(raw_name)
            if (not raw_name or "\0" in raw_name or
                    raw_name.startswith(("/", "\\")) or
                    "\\" in raw_name or name.is_absolute() or
                    any(part in ("", ".", "..") for part in name.parts)):
                raise RuntimeError(f"unsafe source archive path: {raw_name!r}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    f"unsafe source archive type for {raw_name!r}: "
                    f"type={member.type!r}")
            key = name.as_posix()
            if key in seen:
                raise RuntimeError(f"duplicate source archive member: {key}")
            seen.add(key)
            target = destination.joinpath(*name.parts)
            resolved_parent = target.parent.resolve()
            if resolved_parent != root and root not in resolved_parent.parents:
                raise RuntimeError(f"source archive escapes destination: {raw_name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                if not target.is_dir() or target.is_symlink():
                    raise RuntimeError(
                        f"source archive directory collides with non-directory: {key}")
                target.chmod(0o755)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read source archive member: {key}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            actual_size = target.stat().st_size
            if actual_size != member.size:
                raise RuntimeError(
                    f"source archive member size mismatch for {key}: "
                    f"expected={member.size} actual={actual_size}")
            total_bytes += actual_size
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    for required in ("SSSP/main.cu", "SSSP/csr_graph.cu",
                     "SSSP/sssp_run.cu", "core/include/common.h"):
        path = destination / required
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source archive lacks regular {required}")
    return {"members": len(seen), "regular_file_bytes": total_bytes}


def tree_manifest(root):
    rows = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"derived tree contains non-regular file: {path}")
        digest = sha256(path)
        row = {"path": relative, "bytes": path.stat().st_size,
               "sha256": digest}
        rows.append(row)
        aggregate.update(relative.encode() + b"\0" + digest.encode() + b"\n")
    return {"tree_sha256": aggregate.hexdigest(), "files": rows}


def macro_default(text, name):
    matches = re.findall(
        rf"^[ \t]*#[ \t]*define[ \t]+{re.escape(name)}"
        rf"(?![A-Za-z0-9_])(?:[ \t]+([^/\s]+))?",
        text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one source default for {name}, got {len(matches)}")
    return matches[0] or None


def verify_dual_scope(source):
    observed = {}
    for relative, expected in DUAL_DEFAULTS.items():
        text = (source / relative).read_text()
        observed[relative] = {}
        for name, value in expected.items():
            actual = macro_default(text, name)
            if actual != value:
                raise RuntimeError(
                    f"frozen numeric scope changed: {name}={actual!r}, "
                    f"expected {value!r}")
            observed[relative][name] = actual
    return observed


def patch_numeric_source(source, role, evidence_dir):
    path = source / "SSSP/sssp_run.cu"
    before = path.read_text()
    if "MLMQ_CHECKED_ADD_DIAG" in before or \
            "diagnostic_checked_candidate_add" in before:
        raise RuntimeError(f"{role} source already contains the numeric diagnostic")
    if before.count(HELPER_ANCHOR) != 1:
        raise RuntimeError(
            f"{role} helper anchor count is {before.count(HELPER_ANCHOR)}, expected 1")
    replacements = DUAL_REPLACEMENTS if role == "dual" else SINGLE_REPLACEMENTS
    anchor_counts = {}
    after = before.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    for old, new, description in replacements:
        count = after.count(old)
        anchor_counts[description] = count
        if count != 1:
            raise RuntimeError(
                f"{description}: expected one source anchor, got {count}")
        after = after.replace(old, new, 1)
    expected_calls = len(replacements)
    actual_calls = after.count("diagnostic_checked_candidate_add(") - 1
    if actual_calls != expected_calls:
        raise RuntimeError(
            f"{role} checked-add call count {actual_calls}, expected {expected_calls}")
    path.write_text(after)
    patch_path = evidence_dir / f"{role}_numeric_add.patch"
    patch_path.write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="SSSP/sssp_run.cu.clean",
        tofile="SSSP/sssp_run.cu.numeric-diagnostic")))
    return {
        "source": str(path),
        "before_sha256": bytes_sha256(before.encode()),
        "after_sha256": sha256(path),
        "patch": str(patch_path),
        "patch_sha256": sha256(patch_path),
        "helper_anchor_count": 1,
        "replacement_anchor_counts": anchor_counts,
        "checked_add_call_count": actual_calls,
    }


def load_command(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read compile command {path}: {error}") from error
    if not isinstance(value, list) or not value or not all(
            isinstance(item, str) for item in value):
        raise RuntimeError(f"compile command is not a nonempty string array: {path}")
    return value


def copy_verified_artifact(source, destination, expected_sha256):
    """Copy one validated pair artifact and re-bind the bytes actually used."""
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"pair artifact is not a regular file: {source}")
    shutil.copy2(source, destination)
    actual = sha256(destination)
    if actual != expected_sha256:
        raise RuntimeError(
            f"pair artifact changed while being copied: {source}; "
            f"expected={expected_sha256} actual={actual}")
    return {
        "formal_path": str(source),
        "evidence_copy": str(destination),
        "sha256": actual,
    }


def command_defines(command):
    definitions = {}
    for argument in command:
        match = re.fullmatch(r"-D([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?", argument)
        if match:
            definitions.setdefault(match.group(1), []).append(match.group(2))
    return definitions


def derive_compile_command(original, source, binary, role=None):
    definitions = command_defines(original)
    duplicate_definitions = {
        name: values for name, values in definitions.items() if len(values) != 1}
    if duplicate_definitions:
        raise RuntimeError(
            f"formal command repeats compile definitions: {duplicate_definitions}")
    if role == "dual":
        overrides = sorted(DEFAULT_ONLY_MACROS.intersection(definitions))
        if overrides:
            raise RuntimeError(
                "dual formal command overrides source defaults that define the "
                f"numeric scope: {overrides}")
    existing = [item for item in original
                if item.startswith("-DMLMQ_CHECKED_ADD_DIAG")]
    if existing:
        raise RuntimeError(f"formal command already has diagnostic macro: {existing}")
    inherited_register_caps = [
        item for item in original if "maxrregcount" in item.lower()]
    if inherited_register_caps:
        raise RuntimeError(
            "formal command already has a register cap; the numeric diagnostic "
            f"must record its build adjustment unambiguously: {inherited_register_caps}")
    source_hits = {name: 0 for name in ("main.cu", "csr_graph.cu", "sssp_run.cu")}
    include_hits = 0
    output_hits = 0
    translated = []
    replace_output = False
    for index, argument in enumerate(original):
        if replace_output:
            translated.append(str(binary))
            output_hits += 1
            replace_output = False
            continue
        if argument == "-o":
            translated.append(argument)
            replace_output = True
            continue
        candidate = Path(argument)
        if candidate.name in source_hits and candidate.parent.name == "SSSP":
            translated.append(str(source / "SSSP" / candidate.name))
            source_hits[candidate.name] += 1
            continue
        if argument.startswith("-I"):
            include = Path(argument[2:])
            if include.name == "include" and include.parent.name == "core":
                translated.append("-I" + str(source / "core/include"))
                include_hits += 1
                continue
        translated.append(argument)
    if replace_output:
        raise RuntimeError("formal compile command ends after -o")
    if source_hits != {"main.cu": 1, "csr_graph.cu": 1, "sssp_run.cu": 1}:
        raise RuntimeError(f"formal source argument counts changed: {source_hits}")
    if include_hits != 1 or output_hits != 1:
        raise RuntimeError(
            f"formal command path counts changed: include={include_hits} "
            f"output={output_hits}")
    translated.append(DIAGNOSTIC_DEFINE)
    if translated.count(DIAGNOSTIC_DEFINE) != 1:
        raise RuntimeError("numeric diagnostic macro is not unique")
    if role == "dual":
        # The checked-add helper raises the persistent W512 kernel's register
        # allocation above the launchable limit.  This spill cap is confined to
        # the correctness-only derived binary; its timings are never evidence.
        translated.append(DIAGNOSTIC_REGISTER_FLAG)
    if translated.count(DIAGNOSTIC_REGISTER_FLAG) != (1 if role == "dual" else 0):
        raise RuntimeError("numeric diagnostic register cap is not role-exact")
    return translated, {
        "source_argument_counts": source_hits,
        "core_include_replacements": include_hits,
        "output_replacements": output_hits,
        "diagnostic_define_count": translated.count(DIAGNOSTIC_DEFINE),
        "diagnostic_max_registers": (
            DIAGNOSTIC_MAX_REGISTERS if role == "dual" else None),
        "diagnostic_register_flag_count": translated.count(
            DIAGNOSTIC_REGISTER_FLAG),
        "formal_defines": definitions,
        "only_source_semantic_change": (
            "checked int64 candidate addition with device assert"),
        "diagnostic_build_adjustments": (
            ["dual-only register cap retains W512 launchability; timings forbidden"]
            if role == "dual" else []),
        "performance_claim_allowed": False,
    }


def command_record(out, name, command, *, environment=None,
                   recorded_environment=None, timeout=1200):
    command = [str(item) for item in command]
    environment = dict(os.environ if environment is None else environment)
    command_path = out / f"{name}.command.json"
    log_path = out / f"{name}.log"
    result_path = out / f"{name}.result.json"
    write_json(command_path, {
        "command": command,
        "cwd": str(ROOT),
        "environment": (selected_environment(environment)
                        if recorded_environment is None
                        else dict(recorded_environment)),
        "started_utc": now_utc(),
        "timeout_seconds": timeout,
    })
    started = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=timeout)
        rc, raw = process.returncode, process.stdout
    except subprocess.TimeoutExpired as error:
        timed_out = True
        rc, raw = 124, error.stdout or ""
        if isinstance(raw, bytes):
            raw = raw.decode(errors="replace")
        raw += f"\nTIMEOUT_SECONDS={timeout}\n"
    except OSError as error:
        rc, raw = 127, f"EXEC_ERROR={error!r}\n"
    elapsed = time.monotonic() - started
    log_path.write_text(
        raw + f"\nRUN_RC={rc}\nHOST_ELAPSED_SECONDS={elapsed:.6f}\n")
    result = {
        "command": command,
        "command_record": str(command_path),
        "command_record_sha256": sha256(command_path),
        "finished_utc": now_utc(),
        "host_elapsed_seconds_diagnostic_only": elapsed,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "rc": rc,
        "timed_out": timed_out,
    }
    write_json(result_path, result)
    return result, raw


def validate_spec(raw_spec, file_cache=None):
    file_cache = {} if file_cache is None else file_cache
    name, graph_s, oracle_s, source_s, delta_s, cut_s = raw_spec
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid spec name: {name!r}")
    if name not in FROZEN_QUERIES:
        raise ValueError(
            f"unknown frozen spec {name!r}; expected {sorted(FROZEN_QUERIES)}")
    graph = Path(graph_s).resolve(strict=True)
    oracle = Path(oracle_s).resolve(strict=True)
    if not graph.is_file() or not oracle.is_file():
        raise ValueError(f"spec {name} graph/oracle must be regular files")
    try:
        source, delta, cut = map(int, (source_s, delta_s, cut_s))
    except ValueError as error:
        raise ValueError(f"spec {name} source/delta/cut must be integers") from error
    graph_key = ("graph", graph)
    if graph_key not in file_cache:
        file_cache[graph_key] = (read_gr_header(graph), sha256(graph))
    header, graph_hash = file_cache[graph_key]
    oracle_key = ("oracle", oracle)
    if oracle_key not in file_cache:
        file_cache[oracle_key] = sha256(oracle)
    actual = {
        "graph_sha256": graph_hash,
        "oracle_sha256": file_cache[oracle_key],
        "vertices": header["vertices"],
        "source": source,
        "delta": delta,
        "cut_percent": cut,
    }
    expected = FROZEN_QUERIES[name]
    if actual != expected:
        raise ValueError(
            f"spec {name} differs from frozen contract: "
            f"actual={actual} expected={expected}")
    if oracle.stat().st_size != 4 * header["vertices"]:
        raise ValueError(f"spec {name} oracle byte count is invalid")
    return {
        "name": name, "graph": graph, "oracle": oracle,
        "source": source, "delta": delta, "cut_percent": cut,
        "header": header,
        "hashes": {
            "graph": actual["graph_sha256"],
            "oracle": actual["oracle_sha256"],
        },
    }


def load_specs(raw_specs):
    if raw_specs is None or len(raw_specs) != len(FROZEN_QUERIES):
        raise ValueError(
            f"exactly {len(FROZEN_QUERIES)} --spec entries are required")
    file_cache = {}
    specs = [validate_spec(row, file_cache) for row in raw_specs]
    names = [row["name"] for row in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate spec names: {names}")
    if set(names) != set(FROZEN_QUERIES):
        raise ValueError(
            f"frozen spec set differs: got={sorted(names)} "
            f"expected={sorted(FROZEN_QUERIES)}")
    return specs


def frozen_input_integrity(specs):
    """Re-hash every distinct graph/oracle and compare with the frozen set."""
    cache = {}
    rows = {}
    errors = []
    for spec in specs:
        rows[spec["name"]] = {}
        for kind in ("graph", "oracle"):
            path = spec[kind]
            if path not in cache:
                if path.is_symlink() or not path.is_file():
                    cache[path] = {"path": str(path), "regular_file": False}
                else:
                    cache[path] = {
                        "path": str(path), "regular_file": True,
                        "bytes": path.stat().st_size, "sha256": sha256(path),
                    }
            observed = cache[path]
            rows[spec["name"]][kind] = observed
            expected = spec["hashes"][kind]
            if (not observed.get("regular_file") or
                    observed.get("sha256") != expected):
                errors.append(
                    f"{spec['name']} {kind} changed: "
                    f"observed={observed.get('sha256')} expected={expected}")
    return {"valid": not errors, "files": rows, "errors": errors}


def parse_numeric_run(raw, *, role, vertices, source, blocks, delta,
                      cut_percent):
    gpu_count = 1 if role == "single" else 2
    errors = []
    rows = parse_bench(raw)
    valid_rows, reason = validate(
        rows, "MLMQ", gpu_count, source, QUEUE, 0, 1)
    if not valid_rows:
        errors.append(reason)
    oracle = parse_oracle_contract(raw, 1, vertices)
    errors.extend(oracle["errors"])
    found_markers = [marker for marker in FAILURE_MARKERS if marker in raw]
    if re.search(r"\bassert(?:ion)?\b", raw, re.IGNORECASE):
        found_markers.append("assert")
    if re.search(r"\b(?:FAIL|FAILED|FAILURE)\b", raw, re.IGNORECASE):
        found_markers.append("FAIL")
    if re.search(r"CUDA (?:error|ERROR)|illegal memory access", raw):
        found_markers.append("CUDA runtime error")
    if found_markers:
        errors.append(f"failure markers present: {found_markers}")
    partitions = [tuple(map(int, match.groups())) for match in re.finditer(
        r"GPU(\d+) partition: \[(\d+), (\d+)\)", raw)]
    if role == "dual":
        dual = parse_dual_contract(
            raw, 1, blocks, delta, QUEUE_TYPE_IDS[QUEUE],
            WINDOW_MODE, WINDOW_MIN, WINDOW_MAX, IDLE_BACKOFF,
            require_l2_final=True)
        errors.extend(dual["errors"])
        cut = vertices * cut_percent // 100
        expected_partitions = [(0, 0, cut), (1, cut, vertices)]
        if sorted(partitions) != expected_partitions:
            errors.append(
                f"effective partitions differ: observed={sorted(partitions)} "
                f"expected={expected_partitions}")
        no_l3_configs = re.findall(r"^NO_L3_CONFIG ", raw, re.MULTILINE)
        if no_l3_configs:
            errors.append("dual diagnostic unexpectedly logged NO_L3_CONFIG")
        role_contract = dual
    else:
        capacity = parse_l2_capacity_contract(
            raw, 1, require_frozen=True)
        errors.extend(capacity["errors"])
        expected_partitions = []
        if partitions:
            errors.append("independent single unexpectedly logged GPU partitions")
        config = re.findall(
            r"^NO_L3_CONFIG workers=(\d+) delta=(\d+) queue=(\S+)$",
            raw, re.MULTILINE)
        launches = [tuple(map(int, groups)) for groups in re.findall(
            r"^NO_L3_LAUNCH work_blocks=(\d+) delta=(\d+) repeat=(\d+)$",
            raw, re.MULTILINE)]
        if config != [("512", str(delta), QUEUE)]:
            errors.append(f"invalid NO_L3_CONFIG rows: {config}")
        if launches != [(blocks, delta, 0)]:
            errors.append(f"invalid NO_L3_LAUNCH rows: {launches}")
        if re.search(r"^L3_CONFIG ", raw, re.MULTILINE):
            errors.append("independent single unexpectedly logged L3_CONFIG")
        role_contract = {
            "no_l3_config": config,
            "no_l3_launch": launches,
        }
    return {
        "valid": not errors,
        "purpose": "transient candidate-add correctness only; timings are not evidence",
        "bench_rows": rows,
        "oracle": oracle,
        "role_contract": role_contract,
        "partitions": partitions,
        "expected_partitions": expected_partitions,
        "failure_markers": found_markers,
        "errors": errors,
    }


def gpu_process_snapshot(path):
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=30)
    path.write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"GPU process snapshot failed with rc={process.returncode}")
    return process.stdout.strip()


def numeric_runtime_environment(spec, blocks, base_environment=None):
    source = os.environ if base_environment is None else base_environment
    environment = {key: value for key, value in source.items()
                   if not key.startswith(ENV_PREFIXES)}
    # Cooperating persistent kernels are launched on multiple streams.  A
    # blocking launch can wait on the first persistent kernel before the peer
    # and manager kernels needed for its progress have even been launched.
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    environment.update({
        "MLMQ_BENCH": "1",
        "MLMQ_WORK_BLOCKS": str(blocks),
        "MLMQ_CUT_PERCENT": str(spec["cut_percent"]),
        "MLMQ_FINAL_AUDIT": "all",
        "BENCH_DELTA": str(spec["delta"]),
        "BENCH_SOURCE": str(spec["source"]),
        "BENCH_WARMUPS": "0",
        "BENCH_REPEATS": "1",
        "BENCH_QUEUE": QUEUE,
        "L3_SUPPLEMENT_ORACLE": str(spec["oracle"]),
    })
    return environment


def run_solver(output, spec, role, binary, blocks, timeout):
    prefix = output / f"{spec['name']}_{role}"
    before = gpu_process_snapshot(prefix.with_suffix(".gpu_before.log"))
    if before:
        raise RuntimeError(f"GPU process present before {spec['name']} {role}: {before}")
    environment = numeric_runtime_environment(spec, blocks)
    gpu_count = 1 if role == "single" else 2
    command = [binary, "-i", spec["graph"], "-n", str(gpu_count),
               "-d", str(spec["delta"])]
    result, raw = command_record(
        output, f"{spec['name']}_{role}", command,
        environment=environment, timeout=timeout)
    after = gpu_process_snapshot(prefix.with_suffix(".gpu_after.log"))
    contract = parse_numeric_run(
        raw, role=role, vertices=spec["header"]["vertices"],
        source=spec["source"], blocks=blocks, delta=spec["delta"],
        cut_percent=spec["cut_percent"])
    errors = list(contract["errors"])
    if result["rc"] != 0:
        errors.append(f"process rc={result['rc']}")
    if result["timed_out"]:
        errors.append("process timed out")
    if after:
        errors.append(f"GPU process remained after run: {after}")
    result.update({
        "valid": not errors,
        "role": role,
        "requested_gpu_count": gpu_count,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "graph": str(spec["graph"]),
        "graph_sha256": spec["hashes"]["graph"],
        "oracle": str(spec["oracle"]),
        "oracle_sha256": spec["hashes"]["oracle"],
        "source": spec["source"],
        "contract": contract,
        "gpu_processes_before": before,
        "gpu_processes_after": after,
        "errors": errors,
    })
    write_json(prefix.with_suffix(".result.json"), result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-build", required=True, type=Path,
                        help="successful clean-HEAD build_l3_30h_pair.sh output")
    parser.add_argument(
        "--spec", action="append", nargs=6, metavar=SPEC_FIELDS,
        help=("repeat exactly five times: NAME GRAPH ORACLE SOURCE DELTA CUT; "
              "NAME must be one of " + ",".join(FROZEN_QUERIES)))
    parser.add_argument("--out", required=True, type=Path,
                        help="new diagnostic evidence directory outside the repository")
    parser.add_argument("--blocks", type=int, default=107)
    parser.add_argument("--build-timeout", type=int, default=1800)
    parser.add_argument("--run-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("numeric checks must run inside a Slurm allocation")
    if args.blocks <= 0 or args.build_timeout <= 0 or args.run_timeout <= 0:
        parser.error("blocks and timeouts must be positive")
    if args.blocks != 107:
        parser.error("the frozen numeric query set requires exactly 107 blocks")
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"output path already exists: {args.out}")
    try:
        args.specs = load_specs(args.spec)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    git_info = git_snapshot(ROOT)
    if (git_info["head"]["rc"] != 0 or git_info["status"]["rc"] != 0 or
            git_info["status"]["output"]):
        raise SystemExit(
            "numeric checks require the final clean worktree; status is:\n" +
            git_info["status"]["output"])
    output = args.out.resolve(strict=False)
    if output == ROOT or ROOT in output.parents:
        raise SystemExit("numeric-check output must be outside the repository")
    try:
        allocation = slurm_gpu_evidence(os.environ)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    pair = args.pair_build.resolve(strict=True)
    dual_formal = (pair / "dual_build/mlmq").resolve(strict=True)
    single_formal = (pair / "single_build/mlmq").resolve(strict=True)
    try:
        pair_evidence = validate_formal_pair_build(
            pair, dual_formal, single_formal,
            git_info["head"]["output"], ROOT)
    except SystemExit as error:
        raise SystemExit(f"pair-build validation failed: {error}") from error
    build_environment = dict(pair_evidence["version"]["build_environment"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    shutil.copy2(__file__, output / "runner.py")
    manifest = {
        "schema": 1,
        "status": "starting",
        "created_utc": now_utc(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "allocation": allocation,
        "environment": selected_environment(os.environ),
        "git": git_info,
        "pair_build": pair_evidence,
        "configuration": {
            "blocks": args.blocks,
            "l2_buckets": FORMAL_L2_BUCKETS,
            "l2_bucket_max": FORMAL_L2_BUCKET_MAX,
            "l2_batch_size": FORMAL_L2_BATCH_SIZE,
            "queue": QUEUE,
            "window_mode": WINDOW_MODE,
            "window_min": WINDOW_MIN,
            "window_max": WINDOW_MAX,
            "idle_backoff": IDLE_BACKOFF,
            "warmups": 0,
            "repeats_per_binary_per_spec": 1,
            "cuda_launch_blocking": False,
            "cuda_launch_blocking_policy": (
                "explicitly removed because cooperating persistent kernels "
                "must be launched concurrently on multiple streams"
            ),
            "single_diagnostic_max_registers": None,
            "dual_diagnostic_max_registers": DIAGNOSTIC_MAX_REGISTERS,
            "dual_diagnostic_register_policy": (
                "diagnostic-only spill cap retains W512 launchability after "
                "checked-add instrumentation; timings are forbidden"
            ),
            "purpose": "checked transient candidate additions only",
            "performance_claim_allowed": False,
            "numeric_scope": {
                "instrumented": [
                    "dual ordinary large-degree edge expansion",
                    "dual ordinary small-degree edge expansion",
                    "independent-single ordinary large-degree edge expansion",
                    "independent-single ordinary small-degree edge expansion",
                ],
                "no_candidate_add": [
                    "source initialization",
                    "disabled priority bootstrap bucket positioning",
                    "active worker-recovery/backstop dirty re-publication",
                    "BULK candidate transport and receive",
                ],
                "excluded_disabled_paths": [
                    "L3_OWNER_COMMIT", "GHOST_DEPTH", "SP_ASYNC_BF",
                    "L3_TILE_LOAN", "L3_CONTINUATION", "L3_REGION_RELAX",
                    "L3_CHAIN_PARTITION", "L3_CHAIN_SHORTCUTS",
                    "L0_SOURCE_SNAPSHOT", "L0_DIRECT_SMALL",
                ],
            },
        },
        "inputs": {
            "runner": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256(Path(__file__).resolve())},
            "formal_pair": str(pair),
            "formal_binaries_not_executed": True,
            "specs": [{
                "name": row["name"], "graph": str(row["graph"]),
                "oracle": str(row["oracle"]), "source": row["source"],
                "delta": row["delta"], "cut_percent": row["cut_percent"],
                "header": row["header"], "hashes": row["hashes"],
            } for row in args.specs],
        },
    }
    write_json(output / "manifest.json", manifest)

    problems = []
    preflight = {}
    for name, command in (
            ("nvidia_smi", ["nvidia-smi"]),
            ("gpu_query", ["nvidia-smi",
                           "--query-gpu=index,uuid,name,compute_cap",
                           "--format=csv,noheader,nounits"]),
            ("topology", ["nvidia-smi", "topo", "-m"]),
            ("compute_apps", ["nvidia-smi", "--query-compute-apps=pid",
                              "--format=csv,noheader"])):
        record, raw = command_record(output, name, command, timeout=30)
        preflight[name] = record
        if record["rc"] != 0:
            problems.append(f"preflight {name} rc={record['rc']}")
        if name == "gpu_query":
            try:
                record["parsed_gpus"] = parse_gpu_query(
                    raw, require_a100=True)
            except RuntimeError as error:
                problems.append(str(error))
        if name == "compute_apps" and raw.strip():
            problems.append("GPU compute application present before checks")
    manifest["preflight"] = preflight
    if problems:
        manifest.update(status="preflight_failed", problems=problems)
        write_json(output / "manifest.json", manifest)
        write_json(output / "results.json", {"valid": False, "problems": problems})
        return 2

    sources_dir = output / "derived_sources"
    sources_dir.mkdir()
    formal_inputs_dir = output / "formal_inputs"
    formal_inputs_dir.mkdir()
    derivations = {}
    binaries_dir = output / "derived_binaries"
    binaries_dir.mkdir()
    builds = {}
    for role in ("single", "dual"):
        source = sources_dir / role
        role_inputs = formal_inputs_dir / role
        role_inputs.mkdir()
        verified = pair_evidence["components"][role]["verified_sha256"]
        archive_source = pair / f"{role}_build/source.tgz"
        archive = role_inputs / "source.tgz"
        archive_binding = copy_verified_artifact(
            archive_source, archive, verified["source.tgz"])
        command_source = pair / f"{role}_build/command.json"
        command_copy = role_inputs / "command.json"
        command_binding = copy_verified_artifact(
            command_source, command_copy, verified["command.json"])
        extraction = safe_extract_source(archive, source)
        before_tree = tree_manifest(source)
        scope = verify_dual_scope(source) if role == "dual" else {
            "note": "independent committed no-L3 source; only ordinary large/small expansions patched"
        }
        patch = patch_numeric_source(source, role, output)
        after_tree = tree_manifest(source)
        original_command = load_command(command_copy)
        binary = binaries_dir / f"mlmq_numeric_{role}"
        command, translation = derive_compile_command(
            original_command, source, binary, role=role)
        build, _ = command_record(
            output, f"build_{role}", command,
            environment=build_environment,
            recorded_environment=build_environment,
            timeout=args.build_timeout)
        if build["rc"] != 0 or not binary.is_file():
            problems.append(f"{role} diagnostic build failed rc={build['rc']}")
        else:
            build["binary"] = str(binary)
            build["binary_sha256"] = sha256(binary)
        write_json(output / f"build_{role}.result.json", build)
        builds[role] = build
        derivations[role] = {
            "formal_source_archive": archive_binding,
            "formal_command_artifact": command_binding,
            "extraction": extraction,
            "source_tree_before": before_tree,
            "source_tree_after": after_tree,
            "patch": patch,
            "scope": scope,
            "formal_command": str(command_copy),
            "formal_command_sha256": sha256(command_copy),
            "derived_command": command,
            "command_translation": translation,
            "derived_build_environment": build_environment,
            "formal_source_modified": False,
            "repository_source_modified": False,
        }
    manifest["derivations"] = derivations
    manifest["builds"] = builds
    write_json(output / "manifest.json", manifest)
    if problems:
        manifest.update(status="build_failed", problems=problems)
        write_json(output / "manifest.json", manifest)
        write_json(output / "results.json", {"valid": False, "problems": problems})
        return 2

    results = []
    binaries = {
        "single": binaries_dir / "mlmq_numeric_single",
        "dual": binaries_dir / "mlmq_numeric_dual",
    }
    for spec in args.specs:
        for role in ("single", "dual"):
            try:
                result = run_solver(
                    output, spec, role, binaries[role], args.blocks,
                    args.run_timeout)
            except Exception as error:
                result = {
                    "valid": False, "role": role, "spec": spec["name"],
                    "errors": [repr(error)],
                }
            results.append({"spec": spec["name"], **result})
            if not result.get("valid"):
                problems.append(
                    f"{spec['name']} {role}: " +
                    "; ".join(result.get("errors", ["unknown failure"])))
            write_json(output / "results.json", {
                "valid": not problems,
                "completed_runs": len(results),
                "expected_runs": 2 * len(args.specs),
                "performance_claim_allowed": False,
                "runs": results,
                "problems": problems,
            })

    try:
        data_integrity_after = frozen_input_integrity(args.specs)
    except Exception as error:
        data_integrity_after = {
            "valid": False, "files": {}, "errors": [repr(error)]}
    problems.extend(data_integrity_after["errors"])
    compiler_path = Path(pair_evidence["compiler"]["path"])
    compiler_after = {
        "path": str(compiler_path),
        "sha256": sha256(compiler_path),
        "expected_sha256": pair_evidence["compiler"]["sha256"],
    }
    compiler_after["valid"] = (
        compiler_after["sha256"] == compiler_after["expected_sha256"])
    if not compiler_after["valid"]:
        problems.append("formal compiler changed during numeric checks")
    valid = not problems and len(results) == 2 * len(args.specs)
    manifest.update(
        status="complete" if valid else "failed",
        completed_utc=now_utc(),
        numeric_gate=(
            "PASS_FOR_FROZEN_QUERY_SET" if valid
            else "FAIL_FOR_FROZEN_QUERY_SET"),
        performance_claim_allowed=False,
        frozen_input_integrity_after=data_integrity_after,
        compiler_integrity_after=compiler_after,
        problems=problems,
    )
    write_json(output / "manifest.json", manifest)
    write_json(output / "results.json", {
        "valid": valid,
        "completed_runs": len(results),
        "expected_runs": 2 * len(args.specs),
        "numeric_gate": manifest["numeric_gate"],
        "scope": (
            "exact final clean pair, exact five frozen graph/oracle/source "
            "queries, ordinary L1SLF_L2DQ/default-BULK large/small expansion; "
            "not arbitrary inputs or disabled owner-commit/ghost/tile-loan paths"),
        "performance_claim_allowed": False,
        "frozen_input_integrity_after": data_integrity_after,
        "compiler_integrity_after": compiler_after,
        "runs": results,
        "problems": problems,
    })
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
