#!/usr/bin/env python3
"""Run the frozen eight-graph G/G+ final-SHA L3 regression.

The program is an outer evidence driver.  It validates the historical matrix
manifest, builds one canonical pair from the current clean HEAD inside this
process, then invokes ``run_l3_30h.py`` in its paired exploratory mode once for
each of the sixteen physical inputs.  Each inner invocation still runs an
independent no-L3 single-GPU binary and the L3 dual-GPU binary in AB/BA order
for two rounds (one warmup plus five formal queries per process).  External
pair artifacts are deliberately not accepted.

``--dry-run`` performs the static frozen-manifest/plan check only.  A real run
requires a clean committed checkout, a pair built from that same HEAD, exactly
two visible A100 CC 8.0 GPUs in the ``a100`` Slurm partition, and an output
directory outside the repository.  Cases are always launched serially.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/multigpu/run_l3_30h.py"
DEFAULT_MATRIX_MANIFEST = (
    ROOT / "evidence/l3_latest_rerun_38082/matrix/manifest.json"
)

GRAPH_NAMES = ("NY", "BAY", "COL", "FLA", "CAL", "E", "W", "USA")
VIEWS = (("G", "original", "G"), ("G+", "augmented", "G_plus"))
EXPECTED_VARIANTS = {
    "M1-original": ["single", "original", 1],
    "M1-shortcut": ["single", "augmented", 1],
    "M2-original": ["dual", "original", 2],
    "M2-shortcut": ["dual", "augmented", 2],
}

WARMUPS = 1
FORMAL_REPEATS = 5
ROUNDS = 2
WORKERS = 512
BLOCKS = 107
DELTA = 200000
QUEUE = "L1SLF_L2DQ"
QUEUE_TYPE_ID = 19
TARGET_SPEEDUP = 1.20
WINDOW_MODE = 2
WINDOW_MIN = 25000
WINDOW_MAX = 25000
IDLE_BACKOFF = 0
FORMAL_TOOL_PATH = "/usr/local/cuda/bin:/usr/bin:/bin"
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
PROCESS_SAMPLES = WARMUPS + FORMAL_REPEATS
PROCESSES_PER_CASE = ROUNDS * 2
FORMAL_SAMPLES_PER_CONFIGURATION = ROUNDS * FORMAL_REPEATS

EXPECTED_RX = {
    "rx_priority_bootstrap": 0,
    "rx_express": 0,
    "rx_express_enabled": 0,
    "rx_express_slots": 64,
    "rx_express_batch": 32,
    "rx_l2_pull": 0,
    "rx_l2_pull_claim": 0,
    "rx_l2_pull_enabled": 0,
}

PAIR_INTEGRITY_FILES = (
    "status.json",
    "version.json",
    "provenance.json",
    "dual_build/mlmq",
    "dual_build/source.tgz",
    "dual_build/command.json",
    "dual_build/build.log",
    "dual_build/status.json",
    "dual_build/hashes.json",
    "dual_build/version.json",
    "dual_build/provenance.json",
    "single_build/mlmq",
    "single_build/source.tgz",
    "single_build/command.json",
    "single_build/build.log",
    "single_build/status.json",
    "single_build/hashes.json",
    "single_build/version.json",
    "single_build/provenance.json",
)


class ContractError(RuntimeError):
    """A frozen-input, execution, or evidence contract was violated."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read {description} {path}: {error}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def require_exact_keys(value: Any, expected: set[str], description: str) -> None:
    require(isinstance(value, dict), f"{description} must be a JSON object")
    actual = set(value)
    require(
        actual == expected,
        f"{description} keys differ: missing={sorted(expected - actual)} "
        f"extra={sorted(actual - expected)}",
    )


def require_int(value: Any, description: str) -> int:
    require(type(value) is int, f"{description} must be an integer")
    return value


def require_sha256(value: Any, description: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{description} must be a lowercase SHA256 digest",
    )
    return value


def read_gr_header(path: Path) -> dict[str, int]:
    with path.open("rb") as stream:
        raw = stream.read(32)
    require(len(raw) == 32, f"graph is shorter than its GR header: {path}")
    version, edge_size, vertices, edges = struct.unpack("<4Q", raw)
    require(
        version == 1 and edge_size == 4 and vertices > 0,
        f"unsupported GR header version={version} edge_size={edge_size} "
        f"vertices={vertices}: {path}",
    )
    return {
        "version": version,
        "edge_size": edge_size,
        "vertices": vertices,
        "edges": edges,
    }


def git_output(*arguments: str, binary: bool = False) -> bytes | str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if process.returncode:
        detail = process.stderr.decode(errors="replace").strip()
        raise ContractError(
            f"git {' '.join(arguments)} failed with rc={process.returncode}: {detail}"
        )
    return process.stdout if binary else process.stdout.decode().rstrip("\n")


def clean_head(expected_head: str | None = None) -> dict[str, Any]:
    head = str(git_output("rev-parse", "HEAD"))
    status = str(git_output("status", "--porcelain=v1", "--untracked-files=all"))
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "invalid git HEAD")
    require(not status, f"formal regression requires a clean worktree:\n{status}")
    if expected_head is not None:
        require(head == expected_head, f"git HEAD changed: {head} != {expected_head}")
    return {"head": head, "status_porcelain": []}


def require_committed_file(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise ContractError(f"required committed file is outside repository: {path}") from error
    raw = resolved.read_bytes()
    head_raw = git_output("show", f"HEAD:{relative.as_posix()}", binary=True)
    require(raw == head_raw, f"{relative} differs from clean HEAD")
    return {
        "path": str(resolved),
        "head_path": relative.as_posix(),
        "sha256": sha256_bytes(raw),
    }


def validate_matrix_manifest_schema(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(path, "eight-graph matrix manifest")
    require(isinstance(manifest, dict), "matrix manifest must be a JSON object")
    require(manifest.get("job") == "38082", "matrix manifest is not frozen job 38082")
    require(manifest.get("host") == "ada-A100", "matrix manifest host is not ada-A100")

    expected_scalars = {
        "warmups": WARMUPS,
        "formal": FORMAL_REPEATS,
        "rounds": ROUNDS,
        "workers": WORKERS,
        "blocks": BLOCKS,
        "delta": DELTA,
    }
    for key, expected in expected_scalars.items():
        require(
            manifest.get(key) == expected,
            f"matrix manifest {key}={manifest.get(key)!r}, expected {expected}",
        )
    require(
        manifest.get("variants") == EXPECTED_VARIANTS,
        "matrix manifest variants differ from the frozen M1/M2 G/G+ contract",
    )

    graphs = manifest.get("graphs")
    require(isinstance(graphs, dict), "matrix manifest graphs must be an object")
    require(
        tuple(graphs) == GRAPH_NAMES,
        f"matrix graph order/names are {tuple(graphs)}, expected {GRAPH_NAMES}",
    )

    cases: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for graph_name in GRAPH_NAMES:
        graph = graphs[graph_name]
        required = {
            "source", "cut", "cut_percent", "vertices", "original",
            "augmented", "oracle", "hashes",
        }
        require_exact_keys(graph, required, f"graphs.{graph_name}")
        vertices = require_int(graph["vertices"], f"{graph_name}.vertices")
        source = require_int(graph["source"], f"{graph_name}.source")
        cut = require_int(graph["cut"], f"{graph_name}.cut")
        cut_percent = require_int(
            graph["cut_percent"], f"{graph_name}.cut_percent"
        )
        require(vertices > 0, f"{graph_name} has no vertices")
        require(0 <= source < vertices, f"{graph_name} source is out of range")
        require(1 <= cut_percent <= 99, f"{graph_name} cut percent is invalid")
        require(
            cut == vertices * cut_percent // 100 and 0 < cut < vertices,
            f"{graph_name} cut={cut} is not floor(vertices*cut_percent/100)",
        )
        hashes = graph["hashes"]
        require_exact_keys(
            hashes, {"original", "augmented", "oracle"}, f"{graph_name}.hashes"
        )
        for field in ("original", "augmented", "oracle"):
            value = graph[field]
            require(
                isinstance(value, str) and Path(value).is_absolute(),
                f"{graph_name}.{field} must be a nonempty absolute path",
            )
            require_sha256(hashes[field], f"{graph_name}.hashes.{field}")
        require(
            graph["original"] != graph["augmented"],
            f"{graph_name} G and G+ resolve to the same manifest path",
        )
        require(
            graph["hashes"]["original"] != graph["hashes"]["augmented"],
            f"{graph_name} G and G+ unexpectedly have the same hash",
        )
        require(
            graph["oracle"] not in seen_paths,
            f"oracle path is unexpectedly shared across graph names: {graph['oracle']}",
        )
        seen_paths.add(graph["oracle"])
        for view, field, view_slug in VIEWS:
            cases.append({
                "case_id": f"{graph_name}_{view_slug}",
                "graph": graph_name,
                "view": view,
                "view_slug": view_slug,
                "manifest_field": field,
                "graph_path": graph[field],
                "graph_sha256": graph["hashes"][field],
                "oracle_path": graph["oracle"],
                "oracle_sha256": graph["hashes"]["oracle"],
                "source": source,
                "cut": cut,
                "cut_percent": cut_percent,
                "vertices": vertices,
            })
    require(len(cases) == 16, f"expected 16 frozen cases, got {len(cases)}")
    return manifest, cases


def validate_input_files(cases: list[dict[str, Any]]) -> dict[str, Any]:
    cached_hashes: dict[Path, str] = {}
    cached_headers: dict[Path, dict[str, int]] = {}
    evidence: dict[str, Any] = {}

    def regular(path_string: str, description: str) -> Path:
        requested = Path(path_string)
        require(not requested.is_symlink(), f"{description} must not be a symlink")
        try:
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise ContractError(f"cannot resolve {description} {requested}: {error}") from error
        require(resolved.is_file(), f"{description} is not a regular file: {resolved}")
        return resolved

    for case in cases:
        graph = regular(case["graph_path"], f"{case['case_id']} graph")
        oracle = regular(case["oracle_path"], f"{case['case_id']} oracle")
        if graph not in cached_hashes:
            cached_hashes[graph] = sha256(graph)
        if oracle not in cached_hashes:
            cached_hashes[oracle] = sha256(oracle)
        require(
            cached_hashes[graph] == case["graph_sha256"],
            f"{case['case_id']} graph SHA256 mismatch",
        )
        require(
            cached_hashes[oracle] == case["oracle_sha256"],
            f"{case['case_id']} oracle SHA256 mismatch",
        )
        if graph not in cached_headers:
            cached_headers[graph] = read_gr_header(graph)
        header = cached_headers[graph]
        require(
            header["vertices"] == case["vertices"],
            f"{case['case_id']} graph has {header['vertices']} vertices, "
            f"expected {case['vertices']}",
        )
        require(
            oracle.stat().st_size == case["vertices"] * 4,
            f"{case['case_id']} oracle size is not vertices*4",
        )
        case["graph_path"] = str(graph)
        case["oracle_path"] = str(oracle)
        evidence[case["case_id"]] = {
            "graph": {
                "path": str(graph), "sha256": cached_hashes[graph],
                "bytes": graph.stat().st_size, "header": header,
            },
            "oracle": {
                "path": str(oracle), "sha256": cached_hashes[oracle],
                "bytes": oracle.stat().st_size,
            },
        }
    return evidence


def case_command(
    case: dict[str, Any], pair: Path, case_out: Path, timeout_seconds: int
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-s",
        str(RUNNER.resolve()),
        "--sampling", "exploratory",
        "--dual-binary", str((pair / "dual_build/mlmq").resolve()),
        "--single-binary", str((pair / "single_build/mlmq").resolve()),
        "--graph", case["graph_path"],
        "--oracle", case["oracle_path"],
        "--source", str(case["source"]),
        "--delta", str(DELTA),
        "--cut-percent", str(case["cut_percent"]),
        "--blocks", str(BLOCKS),
        "--warmups", str(WARMUPS),
        "--repeats", str(FORMAL_REPEATS),
        "--rounds", str(ROUNDS),
        "--timeout", str(timeout_seconds),
        "--out", str(case_out),
        "--queue", QUEUE,
        "--final-audit", "all",
        "--expected-window-mode", str(WINDOW_MODE),
        "--expected-window-min", str(WINDOW_MIN),
        "--expected-window-max", str(WINDOW_MAX),
        "--expected-idle-backoff", str(IDLE_BACKOFF),
    ]


def dry_run_payload(
    manifest_path: Path,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    out: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    out_path = out.absolute() if out else Path("<OUTSIDE_REPOSITORY_OUTPUT>")
    pair_path = out_path / "pair_build"
    planned = []
    for case in cases:
        case_out = out_path / "cases" / case["graph"] / case["view_slug"]
        planned.append({
            **case,
            "command": case_command(case, pair_path, case_out, timeout_seconds),
        })
    return {
        "mode": "dry_run_static_contract_only",
        "filesystem_inputs_hashed": False,
        "gpu_or_slurm_checked": False,
        "manifest": {
            "path": str(manifest_path.absolute()),
            "sha256": sha256(manifest_path),
            "historical_job": manifest.get("job"),
        },
        "contract": {
            "graphs": list(GRAPH_NAMES),
            "views": [view for view, _, _ in VIEWS],
            "case_count": len(cases),
            "workers": WORKERS,
            "blocks": BLOCKS,
            "delta": DELTA,
            "queue": QUEUE,
            "rounds": ROUNDS,
            "order": ["AB", "BA"],
            "warmups_per_process": WARMUPS,
            "formal_per_process": FORMAL_REPEATS,
            "gpu_execution": "strictly serial; single_no_l3=1, dual_l3=2",
            "pair_build": "fresh canonical clean-HEAD build inside output; external pair forbidden",
        },
        "cases": planned,
    }


def pair_integrity_snapshot(pair: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for name in PAIR_INTEGRITY_FILES:
        path = pair / name
        require(
            not path.is_symlink() and path.is_file(),
            f"pair integrity artifact is missing/not regular: {path}",
        )
        hashes[name] = sha256(path)
    dual = pair / "dual_build/mlmq"
    single = pair / "single_build/mlmq"
    require(os.access(dual, os.X_OK), f"dual binary is not executable: {dual}")
    require(os.access(single, os.X_OK), f"single binary is not executable: {single}")
    require(dual != single, "pair binaries resolve to the same path")
    require(hashes["dual_build/mlmq"] != hashes["single_build/mlmq"],
            "pair binaries have the same SHA256")
    return {
        "directory": str(pair),
        "files": hashes,
        "dual_binary_sha256": hashes["dual_build/mlmq"],
        "single_binary_sha256": hashes["single_build/mlmq"],
    }


def load_canonical_runner() -> Any:
    # Avoid creating ignored bytecode in the audited checkout while loading the
    # canonical validator.  Its validation includes clean-HEAD source archives,
    # exact build argv/macros, compiler/builder hashes, provenance, and all pair
    # artifact hashes.
    sys.dont_write_bytecode = True
    runner_dir = str(RUNNER.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    spec = importlib.util.spec_from_file_location("_l3_30h_pair_validator", RUNNER)
    require(spec is not None and spec.loader is not None,
            f"cannot load canonical runner module: {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (AttributeError, OSError, RuntimeError, SystemExit) as error:
        raise ContractError(f"cannot load canonical runner: {error}") from error
    return module


def validate_pair_with_canonical_runner(
    pair_argument: Path, head: str, module: Any | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Use the runner's pure validator, then take an independent fingerprint."""
    try:
        pair = pair_argument.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"cannot resolve pair build {pair_argument}: {error}") from error
    require(pair.is_dir(), f"pair build is not a directory: {pair}")
    require(pair != ROOT and ROOT not in pair.parents,
            "formal pair build must be outside the repository")
    dual = (pair / "dual_build/mlmq").resolve(strict=True)
    single = (pair / "single_build/mlmq").resolve(strict=True)
    module = module if module is not None else load_canonical_runner()
    try:
        validator = getattr(module, "validate_formal_pair_build")
        canonical = validator(pair, dual, single, head, ROOT)
    except (AttributeError, OSError, RuntimeError, SystemExit) as error:
        raise ContractError(f"canonical pair validation failed: {error}") from error
    require(isinstance(canonical, dict), "canonical pair validator returned no evidence")
    return pair, canonical, pair_integrity_snapshot(pair)


def build_fresh_pair(
    out: Path, head: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the one shared pair in this process, then bind it to clean HEAD."""
    module = load_canonical_runner()
    try:
        builder = getattr(module, "build_formal_pair")
        pair, build_record = builder(ROOT, out)
    except (AttributeError, OSError, RuntimeError, subprocess.SubprocessError,
            SystemExit) as error:
        raise ContractError(f"fresh canonical pair build failed: {error}") from error
    clean_head(head)
    pair, canonical, snapshot = validate_pair_with_canonical_runner(
        Path(pair), head, module
    )
    return pair, build_record, canonical, snapshot


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slurm_allocation_evidence(environment: dict[str, str]) -> dict[str, Any]:
    require(environment.get("SLURM_JOB_ID"), "run must be inside a Slurm allocation")
    require(
        environment.get("SLURM_JOB_PARTITION") == "a100",
        "formal regression requires SLURM_JOB_PARTITION=a100",
    )
    evidence: dict[str, Any] = {
        "SLURM_JOB_ID": environment["SLURM_JOB_ID"],
        "SLURM_JOB_PARTITION": environment["SLURM_JOB_PARTITION"],
    }
    counts: list[tuple[str, int]] = []
    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = environment.get(key)
        if value:
            count = len(csv_items(value))
            evidence[key] = {"value": value, "count": count}
            counts.append((key, count))
    on_node = environment.get("SLURM_GPUS_ON_NODE")
    if on_node:
        count: int | None = None
        if on_node.isdigit():
            count = int(on_node)
        else:
            match = re.search(r":(\d+)(?:\(|$)", on_node)
            if match:
                count = int(match.group(1))
        evidence["SLURM_GPUS_ON_NODE"] = {"value": on_node, "count": count}
        if count is not None:
            counts.append(("SLURM_GPUS_ON_NODE", count))
    require(counts, "Slurm environment does not prove the allocated GPU count")
    require(
        all(count == 2 for _, count in counts),
        f"Slurm allocation is not exactly two GPUs: {counts}",
    )
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        count = len(csv_items(visible))
        evidence["CUDA_VISIBLE_DEVICES"] = {"value": visible, "count": count}
        require(count == 2, "CUDA_VISIBLE_DEVICES does not expose exactly two GPUs")
    return evidence


def run_logged(command: list[str], path: Path, timeout: int = 30) -> str:
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    path.write_text(process.stdout, encoding="utf-8")
    require(
        process.returncode == 0,
        f"command failed with rc={process.returncode}: {' '.join(command)}",
    )
    return process.stdout


def gpu_preflight(out: Path) -> dict[str, Any]:
    allocation = slurm_allocation_evidence(dict(os.environ))
    require(NVIDIA_SMI.is_file() and os.access(NVIDIA_SMI, os.X_OK),
            f"canonical nvidia-smi is unavailable: {NVIDIA_SMI}")
    full = run_logged([str(NVIDIA_SMI)], out / "gpu_initial.log")
    query = run_logged(
        [
            str(NVIDIA_SMI), "--query-gpu=index,uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        out / "gpu_query.csv",
    )
    rows = []
    for raw in csv.reader(query.splitlines(), skipinitialspace=True):
        require(len(raw) == 4, f"malformed nvidia-smi row: {raw!r}")
        index, uuid, name, compute_capability = (value.strip() for value in raw)
        row = {
            "index": index,
            "uuid": uuid,
            "name": name,
            "compute_capability": compute_capability,
        }
        require(
            "A100" in name and compute_capability == "8.0",
            f"formal regression requires A100 CC 8.0, got {row}",
        )
        rows.append(row)
    require(len(rows) == 2, f"nvidia-smi exposes {len(rows)} GPUs, expected two")
    require(len({row["index"] for row in rows}) == 2,
            "nvidia-smi GPU indices are not unique")
    require(len({row["uuid"] for row in rows}) == 2,
            "nvidia-smi GPU UUIDs are not unique")
    topology = run_logged(
        [str(NVIDIA_SMI), "topo", "-m"], out / "gpu_topology.log"
    )
    applications = run_logged(
        [str(NVIDIA_SMI), "--query-compute-apps=pid", "--format=csv,noheader"],
        out / "gpu_apps_initial.log",
    ).strip()
    require(not applications, f"GPU compute applications exist before sampling: {applications}")
    return {
        "allocation": allocation,
        "visible_gpu_count": len(rows),
        "visible_gpus": rows,
        "compute_apps_before": [],
        "nvidia_smi": {"path": str(NVIDIA_SMI), "sha256": sha256(NVIDIA_SMI)},
        "snapshot_bytes": len(full.encode()),
        "topology_bytes": len(topology.encode()),
    }


def gpu_postflight(out: Path, initial: dict[str, Any]) -> dict[str, Any]:
    query = run_logged(
        [
            str(NVIDIA_SMI), "--query-gpu=index,uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        out / "gpu_query_final.csv",
    )
    rows = []
    for raw in csv.reader(query.splitlines(), skipinitialspace=True):
        require(len(raw) == 4, f"malformed final nvidia-smi row: {raw!r}")
        index, uuid, name, compute_capability = (value.strip() for value in raw)
        rows.append({
            "index": index,
            "uuid": uuid,
            "name": name,
            "compute_capability": compute_capability,
        })
    require(rows == initial["visible_gpus"],
            "visible GPU identity changed during the regression")
    applications = run_logged(
        [str(NVIDIA_SMI), "--query-compute-apps=pid", "--format=csv,noheader"],
        out / "gpu_apps_final.log",
    ).strip()
    require(not applications,
            f"GPU compute applications remain after sampling: {applications}")
    return {"visible_gpus": rows, "compute_apps_after": []}


def runtime_environment() -> tuple[dict[str, str], list[str]]:
    allowed = {
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if key in allowed or key.startswith("SLURM_")
    }
    environment["LANG"] = "C"
    environment["LC_ALL"] = "C"
    environment["PATH"] = FORMAL_TOOL_PATH
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    cleared = sorted(key for key in os.environ if key not in environment)
    return environment, cleared


def parse_integer_kv_line(
    line: str, prefix: str, expected_fields: set[str]
) -> tuple[dict[str, int] | None, list[str]]:
    tokens = line.split()
    errors: list[str] = []
    if not tokens or tokens[0] != prefix:
        return None, [f"line does not begin with {prefix}: {line!r}"]
    values: dict[str, int] = {}
    for token in tokens[1:]:
        if "=" not in token:
            errors.append(f"{prefix} token lacks '=': {token!r}")
            continue
        key, raw = token.split("=", 1)
        if key in values:
            errors.append(f"{prefix} repeats field {key}")
            continue
        try:
            values[key] = int(raw)
        except ValueError:
            errors.append(f"{prefix} field {key} is not an integer: {raw!r}")
    actual = set(values)
    if actual != expected_fields:
        errors.append(
            f"{prefix} fields differ: missing={sorted(expected_fields - actual)} "
            f"extra={sorted(actual - expected_fields)}"
        )
    return (None if errors else values), errors


L3_CONFIG_FIELDS = {
    "gpu", "work_blocks", "delta", "queue_type", "window_mode", "window_min",
    "window_max", "idle_backoff", "worker_recovery", "term_wait_ack",
    *EXPECTED_RX.keys(),
}
ACK_FIELDS = {"gpu", "active_slots", "capacity", "work_blocks", "warps_per_block"}
L2_CAPACITY_FIELDS = {
    "budget", "record_bytes", "buckets", "per_bucket", "allocated_records",
    "counter_bits",
}
L2_FINAL_FIELDS = {
    "gpu", "buckets", "reads", "writes", "completed", "guarded_writes",
    "max_bucket_writes", "per_bucket_capacity", "total_capacity", "counter_bits",
    "overflow_guard", "overflow_detected", "no_wrap",
}


def validate_capacity(row: dict[str, int], description: str) -> list[str]:
    errors: list[str] = []
    if row["budget"] <= 0 or row["record_bytes"] <= 0 or row["buckets"] <= 0:
        errors.append(f"{description} has nonpositive dimensions: {row}")
        return errors
    allocated = row["budget"] // row["record_bytes"]
    per_bucket = allocated // row["buckets"] // 512 * 512
    if (
        row["allocated_records"] != allocated
        or row["per_bucket"] != per_bucket
        or per_bucket <= 0
        or row["counter_bits"] != 32
    ):
        errors.append(f"{description} differs from the exact 512-record formula: {row}")
    return errors


def parse_bench_line(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            row[key] = value
    return row


def positive_float(raw: Any, description: str, errors: list[str]) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        errors.append(f"{description} is not numeric: {raw!r}")
        return None
    if not math.isfinite(value) or value <= 0:
        errors.append(f"{description} is not finite and positive: {raw!r}")
        return None
    return value


def validate_raw_process_log(
    path: Path,
    *,
    gpu_count: int,
    source: int,
    cut: int,
    vertices: int,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return {"valid": False, "errors": [f"cannot read raw log {path}: {error}"]}

    for marker in (
        "Error at node", "BENCH_REQUIRE_FAILURE", "ORACLE_MISMATCH",
        "ORACLE_SIZE_ERROR", "FINAL_AUDIT_ERROR", "L2_FINAL_UNSUPPORTED",
        "L3_CHAIN_SETUP",
    ):
        if marker in raw:
            errors.append(f"raw log contains forbidden marker {marker!r}")

    pending: dict[str, list[Any]] = {
        "wide": [], "l3": [], "ack": [], "l2": [], "launch": [], "audit": [],
    }
    groups: list[dict[str, list[Any]]] = []
    bench_rows: list[dict[str, str]] = []
    capacities: list[dict[str, int] | None] = []
    no_l3_configs: list[tuple[int, int, str]] = []

    wide_re = re.compile(r"^WIDE_ORACLE vertices=(\d+) correct=(\d+)$")
    no_l3_re = re.compile(r"^NO_L3_CONFIG workers=(\d+) delta=(\d+) queue=(\S+)$")
    audit_re = re.compile(
        r"^FINAL_AUDIT mismatches=(\d+) residual_edges=(\d+) "
        r"cross_residual_edges=(\d+)$"
    )

    for line in raw.splitlines():
        if line.startswith("WIDE_ORACLE "):
            match = wide_re.fullmatch(line)
            if match is None:
                errors.append(f"malformed WIDE_ORACLE line: {line!r}")
                pending["wide"].append(None)
            else:
                pending["wide"].append(tuple(map(int, match.groups())))
        elif line.startswith("L3_CONFIG "):
            row, row_errors = parse_integer_kv_line(line, "L3_CONFIG", L3_CONFIG_FIELDS)
            errors.extend(row_errors)
            pending["l3"].append(row)
        elif line.startswith("L3_WORKER_ACK "):
            row, row_errors = parse_integer_kv_line(line, "L3_WORKER_ACK", ACK_FIELDS)
            errors.extend(row_errors)
            pending["ack"].append(row)
        elif line.startswith("L2_FINAL "):
            row, row_errors = parse_integer_kv_line(line, "L2_FINAL", L2_FINAL_FIELDS)
            errors.extend(row_errors)
            pending["l2"].append(row)
        elif line.startswith("L2_CAPACITY "):
            row, row_errors = parse_integer_kv_line(
                line, "L2_CAPACITY", L2_CAPACITY_FIELDS
            )
            errors.extend(row_errors)
            capacities.append(row)
        elif line.startswith("NO_L3_CONFIG "):
            match = no_l3_re.fullmatch(line)
            if match is None:
                errors.append(f"malformed NO_L3_CONFIG line: {line!r}")
            else:
                workers, delta, queue = match.groups()
                no_l3_configs.append((int(workers), int(delta), queue))
        elif line.startswith("NO_L3_LAUNCH "):
            row, row_errors = parse_integer_kv_line(
                line, "NO_L3_LAUNCH", {"work_blocks", "delta", "repeat"}
            )
            errors.extend(row_errors)
            pending["launch"].append(row)
        elif line.startswith("FINAL_AUDIT "):
            match = audit_re.fullmatch(line)
            if match is None:
                errors.append(f"malformed FINAL_AUDIT line: {line!r}")
                pending["audit"].append(None)
            else:
                pending["audit"].append(tuple(map(int, match.groups())))
        elif line.startswith("BENCH ") and "algorithm=MLMQ" in line:
            bench_rows.append(parse_bench_line(line))
            groups.append(pending)
            pending = {
                "wide": [], "l3": [], "ack": [], "l2": [], "launch": [], "audit": [],
            }

    if len(bench_rows) != PROCESS_SAMPLES:
        errors.append(
            f"expected {PROCESS_SAMPLES} BENCH rows, got {len(bench_rows)}"
        )
    for index, row in enumerate(bench_rows):
        expected = {
            "algorithm": "MLMQ",
            "gpu_count": str(gpu_count),
            "source": str(source),
            "repeat": str(index),
            "warmup": str(int(index < WARMUPS)),
            "queue": QUEUE,
            "correct": "1",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(
                    f"BENCH row {index} {key}={row.get(key)!r}, expected {value!r}"
                )
        positive_float(row.get("solve_ms"), f"BENCH row {index} solve_ms", errors)
        positive_float(
            row.get("query_wall_ms"), f"BENCH row {index} query_wall_ms", errors
        )

    expected_capacity_count = 1 if gpu_count == 1 else 2
    if len(capacities) != expected_capacity_count:
        errors.append(
            f"expected {expected_capacity_count} L2_CAPACITY rows, got {len(capacities)}"
        )
    valid_capacities = [row for row in capacities if row is not None]
    for index, row in enumerate(valid_capacities):
        errors.extend(validate_capacity(row, f"L2_CAPACITY row {index}"))
    if gpu_count == 2 and len(valid_capacities) == 2:
        if valid_capacities[0] != valid_capacities[1]:
            errors.append("dual GPUs reported different L2_CAPACITY rows")
    capacity = valid_capacities[0] if valid_capacities else None

    for sample_index, group in enumerate(groups):
        if group["wide"] != [(vertices, 1)]:
            errors.append(
                f"sample {sample_index} oracle group is {group['wide']!r}, "
                f"expected [{(vertices, 1)!r}]"
            )
        if gpu_count == 1:
            if group["l3"] or group["ack"] or group["l2"] or group["audit"]:
                errors.append(f"single sample {sample_index} contains L3-only evidence")
            expected_launch = {
                "work_blocks": BLOCKS, "delta": DELTA, "repeat": sample_index,
            }
            if group["launch"] != [expected_launch]:
                errors.append(
                    f"single sample {sample_index} launch is {group['launch']!r}, "
                    f"expected {[expected_launch]!r}"
                )
            continue

        if group["launch"]:
            errors.append(f"dual sample {sample_index} contains NO_L3_LAUNCH")
        if len(group["l3"]) != 2 or any(row is None for row in group["l3"]):
            errors.append(f"dual sample {sample_index} lacks two valid L3_CONFIG rows")
        else:
            if sorted(row["gpu"] for row in group["l3"]) != [0, 1]:
                errors.append(f"dual sample {sample_index} L3_CONFIG GPUs are not 0/1")
            for row in group["l3"]:
                expected = {
                    "work_blocks": BLOCKS,
                    "delta": DELTA,
                    "queue_type": QUEUE_TYPE_ID,
                    "window_mode": WINDOW_MODE,
                    "window_min": WINDOW_MIN,
                    "window_max": WINDOW_MAX,
                    "idle_backoff": IDLE_BACKOFF,
                    "worker_recovery": 1,
                    "term_wait_ack": 1,
                    **EXPECTED_RX,
                }
                mismatches = {
                    key: {"actual": row[key], "expected": value}
                    for key, value in expected.items() if row[key] != value
                }
                if mismatches:
                    errors.append(
                        f"dual sample {sample_index} gpu {row['gpu']} frozen "
                        f"L3_CONFIG mismatch: {mismatches}"
                    )
        if len(group["ack"]) != 2 or any(row is None for row in group["ack"]):
            errors.append(f"dual sample {sample_index} lacks two valid ACK rows")
        else:
            if sorted(row["gpu"] for row in group["ack"]) != [0, 1]:
                errors.append(f"dual sample {sample_index} ACK GPUs are not 0/1")
            for row in group["ack"]:
                if row != {
                    "gpu": row["gpu"],
                    "active_slots": BLOCKS * 16,
                    "capacity": BLOCKS * 16,
                    "work_blocks": BLOCKS,
                    "warps_per_block": 16,
                }:
                    errors.append(
                        f"dual sample {sample_index} gpu {row['gpu']} invalid W512 ACK: {row}"
                    )
        if len(group["l2"]) != 2 or any(row is None for row in group["l2"]):
            errors.append(f"dual sample {sample_index} lacks two valid L2_FINAL rows")
        else:
            if sorted(row["gpu"] for row in group["l2"]) != [0, 1]:
                errors.append(f"dual sample {sample_index} L2_FINAL GPUs are not 0/1")
            for row in group["l2"]:
                valid = (
                    row["buckets"] > 0
                    and row["reads"] >= 0
                    and row["reads"] == row["writes"] == row["completed"]
                    and row["writes"] == row["guarded_writes"]
                    and 0 <= row["max_bucket_writes"] <= row["writes"]
                    and row["per_bucket_capacity"] > 0
                    and row["max_bucket_writes"] <= row["per_bucket_capacity"]
                    and row["total_capacity"]
                    == row["buckets"] * row["per_bucket_capacity"]
                    and 0 < row["total_capacity"] <= 2147483647
                    and row["counter_bits"] == 32
                    and row["overflow_guard"] == 1
                    and row["overflow_detected"] == 0
                    and row["no_wrap"] == 1
                )
                if capacity is not None:
                    valid = valid and (
                        row["buckets"] == capacity["buckets"]
                        and row["per_bucket_capacity"] == capacity["per_bucket"]
                        and row["total_capacity"]
                        == capacity["buckets"] * capacity["per_bucket"]
                        and row["total_capacity"] <= capacity["allocated_records"]
                        and row["counter_bits"] == capacity["counter_bits"]
                    )
                if not valid:
                    errors.append(
                        f"dual sample {sample_index} gpu {row['gpu']} invalid L2_FINAL: {row}"
                    )
        if group["audit"] != [(0, 0, 0)]:
            errors.append(
                f"dual sample {sample_index} final audit is {group['audit']!r}, "
                "expected one all-zero row"
            )

    if gpu_count == 1:
        if no_l3_configs != [(WORKERS, DELTA, QUEUE)]:
            errors.append(
                f"single NO_L3_CONFIG is {no_l3_configs!r}, expected "
                f"{[(WORKERS, DELTA, QUEUE)]!r}"
            )
        if pending != {
            "wide": [(vertices, 1)], "l3": [], "ack": [], "l2": [],
            "launch": [], "audit": [],
        }:
            errors.append(f"unexpected trailing single contract evidence: {pending}")
    else:
        if no_l3_configs:
            errors.append("dual log unexpectedly contains NO_L3_CONFIG")
        if any(pending.values()):
            errors.append(f"dual contract evidence remains after final BENCH: {pending}")

    partitions = [
        tuple(map(int, match.groups()))
        for match in re.finditer(r"GPU(\d+) partition: \[(\d+), (\d+)\)", raw)
    ]
    expected_partitions = [] if gpu_count == 1 else [(0, 0, cut), (1, cut, vertices)]
    if sorted(set(partitions)) != expected_partitions:
        errors.append(
            f"effective partitions are {sorted(set(partitions))}, "
            f"expected {expected_partitions}"
        )
    if len(re.findall(r"^RUN_RC=0$", raw, flags=re.MULTILINE)) != 1:
        errors.append("raw log does not contain exactly one RUN_RC=0 line")

    return {
        "valid": not errors,
        "errors": errors,
        "bench_rows": bench_rows,
        "bench_count": len(bench_rows),
        "capacity_rows": capacities,
        "partition_rows": partitions,
        "oracle_per_query_count": sum(len(group["wide"]) for group in groups),
        "trailing_oracle_count": len(pending["wide"]),
        "l2_final_count": sum(len(group["l2"]) for group in groups),
        "final_audit_count": sum(len(group["audit"]) for group in groups),
        "sha256": sha256(path),
    }


def close_number(actual: Any, expected: float) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def validate_case_output(
    case: dict[str, Any],
    case_out: Path,
    command: list[str],
    runner_rc: int,
    *,
    head: str,
    runner_sha256: str,
    pair_snapshot: dict[str, Any],
    slurm_job_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    errors: list[str] = []

    def load(name: str, expected_type: type) -> Any:
        path = case_out / name
        try:
            value = read_json(path, f"case {name}")
        except ContractError as error:
            errors.append(str(error))
            return None
        if not isinstance(value, expected_type):
            errors.append(f"case {name} has type {type(value).__name__}, expected {expected_type.__name__}")
            return None
        return value

    if runner_rc != 0:
        errors.append(f"runner exited with rc={runner_rc}")
    manifest = load("manifest.json", dict)
    records = load("records.json", list)
    samples = load("samples.json", list)
    summary = load("summary.json", dict)

    binary_hashes = {
        "dual_binary": pair_snapshot["dual_binary_sha256"],
        "single_binary": pair_snapshot["single_binary_sha256"],
    }
    expected_configuration = {
        "sampling": "exploratory",
        "source": case["source"],
        "delta": DELTA,
        "cut_percent": case["cut_percent"],
        "blocks": BLOCKS,
        "warmups_per_process": WARMUPS,
        "formal_repeats_per_process": FORMAL_REPEATS,
        "rounds": ROUNDS,
        "timeout_seconds": timeout_seconds,
        "queue": QUEUE,
        "final_audit": "all",
    }
    if manifest is not None:
        if manifest.get("status") != "measurement_valid":
            errors.append(f"case manifest status={manifest.get('status')!r}")
        if manifest.get("measurement_valid") is not True:
            errors.append("case manifest measurement_valid is not true")
        if manifest.get("all_processes_valid") is not True:
            errors.append("case manifest all_processes_valid is not true")
        if manifest.get("process_count") != PROCESSES_PER_CASE:
            errors.append(
                f"case manifest process_count={manifest.get('process_count')!r}, "
                f"expected {PROCESSES_PER_CASE}"
            )
        if manifest.get("slurm_job_id") != slurm_job_id:
            errors.append("case Slurm job ID differs from outer allocation")
        if manifest.get("command") != command[2:]:
            errors.append("case manifest command differs from the archived outer command")
        configuration = manifest.get("configuration", {})
        for key, expected in expected_configuration.items():
            if configuration.get(key) != expected:
                errors.append(
                    f"case configuration {key}={configuration.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if configuration.get("gpu_contract") != {
            "single_no_l3": 1, "dual_l3": 2, "fallback_allowed": False,
        }:
            errors.append("case GPU contract permits fallback or wrong GPU counts")
        if configuration.get("expected_window") != {
            "mode": WINDOW_MODE,
            "min_cycles": WINDOW_MIN,
            "max_cycles": WINDOW_MAX,
            "idle_backoff": IDLE_BACKOFF,
        }:
            errors.append("case expected-window contract differs from the frozen values")
        git = manifest.get("git", {})
        if git.get("head") != {"rc": 0, "output": head}:
            errors.append("case runner did not record the frozen clean HEAD")
        if git.get("status") != {"rc": 0, "output": ""}:
            errors.append("case runner observed a dirty worktree")
        inputs = manifest.get("inputs", {})
        expected_inputs = {
            "dual_binary": (
                str((Path(pair_snapshot["directory"]) / "dual_build/mlmq").resolve()),
                binary_hashes["dual_binary"],
            ),
            "single_binary": (
                str((Path(pair_snapshot["directory"]) / "single_build/mlmq").resolve()),
                binary_hashes["single_binary"],
            ),
            "graph": (case["graph_path"], case["graph_sha256"]),
            "oracle": (case["oracle_path"], case["oracle_sha256"]),
            "runner": (str(RUNNER.resolve()), runner_sha256),
        }
        for label, (path, digest) in expected_inputs.items():
            row = inputs.get(label, {})
            if row.get("path") != path or row.get("sha256") != digest:
                errors.append(
                    f"case input {label} identity differs: "
                    f"path={row.get('path')!r} sha256={row.get('sha256')!r}"
                )
        graph_header = inputs.get("graph", {}).get("header", {})
        if graph_header.get("vertices") != case["vertices"]:
            errors.append("case runner graph vertex count differs from frozen manifest")
        if inputs.get("oracle", {}).get("bytes") != case["vertices"] * 4:
            errors.append("case runner oracle size differs from vertices*4")
        preflight = manifest.get("preflight", {})
        if preflight.get("visible_gpu_count") != 2:
            errors.append("case runner did not observe exactly two GPUs")
        if preflight.get("compute_apps_before") != []:
            errors.append("case runner observed another GPU compute process")
        visible_gpus = preflight.get("visible_gpus")
        if not isinstance(visible_gpus, list) or len(visible_gpus) != 2:
            errors.append("case runner lacks two per-GPU identity rows")
        else:
            for row in visible_gpus:
                if not isinstance(row, dict) or (
                    "A100" not in str(row.get("name"))
                    or str(row.get("compute_capability")) != "8.0"
                ):
                    errors.append(f"case runner observed a non-A100/CC8.0 GPU: {row!r}")
            if all(isinstance(row, dict) for row in visible_gpus):
                if len({str(row.get("index")) for row in visible_gpus}) != 2:
                    errors.append("case runner GPU indices are not unique")
                if len({str(row.get("uuid")) for row in visible_gpus}) != 2:
                    errors.append("case runner GPU UUIDs are not unique")

    expected_processes = [
        (0, 0, "single_no_l3", 1),
        (0, 1, "dual_l3", 2),
        (1, 0, "dual_l3", 2),
        (1, 1, "single_no_l3", 1),
    ]
    log_evidence: list[dict[str, Any]] = []
    formal_values: dict[str, list[float]] = {
        "single_no_l3": [], "dual_l3": [],
    }
    query_values: dict[str, list[float]] = {
        "single_no_l3": [], "dual_l3": [],
    }
    if records is not None:
        if len(records) != PROCESSES_PER_CASE:
            errors.append(f"records.json has {len(records)} rows, expected {PROCESSES_PER_CASE}")
        for index, expected in enumerate(expected_processes):
            if index >= len(records) or not isinstance(records[index], dict):
                errors.append(f"records.json lacks object row {index}")
                continue
            record = records[index]
            round_id, sequence, label, gpu_count = expected
            expected_prefix = case_out / f"round{round_id:03d}_{sequence}_{label}"
            for key, value in {
                "round": round_id,
                "sequence_in_round": sequence,
                "configuration": label,
                "requested_gpu_count": gpu_count,
                "prefix": str(expected_prefix),
                "rc": 0,
                "valid": True,
                "reason": "ok",
            }.items():
                if record.get(key) != value:
                    errors.append(
                        f"record {index} {key}={record.get(key)!r}, expected {value!r}"
                    )
            binary = (
                Path(pair_snapshot["directory"]) /
                ("single_build/mlmq" if gpu_count == 1 else "dual_build/mlmq")
            ).resolve()
            expected_solver_command = [
                str(binary), "-i", case["graph_path"], "-n", str(gpu_count),
                "-d", str(DELTA),
            ]
            if record.get("command") != expected_solver_command:
                errors.append(f"record {index} solver command differs from frozen command")
            record_samples = record.get("samples")
            if not isinstance(record_samples, list) or len(record_samples) != PROCESS_SAMPLES:
                errors.append(
                    f"record {index} sample count is not {PROCESS_SAMPLES}"
                )
                record_samples = []
            individual_path = expected_prefix.with_suffix(".json")
            if individual_path.is_file():
                try:
                    if read_json(individual_path, "individual process record") != record:
                        errors.append(f"record {index} differs from {individual_path.name}")
                except ContractError as error:
                    errors.append(str(error))
            else:
                errors.append(f"missing individual process record {individual_path}")
            for suffix in (".gpu_before.log", ".gpu_after.log"):
                snapshot = expected_prefix.with_suffix(suffix)
                if not snapshot.is_file() or snapshot.stat().st_size == 0:
                    errors.append(f"missing/empty GPU snapshot {snapshot}")

            log_result = validate_raw_process_log(
                expected_prefix.with_suffix(".log"),
                gpu_count=gpu_count,
                source=case["source"],
                cut=case["cut"],
                vertices=case["vertices"],
            )
            log_evidence.append({
                "record_index": index,
                "round": round_id,
                "sequence": sequence,
                "configuration": label,
                "path": str(expected_prefix.with_suffix(".log")),
                **log_result,
            })
            errors.extend(f"record {index}: {error}" for error in log_result["errors"])
            if record_samples != log_result.get("bench_rows"):
                errors.append(f"record {index} JSON samples differ from raw BENCH rows")
            effective = record.get("effective_contract")
            if not isinstance(effective, dict) or effective.get("errors") != []:
                errors.append(f"record {index} effective_contract has errors")
            else:
                oracle = effective.get("oracle_contract", {})
                if (
                    oracle.get("valid") is not True
                    or oracle.get("expected_samples") != PROCESS_SAMPLES
                    or oracle.get("observed_samples") != PROCESS_SAMPLES
                    or oracle.get("errors") != []
                ):
                    errors.append(f"record {index} serialized oracle contract is invalid")
                if gpu_count == 2:
                    dual = effective.get("dual_contract", {})
                    if (
                        not isinstance(dual, dict)
                        or dual.get("valid") is not True
                        or dual.get("expected_samples") != PROCESS_SAMPLES
                        or dual.get("observed_samples") != PROCESS_SAMPLES
                        or dual.get("l2_final_present") is not True
                        or dual.get("l2_capacity_present") is not True
                        or dual.get("errors") != []
                    ):
                        errors.append(f"record {index} serialized dual contract is invalid")
                else:
                    if effective.get("l3_config_lines") != 0:
                        errors.append(f"record {index} single unexpectedly logged L3_CONFIG")
                    if effective.get("no_l3_config_lines") != 1:
                        errors.append(f"record {index} single lacks NO_L3_CONFIG")
                    if effective.get("no_l3_launch_lines") != PROCESS_SAMPLES:
                        errors.append(f"record {index} single lacks NO_L3_LAUNCH rows")

            for sample_index, sample in enumerate(record_samples):
                if not isinstance(sample, dict):
                    errors.append(f"record {index} sample {sample_index} is not an object")
                    continue
                expected_sample = {
                    "algorithm": "MLMQ",
                    "gpu_count": str(gpu_count),
                    "source": str(case["source"]),
                    "repeat": str(sample_index),
                    "warmup": str(int(sample_index < WARMUPS)),
                    "queue": QUEUE,
                    "correct": "1",
                }
                for key, value in expected_sample.items():
                    if sample.get(key) != value:
                        errors.append(
                            f"record {index} sample {sample_index} {key} differs"
                        )
                solve = positive_float(
                    sample.get("solve_ms"),
                    f"record {index} sample {sample_index} solve_ms", errors,
                )
                query = positive_float(
                    sample.get("query_wall_ms"),
                    f"record {index} sample {sample_index} query_wall_ms", errors,
                )
                if sample_index >= WARMUPS:
                    if solve is not None:
                        formal_values[label].append(solve)
                    if query is not None:
                        query_values[label].append(query)

    if samples is not None:
        expected_total = PROCESSES_PER_CASE * PROCESS_SAMPLES
        if len(samples) != expected_total:
            errors.append(f"samples.json has {len(samples)} rows, expected {expected_total}")
        if any(not isinstance(row, dict) or row.get("run_valid") is not True for row in samples):
            errors.append("samples.json contains a non-object or invalid-run sample")

    extracted: dict[str, Any] = {
        "T1_single_no_l3_solve_ms": formal_values["single_no_l3"],
        "T2_dual_l3_solve_ms": formal_values["dual_l3"],
        "T1_single_no_l3_query_wall_ms": query_values["single_no_l3"],
        "T2_dual_l3_query_wall_ms": query_values["dual_l3"],
    }
    if all(len(formal_values[label]) == FORMAL_SAMPLES_PER_CONFIGURATION
           for label in formal_values):
        t1 = median(formal_values["single_no_l3"])
        t2 = median(formal_values["dual_l3"])
        q1 = median(query_values["single_no_l3"])
        q2 = median(query_values["dual_l3"])
        extracted.update({
            "T1_median_solve_ms": t1,
            "T2_median_solve_ms": t2,
            "speedup_T1_over_T2": t1 / t2,
            "T1_median_query_wall_ms": q1,
            "T2_median_query_wall_ms": q2,
            "query_wall_speedup_T1_over_T2": q1 / q2,
            "target": TARGET_SPEEDUP,
            "target_met": t1 / t2 >= TARGET_SPEEDUP,
        })
    else:
        errors.append(
            "formal solve samples per configuration differ from 10: "
            f"{ {key: len(value) for key, value in formal_values.items()} }"
        )

    if summary is not None:
        if summary.get("sampling") != "exploratory":
            errors.append("case summary sampling label is not exploratory")
        if summary.get("measurement_valid") is not True:
            errors.append("case summary measurement_valid is not true")
        rounds = summary.get("rounds")
        if not isinstance(rounds, list) or len(rounds) != ROUNDS:
            errors.append("case summary does not contain exactly two rounds")
        else:
            for round_id, row in enumerate(rounds):
                if not isinstance(row, dict) or row.get("round") != round_id or row.get("valid") is not True:
                    errors.append(f"case summary round {round_id} is invalid")
                    continue
                for label in ("T1", "T2"):
                    entry = row.get(label, {})
                    if (
                        entry.get("valid") is not True
                        or entry.get("formal_sample_count") != FORMAL_REPEATS
                        or entry.get("expected_formal_samples") != FORMAL_REPEATS
                    ):
                        errors.append(f"case summary round {round_id} {label} sample contract differs")
        combined = summary.get("combined", {})
        if combined.get("valid") is not True:
            errors.append("case combined summary is invalid")
        for label in ("T1", "T2"):
            entry = combined.get(label, {})
            if (
                entry.get("valid") is not True
                or entry.get("formal_sample_count") != FORMAL_SAMPLES_PER_CONFIGURATION
                or entry.get("expected_formal_samples") != FORMAL_SAMPLES_PER_CONFIGURATION
            ):
                errors.append(f"case combined {label} sample contract differs")
        if "speedup_T1_over_T2" in extracted:
            speedup = extracted["speedup_T1_over_T2"]
            if not close_number(combined.get("S_solve_T1_over_T2"), speedup):
                errors.append("case combined solve speedup differs from raw samples")
            target_met = speedup >= TARGET_SPEEDUP
            if combined.get("target_met") is not target_met:
                errors.append("case combined target_met differs from raw samples")
            if summary.get("target_met") is not target_met:
                errors.append("case top-level target_met differs from raw samples")
            if summary.get("numeric_target_met") is not target_met:
                errors.append("case numeric_target_met differs from raw samples")
            if summary.get("formal_integrity_valid") is not True:
                errors.append("case runner integrity gate is not true")
            if manifest is not None and manifest.get("target_met") is not target_met:
                errors.append("case manifest target_met differs from raw samples")
            if manifest is not None and manifest.get("numeric_target_met") is not target_met:
                errors.append("case manifest numeric_target_met differs from raw samples")
            if manifest is not None and manifest.get("formal_integrity_valid") is not True:
                errors.append("case manifest runner integrity gate is not true")

    return {
        "case_id": case["case_id"],
        "graph": case["graph"],
        "view": case["view"],
        "view_slug": case["view_slug"],
        "source": case["source"],
        "cut": case["cut"],
        "cut_percent": case["cut_percent"],
        "vertices": case["vertices"],
        "graph_path": case["graph_path"],
        "graph_sha256": case["graph_sha256"],
        "oracle_path": case["oracle_path"],
        "oracle_sha256": case["oracle_sha256"],
        "runner_rc": runner_rc,
        "measurement_valid": not errors,
        "target_met": extracted.get("target_met"),
        "errors": errors,
        "metrics": extracted,
        "process_log_evidence": log_evidence,
        "case_output": str(case_out),
        "case_manifest_sha256": sha256(case_out / "manifest.json")
            if (case_out / "manifest.json").is_file() else None,
        "case_records_sha256": sha256(case_out / "records.json")
            if (case_out / "records.json").is_file() else None,
        "case_summary_sha256": sha256(case_out / "summary.json")
            if (case_out / "summary.json").is_file() else None,
    }


def geometric_mean(values: list[float]) -> float:
    require(values and all(math.isfinite(value) and value > 0 for value in values),
            "geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in case_results:
        metrics = result.get("metrics", {})
        rows.append({
            "case_id": result["case_id"],
            "graph": result["graph"],
            "view": result["view"],
            "measurement_valid": result["measurement_valid"],
            "T1_single_no_l3_median_solve_ms": metrics.get("T1_median_solve_ms"),
            "T2_dual_l3_median_solve_ms": metrics.get("T2_median_solve_ms"),
            "speedup_T1_over_T2": metrics.get("speedup_T1_over_T2"),
            "T1_single_no_l3_median_query_wall_ms": metrics.get(
                "T1_median_query_wall_ms"
            ),
            "T2_dual_l3_median_query_wall_ms": metrics.get(
                "T2_median_query_wall_ms"
            ),
            "query_wall_speedup_T1_over_T2": metrics.get(
                "query_wall_speedup_T1_over_T2"
            ),
            "target": TARGET_SPEEDUP,
            "target_met": result.get("target_met"),
            "error_count": len(result.get("errors", [])),
            "case_output": result.get("case_output"),
        })

    geomeans: dict[str, Any] = {}
    for view in ("G", "G+"):
        selected = [row for row in rows if row["view"] == view]
        valid = (
            len(selected) == len(GRAPH_NAMES)
            and all(row["measurement_valid"] for row in selected)
        )
        speedups = [row["speedup_T1_over_T2"] for row in selected]
        wall_speedups = [row["query_wall_speedup_T1_over_T2"] for row in selected]
        if valid and all(isinstance(value, (int, float)) for value in speedups):
            geomeans[view] = {
                "valid": True,
                "graph_count": len(selected),
                "solve_speedup_geomean_T1_over_T2": geometric_mean(speedups),
                "query_wall_speedup_geomean_T1_over_T2": geometric_mean(wall_speedups),
            }
        else:
            geomeans[view] = {
                "valid": False,
                "graph_count": len(selected),
                "solve_speedup_geomean_T1_over_T2": None,
                "query_wall_speedup_geomean_T1_over_T2": None,
                "reason": "one or more physical-input measurements are invalid",
            }

    all_valid = len(rows) == 16 and all(row["measurement_valid"] for row in rows)
    all_speedups = [row["speedup_T1_over_T2"] for row in rows]
    pooled = (
        geometric_mean(all_speedups)
        if all_valid and all(isinstance(value, (int, float)) for value in all_speedups)
        else None
    )
    target_rows = [row for row in rows if row["target_met"] is not None]
    return {
        "measurement_valid": all_valid,
        "measurement_validity_is_independent_of_target": True,
        "formula": "S = median(10 T1 solve_ms) / median(10 T2 solve_ms) per case",
        "geomean_formula": "exp(mean(log(per-case S)))",
        "case_count": len(rows),
        "expected_case_count": 16,
        "expected_processes": 16 * PROCESSES_PER_CASE,
        "expected_queries_including_warmups": 16 * PROCESSES_PER_CASE * PROCESS_SAMPLES,
        "expected_formal_queries": 16 * 2 * FORMAL_SAMPLES_PER_CONFIGURATION,
        "target": TARGET_SPEEDUP,
        "target_met_case_count": sum(row["target_met"] is True for row in rows),
        "target_evaluated_case_count": len(target_rows),
        "all_cases_target_met": (
            len(target_rows) == 16 and all(row["target_met"] is True for row in rows)
        ),
        "geomeans": geomeans,
        "all_16_solve_speedup_geomean_T1_over_T2": pooled,
        "cases": rows,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "graph", "view", "measurement_valid",
        "T1_single_no_l3_median_solve_ms", "T2_dual_l3_median_solve_ms",
        "speedup_T1_over_T2", "T1_single_no_l3_median_query_wall_ms",
        "T2_dual_l3_median_query_wall_ms", "query_wall_speedup_T1_over_T2",
        "target", "target_met", "error_count", "case_output",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def artifact_hashes(out: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(out.rglob("*")):
        if path == out / "artifact_sha256.json":
            continue
        require(not path.is_symlink(), f"output contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(out).as_posix()] = sha256(path)
    return result


def ensure_frozen_during_sampling(
    *,
    head: str,
    runner_sha256: str,
    matrix_manifest: Path,
    matrix_manifest_sha256: str,
    pair: Path,
    pair_snapshot: dict[str, Any],
) -> None:
    clean_head(head)
    require(sha256(RUNNER) == runner_sha256, "run_l3_30h.py changed during sampling")
    require(
        sha256(matrix_manifest) == matrix_manifest_sha256,
        "frozen eight-graph manifest changed during sampling",
    )
    for relative, expected in (
        ("dual_build/mlmq", pair_snapshot["dual_binary_sha256"]),
        ("single_build/mlmq", pair_snapshot["single_binary_sha256"]),
    ):
        require(sha256(pair / relative) == expected, f"pair binary changed: {relative}")


def run_case_subprocess(
    command: list[str], log: Path, environment: dict[str, str], case_timeout: int
) -> tuple[int, bool, float]:
    import time

    started = time.monotonic()
    timed_out = False
    with log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=case_timeout,
            )
            rc = process.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            timed_out = True
            stream.write(f"\nOUTER_CASE_TIMEOUT_SECONDS={case_timeout}\n")
        except OSError as error:
            rc = 125
            stream.write(f"\nOUTER_CASE_LAUNCH_ERROR={error!r}\n")
    return rc, timed_out, time.monotonic() - started


def execute(args: argparse.Namespace, manifest: dict[str, Any],
            cases: list[dict[str, Any]]) -> int:
    require(
        args.graph_manifest.resolve(strict=True)
        == DEFAULT_MATRIX_MANIFEST.resolve(strict=True),
        f"real regression requires canonical manifest {DEFAULT_MATRIX_MANIFEST}",
    )
    git = clean_head()
    head = git["head"]
    committed = {
        "driver": require_committed_file(Path(__file__)),
        "runner": require_committed_file(RUNNER),
        "matrix_manifest": require_committed_file(args.graph_manifest),
    }
    runner_sha256 = committed["runner"]["sha256"]
    matrix_manifest_sha256 = committed["matrix_manifest"]["sha256"]

    require(args.out is not None, "--out is required for a real run")
    out_requested = args.out
    require(not out_requested.exists() and not out_requested.is_symlink(),
            f"output already exists: {out_requested}")
    out_parent = out_requested.parent.resolve(strict=True)
    out = out_parent / out_requested.name
    require(out != ROOT and ROOT not in out.parents,
            "formal regression output must be outside the repository")
    out.mkdir()
    write_json(out / "status.json", {
        "status": "preflight_pending",
        "measurement_valid": False,
        "created_utc": now_utc(),
    })
    write_json(out / "identity_preflight.json", {
        "repository": {"root": str(ROOT), **git},
        "committed_files": committed,
    })

    try:
        input_evidence_start = validate_input_files(cases)
        write_json(out / "input_evidence_start.json", input_evidence_start)
        write_json(out / "status.json", {
            "status": "building_fresh_pair",
            "measurement_valid": False,
            "created_utc": now_utc(),
        })
        pair, pair_build_record, canonical_pair, pair_start = build_fresh_pair(
            out, head
        )
        git_after_pair_build = clean_head(head)
        write_json(out / "pair_validation.json", canonical_pair)
        write_json(out / "pair_integrity_start.json", pair_start)
        preflight = gpu_preflight(out)
        environment, cleared_environment = runtime_environment()
        case_timeout = args.timeout * PROCESSES_PER_CASE + 600

        commands = []
        for case in cases:
            case_out = out / "cases" / case["graph"] / case["view_slug"]
            commands.append({
                "case_id": case["case_id"],
                "command": case_command(case, pair, case_out, args.timeout),
                "cwd": str(ROOT),
                "outer_timeout_seconds": case_timeout,
            })
        write_json(out / "commands.json", commands)
        write_json(out / "frozen_cases.json", cases)
        run_manifest = {
            "schema": 1,
            "purpose": (
                "final-clean-SHA eight-graph G/G+ performance/correctness regression; "
                "not primary target selection"
            ),
            "created_utc": now_utc(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "repository": {"root": str(ROOT), **git},
            "repository_after_pair_build": git_after_pair_build,
            "committed_files": committed,
            "historical_matrix_manifest": {
                "path": str(args.graph_manifest.resolve()),
                "sha256": matrix_manifest_sha256,
                "source_job": manifest.get("job"),
            },
            "configuration": {
                "graphs": list(GRAPH_NAMES),
                "physical_views": {"G": "original", "G+": "augmented"},
                "case_count": 16,
                "single": "independent no-L3, one GPU",
                "dual": "L3, two GPUs",
                "round_order": ["single_no_l3 then dual_l3", "dual_l3 then single_no_l3"],
                "warmups_per_process": WARMUPS,
                "formal_per_process": FORMAL_REPEATS,
                "rounds": ROUNDS,
                "workers": WORKERS,
                "blocks": BLOCKS,
                "delta": DELTA,
                "queue": QUEUE,
                "timeout_per_process_seconds": args.timeout,
                "final_audit": "all",
                "target_speedup": TARGET_SPEEDUP,
                "target_failure_invalidates_measurement": False,
                "gpu_serialization": "one runner subprocess at a time; no case concurrency",
            },
            "preflight": preflight,
            "pair_validation": canonical_pair,
            "fresh_pair_build_record": pair_build_record,
            "pair_integrity_start": pair_start,
            "input_evidence_start": input_evidence_start,
            "runtime_environment": environment,
            "cleared_environment_variable_names": cleared_environment,
            "python": {
                "executable": str(Path(sys.executable).resolve()),
                "version": sys.version,
                "sha256": sha256(Path(sys.executable).resolve()),
            },
            "commands_file": str(out / "commands.json"),
        }
        write_json(out / "run_manifest.json", run_manifest)
        write_json(out / "status.json", {
            "status": "sampling",
            "measurement_valid": False,
            "started_utc": now_utc(),
            "completed_cases": 0,
            "expected_cases": 16,
        })

        case_results: list[dict[str, Any]] = []
        for index, (case, command_record) in enumerate(zip(cases, commands)):
            ensure_frozen_during_sampling(
                head=head,
                runner_sha256=runner_sha256,
                matrix_manifest=args.graph_manifest,
                matrix_manifest_sha256=matrix_manifest_sha256,
                pair=pair,
                pair_snapshot=pair_start,
            )
            case_out = out / "cases" / case["graph"] / case["view_slug"]
            case_out.parent.mkdir(parents=True, exist_ok=True)
            require(not case_out.exists(), f"case output already exists: {case_out}")
            driver_dir = out / "driver_logs"
            driver_dir.mkdir(exist_ok=True)
            driver_log = driver_dir / f"{case['case_id']}.log"
            command_record.update({
                "started_utc": now_utc(),
                "driver_log": str(driver_log),
            })
            write_json(driver_dir / f"{case['case_id']}.command.json", command_record)
            print(
                f"CASE_START {index + 1}/16 id={case['case_id']} "
                f"graph={case['graph']} view={case['view']}",
                flush=True,
            )
            rc, timed_out, elapsed = run_case_subprocess(
                command_record["command"], driver_log, environment, case_timeout
            )
            command_record.update({
                "finished_utc": now_utc(),
                "rc": rc,
                "timed_out": timed_out,
                "elapsed_seconds": elapsed,
                "driver_log_sha256": sha256(driver_log),
            })
            write_json(driver_dir / f"{case['case_id']}.command.json", command_record)
            try:
                result = validate_case_output(
                    case,
                    case_out,
                    command_record["command"],
                    rc,
                    head=head,
                    runner_sha256=runner_sha256,
                    pair_snapshot=pair_start,
                    slurm_job_id=os.environ["SLURM_JOB_ID"],
                    timeout_seconds=args.timeout,
                )
            except Exception as error:
                result = {
                    "case_id": case["case_id"],
                    "graph": case["graph"],
                    "view": case["view"],
                    "view_slug": case["view_slug"],
                    "source": case["source"],
                    "cut": case["cut"],
                    "cut_percent": case["cut_percent"],
                    "vertices": case["vertices"],
                    "graph_path": case["graph_path"],
                    "graph_sha256": case["graph_sha256"],
                    "oracle_path": case["oracle_path"],
                    "oracle_sha256": case["oracle_sha256"],
                    "runner_rc": rc,
                    "measurement_valid": False,
                    "target_met": None,
                    "errors": [
                        "outer case evidence validation raised "
                        f"{type(error).__name__}: {error}"
                    ],
                    "metrics": {},
                    "process_log_evidence": [],
                    "case_output": str(case_out),
                }
            result["driver"] = command_record
            case_results.append(result)
            write_json(out / "case_summaries" / f"{case['case_id']}.json", result)
            write_json(out / "case_index.json", case_results)
            write_json(out / "status.json", {
                "status": "sampling",
                "measurement_valid": False,
                "started_utc": run_manifest["created_utc"],
                "updated_utc": now_utc(),
                "completed_cases": len(case_results),
                "expected_cases": 16,
                "invalid_cases_so_far": sum(
                    result["measurement_valid"] is not True for result in case_results
                ),
            })
            print(
                f"CASE_DONE {index + 1}/16 id={case['case_id']} rc={rc} "
                f"measurement_valid={result['measurement_valid']} "
                f"target_met={result.get('target_met')} "
                f"errors={len(result['errors'])}",
                flush=True,
            )

        postflight = gpu_postflight(out, preflight)
        write_json(out / "gpu_postflight.json", postflight)
        ensure_frozen_during_sampling(
            head=head,
            runner_sha256=runner_sha256,
            matrix_manifest=args.graph_manifest,
            matrix_manifest_sha256=matrix_manifest_sha256,
            pair=pair,
            pair_snapshot=pair_start,
        )
        git_after_sampling = clean_head(head)
        write_json(out / "repository_after_sampling.json", {
            "root": str(ROOT), **git_after_sampling,
        })
        pair_end = pair_integrity_snapshot(pair)
        require(pair_end == pair_start, "pair artifacts changed during the regression")
        write_json(out / "pair_integrity_end.json", pair_end)
        input_evidence_end = validate_input_files(cases)
        require(
            input_evidence_end == input_evidence_start,
            "graph/oracle inputs changed during the regression",
        )
        write_json(out / "input_evidence_end.json", input_evidence_end)

        aggregate = aggregate_cases(case_results)
        aggregate["gpu_postflight"] = postflight
        aggregate["repository_after_sampling"] = git_after_sampling
        aggregate["pair_integrity_end"] = pair_end
        aggregate["input_evidence_end"] = input_evidence_end
        aggregate["completed_utc"] = now_utc()
        write_json(out / "summary.json", aggregate)
        write_summary_csv(out / "summary.csv", aggregate["cases"])
        write_json(out / "complete.json", {
            "measurement_valid": aggregate["measurement_valid"],
            "cases": len(case_results),
            "expected_processes": 16 * PROCESSES_PER_CASE,
            "expected_queries": 16 * PROCESSES_PER_CASE * PROCESS_SAMPLES,
            "expected_formal_queries": 16 * 2 * FORMAL_SAMPLES_PER_CONFIGURATION,
            "validated_processes": (
                16 * PROCESSES_PER_CASE if aggregate["measurement_valid"] else None
            ),
            "validated_queries": (
                16 * PROCESSES_PER_CASE * PROCESS_SAMPLES
                if aggregate["measurement_valid"] else None
            ),
            "validated_formal_queries": (
                16 * 2 * FORMAL_SAMPLES_PER_CONFIGURATION
                if aggregate["measurement_valid"] else None
            ),
            "geomeans": aggregate["geomeans"],
            "all_16_solve_speedup_geomean_T1_over_T2": aggregate[
                "all_16_solve_speedup_geomean_T1_over_T2"
            ],
            "target_met_case_count": aggregate["target_met_case_count"],
            "all_cases_target_met": aggregate["all_cases_target_met"],
            "target_result_affects_exit_status": False,
        })
        status = {
            "status": (
                "measurement_valid" if aggregate["measurement_valid"]
                else "measurement_invalid"
            ),
            "measurement_valid": aggregate["measurement_valid"],
            "target_met_case_count": aggregate["target_met_case_count"],
            "all_cases_target_met": aggregate["all_cases_target_met"],
            "completed_cases": len(case_results),
            "expected_cases": 16,
            "completed_utc": now_utc(),
            "summary": str(out / "summary.json"),
        }
        write_json(out / "status.json", status)
        write_json(out / "artifact_sha256.json", artifact_hashes(out))
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0 if aggregate["measurement_valid"] else 1
    except BaseException as error:
        failure = {
            "status": "driver_failed",
            "measurement_valid": False,
            "failure": repr(error),
            "failure_type": type(error).__name__,
            "failed_utc": now_utc(),
        }
        write_json(out / "status.json", failure)
        write_json(out / "failure.json", failure)
        try:
            write_json(out / "artifact_sha256.json", artifact_hashes(out))
        except Exception:
            pass
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f"EIGHT_GRAPH_REGRESSION_FAILED {error}", file=sys.stderr, flush=True)
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-manifest",
        type=Path,
        default=DEFAULT_MATRIX_MANIFEST,
        help="frozen job-38082 matrix manifest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "new output directory outside the repository; the canonical pair is "
            "built fresh inside it; required unless --dry-run"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=400,
        help="timeout in seconds for each single or dual solver process (default: 400)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate the static manifest/16-case plan only; write no output "
            "and inspect no Slurm/GPU state"
        ),
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.dry_run and args.out is None:
        parser.error("a real run requires --out")
    return args


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.graph_manifest.resolve(strict=True)
        manifest, cases = validate_matrix_manifest_schema(manifest_path)
        args.graph_manifest = manifest_path
        if args.dry_run:
            payload = dry_run_payload(
                manifest_path, manifest, cases, args.out, args.timeout
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        return execute(args, manifest, cases)
    except ContractError as error:
        print(f"EIGHT_GRAPH_CONTRACT_ERROR {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
