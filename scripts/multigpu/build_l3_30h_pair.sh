#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ $# -ne 1 ]]; then
    echo "usage: $0 NEW_OUTPUT_DIR" >&2
    exit 2
fi

REQUESTED_OUT=$1
if [[ -e "$REQUESTED_OUT" || -L "$REQUESTED_OUT" ]]; then
    echo "output path already exists: $REQUESTED_OUT" >&2
    exit 2
fi
mkdir -p "$(dirname "$REQUESTED_OUT")"
mkdir "$REQUESTED_OUT"
OUT=$(cd -P "$REQUESTED_OUT" && pwd -P)

NVCC=${NVCC:-nvcc}
BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR:-/a100-data/wyh/boost_1_87_0}
export NVCC BOOST_INCLUDE_DIR

BLOCKED_BUILD_ENV=(
    NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CPATH C_INCLUDE_PATH
    CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH LIBRARY_PATH LD_PRELOAD
    LD_LIBRARY_PATH CUDAFLAGS CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBS
    COMPILER_PATH GCC_EXEC_PREFIX CUDA_HOME CUDA_PATH CUDAHOSTCXX NVCC_CCBIN
)
for variable in "${BLOCKED_BUILD_ENV[@]}"; do
    if [[ -v "$variable" ]]; then
        echo "compiler-influencing environment variable is forbidden: $variable" >&2
        exit 2
    fi
done

DUAL_BUILDER="$ROOT/scripts/multigpu/build_l3_paper_config.sh"
SINGLE_REL="evidence/l3_supplement_20260921_v2/single_source"
SINGLE_BASE="$ROOT/$SINGLE_REL"
SINGLE_SOURCE="$OUT/single_source"
DUAL_BUILD="$OUT/dual_build"
SINGLE_BUILD="$OUT/single_build"
PAIR_COMPLETE=false

write_pair_status() {
    local rc=$1
    local state=$2
    local dual_rc=${3:-}
    local single_rc=${4:-}
    python3 - "$OUT/status.json" "$rc" "$state" "$dual_rc" "$single_rc" <<'PY'
import datetime
import json
import sys

path, rc, state, dual_rc, single_rc = sys.argv[1:]
payload = {
    "rc": int(rc),
    "state": state,
    "dual_rc": None if dual_rc == "" else int(dual_rc),
    "single_rc": None if single_rc == "" else int(single_rc),
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

write_build_status() {
    local path=$1
    local rc=$2
    local role=$3
    local builder=$4
    python3 - "$path" "$rc" "$role" "$builder" <<'PY'
import datetime
import json
import sys

path, rc, role, builder = sys.argv[1:]
payload = {
    "rc": int(rc),
    "role": role,
    "builder": builder,
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

write_hashes() {
    local build_dir=$1
    python3 - "$build_dir" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = (
    "mlmq",
    "source.tgz",
    "command.json",
    "build.log",
    "status.json",
    "version.json",
    "provenance.json",
)
hashes = {}
for name in names:
    path = root / name
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    else:
        hashes[name] = None
with (root / "hashes.json").open("w", encoding="utf-8") as handle:
    json.dump(hashes, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

on_exit() {
    local rc=$?
    if [[ "$PAIR_COMPLETE" != true ]]; then
        write_pair_status "$rc" unexpected_failure "" "" 2>/dev/null || true
    fi
}
trap on_exit EXIT
write_pair_status 1 preparing "" ""

if [[ ! -x "$DUAL_BUILDER" ]]; then
    echo "dual builder is missing or not executable: $DUAL_BUILDER" >&2
    exit 2
fi
if [[ ! -d "$SINGLE_BASE/SSSP" || ! -d "$SINGLE_BASE/core" ]]; then
    echo "committed no-L3 single source is incomplete: $SINGLE_BASE" >&2
    exit 2
fi
if ! git -C "$ROOT" cat-file -e "HEAD:$SINGLE_REL/SSSP/main.cu" || \
    ! git -C "$ROOT" cat-file -e "HEAD:$SINGLE_REL/core/include/common.h"; then
    echo "the no-L3 single source is absent from HEAD" >&2
    exit 2
fi

git -C "$ROOT" status --porcelain=v1 > "$OUT/git_status.txt"
if command -v "$NVCC" >/dev/null 2>&1; then
    "$NVCC" --version > "$OUT/nvcc_version.txt" 2>&1 || true
else
    printf 'nvcc not found: %s\n' "$NVCC" > "$OUT/nvcc_version.txt"
fi

python3 - "$OUT/version.json" "$ROOT" "$NVCC" "$BOOST_INCLUDE_DIR" \
    "$DUAL_BUILDER" "$ROOT/scripts/multigpu/build_l3_30h_pair.sh" \
    "$ROOT/scripts/multigpu/prepare_l3_30h_single.py" <<'PY'
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

path, root_s, nvcc, boost, dual_builder_s, pair_builder_s, adapter_s = sys.argv[1:]
root = pathlib.Path(root_s)
dual_builder = pathlib.Path(dual_builder_s)
pair_builder = pathlib.Path(pair_builder_s)
adapter = pathlib.Path(adapter_s)
nvcc_resolved = shutil.which(nvcc) if "/" not in nvcc else str(pathlib.Path(nvcc).resolve())
if not nvcc_resolved:
    nvcc_resolved = nvcc
nvcc_path = pathlib.Path(nvcc_resolved)

def git(*args, allow_failure=False):
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode and not allow_failure:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout.strip() if proc.returncode == 0 else None

payload = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository_root": str(root),
    "head": git("rev-parse", "HEAD"),
    "branch": git("branch", "--show-current"),
    "describe": git("describe", "--always", "--dirty", "--tags", allow_failure=True),
    "git_status_porcelain": (pathlib.Path(path).parent / "git_status.txt").read_text(encoding="utf-8").splitlines(),
    "nvcc": nvcc,
    "nvcc_resolved": str(nvcc_path),
    "nvcc_sha256": hashlib.sha256(nvcc_path.read_bytes()).hexdigest()
                   if nvcc_path.is_file() else None,
    "boost_include_dir": boost,
    "dual_builder": str(dual_builder.relative_to(root)),
    "dual_builder_sha256": hashlib.sha256(dual_builder.read_bytes()).hexdigest(),
    "pair_builder": str(pair_builder.relative_to(root)),
    "pair_builder_sha256": hashlib.sha256(pair_builder.read_bytes()).hexdigest(),
    "single_adapter": str(adapter.relative_to(root)),
    "single_adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    "build_environment": {
        key: os.environ.get(key) for key in (
            "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR",
            "NVCC", "BOOST_INCLUDE_DIR")
        if os.environ.get(key) is not None
    },
    "blocked_build_environment_variables": [
        "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH", "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH", "LD_PRELOAD",
        "LD_LIBRARY_PATH", "CUDAFLAGS", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
        "LDFLAGS", "LIBS", "COMPILER_PATH", "GCC_EXEC_PREFIX", "CUDA_HOME",
        "CUDA_PATH", "CUDAHOSTCXX", "NVCC_CCBIN",
    ],
    "blocked_build_environment_present": [],
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# Start from the committed independent no-L3 implementation.  Then replace its
# core tree with the current shared core so any queue/core repair is present in
# both binaries.  The post-copy diff is an enforced byte-for-byte check.
mkdir "$SINGLE_SOURCE"
git -C "$ROOT" archive --format=tar HEAD "$SINGLE_REL" | \
    tar -xf - -C "$SINGLE_SOURCE" --strip-components=3
core_before=different
if diff -qr "$ROOT/core" "$SINGLE_SOURCE/core" > "$OUT/core_snapshot_diff.log"; then
    core_before=identical
    printf 'IDENTICAL: committed single snapshot core equals current core\n' \
        > "$OUT/core_snapshot_diff.log"
else
    diff_rc=$?
    if [[ $diff_rc -gt 1 ]]; then
        echo "failed to compare current core with the committed single snapshot" >&2
        exit "$diff_rc"
    fi
fi
rm -rf "$SINGLE_SOURCE/core"
cp -a "$ROOT/core" "$SINGLE_SOURCE/core"
if diff -qr "$ROOT/core" "$SINGLE_SOURCE/core" > "$OUT/core_post_sync_diff.log"; then
    printf 'IDENTICAL: single staging core equals current core after sync\n' \
        > "$OUT/core_post_sync_diff.log"
else
    diff_rc=$?
    echo "single staging core does not match current core after sync" >&2
    exit "$diff_rc"
fi
if [[ -d "$SINGLE_SOURCE/SSSP/l3" ]]; then
    echo "no-L3 single snapshot unexpectedly contains SSSP/l3" >&2
    exit 2
fi
python3 "$ROOT/scripts/multigpu/prepare_l3_30h_single.py" "$SINGLE_SOURCE" \
    > "$OUT/single_adapter.log"

python3 - "$OUT/provenance.json" "$ROOT" "$OUT" "$SINGLE_BASE" "$core_before" <<'PY'
import json
import pathlib
import sys

path, root_s, out_s, single_base_s, core_before = sys.argv[1:]
root = pathlib.Path(root_s)
out = pathlib.Path(out_s)
single_base = pathlib.Path(single_base_s)
payload = {
    "pair_contract": {
        "dual": "current dual-GPU L3 paper configuration",
        "single": "independent committed no-L3 single-GPU source",
        "single_is_dual_n1": False,
    },
    "dual": {
        "source": ["SSSP", "core"],
        "source_root": str(root),
        "builder": "scripts/multigpu/build_l3_paper_config.sh",
        "macro_authority": "builder command.json",
    },
    "single": {
        "base_snapshot": str(single_base.relative_to(root)),
        "snapshot_materialization": "git archive HEAD (ignores working-tree edits)",
        "staging": str((out / "single_source").relative_to(out)),
        "snapshot_core_matches_current_before_sync": core_before == "identical",
        "core_sync_action": "replaced staging core with current repository core",
        "post_sync_check": "byte-for-byte recursive diff passed",
        "performance_adapter": "scripts/multigpu/prepare_l3_30h_single.py",
        "performance_adapter_effect": "explicit validated MLMQ_WORK_BLOCKS and NO_L3_LAUNCH log",
        "l3_compile_defines": [],
    },
    "evidence": {
        "pre_sync_core_diff": "core_snapshot_diff.log",
        "post_sync_core_diff": "core_post_sync_diff.log",
        "git_status": "git_status.txt",
        "nvcc_version": "nvcc_version.txt",
        "single_adapter_log": "single_adapter.log",
    },
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

# Archive exactly the source trees each build consumes before invoking nvcc.
tar -czf "$OUT/.dual_source.tgz" -C "$ROOT" SSSP core
mkdir "$SINGLE_BUILD"
tar -czf "$SINGLE_BUILD/source.tgz" -C "$SINGLE_SOURCE" SSSP core

# The dual build is delegated to the canonical builder so its complete paper
# macro vector remains the single source of truth.
if "$DUAL_BUILDER" "$DUAL_BUILD" > "$OUT/.dual_driver.log" 2>&1; then
    dual_rc=0
else
    dual_rc=$?
fi
if [[ ! -d "$DUAL_BUILD" ]]; then
    mkdir "$DUAL_BUILD"
fi
mv "$OUT/.dual_source.tgz" "$DUAL_BUILD/source.tgz"
mv "$OUT/.dual_driver.log" "$DUAL_BUILD/driver.log"
if [[ ! -f "$DUAL_BUILD/build.log" ]]; then
    cp "$DUAL_BUILD/driver.log" "$DUAL_BUILD/build.log"
fi
if [[ ! -f "$DUAL_BUILD/command.json" ]]; then
    python3 - "$DUAL_BUILD/command.json" "$DUAL_BUILDER" "$DUAL_BUILD" <<'PY'
import json
import sys
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump([sys.argv[2], sys.argv[3]], handle, indent=2)
    handle.write("\n")
PY
fi
write_build_status "$DUAL_BUILD/status.json" "$dual_rc" dual_l3_paper \
    scripts/multigpu/build_l3_paper_config.sh
cp "$OUT/version.json" "$DUAL_BUILD/version.json"
cp "$OUT/provenance.json" "$DUAL_BUILD/provenance.json"
if [[ -f "$DUAL_BUILD/mlmq" ]]; then
    sha256sum "$DUAL_BUILD/mlmq" > "$DUAL_BUILD/mlmq.sha256"
fi
write_hashes "$DUAL_BUILD"

# Build the separate no-L3 source directly; there is deliberately no -n 1 and
# no L3 compile definition in this command.
single_cmd=(
    "$SINGLE_SOURCE/SSSP/main.cu"
    "$SINGLE_SOURCE/SSSP/csr_graph.cu"
    "$SINGLE_SOURCE/SSSP/sssp_run.cu"
    -o "$SINGLE_BUILD/mlmq"
    -DWORK_COUNT=false
    -DMLMQ_WORKER_THREADS=512
    -DDQ_COUNTER_OVERFLOW_GUARD=true
    -O3 -m64 -gencode=arch=compute_80,code=sm_80 -rdc=true
    -lcuda -lcudart -w
    "-I$SINGLE_SOURCE/core/include"
    "-I$BOOST_INCLUDE_DIR"
    -lcusparse
)
python3 - "$SINGLE_BUILD/command.json" "$NVCC" "${single_cmd[@]}" <<'PY'
import json
import sys

command = [sys.argv[2], *sys.argv[3:]]
if any(arg.startswith("-DL3_") for arg in command):
    raise SystemExit("independent single command contains an L3 define")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(command, handle, indent=2)
    handle.write("\n")
PY

if ! command -v "$NVCC" >/dev/null 2>&1; then
    printf 'nvcc not found: %s\n' "$NVCC" > "$SINGLE_BUILD/build.log"
    single_rc=127
elif [[ ! -d "$BOOST_INCLUDE_DIR" ]]; then
    printf 'Boost include directory not found: %s\n' "$BOOST_INCLUDE_DIR" \
        > "$SINGLE_BUILD/build.log"
    single_rc=2
elif "$NVCC" "${single_cmd[@]}" > "$SINGLE_BUILD/build.log" 2>&1; then
    single_rc=0
else
    single_rc=$?
fi
write_build_status "$SINGLE_BUILD/status.json" "$single_rc" single_no_l3 \
    independent_nvcc
cp "$OUT/version.json" "$SINGLE_BUILD/version.json"
cp "$OUT/provenance.json" "$SINGLE_BUILD/provenance.json"
if [[ -f "$SINGLE_BUILD/mlmq" ]]; then
    sha256sum "$SINGLE_BUILD/mlmq" > "$SINGLE_BUILD/mlmq.sha256"
fi
write_hashes "$SINGLE_BUILD"

# Prove that the active dual source did not change between the pre-build
# archive and the end of compilation.  Formal consumers also require the
# before/after Git snapshots to be the same clean HEAD.
SOURCE_CHECK=$(mktemp -d "$OUT/.dual_source_check.XXXXXX")
tar -xzf "$DUAL_BUILD/source.tgz" -C "$SOURCE_CHECK"
dual_source_post_build_match=false
if diff -qr "$ROOT/SSSP" "$SOURCE_CHECK/SSSP" \
        > "$OUT/dual_source_post_build_diff.log" && \
   diff -qr "$ROOT/core" "$SOURCE_CHECK/core" \
        >> "$OUT/dual_source_post_build_diff.log"; then
    dual_source_post_build_match=true
    printf 'IDENTICAL: archived dual source equals repository after build\n' \
        > "$OUT/dual_source_post_build_diff.log"
fi
rm -rf "$SOURCE_CHECK"

python3 - "$OUT/version.json" "$ROOT" "$dual_source_post_build_match" <<'PY'
import json
import pathlib
import subprocess
import sys

path, root_s, source_match_s = sys.argv[1:]
root = pathlib.Path(root_s)

def git(*args):
    proc = subprocess.run(
        ["git", "-C", str(root), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout.strip()

with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
payload["head_after_build"] = git("rev-parse", "HEAD")
payload["git_status_porcelain_after_build"] = git(
    "status", "--porcelain=v1").splitlines()
payload["dual_source_post_build_match"] = source_match_s == "true"
payload["dual_source_post_build_diff"] = "dual_source_post_build_diff.log"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
cp "$OUT/version.json" "$DUAL_BUILD/version.json"
cp "$OUT/version.json" "$SINGLE_BUILD/version.json"
write_hashes "$DUAL_BUILD"
write_hashes "$SINGLE_BUILD"

overall_rc=0
if [[ $dual_rc -ne 0 ]]; then
    overall_rc=$dual_rc
elif [[ $single_rc -ne 0 ]]; then
    overall_rc=$single_rc
elif [[ "$dual_source_post_build_match" != true ]]; then
    overall_rc=3
fi
if [[ $overall_rc -eq 0 ]]; then
    pair_state=complete
else
    pair_state=build_failed
fi
write_pair_status "$overall_rc" "$pair_state" "$dual_rc" "$single_rc"
PAIR_COMPLETE=true
if [[ $overall_rc -ne 0 ]]; then
    echo "paired build failed: dual_rc=$dual_rc single_rc=$single_rc; evidence kept in $OUT" >&2
fi
exit "$overall_rc"
