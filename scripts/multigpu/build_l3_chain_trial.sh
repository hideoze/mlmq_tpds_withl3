#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ $# -ne 1 ]]; then
    echo "usage: $0 NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
REQUESTED_OUT=$1
if [[ -e "$REQUESTED_OUT" || -L "$REQUESTED_OUT" ]]; then
    echo "output already exists: $REQUESTED_OUT" >&2
    exit 2
fi
if [[ -n "$(git -C "$ROOT" status --porcelain=v1)" ]]; then
    echo "chain trial build requires a clean committed worktree" >&2
    git -C "$ROOT" status --short >&2
    exit 2
fi

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

NVCC=${NVCC:-nvcc}
BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR:-/a100-data/wyh/boost_1_87_0}
if ! command -v "$NVCC" >/dev/null 2>&1; then
    echo "nvcc not found: $NVCC" >&2
    exit 2
fi
if [[ ! -d "$BOOST_INCLUDE_DIR" ]]; then
    echo "Boost include directory not found: $BOOST_INCLUDE_DIR" >&2
    exit 2
fi
NVCC=$(readlink -f "$(command -v "$NVCC")")

mkdir -p "$(dirname "$REQUESTED_OUT")"
mkdir "$REQUESTED_OUT"
OUT=$(cd -P "$REQUESTED_OUT" && pwd -P)
mkdir "$OUT/tools"

g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    "$ROOT/scripts/multigpu/test_l3_chain_partition.cpp" \
    -o "$OUT/tools/test_l3_chain_partition"
"$OUT/tools/test_l3_chain_partition" > "$OUT/tools/test_l3_chain_partition.log"
g++ -std=c++17 -O2 -Wall -Wextra -pedantic \
    "$ROOT/scripts/multigpu/analyze_l3_chain_partitions.cpp" \
    -o "$OUT/tools/analyze_l3_chain_partitions"

# The existing formal pair supplies A (independent no-L3 single) and D (the
# current best term-only=false dual reference) from the same clean source SHA.
NVCC="$NVCC" BOOST_INCLUDE_DIR="$BOOST_INCLUDE_DIR" \
    "$ROOT/scripts/multigpu/build_l3_30h_pair.sh" "$OUT/reference_pair" \
    > "$OUT/reference_pair_driver.log" 2>&1

COMMON=(
    "$ROOT/SSSP/main.cu"
    "$ROOT/SSSP/csr_graph.cu"
    "$ROOT/SSSP/sssp_run.cu"
    -DWORK_COUNT=false
    -DMLMQ_WORKER_THREADS=512
    -DBNUM=8
    -DBUCKET_MAX=4
    -Dl2_batch_size=8
    -DL3_COOPERATIVE_COLLECT=true
    -DL3_DIRECT_RX=true
    -DL3_RETAIN_TX=true
    -DL3_WINDOW_MODE=2
    -DL3_WINDOW_MIN_CYCLES=25000ull
    -DL3_WINDOW_MAX_CYCLES=25000ull
    -DL3_IDLE_BACKOFF=false
    -DL3_WORKER_RECOVERY=true
    -DL3_TERM_WAIT_ACK=true
    -DL3_ACK_SCAN=true
    -DL3_ACK_WIDE_SCAN=true
    -DL3_BOUNDARY_INDEX=true
    -DL3_L2_FINAL_COUNTS=true
    -DDQ_COUNTER_OVERFLOW_GUARD=true
    -DL3_CHAIN_SHORTCUTS=false
    -DL3_IDLE_TOKEN_PROBE=true
    -DL3_TERM_ONLY_WORKER=true
    -O3 -m64 -gencode=arch=compute_80,code=sm_80 -rdc=true
    -Xptxas -v
    -lcuda -lcudart -w
    "-I$ROOT/core/include" "-I$BOOST_INCLUDE_DIR" -lcusparse
)

build_variant() {
    local name=$1
    shift
    local directory="$OUT/$name"
    mkdir "$directory"
    local command=("$NVCC" "${COMMON[@]}" "$@" -o "$directory/mlmq")
    python3 - "$directory/command.json" "${command[@]}" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2) + "\n")
PY
    tar -czf "$directory/source.tgz" -C "$ROOT" SSSP core
    set +e
    "${command[@]}" > "$directory/build.log" 2>&1
    local rc=$?
    set -e
    printf '{"rc":%d}\n' "$rc" > "$directory/status.json"
    if [[ $rc -ne 0 ]]; then
        cat "$directory/build.log" >&2
        return "$rc"
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --dump-resource-usage "$directory/mlmq" \
            > "$directory/resource_usage.txt" 2>&1 || true
    fi
    python3 - "$directory/version.json" "$ROOT" "$name" "$NVCC" "$BOOST_INCLUDE_DIR" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
path, root_s, name, nvcc, boost = sys.argv[1:]
root = pathlib.Path(root_s)
payload = {
    "schema": 1,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "variant": name,
    "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
    "git_status": subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"], text=True).splitlines(),
    "nvcc": nvcc,
    "nvcc_sha256": hashlib.sha256(pathlib.Path(nvcc).read_bytes()).hexdigest(),
    "boost_include_dir": boost,
    "worker_threads": 512,
    "term_only_worker": True,
    "chain_partition": name != "dual_control",
    "degree_gate": True,
    "diagnostics": name == "dual_chain_diag",
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    sha256sum "$directory/mlmq" "$directory/source.tgz" \
        "$directory/command.json" "$directory/build.log" \
        "$directory/status.json" "$directory/version.json" \
        > "$directory/SHA256SUMS"
}

build_variant dual_control \
    -DL3_CHAIN_PARTITION=false \
    -DL3_CHAIN_PARTITION_DEGREE_GATE=true \
    -DL3_CHAIN_PARTITION_DIAG=false
build_variant dual_chain \
    -DL3_CHAIN_PARTITION=true \
    -DL3_CHAIN_PARTITION_DEGREE_GATE=true \
    -DL3_CHAIN_PARTITION_DIAG=false
build_variant dual_chain_diag \
    -DL3_CHAIN_PARTITION=true \
    -DL3_CHAIN_PARTITION_DEGREE_GATE=true \
    -DL3_CHAIN_PARTITION_DIAG=true

# B and C are an attribution pair: after normalizing the one treatment macro,
# their recorded compiler argv must be byte-for-byte identical.
python3 - "$OUT/dual_control/command.json" "$OUT/dual_chain/command.json" \
    "$OUT/bc_command_contract.json" <<'PY'
import json
import pathlib
import sys
control_path, chain_path, output_path = map(pathlib.Path, sys.argv[1:])
control = json.loads(control_path.read_text())
chain = json.loads(chain_path.read_text())
def normalize(command, treatment):
    normalized = []
    output_next = False
    for value in command:
        if output_next:
            normalized.append("OUTPUT_BINARY")
            output_next = False
        elif value == "-o":
            normalized.append(value)
            output_next = True
        elif value == treatment:
            normalized.append("-DL3_CHAIN_PARTITION=TREATMENT")
        else:
            normalized.append(value)
    return normalized
control_normalized = normalize(control, "-DL3_CHAIN_PARTITION=false")
chain_normalized = normalize(chain, "-DL3_CHAIN_PARTITION=true")
payload = {
    "schema": 1,
    "valid": control_normalized == chain_normalized,
    "only_treatment_macro_differs": control_normalized == chain_normalized,
    "treatment": "L3_CHAIN_PARTITION false in B, true in C",
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if not payload["valid"]:
    raise SystemExit("B/C compiler commands differ beyond L3_CHAIN_PARTITION")
PY

python3 - "$OUT/manifest.json" "$ROOT" "$NVCC" "$BOOST_INCLUDE_DIR" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
path, root_s, nvcc, boost = sys.argv[1:]
root = pathlib.Path(root_s)
out = pathlib.Path(path).parent
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
artifacts = {}
for relative in (
    "reference_pair/single_build/mlmq",
    "reference_pair/dual_build/mlmq",
    "dual_control/mlmq", "dual_chain/mlmq", "dual_chain_diag/mlmq",
    "tools/test_l3_chain_partition", "tools/analyze_l3_chain_partitions",
    "tools/test_l3_chain_partition.log", "bc_command_contract.json",
):
    artifact = out / relative
    artifacts[relative] = {"bytes": artifact.stat().st_size, "sha256": sha(artifact)}
payload = {
    "schema": 1,
    "status": "complete",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "root": str(root),
    "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
    "git_status": subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1"], text=True).splitlines(),
    "nvcc": nvcc,
    "boost_include_dir": boost,
    "artifacts": artifacts,
    "roles": {
        "A_mlmq": "reference_pair/single_build/mlmq",
        "D_best": "reference_pair/dual_build/mlmq",
        "B_control": "dual_control/mlmq",
        "C_chain": "dual_chain/mlmq",
        "C_diagnostic": "dual_chain_diag/mlmq",
    },
}
pathlib.Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
sha256sum "$OUT/manifest.json" > "$OUT/manifest.sha256"
printf 'L3_CHAIN_BUILD_PASS out=%s head=%s\n' "$OUT" "$(git -C "$ROOT" rev-parse HEAD)"
