#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -lt 5 || $# -gt 7 ]]; then
    echo "usage: $0 BASE_PAIR_DIR NEW_OUTPUT_DIR IDLE_BACKOFF MIN_CYCLES MAX_CYCLES [none|work-wait|compact] [MAX_REGISTERS]" >&2
    exit 2
fi

BASE=$(realpath "$1")
OUT_REQUEST=$2
IDLE_BACKOFF=$3
MIN_CYCLES=$4
MAX_CYCLES=$5
VARIANT_MODE=${6:-none}
MAX_REGISTERS=${7:-0}
if [[ "$IDLE_BACKOFF" != true && "$IDLE_BACKOFF" != false ]]; then
    echo "IDLE_BACKOFF must be true or false" >&2
    exit 2
fi
if [[ ! "$MIN_CYCLES" =~ ^[1-9][0-9]*$ || ! "$MAX_CYCLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "window cycles must be positive integers" >&2
    exit 2
fi
if (( MAX_CYCLES < MIN_CYCLES )); then
    echo "MAX_CYCLES must be >= MIN_CYCLES" >&2
    exit 2
fi
if [[ "$VARIANT_MODE" != none && "$VARIANT_MODE" != work-wait && "$VARIANT_MODE" != compact ]]; then
    echo "variant mode must be none, work-wait, or compact" >&2
    exit 2
fi
if [[ ! "$MAX_REGISTERS" =~ ^[0-9]+$ ]]; then
    echo "MAX_REGISTERS must be zero (uncapped) or a positive integer" >&2
    exit 2
fi
WORKER_THREADS=512
if [[ "$VARIANT_MODE" == work-wait ]]; then
    # Full per-warp diagnostics raise register pressure enough that the W512
    # worker cannot launch on A100.  W320 retains one resident block per SM
    # and is diagnostic-only; its timings are never mixed with performance.
    WORKER_THREADS=320
fi
if [[ ! -x "$BASE/single_build/mlmq" ]]; then
    echo "base pair lacks an executable independent single: $BASE" >&2
    exit 2
fi
if [[ -e "$OUT_REQUEST" || -L "$OUT_REQUEST" ]]; then
    echo "output already exists: $OUT_REQUEST" >&2
    exit 2
fi
mkdir -p "$(dirname "$OUT_REQUEST")"
mkdir "$OUT_REQUEST"
OUT=$(cd "$OUT_REQUEST" && pwd)
DUAL="$OUT/dual_build"
mkdir "$DUAL"
cp -a "$BASE/single_build" "$OUT/single_build"

NVCC=${NVCC:-nvcc}
BOOST_INCLUDE_DIR=${BOOST_INCLUDE_DIR:-/a100-data/wyh/boost_1_87_0}
command=(
    "$NVCC"
    "$ROOT/SSSP/main.cu"
    "$ROOT/SSSP/csr_graph.cu"
    "$ROOT/SSSP/sssp_run.cu"
    -o "$DUAL/mlmq"
    -DWORK_COUNT=false
    "-DMLMQ_WORKER_THREADS=$WORKER_THREADS"
    -DL3_COOPERATIVE_COLLECT=true
    -DL3_DIRECT_RX=true
    -DL3_RETAIN_TX=true
    -DL3_WINDOW_MODE=2
    -DL3_WORKER_RECOVERY=true
    -DL3_TERM_WAIT_ACK=true
    -DL3_ACK_SCAN=true
    -DL3_ACK_WIDE_SCAN=true
    -DL3_BOUNDARY_INDEX=true
    -DL3_L2_FINAL_COUNTS=true
    -DL3_CHAIN_SHORTCUTS=false
    -DL3_IDLE_TOKEN_PROBE=true
    "-DL3_IDLE_BACKOFF=$IDLE_BACKOFF"
    "-DL3_WINDOW_MIN_CYCLES=${MIN_CYCLES}ull"
    "-DL3_WINDOW_MAX_CYCLES=${MAX_CYCLES}ull"
    -O3 -m64 -gencode=arch=compute_80,code=sm_80 -rdc=true
    -lcuda -lcudart -w
    "-I$ROOT/core/include" "-I$BOOST_INCLUDE_DIR" -lcusparse
)
if [[ "$VARIANT_MODE" == work-wait ]]; then
    command+=(
        -DL3_WORK_DIAG=true
        -DL3_WAIT_DIAG=true
    )
fi
if [[ "$VARIANT_MODE" == compact ]]; then
    command+=( -DL3_COMPACT_CANDIDATES=true )
fi
if (( MAX_REGISTERS > 0 )); then
    command+=( "--ptxas-options=-maxrregcount=$MAX_REGISTERS" )
fi
python3 - "$DUAL/command.json" "${command[@]}" <<'PY'
import json
import sys
with open(sys.argv[1], "w") as stream:
    json.dump(sys.argv[2:], stream, indent=2)
    stream.write("\n")
PY

python3 - "$OUT/version.json" "$ROOT" "$BASE" "$IDLE_BACKOFF" "$MIN_CYCLES" "$MAX_CYCLES" "$VARIANT_MODE" "$WORKER_THREADS" "$MAX_REGISTERS" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

path, root_s, base_s, idle, minimum, maximum, variant_mode, worker_threads, max_registers = sys.argv[1:]
root = Path(root_s)
single = Path(base_s) / "single_build/mlmq"
payload = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "branch": subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip(),
    "git_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True).splitlines(),
    "base_pair": str(Path(base_s)),
    "single_reused_sha256": hashlib.sha256(single.read_bytes()).hexdigest(),
    "variant": {
        "idle_backoff": idle == "true",
        "window_mode": 2,
        "window_min_cycles": int(minimum),
        "window_max_cycles": int(maximum),
        "variant_mode": variant_mode,
        "diagnostics": "work-wait" if variant_mode == "work-wait" else "none",
        "compact_candidates": variant_mode == "compact",
        "worker_threads": int(worker_threads),
        "max_registers": int(max_registers),
    },
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY
cp "$OUT/version.json" "$DUAL/version.json"
cp "$OUT/version.json" "$OUT/single_build/variant_pair_version.json"
tar -czf "$DUAL/source.tgz" -C "$ROOT" SSSP core

if ! command -v "$NVCC" >/dev/null 2>&1; then
    printf 'nvcc not found: %s\n' "$NVCC" > "$DUAL/build.log"
    rc=127
elif [[ ! -d "$BOOST_INCLUDE_DIR" ]]; then
    printf 'Boost include directory not found: %s\n' "$BOOST_INCLUDE_DIR" > "$DUAL/build.log"
    rc=2
elif "${command[@]}" > "$DUAL/build.log" 2>&1; then
    rc=0
else
    rc=$?
fi
python3 - "$DUAL/status.json" "$rc" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({"rc": int(sys.argv[2])}, indent=2) + "\n")
PY
if [[ $rc -eq 0 ]]; then
    sha256sum "$DUAL/mlmq" > "$DUAL/mlmq.sha256"
fi
sha256sum "$DUAL/command.json" "$DUAL/source.tgz" "$DUAL/version.json" \
    > "$DUAL/artifacts.sha256"
python3 - "$OUT/status.json" "$rc" <<'PY'
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    "rc": int(sys.argv[2]),
    "dual_rc": int(sys.argv[2]),
    "single_reused": True,
}, indent=2) + "\n")
PY
exit "$rc"
