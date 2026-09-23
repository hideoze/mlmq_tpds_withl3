#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -ne 1 ]]; then
    echo "usage: $0 NEW_OUTPUT_DIR" >&2
    exit 2
fi

OUT=$1
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)
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

cmd=("$ROOT/SSSP/main.cu" "$ROOT/SSSP/csr_graph.cu" "$ROOT/SSSP/sssp_run.cu"
    -o "$OUT/mlmq"
    -DWORK_COUNT=false
    -DMLMQ_WORKER_THREADS=512
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
    -O3 -m64 -gencode=arch=compute_80,code=sm_80 -rdc=true
    -lcuda -lcudart -w
    "-I$ROOT/core/include" "-I$BOOST_INCLUDE_DIR" -lcusparse)

python3 - "$OUT/command.json" "$NVCC" "${cmd[@]}" <<'PY'
import json,sys
with open(sys.argv[1],"w") as f:
    json.dump([sys.argv[2],*sys.argv[3:]],f,indent=2)
    f.write("\n")
PY
printf 'root=%s\n' "$ROOT" > "$OUT/build.env"
printf 'boost_include_dir=%s\n' "$BOOST_INCLUDE_DIR" >> "$OUT/build.env"
"$NVCC" "${cmd[@]}" > "$OUT/build.log" 2>&1 || {
    rc=$?
    printf '{"rc":%d}\n' "$rc" > "$OUT/status.json"
    cat "$OUT/build.log" >&2
    exit "$rc"
}
printf '{"rc":0}\n' > "$OUT/status.json"
sha256sum "$OUT/mlmq" > "$OUT/mlmq.sha256"
