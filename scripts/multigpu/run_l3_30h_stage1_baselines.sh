#!/usr/bin/env bash
set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -ne 3 ]]; then
    echo "usage: $0 PAIR_BUILD_DIR ORACLE_DIR NEW_OUTPUT_DIR" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a Slurm allocation" >&2
    exit 2
fi

BUILD=$(realpath "$1")
ORACLES=$(realpath "$2")
OUT=$3
if [[ -e "$OUT" ]]; then
    echo "output already exists: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
STATUS="$OUT/driver_status.tsv"
printf 'case\trc\tgraph\tsource\tdelta\tcut_percent\tblocks\n' > "$STATUS"

run_case() {
    local name=$1
    local graph=$2
    local oracle=$3
    local source=$4
    local delta=$5
    local cut=$6
    local case_out="$OUT/$name"
    python3 "$ROOT/scripts/multigpu/run_l3_30h.py" \
        --sampling exploratory \
        --dual-binary "$BUILD/dual_build/mlmq" \
        --single-binary "$BUILD/single_build/mlmq" \
        --graph "$graph" \
        --oracle "$oracle" \
        --source "$source" \
        --delta "$delta" \
        --cut-percent "$cut" \
        --blocks 107 \
        --warmups 1 \
        --repeats 3 \
        --rounds 1 \
        --timeout 300 \
        --out "$case_out" \
        > "$OUT/$name.driver.log" 2>&1
    local rc=$?
    printf '%s\t%d\t%s\t%d\t%d\t%d\t107\n' \
        "$name" "$rc" "$graph" "$source" "$delta" "$cut" >> "$STATUS"
    echo "STAGE1_CASE name=$name rc=$rc"
}

USA_BASE=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA
run_case usa_gplus \
    "$USA_BASE/augmented.gr" "$USA_BASE/oracle.i32" 11973673 200000 60
run_case delaunay_n23 \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/delaunay_n23.gr \
    "$ORACLES/delaunay_n23_s0.i32" 0 64 50
run_case rgg_n_2_20_s0 \
    /a100-data1/Dingzhong/mlmq/data/rgg_n_2_20_s0.gr \
    "$ORACLES/rgg_n_2_20_s0_s0.i32" 0 64 50
run_case atmosmodm \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/atmosmodm.gr \
    "$ORACLES/atmosmodm_s0.i32" 0 200000 50
run_case rmat22 \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/rmat22.gr \
    "$ORACLES/rmat22_s0.i32" 0 200000 50

if awk -F '\t' 'NR > 1 && $2 != 0 { failed=1 } END { exit failed }' "$STATUS"; then
    echo "L3_30H_STAGE1_BASELINES PASS out=$OUT"
    exit 0
fi
echo "L3_30H_STAGE1_BASELINES COMPLETE_WITH_FAILURES out=$OUT" >&2
exit 1
