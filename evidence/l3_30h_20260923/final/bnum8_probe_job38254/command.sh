#!/usr/bin/env bash
set -u

repo=/mnt/709/data3/home/Dingzhong/.codex/worktrees/bbd4/mlmq_tpds_withl3
binary="$repo/tmp/l3_bnum8_probe/mlmq"
graph=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/augmented.gr
oracle=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/oracle.i32
out_dir="$repo/tmp/l3_bnum8_probe/job_${SLURM_JOB_ID:-unknown}"

mkdir -p "$out_dir"
sha256sum "$binary"

export PATH=/usr/local/cuda/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=0,1
export LANG=C
export LC_ALL=C
export MLMQ_BENCH=1
export MLMQ_CUT_PERCENT=60
export MLMQ_FINAL_AUDIT=failure
export MLMQ_WORK_BLOCKS=107
export BENCH_DELTA=400000
export BENCH_SOURCE=11973673
export BENCH_WARMUPS=1
export BENCH_REPEATS=5
export BENCH_QUEUE=L1SLF_L2DQ
export L3_SUPPLEMENT_ORACLE="$oracle"

for process_index in 1 2 3; do
    log="$out_dir/process_${process_index}.log"
    printf 'BNUM8_PROCESS_START index=%s log=%s\n' "$process_index" "$log"
    "$binary" -i "$graph" -n 2 -d 400000 >"$log" 2>&1
    rc=$?
    /usr/bin/grep -E 'L2_CAPACITY|L2_FINAL|WIDE_ORACLE|BENCH algorithm|Error at node|CUDA error|FINAL_AUDIT_ERROR' "$log" || true
    printf 'BNUM8_PROCESS_END index=%s rc=%s\n' "$process_index" "$rc"
    if [[ "$rc" -ne 0 ]]; then
        tail -80 "$log"
        exit "$rc"
    fi
done
