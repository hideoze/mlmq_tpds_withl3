#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -ne 1 ]]; then
    echo "usage: $0 NEW_OUTPUT_DIR" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "run inside a Slurm allocation" >&2
    exit 2
fi

OUT=$1
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
OUT=$(cd "$OUT" && pwd)

date --iso-8601=seconds > "$OUT/start_time.txt"
{
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
    echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
    echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "HOSTNAME=$(hostname)"
    echo "PWD=$PWD"
} > "$OUT/allocation.env"

scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm_job.txt"
nvidia-smi > "$OUT/nvidia_smi.log"
nvidia-smi -L > "$OUT/gpu_list.log"
nvidia-smi topo -m > "$OUT/topology.log"
nvidia-smi --query-gpu=index,uuid,name,memory.total,compute_mode,pstate,temperature.gpu \
    --format=csv,noheader > "$OUT/gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader > "$OUT/apps_before.csv"
if [[ -s "$OUT/apps_before.csv" ]]; then
    echo "GPU already occupied inside allocation" >&2
    exit 3
fi

NVCC=${NVCC:-nvcc}
"$NVCC" --version > "$OUT/nvcc_version.log"
compile=("$NVCC" "$ROOT/scripts/multigpu/l3_30h_preflight.cu" -O2
         -gencode=arch=compute_80,code=sm_80 -o "$OUT/cuda_preflight")
printf '%q ' "${compile[@]}" > "$OUT/compile_command.txt"
printf '\n' >> "$OUT/compile_command.txt"
"${compile[@]}" > "$OUT/compile.log" 2>&1
"$OUT/cuda_preflight" > "$OUT/cuda_preflight.log" 2>&1
grep -q '^CUDA_PREFLIGHT PASS$' "$OUT/cuda_preflight.log"

{
    command -v nvshmrun || true
    command -v nvshmem-info || true
    command -v mpirun || true
    ldconfig -p 2>/dev/null | grep -i nvshmem || true
    find /usr /opt -maxdepth 6 \
        \( -name nvshmem.h -o -name 'libnvshmem*.so*' \) -print 2>/dev/null || true
} > "$OUT/nvshmem_probe.log"

for path in \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/delaunay_n20.gr \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/delaunay_n23.gr \
    /a100-data1/Dingzhong/mlmq/data/rgg_n_2_20_s0.gr \
    /a100-data1/Dingzhong/mlmq/data/nlpkkt80.gr \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/atmosmodm.gr \
    /a100-data1/Dingzhong/mlmq/data/sssp-int/rmat22.gr
do
    realpath "$path"
    stat -c '%s %n' "$path"
done > "$OUT/dataset_paths.log"

python3 - "$OUT/manifest.json" "$ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

out = Path(sys.argv[1])
root = Path(sys.argv[2])
plan = root / "L3_30h_performance_plan.md"
result = {
    "status": "PASS",
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "start_time": (out.parent / "start_time.txt").read_text().strip(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
    "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    "nvshmem_detected": any(
        "nvshmem" in line.lower()
        for line in (out.parent / "nvshmem_probe.log").read_text().splitlines()
    ),
    "gpu_preflight": "PASS",
    "exactly_two_visible_gpus": True,
    "gpu_apps_before": [],
}
out.write_text(json.dumps(result, indent=2) + "\n")
PY

date --iso-8601=seconds > "$OUT/end_time.txt"
sha256sum "$OUT"/* > "$OUT/files.sha256"
echo "L3_30H_PREFLIGHT PASS out=$OUT job=$SLURM_JOB_ID"
