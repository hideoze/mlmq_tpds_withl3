#!/usr/bin/env bash
# Independently confirm the final-SHA primary USA result in a new two-A100
# Slurm allocation.  JOB_A_OUTPUT_DIR is required so the job IDs can be proven
# distinct rather than merely assumed to be distinct.

set -Eeuo pipefail
umask 077

readonly PYTHON=/usr/bin/python3
readonly NVIDIA_SMI=/usr/bin/nvidia-smi

usage() {
    printf 'usage: %s [--dry-run] JOB_A_OUTPUT_DIR NEW_OUTPUT_DIR\n' "$0" >&2
}

die() {
    printf 'JOB_B_ERROR %s\n' "$*" >&2
    exit 2
}

utc_now() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

print_command() {
    local item
    printf 'cd -- %q\n' "$repo_root"
    printf 'exec'
    for item in "$@"; do printf ' %q' "$item"; done
    printf '\n'
}

assert_clean_head() {
    local current_head current_status
    current_head=$(git -C "$repo_root" rev-parse HEAD)
    current_status=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
    [[ "$current_head" == "$frozen_head" ]] ||
        die "repository HEAD changed: $current_head != $frozen_head"
    [[ -z "$current_status" ]] || die "repository is not clean:\n$current_status"
}

preflight_allocation() {
    local value key count found=0 row name capability gpu_query apps
    local -a items=() rows=()
    [[ -n "${SLURM_JOB_ID:-}" ]] || die 'SLURM_JOB_ID is not set'
    [[ "${SLURM_JOB_PARTITION:-}" == a100 ]] ||
        die "SLURM_JOB_PARTITION must be a100, got ${SLURM_JOB_PARTITION:-<unset>}"
    for key in SLURM_STEP_GPUS SLURM_JOB_GPUS; do
        value=${!key:-}
        if [[ -n "$value" ]]; then
            IFS=',' read -r -a items <<< "$value"
            (( ${#items[@]} == 2 )) || die "$key must name exactly two GPUs: $value"
            found=1
        fi
    done
    value=${SLURM_GPUS_ON_NODE:-}
    if [[ "$value" =~ ^2$ || "$value" =~ :2(\(|$) ]]; then found=1; fi
    (( found == 1 )) || die 'cannot prove an exactly-two-GPU Slurm allocation'
    if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
        IFS=',' read -r -a items <<< "$CUDA_VISIBLE_DEVICES"
        (( ${#items[@]} == 2 )) ||
            die "CUDA_VISIBLE_DEVICES must expose exactly two GPUs: $CUDA_VISIBLE_DEVICES"
    fi
    [[ -x "$NVIDIA_SMI" && ! -L "$NVIDIA_SMI" ]] || die 'canonical nvidia-smi unavailable'
    gpu_query=$($NVIDIA_SMI --query-gpu=index,uuid,name,compute_cap \
        --format=csv,noheader,nounits)
    mapfile -t rows <<< "$gpu_query"
    (( ${#rows[@]} == 2 )) || die "nvidia-smi exposes ${#rows[@]} GPUs, expected 2"
    for row in "${rows[@]}"; do
        IFS=',' read -r _ _ name capability <<< "$row"
        [[ "$name" == *A100* && "${capability//[[:space:]]/}" == 8.0 ]] ||
            die "formal GPU must be A100 CC 8.0, got: $row"
    done
    apps=$($NVIDIA_SMI --query-compute-apps=pid --format=csv,noheader)
    [[ -z "${apps//[[:space:]]/}" ]] || die "GPU compute applications already present: $apps"
    allocation_gpu_query=$gpu_query
}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
(( $# == 2 )) || { usage; exit 2; }

[[ ! -L "${BASH_SOURCE[0]}" ]] || die 'the job script must not be a symlink'
script_path=$(realpath -e -- "${BASH_SOURCE[0]}")
script_dir=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(realpath -e -- "$(git -C "$script_dir" rev-parse --show-toplevel)")
[[ "$script_dir" == "$repo_root/scripts/multigpu" ]] || die 'unexpected script location'

job_a_output=$(realpath -e -- "$1")
[[ -d "$job_a_output" ]] || die "Job A output is not a directory: $job_a_output"
requested_output=$2
[[ "$requested_output" == /* ]] || die 'NEW_OUTPUT_DIR must be an absolute path'
output_root=$(realpath -m -- "$requested_output")
case "$output_root" in
    "$repo_root"|"$repo_root"/*) die "formal output must be outside repository: $output_root" ;;
esac
[[ ! -e "$output_root" && ! -L "$output_root" ]] ||
    die "output path already exists: $output_root"

runner="$repo_root/scripts/multigpu/run_l3_30h.py"
formal_contract="$repo_root/evidence/l3_30h_20260923/final/formal_input_contract.json"
usa_graph=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/augmented.gr
usa_oracle=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/oracle.i32
for path in "$runner" "$formal_contract" "$usa_graph" "$usa_oracle"; do
    [[ ! -L "$path" && -f "$path" ]] || die "missing/non-regular input: $path"
done

primary_out="$output_root/01_primary_confirmation"
primary_command=(
    "$PYTHON" -s "$runner"
    --sampling formal
    --input-contract "$formal_contract"
    --graph "$usa_graph"
    --oracle "$usa_oracle"
    --source 11973673
    --delta 400000
    --cut-percent 60
    --blocks 107
    --warmups 1
    --repeats 5
    --rounds 2
    --timeout 400
    --out "$primary_out"
    --queue L1SLF_L2DQ
    --final-audit failure
    --expected-window-mode 2
    --expected-window-min 25000
    --expected-window-max 25000
    --expected-idle-backoff 0
)

if (( dry_run == 1 )); then
    printf 'JOB_B_DRY_RUN job_a=%s output=%s\n' "$job_a_output" "$output_root"
    print_command "${primary_command[@]}"
    printf 'JOB_B_DRY_RUN_OK no_output_created=1 gpu_checked=0\n'
    exit 0
fi

job_a_status="$job_a_output/orchestration.status"
[[ -f "$job_a_status" && ! -L "$job_a_status" ]] ||
    die "Job A status is missing/non-regular: $job_a_status"
grep -qx 'workflow=job_a' "$job_a_status" || die 'referenced output is not Job A'
grep -qx 'status=COMPLETE' "$job_a_status" || die 'Job A is not complete'
job_a_id=$(awk -F= '$1 == "slurm_job_id" { print substr($0, index($0, "=") + 1) }' "$job_a_status")
[[ -n "$job_a_id" ]] || die 'Job A status lacks its Slurm job ID'

frozen_head=$(git -C "$repo_root" rev-parse HEAD)
[[ "$frozen_head" =~ ^[0-9a-f]{40}$ ]] || die "invalid HEAD: $frozen_head"
job_a_head=$(awk -F= '$1 == "head" { print substr($0, index($0, "=") + 1) }' "$job_a_status")
[[ "$job_a_head" == "$frozen_head" ]] ||
    die "Job B HEAD differs from Job A: $frozen_head != $job_a_head"
assert_clean_head
preflight_allocation
[[ "$SLURM_JOB_ID" != "$job_a_id" ]] ||
    die "Job B must use a fresh Slurm allocation, but both job IDs are $job_a_id"

mkdir -p -- "$(dirname -- "$output_root")"
canonical_parent=$(realpath -e -- "$(dirname -- "$output_root")")
output_root="$canonical_parent/$(basename -- "$output_root")"
[[ ! -e "$output_root" && ! -L "$output_root" ]] || die 'output appeared during preflight'
mkdir -- "$output_root"
mkdir -- "$output_root/driver_logs" "$output_root/metadata"
output_created=1
status_file="$output_root/orchestration.status"

on_exit() {
    local rc=$?
    trap - EXIT
    if (( rc != 0 )) && [[ -n ${output_created:-} ]]; then
        printf 'workflow=job_b\nstatus=FAILED\nrc=%s\nhead=%s\nslurm_job_id=%s\njob_a_slurm_job_id=%s\nupdated_utc=%s\n' \
            "$rc" "$frozen_head" "$SLURM_JOB_ID" "$job_a_id" "$(utc_now)" > "$status_file"
    fi
    exit "$rc"
}
trap on_exit EXIT

printf '%s\n' "$frozen_head" > "$output_root/metadata/head.txt"
printf '%s\n' "$job_a_output" > "$output_root/metadata/job-a-output.txt"
printf '%s\n' "$allocation_gpu_query" > "$output_root/metadata/gpu-query.csv"
env | LC_ALL=C sort | awk -F= \
    '$1 ~ /^SLURM_/ || $1 == "CUDA_VISIBLE_DEVICES" { print }' \
    > "$output_root/metadata/slurm.env"
print_command "${primary_command[@]}" > "$output_root/driver_logs/01_primary_confirmation.command.sh"
printf 'workflow=job_b\nstatus=RUNNING\nrc=0\nhead=%s\nslurm_job_id=%s\njob_a_slurm_job_id=%s\nupdated_utc=%s\n' \
    "$frozen_head" "$SLURM_JOB_ID" "$job_a_id" "$(utc_now)" > "$status_file"

started=$(utc_now)
set +e
(cd -- "$repo_root" && PYTHONDONTWRITEBYTECODE=1 "${primary_command[@]}") \
    > "$output_root/driver_logs/01_primary_confirmation.stdout.log" \
    2> "$output_root/driver_logs/01_primary_confirmation.stderr.log"
rc=$?
set -e
printf 'step=01_primary_confirmation\nstarted_utc=%s\nfinished_utc=%s\nrc=%s\n' \
    "$started" "$(utc_now)" "$rc" \
    > "$output_root/driver_logs/01_primary_confirmation.result.env"
(( rc == 0 )) || exit "$rc"
assert_clean_head

printf 'workflow=job_b\nstatus=COMPLETE\nrc=0\nhead=%s\nslurm_job_id=%s\njob_a_slurm_job_id=%s\nupdated_utc=%s\n' \
    "$frozen_head" "$SLURM_JOB_ID" "$job_a_id" "$(utc_now)" > "$status_file"
printf 'JOB_B_COMPLETE output=%s head=%s slurm_job_id=%s job_a_slurm_job_id=%s\n' \
    "$output_root" "$frozen_head" "$SLURM_JOB_ID" "$job_a_id"
