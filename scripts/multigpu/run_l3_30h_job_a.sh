#!/usr/bin/env bash
# Run the complete final-SHA Job A evidence workflow inside one two-A100 Slurm
# allocation.  GPU-bearing children are deliberately executed one at a time.

set -Eeuo pipefail
umask 077

readonly PYTHON=/usr/bin/python3
readonly NVIDIA_SMI=/usr/bin/nvidia-smi
readonly SHA256SUM=/usr/bin/sha256sum

usage() {
    printf 'usage: %s [--dry-run] NEW_OUTPUT_DIR\n' "$0" >&2
}

die() {
    printf 'JOB_A_ERROR %s\n' "$*" >&2
    exit 2
}

utc_now() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

trim() {
    local value=$1
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

require_regular_file() {
    local path=$1
    local description=$2
    [[ ! -L "$path" && -f "$path" ]] ||
        die "$description is missing, a symlink, or not a regular file: $path"
}

require_sha256() {
    local expected=$1
    local path=$2
    local actual
    actual=$($SHA256SUM -- "$path")
    actual=${actual%% *}
    [[ "$actual" == "$expected" ]] ||
        die "SHA256 mismatch for $path: got $actual expected $expected"
}

assert_clean_head() {
    local current_head current_status
    current_head=$(git -C "$repo_root" rev-parse HEAD)
    current_status=$(git -C "$repo_root" status --porcelain=v1 --untracked-files=all)
    [[ "$current_head" == "$frozen_head" ]] ||
        die "repository HEAD changed: $current_head != $frozen_head"
    [[ -z "$current_status" ]] ||
        die "repository is not clean:\n$current_status"
}

print_command() {
    local item
    printf 'cd -- %q\n' "$repo_root"
    printf 'exec'
    for item in "$@"; do
        printf ' %q' "$item"
    done
    printf '\n'
}

write_status() {
    local state=$1
    local step=$2
    local rc=$3
    local temporary="$output_root/.orchestration.status.tmp"
    {
        printf 'workflow=job_a\n'
        printf 'status=%s\n' "$state"
        printf 'current_step=%s\n' "$step"
        printf 'rc=%s\n' "$rc"
        printf 'head=%s\n' "$frozen_head"
        printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
        printf 'updated_utc=%s\n' "$(utc_now)"
    } > "$temporary"
    mv -- "$temporary" "$output_root/orchestration.status"
}

run_step() {
    local step=$1
    shift
    local command_file="$output_root/driver_logs/$step.command.sh"
    local stdout_file="$output_root/driver_logs/$step.stdout.log"
    local stderr_file="$output_root/driver_logs/$step.stderr.log"
    local result_file="$output_root/driver_logs/$step.result.env"
    local started finished rc

    assert_clean_head
    current_step=$step
    write_status RUNNING "$step" 0
    print_command "$@" > "$command_file"
    started=$(utc_now)
    printf 'JOB_A_STEP_START step=%s utc=%s\n' "$step" "$started"
    if (cd -- "$repo_root" && "$@") > "$stdout_file" 2> "$stderr_file"; then
        rc=0
    else
        rc=$?
    fi
    finished=$(utc_now)
    {
        printf 'step=%s\n' "$step"
        printf 'started_utc=%s\n' "$started"
        printf 'finished_utc=%s\n' "$finished"
        printf 'rc=%s\n' "$rc"
        printf 'stdout=%s\n' "$stdout_file"
        printf 'stderr=%s\n' "$stderr_file"
        printf 'command=%s\n' "$command_file"
    } > "$result_file"
    printf 'JOB_A_STEP_END step=%s rc=%s utc=%s\n' "$step" "$rc" "$finished"
    (( rc == 0 )) || return "$rc"
    assert_clean_head
}

preflight_allocation() {
    local key value count count_evidence=0
    local gpu_query apps row index uuid name capability
    local -a rows=()
    local -a uuids=()

    [[ -n "${SLURM_JOB_ID:-}" ]] || die 'SLURM_JOB_ID is not set'
    [[ "${SLURM_JOB_PARTITION:-}" == a100 ]] ||
        die "SLURM_JOB_PARTITION must be a100, got ${SLURM_JOB_PARTITION:-<unset>}"

    for key in SLURM_STEP_GPUS SLURM_JOB_GPUS; do
        value=${!key:-}
        if [[ -n "$value" ]]; then
            IFS=',' read -r -a allocation_items <<< "$value"
            count=${#allocation_items[@]}
            (( count == 2 )) || die "$key exposes $count GPUs, expected 2: $value"
            count_evidence=1
        fi
    done
    value=${SLURM_GPUS_ON_NODE:-}
    if [[ -n "$value" ]]; then
        if [[ "$value" =~ ^[0-9]+$ ]]; then
            count=$value
        elif [[ "$value" =~ :([0-9]+)(\(|$) ]]; then
            count=${BASH_REMATCH[1]}
        else
            count=''
        fi
        if [[ -n "$count" ]]; then
            (( count == 2 )) ||
                die "SLURM_GPUS_ON_NODE reports $count GPUs, expected 2: $value"
            count_evidence=1
        fi
    fi
    (( count_evidence == 1 )) ||
        die 'cannot prove an exactly-two-GPU Slurm allocation'

    value=${CUDA_VISIBLE_DEVICES:-}
    if [[ -n "$value" ]]; then
        IFS=',' read -r -a allocation_items <<< "$value"
        (( ${#allocation_items[@]} == 2 )) ||
            die "CUDA_VISIBLE_DEVICES must expose two GPUs: $value"
    fi

    [[ -x "$NVIDIA_SMI" && ! -L "$NVIDIA_SMI" ]] ||
        die "canonical nvidia-smi is unavailable: $NVIDIA_SMI"
    gpu_query=$($NVIDIA_SMI \
        --query-gpu=index,uuid,name,compute_cap \
        --format=csv,noheader,nounits)
    mapfile -t rows <<< "$gpu_query"
    (( ${#rows[@]} == 2 )) ||
        die "nvidia-smi exposes ${#rows[@]} GPUs, expected 2"
    for row in "${rows[@]}"; do
        IFS=',' read -r index uuid name capability <<< "$row"
        index=$(trim "$index")
        uuid=$(trim "$uuid")
        name=$(trim "$name")
        capability=$(trim "$capability")
        [[ "$index" =~ ^[0-9]+$ ]] || die "invalid GPU index in: $row"
        [[ "$uuid" == GPU-* ]] || die "invalid GPU UUID in: $row"
        [[ "$name" == *A100* && "$capability" == 8.0 ]] ||
            die "formal GPU must be A100 CC 8.0, got $name CC $capability"
        uuids+=("$uuid")
    done
    [[ "${uuids[0]}" != "${uuids[1]}" ]] || die 'visible GPU UUIDs are not unique'
    apps=$($NVIDIA_SMI --query-compute-apps=pid --format=csv,noheader)
    [[ -z "${apps//[[:space:]]/}" ]] ||
        die "GPU compute applications are already present: $apps"
    allocation_gpu_query=$gpu_query
}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
    shift
fi
(( $# == 1 )) || {
    usage
    exit 2
}

[[ ! -L "${BASH_SOURCE[0]}" ]] || die 'the job script must not be a symlink'
script_path=$(realpath -e -- "${BASH_SOURCE[0]}")
script_dir=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
repo_root=$(realpath -e -- "$repo_root")
[[ "$script_dir" == "$repo_root/scripts/multigpu" ]] ||
    die "script is not under the expected repository location: $script_dir"

requested_output=$1
[[ "$requested_output" == /* ]] || die 'NEW_OUTPUT_DIR must be an absolute path'
output_root=$(realpath -m -- "$requested_output")
[[ "$output_root" != / && "$output_root" != . ]] || die 'unsafe output path'
case "$output_root" in
    "$repo_root"|"$repo_root"/*)
        die "formal output must be outside the repository: $output_root"
        ;;
esac
[[ ! -e "$output_root" && ! -L "$output_root" ]] ||
    die "output path already exists: $output_root"

runner="$repo_root/scripts/multigpu/run_l3_30h.py"
final_checks="$repo_root/scripts/multigpu/run_l3_30h_final_checks.py"
numeric_checks="$repo_root/scripts/multigpu/run_l3_30h_numeric_checks.py"
usa_sources="$repo_root/scripts/multigpu/run_l3_30h_usa_sources.py"
eight_graph="$repo_root/scripts/multigpu/run_l3_30h_eight_graph_regression.py"
formal_contract="$repo_root/evidence/l3_30h_20260923/final/formal_input_contract.json"
matrix_manifest="$repo_root/evidence/l3_latest_rerun_38082/matrix/manifest.json"

usa_graph=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/augmented.gr
usa_primary_oracle=/mnt/709/data3/home/Dingzhong/src/mlmq_tpds_withl3/tmp/l3_no_sync_rerun_20260923/graphs/USA/oracle.i32
atmos_graph=/mnt/a100/data1/Dingzhong/mlmq/data/sssp-int/atmosmodm.gr
rmat_graph=/mnt/a100/data1/Dingzhong/mlmq/data/sssp-int/rmat22.gr
atmos_oracle="$repo_root/tmp/l3_30h_20260923/oracles/atmosmodm_s0.i32"
rmat_oracle="$repo_root/tmp/l3_30h_20260923/oracles/rmat22_s0.i32"
usa_additional_1_oracle="$repo_root/tmp/l3_30h_20260923/formal_oracles/USA/source_old7982448_new18266241.i32"
usa_additional_2_oracle="$repo_root/tmp/l3_30h_20260923/formal_oracles/USA/source_old15964897_new6146689.i32"

for path in \
    "$runner" "$final_checks" "$numeric_checks" "$usa_sources" \
    "$eight_graph" "$formal_contract" "$matrix_manifest" "$usa_graph" \
    "$usa_primary_oracle" "$atmos_graph" "$rmat_graph" "$atmos_oracle" \
    "$rmat_oracle" "$usa_additional_1_oracle" "$usa_additional_2_oracle"; do
    require_regular_file "$path" input
done
[[ -x "$PYTHON" ]] || die "canonical Python is unavailable: $PYTHON"

# Freeze the explicit inputs used outside the committed manifests as early as
# possible.  The child drivers repeat these checks and archive their evidence.
require_sha256 c428695cedd4cdec1e889a73bb53cc445aae53e8d7a7635fcda206df944ae478 "$atmos_graph"
require_sha256 164e61cce40603749d8df44fd5bcff9445a0b4bd83f0045be11670a63234b994 "$rmat_graph"
require_sha256 15e8a91b2db3f4b4c358b3aa23972c6dc843dbdd2dfc831bc75abf5c0cb2be8b "$atmos_oracle"
require_sha256 ef421f45303c022800cc317b672d1a7aeddb02df29d7450230e49a5113ccf60e "$rmat_oracle"
require_sha256 85c273900a89422369a06f5524f784b58ed91ceef36addb4ae3710dff5a1d6eb "$usa_graph"
require_sha256 19b674d79f5bc48d26107facb6327c8f8661d40a5bb80db67dd8feb260757a44 "$usa_primary_oracle"
require_sha256 f86e1b6509badf703eb50125156f3479938a928b2abbb2430910673bcf9ca9e3 "$usa_additional_1_oracle"
require_sha256 84664767003586c247a6657450796b0ee2ffbfde8ba84445f70ba05b9c008294 "$usa_additional_2_oracle"

primary_out="$output_root/01_primary"
final_checks_out="$output_root/02_final_checks"
numeric_out="$output_root/03_numeric_add"
usa_sources_out="$output_root/04_usa_sources"
eight_graph_out="$output_root/05_eight_graph"
primary_pair="$primary_out/pair_build"

primary_command=(
    "$PYTHON" -s "$runner"
    --sampling formal
    --input-contract "$formal_contract"
    --graph "$usa_graph"
    --oracle "$usa_primary_oracle"
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
final_checks_command=(
    "$PYTHON" -s "$final_checks"
    --pair-build "$primary_pair"
    --graph-manifest "$matrix_manifest"
    --out "$final_checks_out"
    --cxx /usr/bin/g++
    --boost-include /a100-data/wyh/boost_1_87_0
    --blocks 107
    --delta 200000
    --queue L1SLF_L2DQ
    --window-mode 2
    --window-min 25000
    --window-max 25000
    --idle-backoff 0
    --timeout 400
)
numeric_command=(
    "$PYTHON" -s "$numeric_checks"
    --pair-build "$primary_pair"
    --spec atmosmodm "$atmos_graph" "$atmos_oracle" 0 200000 50
    --spec rmat22 "$rmat_graph" "$rmat_oracle" 0 200000 50
    --spec usa_primary "$usa_graph" "$usa_primary_oracle" 11973673 400000 60
    --spec usa_additional_1 "$usa_graph" "$usa_additional_1_oracle" 18266241 400000 60
    --spec usa_additional_2 "$usa_graph" "$usa_additional_2_oracle" 6146689 400000 60
    --out "$numeric_out"
    --blocks 107
    --build-timeout 1800
    --run-timeout 600
)
usa_sources_command=(
    "$PYTHON" -s "$usa_sources"
    --out "$usa_sources_out"
    --timeout 400
    --outer-slack 1200
)
eight_graph_command=(
    "$PYTHON" -s "$eight_graph"
    --graph-manifest "$matrix_manifest"
    --out "$eight_graph_out"
    --timeout 400
)

if (( dry_run == 1 )); then
    printf 'JOB_A_DRY_RUN output=%s\n' "$output_root"
    printf '\n[01_primary]\n'
    print_command "${primary_command[@]}"
    printf '\n[02_final_checks]\n'
    print_command "${final_checks_command[@]}"
    printf '\n[03_numeric_add]\n'
    print_command "${numeric_command[@]}"
    printf '\n[04_usa_sources]\n'
    print_command "${usa_sources_command[@]}"
    printf '\n[05_eight_graph]\n'
    print_command "${eight_graph_command[@]}"
    PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -s "$eight_graph" \
        --graph-manifest "$matrix_manifest" --out "$eight_graph_out" \
        --timeout 400 --dry-run > /dev/null
    printf 'JOB_A_DRY_RUN_OK no_output_created=1 gpu_checked=0\n'
    exit 0
fi

frozen_head=$(git -C "$repo_root" rev-parse HEAD)
[[ "$frozen_head" =~ ^[0-9a-f]{40}$ ]] || die "invalid HEAD: $frozen_head"
assert_clean_head
preflight_allocation

mkdir -p -- "$(dirname -- "$output_root")"
canonical_parent=$(realpath -e -- "$(dirname -- "$output_root")")
output_root="$canonical_parent/$(basename -- "$output_root")"
case "$output_root" in
    "$repo_root"|"$repo_root"/*)
        die "formal output resolved inside the repository: $output_root"
        ;;
esac
[[ ! -e "$output_root" && ! -L "$output_root" ]] ||
    die "output path appeared during preflight: $output_root"
mkdir -- "$output_root"
mkdir -- "$output_root/driver_logs" "$output_root/metadata"
output_created=1
current_step=initializing

on_exit() {
    local rc=$?
    trap - EXIT
    if (( rc != 0 )) && [[ -n ${output_created:-} ]]; then
        set +e
        write_status FAILED "$current_step" "$rc"
        set -e
    fi
    exit "$rc"
}
trap on_exit EXIT

printf '%s\n' "$frozen_head" > "$output_root/metadata/head.txt"
git -C "$repo_root" branch --show-current > "$output_root/metadata/branch.txt"
git -C "$repo_root" status --porcelain=v1 --untracked-files=all \
    > "$output_root/metadata/git-status.txt"
printf '%s\n' "$repo_root" > "$output_root/metadata/repository-root.txt"
printf '%s\n' "$script_path" > "$output_root/metadata/job-script.txt"
printf '%s\n' "$(utc_now)" > "$output_root/metadata/started-utc.txt"
printf '%s\n' "$allocation_gpu_query" > "$output_root/metadata/gpu-query.csv"
$NVIDIA_SMI > "$output_root/metadata/nvidia-smi.log"
$NVIDIA_SMI topo -m > "$output_root/metadata/gpu-topology.log"
env | LC_ALL=C sort | awk -F= \
    '$1 ~ /^SLURM_/ || $1 == "CUDA_VISIBLE_DEVICES" { print }' \
    > "$output_root/metadata/slurm.env"
{
    printf 'cd -- %q\n' "$repo_root"
    printf 'exec %q %q\n' "$script_path" "$output_root"
} > "$output_root/metadata/invocation.command.sh"
{
    printf '# Job A executes these GPU workflows strictly serially.\n'
    printf '\n# 01_primary\n'
    print_command "${primary_command[@]}"
    printf '\n# 02_final_checks\n'
    print_command "${final_checks_command[@]}"
    printf '\n# 03_numeric_add\n'
    print_command "${numeric_command[@]}"
    printf '\n# 04_usa_sources\n'
    print_command "${usa_sources_command[@]}"
    printf '\n# 05_eight_graph\n'
    print_command "${eight_graph_command[@]}"
} > "$output_root/metadata/plan.commands.sh"

export PYTHONDONTWRITEBYTECODE=1
write_status RUNNING preflight_complete 0

# The primary formal measurement must remain the first GPU-bearing child.
run_step 01_primary "${primary_command[@]}"
run_step 02_final_checks "${final_checks_command[@]}"
run_step 03_numeric_add "${numeric_command[@]}"
run_step 04_usa_sources "${usa_sources_command[@]}"
run_step 05_eight_graph "${eight_graph_command[@]}"

current_step=complete
printf '%s\n' "$(utc_now)" > "$output_root/metadata/completed-utc.txt"
write_status COMPLETE complete 0
printf 'JOB_A_COMPLETE output=%s head=%s slurm_job_id=%s\n' \
    "$output_root" "$frozen_head" "$SLURM_JOB_ID"
