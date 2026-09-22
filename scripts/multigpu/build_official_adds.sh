#!/usr/bin/env bash
# Extract only known source members; never execute artifact scripts.
set -euo pipefail
archive=${1:?Usage: build_official_adds.sh ARCHIVE NEW_OUTPUT_DIRECTORY}
out=${2:?Missing output directory}
[[ ! -e "$out" ]] || { echo "Output already exists" >&2; exit 2; }
actual=$(md5sum "$archive")
[[ ${actual%% *} == 11fa78b8a387d33486a3f8511191f45e ]] || { echo "Wrong artifact checksum" >&2; exit 2; }
mkdir -p "$out/vendor"
for name in Makefile support.h wl.h common.h kernel.cu support.cu csr_graph.h csr_graph.cu; do
    unzip -q "$archive" "ppopp-code/ads_int/$name" -d "$out/vendor"
done
# Compatibility-only redirect: original include text and all vendor bytes stay intact.
cuda_compiler=$(readlink -f "$(command -v nvcc)")
cuda_include=$(dirname "$(dirname "$cuda_compiler")")/include
[[ -f "$cuda_include/cub/cub.cuh" ]] || { echo "Toolkit CUB missing" >&2; exit 2; }
ln -s "$cuda_include" "$out/vendor/ppopp-code/cub-1.8.0"
nvcc --version > "$out/nvcc_version.txt"
nvcc ADDS/official_bench.cu "$out/vendor/ppopp-code/ads_int/support.cu" \
    "$out/vendor/ppopp-code/ads_int/csr_graph.cu" -I"$out/vendor/ppopp-code" \
    -I/a100-data/wyh/boost_1_87_0/ \
    -O3 -gencode=arch=compute_80,code=sm_80 -rdc=true -lcudadevrt \
    -Xptxas -O3 -Xptxas -v -lcuda -lcudart -o "$out/adds" > "$out/build.log" 2>&1
sha256sum "$out/adds" ADDS/official_bench.cu "$out/vendor/ppopp-code/ads_int/"* \
    > "$out/sha256.txt"
