# MLMQ — multi-level message queue SSSP (frozen L3 supplement version)

Source snapshot of the capacity-fixed frozen version used for the L3
supplementary experiments (job 37327, `L3_supplementary_experiments_results.md`)
plus the six-configuration ADDS comparison
(`L3_adds_samegraph_sixconfig_results.md`, job 37646).

## Layout

- `SSSP/`, `core/` — dual-GPU L3 solver (identical to `dual_source/` in the
  archived `tmp/l3_supplement_20260921_v2/dual_build/source.tgz`).
- `ADDS/` — checksum-pinned official ADDS artifact adapter
  (`ADDS/official_bench.cu`) and original ads_int sources.
- `scripts/multigpu/` — frozen experiment runners
  (`run_supplement_matrix.py`, `run_adds_samegraph_matrix.py`, …).
- `evidence/` — logs, JSON records, source snapshots, build commands and
  SHA256 manifests for jobs 37327/37329/37646. No graph data and no binaries
  are included (see `evidence/l3_supplement_20260921_v2/graphs_hash_manifest.json`
  and `hashes.json` files for byte-level identification).
- `knowledgebase/` — audit artifacts for the supplement batch.

## Paper configuration build (A100, CUDA 12.4, sm_80)

```bash
RUN=tmp/l3_minimal_revision_$(date +%Y%m%d_%H%M%S)
scripts/multigpu/build_l3_paper_config.sh "$RUN/paper_config_build"
bash scripts/multigpu/build_official_adds.sh ADDS_ARTIFACT.zip OUT_DIR  # ADDS
```

The L3 build script records the exact `nvcc` argument vector, build log,
return code, and binary SHA256 in the new output directory. It uses the frozen
paper macros (`MLMQ_WORKER_THREADS=512`, cooperative collect, direct RX,
retained TX, window mode 2, worker recovery, ACK wait/scan, boundary index,
final L2 counts, and `L3_CHAIN_SHORTCUTS=false`) and `sm_80`; it does not rely
on `SSSP/Makefile`, which is a default build without those paper macros.
`BOOST_INCLUDE_DIR` may override the recorded default path
`/a100-data/wyh/boost_1_87_0` when compiling on another installation.

For the paired eight-graph rebuild and measurement, retain the independent
single-GPU baseline source at `tmp/nol3_213_preload/source.tgz`. Use a new
output directory for the matrix pipeline. Run the snippets in the same shell
so `$RUN` keeps pointing to the directory created above:

```bash
BASE="$RUN/matrix"
python3 scripts/multigpu/prepare_supplement.py "$BASE"
python3 scripts/multigpu/prepare_supplement_diagnostics.py "$BASE"
python3 scripts/multigpu/build_supplement_fixtures.py "$BASE"
# Run both commands below inside a two-GPU Slurm allocation.
python3 scripts/multigpu/run_supplement_matrix.py "$BASE"
python3 scripts/multigpu/run_supplement_checks.py "$BASE"
```

The matrix keeps the existing eight graphs, G/G+ variants, fixed sources,
queue `L1SLF_L2DQ`, delta `200000`, `MLMQ_WORK_BLOCKS=107`, one warmup, five
timed queries, and two interleaved rounds. The cut percentages are 60% for NY,
55% for COL, 40% for W, and 50% for the other graphs, as set in
`run_supplement_matrix.py`; each fixed source is read from that graph's
`layout.json` as `source_new_id`. It loads original CSR files from
`tmp/landmark206_matrix/<NAME>/landmark.gr`, exports G+ and a per-vertex oracle
into `BASE/graphs/`, and records resolved sources, cuts, and file hashes in
`BASE/matrix/graphs.json`. Archived numerical comparisons are in
`L3_supplementary_experiments_results.md` and
`L3_adds_samegraph_sixconfig_results.md`; the latter records the separate ADDS
batch and its pinned binary. New MLMQ measurements use the current build
artifacts. Reuse ADDS timings only with their archived batch provenance and
after matching graph hashes, device, configuration, and solve-time boundary.

All GPU checks and measurements must run through Slurm on the target
partition. On ada-709, compilation is allowed but the local driver is
unavailable; do not launch GPU tests there. For A100, use `sm_80` and verify
the assigned GPUs and their topology from inside the allocation.
