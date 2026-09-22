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

## Build (A100, CUDA 12.4, sm_80)

```bash
cd SSSP && make          # dual-GPU mlmq binary
bash scripts/multigpu/build_official_adds.sh ADDS_ARTIFACT.zip OUT_DIR  # ADDS
```

Benchmarks run only inside Slurm on the target partition; the Makefile
gencode must match the target GPU (sm_80 = A100).
