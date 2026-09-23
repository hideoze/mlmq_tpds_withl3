# L3 30-hour stage 0

This directory separates executable evidence from pending work.

- `version_manifest.json`: repository, toolchain, Slurm allocation, 30-hour
  clock, GPU topology, build hashes, and NVSHMEM decision.
- `dataset_manifest.json`: full streaming audit of the six fixed P0/control
  files. All CSR layouts are valid; `nlpkkt80` is rejected because its payload
  contains negative weights.
- `usa_augmented_dataset_manifest.json`: separate full streaming audit of the
  exact frozen USA G+ input (`85c273...`), including every destination and all
  66,684,784 int32 edge weights; the format is valid and weights are nonnegative.
- `oracle_manifest.json`: source-0 int64 oracle results. Oracle binaries remain
  under ignored `tmp/`; only hashes and outcomes are tracked.
- `capacity_manifest.json`: static allocation/index audit and the historical
  evidence boundary.  Its overall status remains `NOT_PROVEN` until the final
  clean pair completes guarded L2 and checked-add GPU validation.
- `gpu_preflight_38119_failed/`: the first real two-GPU allocation. CUDA and
  P2P checks passed, but the shell script stopped during the filesystem probe.
- `gpu_preflight_38122/`: corrected executable preflight, PASS.
- `build_base_attempt1/` and `build_base/`: two successful paired builds. The
  latter (`build_base_v2` in ignored `tmp/`) is the binary pair used by stage 1.

The branch is dirty while implementation and exploratory runs continue, so
these results are not final clean-commit acceptance evidence.

The NVSHMEM decision is scoped to the allocation probes recorded in
`version_manifest.json`: no usable installation was detected in that job's
`PATH`, `/usr`, or `/opt` search.  It is not a machine-wide absence claim.
