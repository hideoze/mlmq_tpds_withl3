# Stage 1 exploratory evidence

Everything in this directory is exploratory evidence collected from a dirty
development worktree.  It may select or reject a candidate, but it is not a
clean-SHA formal acceptance result.

## Status schema boundary

The early sweep driver used `status: PASS` to mean that every child process
returned zero.  It did **not** mean that `S >= 1.20`.  This rc-only schema was
used by jobs 38136, 38138--38140, 38147, 38149, 38159, and 38163; job 38145
explicitly records failures.  Job 38129's `complete` and job 38131's
`driver_status.tsv` are also completion/correctness labels rather than
performance acceptance.

Starting with the Route-B2 diagnostics, the driver separates:

- `measurement_valid`: process, oracle, sample-count, and runtime-contract
  checks passed;
- `target_met`: the solve-time median ratio reached 1.20;
- `acceptance_pass`: both conditions hold (added after the exploratory jobs).

No Stage 1 result is a formal acceptance pass.  Job 38180 is a valid
exploratory measurement but has `target_met=false` for both USA and RGG.

`candidate_ledger.json` and `candidate_ledger.csv` flatten every retained
sweep manifest into 34 auditable cases: 29 valid-but-below-target measurements
and five retained failures.  Missing/failed cases are never encoded as zero
time.  The separate initial five-graph baseline and n20 smoke keep their full
named evidence directories and are summarized in the execution report.

## Evidence paths

Copied manifests preserve the paths emitted during execution, many of which
refer to the ignored `tmp/l3_30h_20260923/` working area.  The raw logs,
records, samples, and summaries themselves are copied under their named job
directories here.  Large generated graphs, binaries, and source tarballs are
intentionally not committed; compact hash/provenance manifests identify them.

The Route-B1 transformation stdout was not retained, so Route B1 remains an
exploratory layout result even though the generated CSR, permutation,
inverse-permutation, payload, and oracle mappings were subsequently checked
in full.  See `route_b1_layout_manifest.json`.
