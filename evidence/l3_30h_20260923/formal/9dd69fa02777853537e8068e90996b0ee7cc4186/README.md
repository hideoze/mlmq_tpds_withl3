# L3 30-hour final evidence

This directory is the compact, repository-safe export of the completed L3
30-hour workflow for formal source commit
`9dd69fa02777853537e8068e90996b0ee7cc4186`.

## Outcome

The execution and measurement contracts passed, but the fixed performance
target did not:

- classification: `VALID_BELOW_TARGET`;
- Job A: Slurm 38304, `COMPLETE`, T1/T2 solve medians
  `73.3605045 / 64.6717355 ms`, `S=1.1343518762`;
- independent Job B: Slurm 38305, `COMPLETE`, T1/T2 solve medians
  `73.3405565 / 64.5799375 ms`, `S=1.1356554270`;
- fixed target: `S >= 1.20`; both accepted formal batches have
  `target_met=false`;
- final checks: 4/4 small fixtures, sequential source/reset, and 8/8 graph
  correctness checks passed;
- checked-add numeric diagnostic: 10/10 runs passed and
  `performance_claim_allowed=false`;
- two fixed additional USA sources were valid at `0.7343201845x` and
  `1.0907633645x`;
- the full-sample eight-graph regression was valid for 16/16 G/G+ cases;
  solve-speedup geometric means were `0.8881843984x` on G and
  `0.9298583563x` on G+, with 0/16 cases meeting 1.20.

`formal_summary.json` and `formal_summary.csv` contain independently
recomputed median/IQR/MAD statistics. `completion_matrix.*` separates
execution validity from the performance target. `formal_attempt_ledger.*`
keeps all seven failed/exploratory predecessors and the two accepted jobs
without mixing samples.

## Evidence boundary

`jobA/` and `jobB/` preserve the allowlisted small JSON, CSV, command,
environment, status, hash, text, and raw log evidence. Logs are stored with
deterministic gzip metadata. Binaries, graphs, oracles, expanded source trees,
generated fixture trees, source archives, and files over 2 MiB are excluded;
their paths and hashes remain in the retained provenance, while exclusion
counts and examples are recorded in `archive_manifest.json`.

The primary Job A/B runners use `sampling=formal`. The additional-source and
eight-graph runners reuse a fresh pair frozen by the clean-SHA Job A outer
workflow, so their inner summaries truthfully retain
`sampling="exploratory"`. They are classified as
`full_sample_clean_sha_extension_regression`: two reverse-order rounds with
one warmup and five timed queries per process under outer formal-equivalent
source, pair, input, correctness, capacity, and postflight gates. They are not
used to select or redefine the primary target.

## Layout

- `formal_summary.*`: accepted primary, robustness, and per-graph metrics.
- `completion_matrix.*`: machine-readable acceptance checklist.
- `formal_attempt_ledger.*`: all retained attempts and evidence eligibility.
- `jobA/`, `jobB/`: compact raw evidence with the original hierarchy.
- `figures/`: 3.5-inch PDF, 400-dpi PNG, plotting metrics, and provenance.
- `paper_update.md`: proposed text and caption; manuscript operations remain
  explicitly `NOT_RUN`.
- `archive_manifest.json`: inclusion/exclusion policy and per-file hashes.
- `SHA256SUMS`: hashes every delivered file except itself.

The detailed engineering narrative and every per-source/per-graph T1/T2 value
are in [L3_30h_performance_report.md](../../../../L3_30h_performance_report.md).

## Verification

From this directory:

```bash
sha256sum -c SHA256SUMS
find . -type f -name '*.gz' -exec gzip -t {} +
```

The exporter validation can be rerun against the retained external source jobs
without writing output:

```bash
python3 ../../../../scripts/multigpu/export_l3_30h_formal.py \
  --job-a /mnt/709/data3/home/Dingzhong/l3_30h_formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobA \
  --job-b /mnt/709/data3/home/Dingzhong/l3_30h_formal/9dd69fa02777853537e8068e90996b0ee7cc4186/jobB \
  --expected-sha 9dd69fa02777853537e8068e90996b0ee7cc4186 \
  --validate-only
```

These commands verify the archived evidence and its source binding; they do
not rerun GPU experiments.
