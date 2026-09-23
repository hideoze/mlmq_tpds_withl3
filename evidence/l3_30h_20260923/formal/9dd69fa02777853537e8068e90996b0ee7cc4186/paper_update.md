# Paper update package

## Status

- Evidence paragraph and caption: `READY_FOR_INSERTION`.
- Figure assets: `figures/l3_30h_final.pdf` and
  `figures/l3_30h_final.png`.
- Plot data and provenance: `figures/metrics.csv` and
  `figures/provenance.json`.
- Manuscript insertion: `NOT_RUN (manuscript source absent)`.
- TeX compilation: `NOT_RUN (manuscript source absent)`.
- PDF/layout inspection: `NOT_RUN (manuscript source absent)`.

The repository contains no `.tex`, `.bib`, manuscript PDF, or compilable paper
project. This package therefore supplies exact replacement text and a
single-column figure without claiming that either has been inserted or
compiled.

## Proposed result paragraph

> On the frozen shortcut-augmented USA input, the final clean revision achieves valid same-input solve-only speedups of **1.134x** in the primary run and **1.136x** in an independent two-A100 allocation. Both results are below the fixed **1.20x** target. Across eight road-network graphs, the geometric-mean speedups are **0.888x** on G and **0.930x** on G+, and none of the 16 graph/view cases reaches 1.20x. Two additional fixed USA sources yield **0.734x** and **1.091x**. Thus, the two-GPU implementation and measurement workflow pass execution and correctness validation, while the performance result is valid but below target.

The acceptance metric is always
`median(T1 independent single-GPU no-L3 solve_ms) /
median(T2 same-query dual-GPU L3 solve_ms)`. Query-wall ratios are reported in
the evidence but must not replace the solve-only acceptance result. Numeric
checked-add timings are correctness-only and must not be cited as performance.

## Proposed figure caption

> **Same-graph dual-GPU L3 scaling on the final clean revision.** Bars show the ratio of the ten-sample median single-GPU no-L3 solve time (T1) to the ten-sample median dual-GPU L3 solve time (T2) for the original graph G and shortcut-augmented graph G+. The dashed line marks parity and the dotted line marks the fixed 1.20x target. All 16 cases pass correctness and measurement-validity gates; the geometric means are 0.888x on G and 0.930x on G+, and no case reaches 1.20x. These full-sample extension regressions use a fresh clean-SHA pair under the Job A outer gates; the primary and independent formal USA G+ results are reported separately.

## Provenance boundary

The figure includes only Job A 38304 eight-graph results bound to formal source
SHA `9dd69fa02777853537e8068e90996b0ee7cc4186`. Failed Jobs 38227, 38238,
38271, 38280, and 38292, dirty exploratory Jobs 38251/38254, historical jobs,
and Job B confirmation samples are not mixed into the bars. Job B is used only
as the independent primary confirmation stated in the paragraph.
