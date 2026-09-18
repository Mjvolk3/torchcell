---
id: 66dps2t6sbkl7jgyrc3exyx
title: S3_closure_recompute
desc: ''
updated: 1789719776295
created: 1789719776295
---

## 2026.09.18 - Recompute interactions and p-values over the S3 pool

Script: `experiments/025-solid-growth/scripts/s3_closure_recompute.py` (stages `scan`, `raw`,
`analyze`), figures and tables in `s3_closure_plots.py`. Results and reading in
[[experiments.025-solid-growth.s3-closure]]; the typeset document is `notes-tex/025-s3-closure/`.

- `scan` reads the 1,121,645 S3 records from the 025 LMDB with 48 workers (under a minute with a
  warm page cache) into `$DATA_ROOT/data/torchcell/experiments/025-solid-growth/s3_closure/s3_records.parquet`:
  per record the stored fitness, its RMS-pooled SD, `n_samples`, `num_duplicates`, the joined
  source names, the stored interaction and p-value, and how many source names the interaction
  entry joined.
- `raw` filters the four Costanzo 2016 files and the Kuzmin 2018 / 2020 tables to the closure
  pairs (2,026,500 digenic rows) and keeps every Kuzmin row (1,436,105) for the trigenic joins.
  Kuzmin 2020 names the combined-fitness columns `Double/triple mutant fitness`.
- `analyze` writes `results/s3_closure_recompute_summary.json`,
  `results/s3_closure_composition.csv`, the four LaTeX tables under
  `notes-tex/025-s3-closure/tables/`, and the three figures below (true-size SVG plus PNG).

Gotchas found on the way: the per-genotype merge keeps no per-source values, so the measured
mean of an essentiality-tainted single is recovered as k/(k-1) times the stored mean; the
stored single-mutant SD is not the quantity the sources propagated (adding it lowers the p
ranking from 0.984 to 0.813); the deduplicator's p-value is a t-test over the duplicate scores
and never reads the source p-values.

![](./assets/images/025-solid-growth/s3_closure_strength.svg)

![](./assets/images/025-solid-growth/s3_closure_confidence.svg)

![](./assets/images/025-solid-growth/s3_closure_hazards.svg)
