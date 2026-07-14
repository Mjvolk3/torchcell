---
id: hfennuvre2o0sj8g5hktmse
title: Traditional_ml Summary_table
desc: ''
updated: 1783986626299
created: 1783986626299
---

## 2026.07.13 - One reconstructable source for the classical-ML SI tables

The 5-fold CV standard deviation only ever existed inside the plotting scripts, which meant no
Supplementary table could be rebuilt from the result CSVs alone. This script is the single place
that reconciles them: for each (dataset, model, size, encoding) it keeps the one configuration
with the highest `val_spearman` (validation-selected, no test leakage), reports that
configuration's test and CV values for every metric, and emits both a tidy machine-readable CSV
(`results/traditional_ml_summary_with_std.csv`, per study) and the paper's SI tables as `.tex`.
No model is refit; it reads existing results only. The tables are `\input` by
`paper/nature-biotech/sections/backmatter.tex` under Supplementary Note 6, and regenerate with:

```bash
python experiments/smf-dmf-tmf-001/traditional_ml-summary_table.py --write-tables
```

Selection is by Spearman for *all* tables on purpose: one configuration is chosen per encoding
and then reported across every metric, so the tables describe the same model rather than three
different per-metric winners. That choice has a visible consequence, below.

### Making the tables fit the journal, and what that exposed

The MSE table ran off the page. The cause was not styling: two SVR fits carry test MSEs of
5.7e10 and 4.6e13, and at four decimals one cell rendered as ~40 characters. A LaTeX `tabular`
cannot wrap, so a single cell sets the width of its whole column. Measured against the real
constraint -- the stock `sn-jnl` submission `\textwidth` of 372pt, far narrower than the 517pt
`editing.tex` overrides to -- the MSE table was 545pt and the Spearman table 402pt, so *both*
were overflowing the actual submission while only the MSE one was visibly broken in the drafting
view.

Three changes, all in `latex_bench_table`:

- **MSE is reported in units of 1e-3.** Every legitimate value is 0.0029--0.0287, so the leading
  `0.00` on every cell was pure width. This is precision-neutral: one decimal after scaling has
  the same resolution as the four raw decimals it replaces (0.0256 -> 25.6). A 1e-2 scale was
  rejected because it drops the entire interaction block below 1 (0.29--0.59), reintroducing the
  leading zeros.
- **Diverged fits are labelled, not hidden.** They are shown in scientific notation with a
  dagger and a caption note. They are not corrupt data -- they follow from the selection rule:
  Spearman is rank-based and scale-invariant, so a badly scale-calibrated SVR can still rank
  well. `random (d=1000)` at 1e4 is the clean illustration, with a healthy rho = 0.458 sitting
  next to an MSE of 5.7e10. MSE is scale-sensitive and explodes.
- **`\tabcolsep` 6pt -> 3pt**, which was reclaiming ~72pt of pure padding across seven columns.

Result: 293pt (MSE) and 366pt (Spearman), both inside 372pt.

### Pearson table added

Supplementary Note 6 quotes Pearson in prose -- one-hot fitness reaching r ~ 0.87--0.90, gene
interactions plateauing near r ~ 0.4 -- with no table a reader could check those claims against.
`extract()` already stored `test_pearson` and `cv_pearson_std` for every row and
`latex_bench_table` was already metric-generic, so the table needed only a label entry plus an
`\input`. It confirms the prose: one-hot fitness peaks at 0.902 (SVR, 1e5) while the entire
interaction block sits at 0.33--0.39.

SI table order is now S3 Spearman, S4 MSE, S5 Pearson, fixed by the order of the `\input` lines
in `backmatter.tex` and by `write_paper_tables()`. Those two must stay in sync.

### Open question

SVR / interaction / `random (d=1000)` at 1e3 reports a CV s.d. of 0.383 against a mean Spearman
of 0.458 -- a spread nearly as large as the value, in the same encoding-and-model cell that
produced one of the diverged MSE fits. The two symptoms likely share a cause and the fits there
are worth re-examining.
