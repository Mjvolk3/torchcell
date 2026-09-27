---
id: seuiclakcst5uuxtzxu50oe
title: Per_dataset_baselines
desc: ''
updated: 1790492169034
created: 1790492169034
---

## 2026.09.27 - Ridge and kNN on every dataset, against each one's own ceiling

Generalizes [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.baseline_ceilings]] from
Vanacloig to all five datasets, because a pooled claim needs a per-dataset reference: a dataset
already near its ceiling has no headroom for pooling to recover, and a dataset with no served
uncertainty has no ceiling to be judged against at all.

**Design.** Gene-by-compound matrix per dataset, hold out whole compounds, predict the held-out
column from the molecule embedding. Leave-one-compound-out for panels of at most 60 compounds
and ten folds grouped by compound otherwise, because Wildenhain's 5,170 compounds would
otherwise be 5,170 fits per encoder. The scheme used is written into every row so the two are
never silently compared as equal. Two targets, raw and gene-centered, with centering refit inside
each fold. The null on the centered target is a similarity-weighted mean of randomly chosen
training compounds, since a gene-mean prediction is identically zero there.

**Ceilings exist only where the source serves an uncertainty.** Vanacloig serves one on every
record, the Hillenmeyer arms on a third and a quarter, Wildenhain on 4 percent, and Hoepfner on
none. Below 200 cells carrying a standard error the ceiling is reported NA rather than estimated
from a biased subset.

**Cross-check.** The Vanacloig row reproduces the earlier script exactly: FCFP4 count ridge at
Spearman 0.311 against a 0.835 ceiling on the centered compound-cold target.

Figure: `notes/assets/images/031-env-chemgen-inhibitor-tolerance/encoder_comparison.svg`.
Result files: `results/per_dataset_baselines.csv`,
`results/per_dataset_baselines_summary.csv`. Rendered as table t13 and Figure 5 of
`notes-tex/031-unified-representation`.

### 2026.09.27 - Results across all five datasets

| dataset | scheme | null | best ridge | best kNN | best overall | ceiling | of ceiling |
|---|---|---|---|---|---|---|---|
| Vanacloig | LOCO, 41 folds | 0.028 | **0.311** | 0.284 | 0.311 ridge fcfp4 | 0.84 | 0.37 |
| Hillenmeyer HOM | 10-fold | 0.030 | 0.105 | **0.141** | 0.141 knn5 fcfp4 | 0.94 | 0.15 |
| Hillenmeyer HET | 10-fold | 0.003 | 0.113 | **0.126** | 0.126 knn1 maccs | 0.79 | 0.16 |
| Hoepfner | 10-fold | 0.023 | **0.189** | 0.169 | 0.189 ridge fcfp4 | none served | |
| Wildenhain | 10-fold | 0.168 | 0.249 | **0.394** | 0.394 knn5 ecfp4 | 0.87 | 0.45 |

**Three findings.**

1. Molecule features carry compound-specific signal in EVERY dataset, not only Vanacloig. Each
   beats its own no-feature null on the centered target.
2. **Which simple model wins is not constant, and this corrects an earlier claim.** The earlier
   Vanacloig-only run concluded "ridge beats kNN on both targets and every encoder". That holds
   on Vanacloig and Hoepfner and is FALSE on the other three: similarity-weighted nearest
   neighbors wins on Wildenhain (0.394 against 0.249), Hillenmeyer HOM and Hillenmeyer HET. The
   margin is largest on the densest compound panel. I am not offering a mechanism for that; it
   is a measured split, not an explained one.
3. Wildenhain's null is 0.168 against 0.003 to 0.030 everywhere else, so its headline 0.394
   should be read against that null rather than against the other datasets' scores.

**Two limits on the ceilings.** Hoepfner serves no uncertainty on any record, so it has no
ceiling at all. The Hillenmeyer and Wildenhain ceilings rest on the 33, 17 and 1 percent of
cells carrying a served standard error, which is the subset measured more than once and
therefore not a random sample. Only the Vanacloig ceiling, at 97 percent of cells, comes from
substantially the whole matrix.

**Encoder count.** Twelve encoders are scored on Vanacloig and eleven on the other four,
because Uni-Mol cannot build a conformer for every compound in them (isolated ions, metal
salts) and the script drops an encoder that cannot represent the whole panel rather than
scoring it on a subset.

**Cost.** The full run is about 25 CPU-hours, dominated by Wildenhain's 5,170-compound panel.
`--figure-only` redraws the figure from the saved summary without repeating it.
