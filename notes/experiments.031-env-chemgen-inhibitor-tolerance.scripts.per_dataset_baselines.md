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
