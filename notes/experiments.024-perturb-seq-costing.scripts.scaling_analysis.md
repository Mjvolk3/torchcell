---
id: uxly0jdxd4weuywm4pg6uui
title: Scaling_analysis
desc: ''
updated: 1787625566145
created: 1787625566146
---

## 2026.08.24 - Delivery, combination recovery, and the environment axis

`experiments/024-perturb-seq-costing/scripts/scaling_analysis.py`

Backs Sec. 7 of the review. Three questions the earlier sections do not ask:

1. **Delivery.** If several guides per cell arrive on several plasmids rather
   than on one array, how many DISTINCT guides does a cell carry? Selection
   conditions on carrying at least one, so the count is a zero-truncated
   Poisson, and two plasmids can carry the same guide.
2. **Recovery.** Must a combination be seen twice? Main effects never require a
   repeat; a named combination's joint transcriptome requires `c` cells carrying
   exactly it. The answers differ by orders of magnitude.
3. **Environments.** Linear in cell demand where named combinations are
   quadratic, which is the whole reason that axis is attractive.

`TARGET_PLASMIDS_PER_CELL` is integers on purpose. It was floats, and a reader
asked what a target of 1.5 was supposed to mean -- fairly, since the column is a
copy number per cell, not a count of genes or of library members.

Feeds tables 17 and 18 and `results/scaling_analysis.json`. `lam_for_target_mean`
is shared with `plot_poisson_primer.py`, so the figure and the table cannot drift.
