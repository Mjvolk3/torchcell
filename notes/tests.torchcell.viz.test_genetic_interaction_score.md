---
id: o5s0cquygqht25f61923vrt
title: Test_genetic_interaction_score
desc: ''
updated: 1790648912222
created: 1790648912222
---

## 2026.09.28 - The genetic interaction score box plot on simulated data (Phase 9)

6 tests. Binning (bins 0, 2, 3, 6, 9), Pearson 0.961 and R squared 0.924 recomputed, tick labels `"-0.40"` to `"0.00"`, the two black neutral reference lines, the N/A title for one valid pair, the simulated layout with exactly 11 NaN under seed 0 and exactly 10 two-value cells (6 at 0.2, 4 at 0.3), and `main()`. The `except ValueError` arms at lines 37 to 43 are unreachable under scipy 1.16 for the same reason as in [[tests.torchcell.viz.test_viz_fitness]]. Coverage 67.4% to 97%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]].
