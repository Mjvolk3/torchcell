---
id: fo4i729i0m0z58jl3uo0fyt
title: Expression_baselines_split_table
desc: ''
updated: 1789290779629
created: 1789290779629
---

## 2026.09.13 - Two generated tables for the expression document

Reads `results/expression_baselines_split/seed<k>.json`, `seed0_fold90.json` and `results/knn_embedding_probe.json`, never recomputes, and writes `tables/expression_baselines_split.tex` (B2 and B3 per split seed, val and test rows, mean and sd over the four draws, the 90/10 column) and `tables/knn_embedding_probe.tex` (best k per embedding against the random floors) into `notes-tex/019-simb-multimodal-expression/`, plus `results/expression_baselines_split/summary.json` with the across-split means. Both tables carry a `%% SOURCE:` line and are input by `sections/3-baselines.tex`, the section that states the baseline equations ([[experiments.019-simb-multimodal.scripts.expression_baselines_split]]).

Across the four draws (ProtT5): B2 val 0.106 +/- 0.026, test 0.110 +/- 0.027; B3 val 0.120 +/- 0.009, test 0.118 +/- 0.051; 90/10 on split 0 moves B2 by +0.002 and B3 by +0.011.
