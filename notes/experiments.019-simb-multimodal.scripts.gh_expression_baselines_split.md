---
id: xcc5i0hbjp2sj6f2yn50lab
title: Gh_expression_baselines_split
desc: ''
updated: 1789288496704
created: 1789288496704
---

## 2026.09.13 - Five CPU tasks on GilaHyper, one per partition

Array over the partitions the v13 round trains on: tasks 0-3 are split seeds 0-3 at 80/10/10 (val-selected, test reported), task 4 is split 0 with the test records folded into train (the 90/10 arm, val only). `-p main`, 8 CPUs, 32 GB, four-hour wall; runs `expression_baselines_split.py` from the submitting checkout. Results: `results/expression_baselines_split/seed<k>[_fold90].json` ([[experiments.019-simb-multimodal.scripts.expression_baselines_split]]).
