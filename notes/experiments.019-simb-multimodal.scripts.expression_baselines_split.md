---
id: fl2ywkhjkg52180a1bftpz8
title: Expression_baselines_split
desc: ''
updated: 1789288488120
created: 1789288488120
---

## 2026.09.13 - B0 to B3 on the partition the model trains on

`expression_baselines.py` scored the Ahlmann-Eltze baselines on the oracle-family split, a private permutation of the raw Kemmeren LMDB, so baseline and model were only approximately on the same strains. This script reads `index_details_seed_<k>.json` (the expression record indices per split that the datamodule wrote) and the phenotype values from the same processed `fig3_core` LMDB the model trains on. Per split seed: B0 per-gene train mean, B1 no change, B2 bilinear ridge over the rank and ridge grids, B3 embedding-neighbor mean, for the four embeddings; rank, ridge and k are selected on val (as the model selects its epoch) and test is reported at the selected cell, with every cell's val and test score stored. `--fold-test-into-train` reproduces the 90/10 arm (val only). Two deliberate differences from the oracle-family version: double-deletion strains are kept (their representation is the mean of the two genes' embeddings) because the model's val set contains them, and the per-feature Pearson drops near-constant columns as the training metric does.

Launched on GilaHyper CPU as a five-task array ([[experiments.019-simb-multimodal.scripts.gh_expression_baselines_split]]); results land in `results/expression_baselines_split/seed<k>[_fold90].json` and are read against the v13 round ([[experiments.019-simb-multimodal.conf.cgt_expr_v13_split]]).
