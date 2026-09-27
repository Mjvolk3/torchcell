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

## 2026.09.27 - Morphology on the same partitions

`--label calmorph --dataset-tag fig3_core` scores the four baselines on the CalMorph label
over the model's own partitions (3,757 / 469 / 469 per seed, 281 features, near-constant
columns dropped as the training metric does). Results land in
`results/morphology_baselines_split_fig3_core/seed<k>.json`; the directory is now keyed by
label so the run cannot overwrite the expression baselines. Validation per-feature Pearson
over seeds 0 to 3, the gate embeddings:

| embedding | B2 bilinear ridge, val | B3 neighbor mean, val |
|---|---|---|
| ProtT5 | 0.040, 0.082, 0.063, 0.053 | 0.057, 0.066, 0.089, 0.073 |
| CaLM | 0.065, 0.070, 0.064, 0.040 | 0.049, 0.038, 0.054, 0.045 |
| species LM 5p+3p | 0.029, -0.005, 0.029, 0.023 | 0.019, 0.027, 0.042, 0.009 |
| chromatin pathways | 0.081, 0.111, 0.061, 0.088 | 0.089, 0.091, 0.092, 0.079 |

The ceiling for the same 278 modeled features is a mean of 0.611 and a median of 0.647
(`morphology_noise_ceiling.json`), and the best trained morphology run on record reaches a
rolling maximum of 0.082 (`vsceij2v`, epoch 27 of 65). So the trained model, the linear
baseline and the neighbor mean all sit at 0.04 to 0.11 against a ceiling of 0.61, and the
manuscript's 0.619 was a placeholder target with no run behind it.
