---
id: w4vtmpwklt9bz2bd60csvpr
title: Additive_baseline_gene_interaction
desc: ''
updated: 1788312489542
created: 1788312489542
---

## 2026.09.01 - Additive Null Baselines on the 010 Split

Fits interaction-free null baselines for `gene_interaction` on the exact 010
train/val/test split, so the equivariant cell graph transformer can be compared
against a model with no capacity to represent gene interaction. Motivated by
Visani, Verma and DeWitt, bioRxiv 2026.04.23.719915.

Reads the pinned build directly rather than through the loaders: `label_df.parquet`
for labels, `is_any_perturbed_gene_index.json` to recover each record's three
perturbed genes, `index_seed_42.json` for the split. Transformer reference metrics
are read from the three checkpoints' re-evaluation runs under
`$DATA_ROOT/wandb-experiments`.

Baselines: B0 train mean, B1 additive per-gene ridge, B2 additive plus recurring
gene pairs, B3 hierarchical empirical mean, B4 recurring gene pairs only, B5 an
embedding-sum MLP on B1's feature space across three seeds. Ridge alpha is chosen
on val and reported on test.

Outputs `results/additive_baseline_gene_interaction.csv`,
`results/additive_baseline_B1_coefficients.npz` (consumed by the two inference-space
scripts), per-model prediction arrays, and the test-Pearson bar panel.

![](assets/images/010-kuzmin-tmi/additive_baseline_test_pearson.svg)

Findings and interpretation:
[[experiments.010-kuzmin-tmi.additive-baseline-analysis]]
