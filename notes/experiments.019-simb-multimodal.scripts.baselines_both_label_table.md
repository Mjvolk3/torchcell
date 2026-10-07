---
id: wtsjqamsbicfgof7wp1vydl
title: Baselines_both_label_table
desc: ''
updated: 1791362357965
created: 1791362357965
---

## 2026.10.07 - Twelve-seed baselines on the both-label store

Summarizes the per-seed files of IGB job 2423302 (`results/baselines_split_fig3_proteome_both_expression_log2_ratio_full/` and `..._both_protein_abundance_full/`, written by [[experiments.019-simb-multimodal.scripts.expression_baselines_split]] with `--require-labels`) into `results/baselines_both_label.json` and `notes-tex/figure-3-gate/tables/baselines_both_label.tex`: bilinear ridge and nearest-neighbor average on ProtT5, the composite stack and random 1024, mean and sd over 12 split seeds, validation and test, cells selected on validation. ProtT5 neighbor average leads both heads: expression 0.110 ± 0.017 / 0.090 ± 0.027, proteome 0.065 ± 0.013 / 0.040 ± 0.024. The v22 window scores (0.062 to 0.075) sit below the expression baselines on the same store.
