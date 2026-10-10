---
id: 24iscy7ctcv61km3mi8gh2l
title: Expression_record
desc: ''
updated: 1791537298195
created: 1791537298195
---

## 2026.10.09 - The 019 record read against the scaling axes

Reads eight committed result files of experiment 019 (`expression_ceiling_all`, `morphology_noise_ceiling`, `short_budget_spread`, `loss_min_vs_pearson_peak`, `joint_checkpoint_readout`, `baselines_both_label`, `v21_readout`, `prediction_shrinkage_probe`) and writes three files for the scaling-laws document (`notes-tex/modeling/scaling-laws`): `tables/floors.tex` (the replicate ceiling per panel, the bound on E in Pearson units), `tables/budget_curve.tex` (the eight long-budget arms rescored at ten budgets, the compute axis in Pearson) and `tables/expression_numbers.tex` (the macros the section's prose quotes: the loss-against-metric disagreement, the v19 joint null, the baseline bar, the best twelve-seed arm, the shrinkage factors by head).

Nothing here is a new measurement; the section it feeds (`sections/6-expression.tex`) says what the record settles (the floor, a budget curve on the wrong statistic, a multimodal data-axis null, a confounded parameter-count null) and specifies the two expression sweeps that would give an exponent. The result files were copied from branch `multimodal-phenotype-retrospective` so the script runs on this branch; they are byte-identical to the committed ones there.
