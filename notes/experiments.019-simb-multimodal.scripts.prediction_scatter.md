---
id: qniv0zk4v5fboyi3650mcko
title: Prediction_scatter
desc: ''
updated: 1791325519658
created: 1791325519658
---

## 2026.10.06 - What the expression predictions look like

The round scores are per-gene Pearson summaries; this figure shows the predictions themselves for the best expression checkpoint on file, the v13 reference (run `wq8y8nd5`, split seed 0, validation Pearson per gene 0.223 on the dump), beside the bilinear ridge (B2) and nearest-neighbor average (B3) on ProtT5, refit on the same partition with the cells [[experiments.019-simb-multimodal.scripts.expression_baselines_split]] selected on validation. 155 validation strains x 6,127 graph genes. Predictions come from the dump written by [[experiments.019-simb-multimodal.scripts.gh_eval_ckpt_predictions]]; loaders are shared with [[experiments.019-simb-multimodal.scripts.variance_stratified_pearson]]. Numbers in `results/prediction_scatter.json`.

![](./assets/images/019-simb-multimodal/prediction_scatter_2026-10-06-17-47-55.svg)

**Figure.** (a) Model prediction against the measured log2 ratio over every strain-gene pair (pooled r 0.45, carried mostly by the per-gene means). (b) The same with each gene's train mean removed from both axes, the view the per-gene Pearson scores: pooled r 0.28, regression slope 0.09; the model's within-gene deviations have a third of the measured spread (median sd ratio 0.32, panel e). (c) B3 on the same axes: full spread (ratio 1.08) at r 0.17. (d) Per-gene Pearson against the gene's train sd, running median per predictor: the model rises from 0.1 on the quietest genes to 0.4 on the most variable, the baselines from 0.1 to 0.2. (e) Per-gene sd(pred)/sd(measured); the dotted line at 0.22 is the model's mean Pearson, the ratio that minimizes squared error, so the model is still 1.4 times too spread for squared error while being three times too narrow against the measurements. (f) The strongest-response validation strain (YER068W deletion, sd over genes 0.56): measured values run from -2 to +2.5; neither predictor leaves +-0.5. (g) A median strain (YJL187C deletion): both predictors reproduce the per-gene mean profile with a small offset. (h) The gene with the highest model Pearson (YKL145W, r 0.62): the ordering of strains is right and the predicted range is 0.0 to 0.15 against a measured -1 to 0.5. (i) The median gene (YHL028W, r 0.22).

What this says: the model predicts the direction a gene moves across strains better than the baselines, and it does not predict how far. The large responses (strong strains, extreme values) are absent from every predictor on this partition.

Where the pooled r of 0.45 in panel a comes from (`pooled_decomposition` in the results file): the per-gene train mean alone, repeated for every strain, correlates 0.40 with the measurements across all pairs, because genes differ in their mean log2 ratio (sd of the gene means 0.09 against a pooled sd of 0.22; 16.7% of the pooled variance). The model's predictions correlate 0.77 with that mean profile. Removing the gene means from both sides leaves 0.28 pooled and 0.22 as the mean per-gene Pearson, so about half of the pooled number is the mean profile and the per-gene Pearson is the honest score.
