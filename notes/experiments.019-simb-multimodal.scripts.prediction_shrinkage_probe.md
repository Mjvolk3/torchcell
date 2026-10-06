---
id: lazcscs6s419vcqr4mczdd4
title: Prediction_shrinkage_probe
desc: ''
updated: 1791270293463
created: 1791270293463
---

## 2026.10.06 - One scalar fixes the squared error on expression; the proteome head is under-dispersed

Question: the validation loss bottoms out early and rises while the per-feature Pearson keeps climbing, and at the Pearson peak no run beats the per-gene mean on squared error. Is that a magnitude problem a single scalar can fix? A predictor with correlation r minimizes squared error at sd(pred) / sd(target) = r.

Method: the six saved validation dumps (`$DATA_ROOT/val-predictions/`, best-validation checkpoints of v13 expression and v14 proteome). Validation strains cut in two halves (seed 0); per-gene intercept and ONE global shrink factor c fit on one half, normalized squared error scored on the other, halves swapped and averaged. No GPU, no retraining. Output `experiments/019-simb-multimodal/results/prediction_shrinkage_probe.json`.

| checkpoint | val strains | Pearson per feature | sd(pred) / sd(target) | fitted c | NMSE, per-gene mean | NMSE, model as is | NMSE, shrunk by c |
|---|---|---|---|---|---|---|---|
| expression V_ref split 2 | 155 | 0.144 | 0.317 | 0.455 | 1.030 | 1.052 | 1.016 |
| expression V_ref split 0 | 155 | 0.225 | 0.319 | 0.697 | 1.036 | 0.994 | 0.984 |
| expression V_concat split 0 | 155 | 0.192 | 0.343 | 0.518 | 1.036 | 1.029 | 0.997 |
| proteome P_concat split 0 | 448 | 0.113 | 0.062 | 1.750 | 1.010 | 0.997 | 1.002 |
| proteome P_ref split 0 | 448 | 0.116 | 0.085 | 1.380 | 1.010 | 0.995 | 0.997 |
| proteome P_ref split 2 | 448 | 0.128 | 0.063 | 1.885 | 1.008 | 0.992 | 0.987 |

The per-gene mean reads above 1 because it is estimated on half the validation strains (about 78 for expression).

- **Expression: the predictions are 1.4 to 2.2 times too spread for their correlation**, the fitted scalar is 0.46 to 0.70, and after shrinking all three checkpoints beat the per-gene mean (by 0.014, 0.052 and 0.039). On split 2 the model as is was WORSE than the per-gene mean. The shrunk error matches what the correlation allows: for split 0, 1.036 times (1 - 0.225 squared) is 0.983 against a measured 0.984. So on these three checkpoints the squared-error deficit is a scale problem and the ordering is intact.
- **The same number says how little is explained: r squared is 2 to 5 percent of per-gene variance across held-out strains.**
- **Proteome: the opposite sign.** Spread ratio 0.06 to 0.09 against a Pearson of 0.11 to 0.13, fitted c 1.4 to 1.9: the head is under-dispersed at its best-validation checkpoint (it peaks early, in the 200 to 400 window), and the model as is already beats the per-gene mean slightly.
- Caveats: three checkpoints per modality on two split seeds, each the best-validation checkpoint (a selected epoch); the dump routine calls the model without the observed-label input, which for these masked-objective runs is the nothing-revealed path the logged k=0 metric also uses, not verified here. Hypothesis (untested): the over-dispersion is magnitude overfitting (train Pearson 0.6 against 0.1 to 0.2 held out), which stronger regularization would reduce at training time; weight decay in this lineage is 1e-8.
