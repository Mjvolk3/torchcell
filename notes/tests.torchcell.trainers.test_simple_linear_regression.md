---
id: n04cjirgwdes7ny2tptlohj
title: Test_simple_linear_regression
desc: ''
updated: 1790756746302
created: 1790756746302
---

## 2026.09.30 - Phase 11: the linear task with hand-set weights

Fixture: a `SimpleLinearModel` with hand-set weights predicting [0.5, 1.5, -0.5] against `y = [1, 0, 0.25]`. Closed forms: MSE 49/48, MAE 11/12, Pearson -3/sqrt(156), Spearman -0.5, gradient norm sqrt(54)/6; after one step the weights are [0.999, -0.999] and the bias 0.499. Twelve tests: loss selection and values, the optimizer, the forward squeeze of a one-set batch to a 0-d tensor (line 118), `clip_grad_norm` on and off (max_norm 0.25, the pre-clip norm), validation with the `mae` loss, test, the `genetic_interaction_score` target, and the artifact schedule.

Findings: the invalid-loss message runs two sentences together, "is not valid.Currently" (lines 74-75); an unknown target raises `UnboundLocalError` (lines 211-214); the sanity-check box-plot leak (line 202); model artifacts are logged only on plotting epochs and one epoch late (`model-global_step-2` holds `epoch=0-step=1.ckpt`; with plotting every 2 epochs nothing is ever logged).
