---
id: f8d3288jv3wsp7jzwbcykpv
title: Test_neo_regression
desc: ''
updated: 1790413710249
created: 1790413710249
---

## 2026.09.26 - neo_regression on a two-layer toy model

`MSEListMLELoss(alpha=0.5)` on [1, 2, 3] vs [3, 2, 1] is 8/3 + 0.5 * 4.2228178933 = 4.7780756133 and `ListMLEMetric` is the sample-weighted mean over updates (3.9138242176 on one 1-list and one 2-list batch), both by hand. The task's loss-name switch, Adam configuration and `forward = top(mean-pool(main(x)))` are pinned exactly. The fit test runs one train and one validation batch with `max_epochs=1` (not `fast_dev_run`, so `ModelCheckpoint` exists; its `best_model_path` is still empty when the module hook runs, so the artifact branch is inert) and records the two `wandb.log` payloads `on_validation_epoch_end` emits at epoch 0, the prediction-stats table and the box-plot image, mocking only `wandb.Table`, `wandb.Image` and `wandb.log`. Phase 2 of [[plan.test-suite-buildout.2026.09.25]].
