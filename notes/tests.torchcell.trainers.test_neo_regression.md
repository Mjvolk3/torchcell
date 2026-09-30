---
id: f8d3288jv3wsp7jzwbcykpv
title: Test_neo_regression
desc: ''
updated: 1790413710249
created: 1790413710249
---

## 2026.09.26 - neo_regression on a two-layer toy model

`MSEListMLELoss(alpha=0.5)` on [1, 2, 3] vs [3, 2, 1] is 8/3 + 0.5 * 4.2228178933 = 4.7780756133 and `ListMLEMetric` is the sample-weighted mean over updates (3.9138242176 on one 1-list and one 2-list batch), both by hand. The task's loss-name switch, Adam configuration and `forward = top(mean-pool(main(x)))` are pinned exactly. The fit test runs one train and one validation batch with `max_epochs=1` (not `fast_dev_run`, so `ModelCheckpoint` exists; its `best_model_path` is still empty when the module hook runs, so the artifact branch is inert) and records the two `wandb.log` payloads `on_validation_epoch_end` emits at epoch 0, the prediction-stats table and the box-plot image, mocking only `wandb.Table`, `wandb.Image` and `wandb.log`. Phase 2 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Phase 12: the remaining branches on a deterministic model

Seven to seventeen tests, 79 to 99 percent. The test stage in closed form (MSE 0.125625, Spearman sqrt(0.9), Pearson 0.9478749590) with the binned table rows; one step (`train/*` at 1.0, Adam moves both weights by +0.1, `val/*` MSE 0.565); clipping called once with max_norm 0.25 on a norm of sqrt(13); the interaction-target box plot for validation and test; the epoch-1 artifact `model-global_step-2` pointing at `epoch=0-step=1.ckpt`; the exact Adam configuration.

Findings: test-stage keys are `test_pearson`/`test_spearman`, not the slash form train and validation use (lines 355, 361); `enable_checkpointing=False` raises on `best_model_path` (324-325), so the value tests turn checkpointing on; `list_mle` on `[B, 1]` predictions is exactly 0, so the weights never move under it (131-132); an unknown target raises `UnboundLocalError` (312-316); an epoch that skips the box plot keeps its stored predictions (302-305); a bin with a single prediction reports StdDev NaN.
