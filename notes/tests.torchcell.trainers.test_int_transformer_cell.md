---
id: 5h4s63rtutz5x331mplj3dy
title: Test_int_transformer_cell
desc: ''
updated: 1790413702783
created: 1790413702783
---

## 2026.09.26 - The live CGT trainer under fast_dev_run on CPU

`RegressionTask` with a one-layer CGT, `LogCoshLoss`, every plotting frequency 0 and `device="cpu"`, on two pre-collated batches (`DataLoader(..., batch_size=None)`). Lightning 2.5 `fast_dev_run=True` runs one training and one validation batch with loggers and checkpointing disabled; the test asserts one global step, that the manual-optimization step moved the weights, finite `train/loss` / `val/loss`, the logged learning rate, and that a recorder on `wandb.log` was never called. Also pinned: `_is_scheduled` with 0 and `None` meaning never (the ZeroDivisionError fix in its docstring), `_get_batch_size` counting genotypes not perturbed genes, the three `configure_optimizers` shapes (bare AdamW, CosineAnnealingLR dict, ReduceLROnPlateau monitoring `val/gene_interaction/MSE`), and the loud failure when no loss function is given. Phase 2 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Phase 15: the remaining branches on a scripted stand-in

Six to twenty-three tests; alone 31 to 74 percent, with the `_methods` and `_schedule` files 87 to 95. Under `fast_dev_run` with a scripted stand-in: exact train and val losses and metrics, the first AdamW step (scale 1.0099), the cosine schedule moving lr from 1e-2 to 5e-3; plot order val then train with the pooled latent (2.2, 2.4); `trainer.test` giving exact test metrics and one plot; 0-dim predictions and targets lifted to [1, 1]; batch-size and device fallbacks; frozen parameters left out of the dummy losses; residual ratios and degree bias accumulating with edge recovery off; all-NaN targets updating no metric; the sanity check never plotting; the oversmoothing log sqrt(2) and the all-NaN box plot skipped; the CUDA cache emptied every 50 batches; optimizer config renaming `learning_rate` to `lr`, an unknown optimizer raising `AttributeError`, the accumulation schedule, empty diagnostic accumulators drawing no plot.

Findings: the default ReduceLROnPlateau scheduler cannot finish one epoch, the step runs and the epoch end raises `AssertionError` (lines 1534/1567 with 1374); the PointDistGraphReg branch drops multi-element loss components the generic branch logs as `_0`, `_1` (862-878 versus 915-923); an inverse transform returning a non-tensor is silently ignored so original-scale metrics are computed on transformed predictions, MSE 1285 (1017); the precision plot is called with empty inner dicts (169).
