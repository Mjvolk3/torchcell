---
id: 5h4s63rtutz5x331mplj3dy
title: Test_int_transformer_cell
desc: ''
updated: 1790413702783
created: 1790413702783
---

## 2026.09.26 - The live CGT trainer under fast_dev_run on CPU

`RegressionTask` with a one-layer CGT, `LogCoshLoss`, every plotting frequency 0 and `device="cpu"`, on two pre-collated batches (`DataLoader(..., batch_size=None)`). Lightning 2.5 `fast_dev_run=True` runs one training and one validation batch with loggers and checkpointing disabled; the test asserts one global step, that the manual-optimization step moved the weights, finite `train/loss` / `val/loss`, the logged learning rate, and that a recorder on `wandb.log` was never called. Also pinned: `_is_scheduled` with 0 and `None` meaning never (the ZeroDivisionError fix in its docstring), `_get_batch_size` counting genotypes not perturbed genes, the three `configure_optimizers` shapes (bare AdamW, CosineAnnealingLR dict, ReduceLROnPlateau monitoring `val/gene_interaction/MSE`), and the loud failure when no loss function is given. Phase 2 of [[plan.test-suite-buildout.2026.09.25]].
