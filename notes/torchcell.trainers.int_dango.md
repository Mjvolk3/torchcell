---
id: i5hnlr4hotlwpwzf5x7897d
title: Int_dango
desc: ''
updated: 1746843364206
created: 1746843364206
---

## 2026.10.02 - One forward per step, NaN targets masked before the loss, honest epoch metrics (issue #616)

### Why the model ran twice

Commit `9f995c828` ("dango works", 2025-05-08) changed `RegressionTask.forward` to return only `{"integrated_embeddings": ...}` from the model's outputs dict, dropping `reconstructions`. The same commit added the `DangoLoss` branch of `_shared_step`, which needs `reconstructions`, so it called `self.model(self.cell_graph, batch)` a second time to get them. It was not for logging: every log read the first call's outputs. Every 005/006 Dango step therefore ran the pretrain GNN, the meta-embedding and the HyperSAGNN twice.

### Changes

- `forward` returns the model's outputs dict unchanged; `_shared_step` takes `reconstructions` from it. One model call per step feeds the loss, both metric spaces, `{stage}/integrated_embeddings_norm` and the plot buffers. Measured on a seeded real `Dango` + `DangoLoss` step (`test_one_forward_step_equals_the_two_forward_protocol_bit_for_bit`): loss, every logged component and the metrics are equal to the two-forward values (float32 `==`), for `LinearUntilUniform(10)` at epoch 4 (loss 0.17869192361831665, captured from the pre-fix code) and `PreThenPost(10)` at epoch 12. Gradients agree to 9.3e-10 (summation order only). A 3-epoch CPU `Trainer.fit` (scratch) recorded 13 forwards for 13 steps.
- The step refuses a prediction count that differs from the target count (`"<stage> batch <i>: the model returned P predictions for T targets"`).
- NaN targets are dropped before the loss with the transformed-metric mask, for every loss; a batch whose targets are all NaN is refused by name.
- `_compute_metrics_safely` is gone (its guard matched messages torchmetrics 1.8.2 never raises). `_log_epoch_metrics` refuses an epoch with no sample by name instead of logging NaN for the monitored `val/gene_interaction/MSE`; one sample logs MSE and RMSE and Pearson NaN. Under `dataloader_profiling` nothing is logged.
- Plot buffers keep ONE gene table (the epoch's last step), never stacked and never indexed by a sample index, so `{train,val}_sample/oversmoothing_integrated_embeddings` is the smoothness of a single table (it used to grow as sqrt(number of collected batches)).
- `grad_accumulation_schedule` other than None and a scheduler `type` other than `ReduceLROnPlateau` are refused at construction; the dead accumulation branches and `current_accumulation_steps` are removed. Every 005/006 Dango config resolves to null and ReduceLROnPlateau.

Tests: [[tests.torchcell.trainers.test_int_dango]].

### Review follow-up (PR #635)

- `experiments/006-kuzmin-tmi/scripts/dango.py` returns after `wandb.finish()` under `execution_mode == "dataloader_profiling"`, since the task logs no validation metrics there and the script read `trainer.callback_metrics["val/gene_interaction/MSE"]` after fit.
- The ReduceLROnPlateau scheduler is built but never stepped: this task uses manual optimization and never calls `lr_schedulers().step()`, so the learning rate stays at its configured value. The docstrings now say so; the behavior is unchanged here and tracked as its own issue. `val/gene_interaction/MSE` is the checkpoint callbacks' monitored key.
