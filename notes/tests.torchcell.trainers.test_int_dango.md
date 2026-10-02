---
id: i42wrxuz5tfsklv1okr3675
title: Test_int_dango
desc: ''
updated: 1790916713875
created: 1790916713875
---

## 2026.10.01 - Dango trainer: exact loss, logs, metrics and findings

Paired with [[torchcell.trainers.int_dango]] (`torchcell/trainers/int_dango.py`), the `RegressionTask` used by the 005/006 Dango scripts.

### Fixture

- `_Scripted` stand-in returns `w * [1, 3]` (scalar parameter `w` = 1), `integrated_embeddings = w * [[3, 4], [0, 1], [6, 8]]` (row norms 5, 1, 10) and one reconstruction per network; it holds an `unused` parameter and records each call. Targets [2, 5].
- Three-gene cell graph, `string12_0_neighborhood` 0->1, 1->2 and `string12_0_fusion` 2->0. Real `DangoLoss` with lambda 0.1 / 1.0 and `LinearUntilUniform(10)` (the 006 production schedule).
- `self.log` is a recorder; a bare `lightning.Trainer` is attached only to set `current_epoch`, `sanity_checking` and `default_root_dir`. Optimizer, `manual_backward` and `clip_grad_norm_` are stubbed for `training_step`; `Visualization`, the box plot and `wandb` are stubbed for plots.

### Expected values

- Reconstruction loss: neighborhood (1 + 0.1 * 1.25) / 9 = 0.125, fusion 1 / 9, mean 0.1180556 (adjacency row = source; the transposed reading would give 0.1791667). Interaction loss (log cosh 1 + log cosh 2) / 2 = 0.8793917.
- Step loss alpha *0.1180556 + (1 - alpha)* 0.8793917 with alpha 1.0, 0.8, 0.5, 0.5 at epochs 0, 4, 10, 12. Logged in order: the five loss components, `<stage>/loss`, `<stage>/integrated_embeddings_norm` (16 / 3); batch_size is `predictions.size(0)`, the number of genotypes (2).
- Metrics: both spaces see [1, 3] vs [2, 5] without a transform (MSE 2.5, Pearson 1). With an inverse transform 2x + 1 on `gene_interaction`, the original-unit metrics see [3, 7] vs `phenotype_values_original` [4, 10] (MSE 5) and the transformed ones see the raw pairs (MSE 2.5).
- `_ensure_no_unused_params_loss` is a 0.0 tensor that reaches every gradless parameter, and the int 0 when all have gradients. `training_step` logs `learning_rate` with batch_size = len(phenotype_values).
- `configure_optimizers` on the 006 config: AdamW(lr 1e-5, weight_decay 1e-6), ReduceLROnPlateau(min, 0.2, 3, 1e-4, rel, 2, 1e-9, 1e-10), monitor `val/gene_interaction/MSE`, interval epoch, frequency 1.

### Findings

- With `DangoLoss` the model is run twice per step (int_dango.py:214 and :262); values agree because Dango has no dropout, compute doubles.
- NaN targets are masked for both metric updates (int_dango.py:351-356, 381-386) but not for the loss (int_dango.py:284-290); one NaN target makes the step loss NaN. Not shown reachable in the 006 data.
- `grad_accumulation_schedule` is only tested for None (int_dango.py:450, 455); `current_accumulation_steps` stays 1. All 005/006 Dango configs set it to null.
- `lr_scheduler_config["type"]` is dropped (int_dango.py:660-663); any type builds ReduceLROnPlateau. All 005/006 configs say ReduceLROnPlateau.
- `_compute_metrics_safely`'s guard (int_dango.py:489-496) is dead with the installed torchmetrics: an empty epoch returns NaN MSE/RMSE/Pearson, a one-sample epoch NaN Pearson.
- Each training step on a plot epoch appends the whole [num_genes, H] embedding table (int_dango.py:409-424; in the ceiling branch it is indexed by sample position, picking gene rows), so `train_sample/oversmoothing_integrated_embeddings` is sqrt(number of collected batches) times the table's smoothness.

### Left uncovered

The defensive "latents key missing" re-creations (int_dango.py:405-408, 418-421, 433-434) and the 2-column / 0-dim branches of `phenotype_values_original` (int_dango.py:245, 250), which no reachable input exercises.

## 2026.10.02 - Audit 1 corrections and added pins

### Finding reach, corrected

- Model run twice per step (int_dango.py:214 and :262): this applies to every 005/006 Dango run, including the manuscript runs behind `paper/nature-biotech/sections/tab-dango-full-runs.tex` and `tab-dango-string-versions.tex`. Reported values are unchanged because `torchcell/models/dango.py` has no dropout (the second forward is identical). Hypothesis (untested): roughly doubled compute per step lowered the Epochs column of the time-capped runs; no run was repeated with the fix to measure it.
- NaN targets: the interaction loss is NaN for any NaN target, but whether the step loss is NaN depends on the schedule. `LinearUntilUniform` (006): NaN at every epoch (0 * NaN at epoch 0). `PreThenPost(10)` (005 default) before its transition: the step loss is the finite reconstruction loss (0.1180556 on the fixture), only `train/interaction_loss` is NaN and `weighted_interaction_loss` is 0; after the transition the step loss is NaN.
- `grad_accumulation_schedule`: no Dango config sets a non-null value. Eleven 005/006 configs set it to null; five 006 configs omit the key (`dango_kuzmin2018_tmi_string12_0_profile.yaml`, `_string12_0_dataloader_profile.yaml`, `_string12_0_086.yaml`, `_string12_0_086_dataloader.yaml`, `_string12_0_086_model.yaml`), and all five inherit it as null through Hydra `defaults` from `dango_kuzmin2018_tmi_string12_0.yaml` (directly or through `_086`).
- Empty-genotype finding (model side): the mid-batch exception is `RuntimeError`, as the model test pins.

### Added pins

- Epoch-end hooks assert all six logged values (MSE 2.5, RMSE sqrt(2.5), Pearson 1.0, original and transformed spaces) for train and val; the model-profiling step asserts `train/loss` 2.5 and `train/integrated_embeddings_norm` 16 / 3.
- `PreThenPost(10)` through `_shared_step`: epochs 0 and 9 give 0.1180556 (alpha 1, weighted interaction 0), epoch 10 gives 0.8793917 (alpha 0, weighted reconstruction 0).
- Prediction/target count mismatch (2 predictions, 3 targets, the trailing empty genotype case): with `DangoLoss` the step raises `RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0` before any metric update; with a shape-blind loss it reaches int_dango.py:355 and raises `IndexError: The shape of the mask [3, 1] at index 0 does not match the shape of the indexed tensor [2, 1] at index 0`.
- `learning_rate` batch size: it is len(phenotype_values) while the step logs use predictions.size(0). They differ only with a count mismatch; with all-NaN targets (both metric updates skipped) and a shape-blind loss the step completes and logs batch_size 2 for `train/loss` and 3 for `learning_rate`.
- Every `torch.manual_seed` runs inside `torch.random.fork_rng()` through an autouse fixture.
