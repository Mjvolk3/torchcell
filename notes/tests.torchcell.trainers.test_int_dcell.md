---
id: 5tyc681jaqeq6lmbtyq8uu2
title: Test_int_dcell
desc: ''
updated: 1790820782092
created: 1790820782092
---

## 2026.09.30 - Paired shapes into DCellLoss (issues #554, #578)

New. `RegressionTask._shared_step` with a weight-free stand-in (root [0, -1, 1], GO:1 [0, 1, 2], GO:2 [0, 2, 1], y = [1, 0, 0.5]) returns 1.7 under `"sum"` and 1.225 under `"mean"`, reachable only with paired `[B]` shapes; the old broadcast path gave 1.075 on this fixture. The returned predictions and targets stay `[B, 1]` for the metrics. Issues #554, #578.

## 2026.10.01 - Phase 19: the rest of RegressionTask

40 cases (2 before; 46 after the 2026.10.02 audit round). Same weight-free stand-in (w = 1, root [0, -1, 1], GO:1 [0, 1, 2], GO:2 [0, 2, 1], y = [1, 0, 0.5]); `self.log`, `optimizers`, `manual_backward` and `clip_grad_norm_` replaced by recorders; no `Trainer.fit`.

- `_shared_step` logs `<stage>/primary_loss` 0.75, `auxiliary_loss` 9.5 / 3, `weighted_auxiliary_loss` 0.95, then `<stage>/loss` 1.7, each with `batch_size=3` (the genotype count, `predictions.size(0)`; issue 596 / PR 587 concern trainers that logged a node count, this one does not) and `sync_dist=True`.
- Metrics receive the root against y: MSE 0.75, RMSE 0.8660254, Pearson 0.5 (centered r = [0, -1, 1], centered y = [0.5, -0.5, 0]: 0.5 / sqrt(2 * 0.5)). With an affine inverse transform 2x + 1 and `phenotype_values_original` [3, 1, 2] the original-scale metrics get [1, -1, 3] (MSE 3) while loss and transformed metrics stay on the transformed scale.
- `training_step`: backward of 1.7, then clipping of `task.parameters()` at the configured max norm, `step`, `zero_grad`; w.grad = 2.7 (root 2/3 *1.5 = 1.0, auxiliary 0.3* 2/3 * (4 + 4.5) = 1.7); `learning_rate` 0.01 logged with `batch_size=3`. With the counter at 2: losses 0.85, one step after two backwards.
- Sample collection per stage (train ceiling subsample by seeded `randperm`, val every plot epoch, test always), `_plot_samples` hand-off (concatenation, ceiling subsample, smoothness sqrt(8) of the centered latents, box plot of column 0, NaN target skips the box plot), the three epoch-end hooks (original then transformed collection, reset, one plot, cleared samples; no plot and no clear during sanity checking), the epoch-start clears, `_compute_metrics_safely` skipping exactly its two messages.
- `configure_optimizers`: AdamW with `learning_rate` renamed to `lr`, exactly `task.parameters()`, ReduceLROnPlateau with the config minus `type`, monitor `val/gene_interaction/MSE` per epoch; the dummy batch (template with column 3 zeroed, ptr [0, 4]; or the one-row fallback); a failing dummy forward adds `model.dummy`; a parameter-free model gets a task-level `dummy`.

Findings (each pinned, with the reproduction in the test):

- int_dcell.py:664-675: the dummy forward feeds ONE sample to DCell/DCellOpt, whose BatchNorm1d refuses it in training mode; the exception is swallowed and every DCell training run registers `model.dummy`, so a checkpoint carries `model.dummy` and `load_from_checkpoint` with a freshly built model (the resume path of the 006 and 005 dcell.py scripts) fails strict loading with `Unexpected key(s) in state_dict: "model.dummy"`; the failed forward also increments term 1's `num_batches_tracked`.
- (Corrected 2026.10.02, see below.) The plateau scheduler is never stepped: the task uses manual optimization and Lightning skips the epoch-end scheduler update, so every DCell run trained at a constant learning rate. The `lr == min_lr` fact in the 006 configs and the 005 main config is true but moot.
- (Corrected 2026.10.02.) The 005 `_cpu` and `_maxing_resources` configs write `threshold: 1e -4`, a string that reaches the scheduler unchanged; it would raise at a first `step`, but nothing ever steps it, so those runs train without error.
- int_dcell.py:257-262, 288-293 vs 221-223: NaN targets are masked for both metric updates but reach `DCellLoss`, so one NaN label makes the step loss NaN. Not measured whether any 005/006 label is NaN.
- int_dcell.py:383, 388, 59: `grad_accumulation_schedule` is only tested for None; its values are never read (the divisor stays 1). All DCell configs set it to null.
- int_dcell.py:701-704: `lr_scheduler_config["type"]` is dropped; ReduceLROnPlateau is always built.
- int_dcell.py:206, 318-376, 489-491: the trainer collects `subsystem_outputs["root"]`, a key neither DCell nor DCellOpt returns, so latents and the oversmoothing log never occur with the real model.
- int_dcell.py:131-134: `hasattr(batch, "gene")` is False for a HeteroData, so the device probe is dead and the batch is moved on every call (a no-op on one device).

Coverage of `torchcell/trainers/int_dcell.py` from this file: 31% -> 89%. Left uncovered: 0-dim prediction and target reshapes, the train-collection latent branch for batches under the ceiling, and the dead device probe.

## 2026.10.02 - Audit round 2 corrections

46 cases. Changes from the independent audit:

- D2 restated. `RegressionTask` sets `automatic_optimization = False` (int_dcell.py:100) and never calls a scheduler's `step`. Lightning 2.5.5's `_update_learning_rates` (training_epoch_loop.py:454-468) returns early under manual optimization, and the scheduler config comes back with `reduce_on_plateau` False and `monitor` None. `test_the_plateau_scheduler_of_a_dcell_run_is_never_stepped` drives four epochs of `training_step`, `validation_step`, `on_validation_epoch_end`, `on_train_epoch_end` and Lightning's own epoch-end update for each shipped config. It records no `ReduceLROnPlateau.step` call, and the learning rate stays at 1e-3. A control run with `automatic_optimization` switched on steps the scheduler twice and lowers the rate to 2e-4. `lr == min_lr` remains a secondary assertion, stated as moot. Reach: every DCell run of 006 and 005 `dcell.py`. Not checked against the wandb run configs.
- D3 restated. `threshold: "1e -4"` survives into the scheduler as a string (`scheduler.threshold == "1e -4"`), and the never-stepped assertion holds. Reach: latent (005 `_cpu`, `_maxing_resources`).
- D1 evidence and reach. `experiments/006-kuzmin-tmi/scripts/dcell_training_gpu_profile.py:139-145` deletes a `dummy` key from real checkpoints and attributes it to an older model. The key comes from this dummy forward. The test now runs for both `DCell` (what every 005/006 config builds) and `DCellOpt`. The resume path (006 `dcell.py:410`, 005 `dcell.py:363`) is latent, because all seven DCell configs set `checkpoint_path: null` (grep of `experiments/00{5,6}*/conf/dcell*.yaml`).
- Reach of the other findings (from the audit table): D4 NaN masked from metrics but not the loss, not measured whether any Kuzmin TMI label is NaN; D5 accumulation values never read, latent (all seven configs null); D6 scheduler `type` ignored, latent and moot; D7 `subsystem_outputs` never returned, now pinned for DCell and DCellOpt, so the oversmoothing log never fires in any run; D8 device probe dead, performance only.
- New pins:
  - Each metric scale masks by its own target's NaNs: transformed rows 1 and 2, original rows 0 and 1.
  - A batch of one reaches the inverse transform as a 0-dim tensor (int_dcell.py:269).
  - A non-tensor inverse result is ignored silently (a Finding at :279): original-scale MSE 14/3 instead of 3. Latent, since the scripts pass `inverse_transform=None`.
  - A multi-column original target is cut to column 0.
- Rewrites: the train-ceiling test now uses ceiling 4 and two batches of 3 (4 rows; the second chunk is the `randperm(3)[:1]` row). The sanity-check hook asserts the six exact names and values. The `configure_optimizers` docstring says the monitor is ignored under manual optimization. Every global reseed now runs inside `torch.random.fork_rng()`.
- Mutants killed: original mask taken from the transformed target, `remaining = ceiling`, column 1 instead of 0, non-tensor inverse used, `squeeze(1)` before the inverse.

## 2026.10.02 - The checkpoint test loads the state dict directly

`test_dcell_in_training_mode_always_gets_a_dummy_that_breaks_checkpoint_reload` failed on the CI runner for both models: its newer Lightning unpickles a checkpoint with `weights_only=True` and stops on the hyperparameters (`Unsupported global: torch_geometric.data.storage.BaseStorage`) before it reaches the state dict. The test now asserts `model.dummy` is in the saved state dict and that `load_state_dict` on a freshly built task raises `Unexpected key(s) in state_dict: "model.dummy"`, which is the strict load `load_from_checkpoint` ends with, on every Lightning version. The CI observation is a second obstacle to resuming a DCell run on a current Lightning and is recorded on issue 615.
