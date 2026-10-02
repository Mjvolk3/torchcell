---
id: 4qzudd3h11gnxjxln5xrycs
title: Test_int_hetero_cell_diffusion
desc: ''
updated: 1790917186474
created: 1790917186474
---

## 2026.10.01 - Phase 19: where DiffusionRegressionTask differs

New file, split from [[tests.torchcell.trainers.test_int_hetero_cell]] (which holds every behavior the two tasks share, parametrized over both) and reusing its stand-ins: p = [1, 3, -2], y = [2, 5, -1], `z_p = [[3, 4], [0, 0], [6, 8]]`, squared error 2.0.

Expected values:

- Train stage with the real `DiffusionLoss(model, lambda_diffusion=2)` and a stand-in `compute_diffusion_loss` of 0.3: the loss asks for `(targets [3, 1], z_p, t_mode="full")`, total 0.6; logs `train/diffusion_loss` 0.3, `train/total_loss` 0.6, `train/loss` 0.6, `train/z_p_norm` 5; the detached 0.6 is kept for `train/avg_diffusion_loss`.
- Validation and test: `F.mse_loss` = 2.0 in model units, `loss_func` never called, `{stage}/inference_mse` logged as a float; only validation keeps it for `val/avg_inference_mse`.
- Component logger: one-element tensor as its value, [1, 2] as its mean 1.5, empty tensor as NaN, plain numbers dropped; a missing `z_p` is passed as `None`.
- Epoch ends: buffered 1.0, 2.0, 4.5 log their mean 2.5 and the buffer is cleared.

Findings (source `torchcell/trainers/int_hetero_cell.py`):

- With the real `GeneInteractionDiff` (diffusion decoder) in training mode the predictions are zeros (`hetero_cell_bipartite_dango_diff_gi.py` lines 242-254) and the train metrics score them: targets [0.5, -0.25, 1.0] give train MSE 0.4375 and Pearson NaN, independent of training.
- The two `_shared_step` bodies drifted: on the train stage the diffusion task averages vector components, drops numeric ones and ignores `graph_reg_loss` (2.0 vs 2.25 for `RegressionTask`); on validation it uses `F.mse_loss` instead of `loss_func`. A missing loss is a bare `assert` (lost under `python -O`) instead of `ValueError`.

## 2026.10.02 - Audit corrections

- `test_two_shared_steps_drifted_apart[val]` now asserts the exact `RegressionTask` side: loss 2.25 and `{"val/vec_0": 1.0, "val/vec_1": 2.0, "val/n": 3.0, "val/graph_reg_loss": 0.25, "val/loss": 2.25, "val/z_p_norm": 5.0}`.
- The real-model test reseeds inside `torch.random.fork_rng()`, so the global RNG is left unchanged.
- Reach (audit 2): the zero-prediction train metrics (H9) are reached by 006 `hetero_cell_bipartite_dango_diff_gi.yaml` via `hetero_cell_bipartite_dango_diff_gi.py`; the validation and test `F.mse_loss` path is reached by the same config; the other drift items are latent.

## 2026.10.02 - Findings retired by the issue #614 fix

- `test_real_diffusion_model_train_placeholders_are_not_scored` replaces the zero-placeholder pin: with the real `GeneInteractionDiff` in training mode two train batches update neither train collection and keep no plot sample, one INFO record announces the skip, and in eval mode the sampled predictions (not all zero) are exactly what the val collection receives.
- `test_both_tasks_share_one_step_except_the_stage_loss` replaces the drift pin: on train both tasks give 2.25 and the same logs; on val the diffusion task logs `val/inference_mse` 2.0 and adds the same graph term (2.25).
- `test_diffusion_train_components_log_like_every_other_loss`: a [1, 2] tensor logs `vec_0`, `vec_1`, a number logs as is, an unnamed loss gets two arguments.
- `test_diffusion_train_requires_a_loss_only_on_the_train_stage`: `ValueError("No loss function provided")`, no bare `assert`.
- New: `test_diffusion_loss_on_a_model_without_z_p_is_refused_by_name`, `test_diffusion_train_epoch_end_logs_no_train_metric_and_steps_the_scheduler` (only `train/avg_diffusion_loss` 2.0 is logged; the scheduler is stepped once).

## 2026.10.02 - Review fixes on PR #637

- New `test_a_non_diffusion_loss_is_refused_at_construction` (`LogCoshLoss`, `nn.MSELoss`, full message).
- The component test now uses the `DiffusionLoss`-typed stand-in, called `(pred, target, z_p)`.
