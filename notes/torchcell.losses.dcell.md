---
id: 5tez5nfisvoq3d66slbqinr
title: Dcell
desc: ''
updated: 1790818895649
created: 1790818895649
---

## 2026.09.30 - Paper sum, declared root key, no broadcast (issues #554, #578)

Issue #554, PR #574. Previous behavior: `DCellLoss.forward` reduced the non-root subsystem MSEs with `torch.stack(...).mean()` before multiplying by alpha, skipped the root by `"GO:ROOT"` or by `torch.equal` with the predictions, and broadcast any shape mismatch silently. Ma et al. 2018 (mirror `torchcell-library/maUsingDeepLearning2018/paper.md` line 196, sha256 `ac837bc358ea4969a72789e66e31380bfdcbd7b8aea98dec47b108ee21d2070c`) define the objective as `(1/N) sum_i (Loss(Linear(O_i^(r)), y_i) + alpha * sum_{t != r} Loss(Linear(O_i^(t)), y_i)) + lambda ||W||_2`, and line 199 gives "the parameter $\alpha$ $_ { ( = 0 . 3 ) }$ balances these two contributions": a SUM over the non-root subsystems. The `lambda ||W||_2` term is the optimizer's `weight_decay` (1e-6 in the configs); the mirror text gives no lambda value.

Fix:

- `aux_reduction: Literal["sum", "mean"]` is a REQUIRED keyword-only argument (two regimes have real runs, so no value is assumed); any other value raises `ValueError`. `DCellRegressionTask` and `DCellRegressionSlimTask` take it too, and every caller states it (the profile scripts use `"mean"` because their committed CSVs were measured under the old loss).
- The root is skipped by its DECLARED key: `DCell` and `DCellOpt` put `outputs["root_key"] = "GO:<root index>"`, and the loss skips that key and `"GO:ROOT"`. `linear_outputs` without `root_key` raises. Identity was not enough: `DCellOpt` builds the prediction and `GO:<root>` by two indexing calls, so an identity skip counted the root (fixture: 0.8298076 against the paper value 0.6840844).
- A prediction or counted head whose shape differs from the target raises `ValueError`; `int_dcell` now passes `[B]` predictions and targets.

**Which runs used which loss.** Every auxiliary-on DCell run in 005 and 006 went through `torchcell.trainers.int_dcell.RegressionTask`, which passed `[B, 1]` predictions and targets against `[B]` heads. So each auxiliary MSE broadcast over a `[B, B]` grid (each term equal to var(o) + var(y) + (mean o - mean y)^2), the old `torch.equal` root test was always False, and the root head `GO:<root index>` was counted as an auxiliary term next to the mean reduction. Neither `aux_reduction="mean"` nor `"sum"` reproduces that loss. On the reviewer's live-shape fixture the old loss measured 0.655583, the new mean 0.673322 and the new sum 0.871484 (PR #574 review, not rerun here); on the `test_int_dcell` fixture the old loss is 1.075 against 1.225 (mean) and 1.7 (sum), computed by script on 2026-09-30. Issue #578 records the affected runs.

Auxiliary-on runs, from the frozen run tables in `experiments/006-kuzmin-tmi/results/dcell_training/`:

- 005: `kuzmin2018_runs.csv` lists 23 auxiliary-on runs of 31, created 2025-05-16 to 2025-05-20. `experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi.yaml` carried `use_auxiliary_losses: true` from fa9a4faf4 (2025-05-12) to 6898b4611 (2025-05-23), so the current `false` in that file does not describe them.
- 006: `runs.csv` lists four auxiliary-on runs, all slurm job 1922684 (`eni948by`, `lrudrans`, `fy516knk`, `c7248f86`). The slurm log `experiments/006-kuzmin-tmi/slurm/output/006-001-dcell_1922684.out` carries the PyTorch target-size broadcast warning (line 285, 4,000 occurrences per the review).

Configs now: `aux_reduction: "mean"` in the three auxiliary-on historical configs (005 `_cpu`, 005 `_maxing_resources`, 006 `_mmli_001`), with a comment that neither value reproduces their runs; `"sum"` in the three auxiliary-off configs, where it is inert and is the paper value if auxiliaries are switched on; and the new `experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi_mmli_002.yaml`, a copy of `_mmli_001` with `"sum"`, for the paper-objective run.

Evidence: [[tests.torchcell.losses.test_losses_dcell]], [[tests.torchcell.models.test_dcell_opt]], [[tests.torchcell.trainers.test_int_dcell]], [[tests.torchcell.models.test_dcell]] (`main` reads the key and logs the closed-form loss of each reduction).
