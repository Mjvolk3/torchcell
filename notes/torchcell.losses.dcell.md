---
id: 5tez5nfisvoq3d66slbqinr
title: Dcell
desc: ''
updated: 1790818895649
created: 1790818895649
---

## 2026.09.30 - Auxiliary reduction follows the paper (sum), root skipped by key

Issue #554. Previous behavior: `DCellLoss.forward` reduced the non-root subsystem MSEs with `torch.stack(...).mean()` before multiplying by alpha. Ma et al. 2018 (mirror `torchcell-library/maUsingDeepLearning2018/paper.md` line 196, sha256 `ac837bc358ea4969a72789e66e31380bfdcbd7b8aea98dec47b108ee21d2070c`) define the objective as `Loss(Linear(O^(r)), y) + alpha * sum_{t != r} Loss(Linear(O^(t)), y)`, and line 199 gives "the parameter $\alpha$ $_ { ( = 0 . 3 ) }$ balances these two contributions": a SUM over the non-root subsystems. Under the mean the effective alpha was 0.3 / (T - 1).

Fix: a constructor argument `aux_reduction: Literal["sum", "mean"] = "sum"`; any other value raises `ValueError("aux_reduction must be 'sum' or 'mean', got ...")`. Also, a non-root head whose tensor happened to equal the root was dropped by a `torch.equal` test. The root is now skipped by key only: `"GO:ROOT"` and the key bound to the same tensor object (`torchcell.models.dcell.DCell` stores the root under `GO:<root index>` and aliases it as `GO:ROOT`, dcell.py 347-350).

Which runs used which reduction. Every DCell run in experiments 005 and 006 to date was built with `DCellLoss(alpha=0.3, use_auxiliary_losses=...)` before this fix, so every one of them that had auxiliary losses on trained with the MEAN, an effective alpha of 0.3 / (T - 1), where T - 1 is the number of non-root GO terms in the filtered hierarchy (logged per run as `model/num_go_terms`); for the GO hierarchy that is near zero, so those runs were close to root-only training. By config: auxiliary losses ON (mean applied) in `experiments/005-kuzmin2018-tmi/conf/dcell_kuzmin2018_tmi_cpu.yaml`, `dcell_kuzmin2018_tmi_maxing_resources.yaml` and `experiments/006-kuzmin-tmi/conf/dcell_kuzmin2018_tmi_mmli_001.yaml`; auxiliary losses OFF (reduction irrelevant, unaffected) in `005/conf/dcell_kuzmin2018_tmi.yaml`, `006/conf/dcell_kuzmin2018_tmi.yaml` and `006/conf/dcell_kuzmin2018_tmi_mmli_000.yaml`. A run launched with a command-line override of `use_auxiliary_losses` follows the override, not the file. The default now follows the paper (`"sum"`); any comparison to the earlier runs must pass `aux_reduction="mean"`.

The 005/006 DCell configs expose the loss under `regression_task.dcell_loss`; each now carries `aux_reduction: "sum"` and the 005/006 `scripts/dcell.py` plus the `main` of `torchcell/models/dcell.py` and `torchcell/models/dcell_opt.py` pass it through. The analysis script `experiments/006-kuzmin-tmi/scripts/dcell_training_wandb.py` does not yet tabulate the reduction; runs without the key were mean.

Evidence: [[tests.torchcell.losses.test_losses_dcell]] (`test_default_is_the_paper_sum_over_non_root_subsystems` 3.7, `test_mean_reduction_reproduces_the_pre_fix_runs` 3.1, `test_root_is_skipped_by_key_and_alias_only` 3.25, `test_unknown_reduction_is_refused`); trainer fixture 1.7 (sum) vs 1.225 (mean) in [[tests.torchcell.trainers.test_dcell_regression]].
