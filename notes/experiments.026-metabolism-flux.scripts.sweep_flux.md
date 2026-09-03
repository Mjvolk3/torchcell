---
id: 24g5f88xxcq00v0lslgwomh
title: Sweep_flux
desc: ''
updated: 1788399396476
created: 1788399396476
---

## 2026.09.02 - Why the overnight window went to replication and a null, not to more epochs

The obvious use of a 12 h, 4 GPU window is a longer training budget. Reading the banked
20-epoch runs in `experiments/026-metabolism-flux/results/flux_arms_gpu*.json` says that
is the wrong experiment, and the reason is three measurements rather than an opinion.

**Validation Pearson peaks early and then decays.** Across all 18 completed (arm, seed)
runs the peak epoch has median 4.5 and maximum 18, out of 20. The mean of the final five
epochs is at or below zero for 13 of the 18. A longer budget buys a worse number, not a
better one, so the lever is regularization rather than epochs. See
[[experiments.026-metabolism-flux.scripts.train_flux]] for the arm definitions these runs
used.

**The reported per-arm score is a maximum over epochs.** That is an upward-biased order
statistic whose bias grows with the number of epochs run. It is not an estimate of an
arm's performance, and two arms trained for different numbers of epochs are not
comparable on it at all.

**The validation set is small enough that its own noise dominates.** The new
`n_val_<head>` counter measures 353 betaxanthin observations in the validation split at
seed 999. A Pearson correlation on n observations has null width $1/\sqrt{n-3}$, which
is 0.0535 here. The five arms are separated by 0.004 to 0.08, so every reported
difference sits inside one null width of every other.

Those three facts have one shape: the comparison as it stands cannot distinguish any arm
from any other, or from nothing at all.

### The three grids

| grid | cells | what it decides |
| --- | --- | --- |
| `null` | `pooled` and `flux_anchored`, training targets permuted, 12 seeds | how large the reported statistic gets when there is provably no signal |
| `reg` | learning rate x weight decay x hidden width on `flux_anchored`, 2 seeds | whether any nearby setting escapes the noise floor instead of peaking at epoch 5 |
| `arms` | all five registered arms, 10 fresh seeds across two jobs | the arm ordering at 13 replicates, a standard error near 0.009 |

The `null` grid is the decisive one. Permuting training targets while leaving validation
real makes each run's maximum over epochs a draw from the null distribution of the exact
statistic the arms are reported with, under the real epoch-to-epoch correlation rather
than an assumed independence. Twelve seeds per arm gives that distribution directly, so
whether 0.087 is a result stops being a judgment call.

The permutation applies to the target and its mask together. Permuting the target alone
would pair a value with another row's observed flag, which changes the missingness
pattern as well as the association and makes the null measure the wrong thing.

### Ordering is load-bearing

Every grid is expanded SEED-MAJOR. Cell-major finishes every seed of one configuration
before starting the next, so a job stopped by the wall clock yields complete data for
some configurations and none for others, which supports no comparison. Seed-major makes
any prefix a balanced experiment. `--max-hours` additionally refuses to start a cell that
is not projected to finish, so a job ends on a complete run rather than losing one to the
scheduler.

Scored by [[experiments.026-metabolism-flux.scripts.analyze_sweep]], which reports the
peak, the peak epoch and the last-five mean side by side and never a bare maximum.
