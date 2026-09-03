---
id: kl1nvqprffawc1zqckzxfcq
title: Analyze_sweep
desc: ''
updated: 1788399405736
created: 1788399405736
---

## 2026.09.02 - Scoring the sweep against its own calibrated null

Reads every results file under `experiments/026-metabolism-flux/results/` written by
[[experiments.026-metabolism-flux.scripts.sweep_flux]], plus the banked 20-epoch files
from [[experiments.026-metabolism-flux.scripts.train_flux]], and reduces each run to
three numbers.

| statistic | what it is | why it is here |
| --- | --- | --- |
| `peak` | maximum validation Pearson over epochs | the statistic the earlier runs used, kept so old and new results compare on equal terms |
| `peak_epoch` | where that maximum fell | a peak at epoch 3 of 30 is an overfitting signature, not a performance one |
| `last5` | mean of the final five epochs | not an order statistic, so it is the honest summary of where training ended |

The banked runs are tagged `banked` rather than pooled with the new ones, because a
maximum over 20 epochs and a maximum over 30 epochs are different order statistics and
averaging them together would compare two different things.

### The empirical p-value

Each arm's p-value is the fraction of null draws at or above that arm's mean peak, with
the standard `+1` finite-sample correction in both terms so it is never reported as
exactly zero on a null of finite size. It needs no normality assumption and no analytic
correction for the epoch maximum, because the null was drawn through that same maximum.

The analytic width $1/\sqrt{n-3}$ is printed alongside it as a cross-check. They answer
slightly different questions: the analytic figure is the width of a single correlation,
while the empirical null is the width of the maximum of the whole epoch sequence, so the
empirical one is the larger and is the one an arm has to clear.

A NaN epoch is dropped rather than scored as zero. Scoring it as zero would pull the
last-five mean toward zero and make a run that broke look like a run that merely
performed badly, and those have different fixes. The distinction is the same one
`masked_pearson` was changed to preserve.

`n_val_<head>` was added with the sweep, so the banked files do not carry it. The
analytic null width is then left absent rather than computed off a count that is not
there.

Figure and summary JSON are timestamped on write, following the repo convention of
iterating with a timestamp and removing it once the plot is settled.
