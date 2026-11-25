---
id: o8c4tm9ag68hxeawr9gv79b
title: Low Data Feasibility Goal
desc: ''
updated: 1795572168929
created: 1795572168929
---

## 2026.11.24 - The goal: how good a Spearman does 375 records support?

The MVP goal ([[experiments.048-pputida-isoprenol-cgt.p-putida-cgt-mvp-goal]])
asked whether a P. putida CGT with a metabolic module can be built at all, and
it can: three arm groups ran and the design-split CGT reached Pearson 0.773 on
44 held-out designs. This is the next question, and it is narrower on purpose.

**The goal.** Get the best held-out **Spearman** this campaign supports, on one
GPU, and record enough of the rest of the metric surface to know what kind of
model produced it.

**Why Spearman is the objective.** The decision the model is for is which
CRISPRi design to build next, and that decision reads the ordering of the
candidates, not their absolute titers. A model that ranks correctly and is
miscalibrated by a constant factor is useful; one that predicts the mean titer
of every strain to within 20 mg/L and gets the ordering wrong is not.

**Why the other metrics are recorded anyway.** Spearman and Pearson are both
invariant to an increasing affine transform of the predictions, so neither can
report that the model has the scale wrong. Every run therefore logs, on train,
validation and test, at every epoch: Spearman, Pearson, standardized MSE, raw
MSE, RMSE, MAE, NMSE and R-squared. NMSE and R-squared are normalized by the
variance of the split being scored, so 1.0 and 0.0 respectively are the scores
of predicting that split's own mean. Standardized MSE uses the train split's
mean and sd instead, which is what the loss minimizes, so the two disagree on a
split whose spread differs from the train split's; they are kept separate rather
than reconciled.

**The thing to beat is not the constant.** Gradient-boosted trees on the
multi-hot guide-target vector score Spearman 0.773 on the design-split test set.
The trees see exactly what the transformer sees about the genotype, minus the
graph and the sequence, so the gap between them is the whole accounting of what
the structure buys. Ridge on the same features scores 0.731. A 17.6M-parameter
model that does not pass the trees has not earned its parameters, and that has
happened before on this stack.

### This is a feasibility trial, and it is labeled as one

465 records over 402 designs is two orders of magnitude less data than any other
arm of this program trains on. The run series is therefore named for what it is
rather than for what we hope it shows: the wandb project is
`torchcell_048-pputida-isoprenol-low-data-trial` and the reports land under
`results/low_data_trial/`. The experiment directory keeps its `048` number and
slug, because that identifier is load-bearing in the dendron note names, the
branch, the open PR and every committed artifact path; renaming the series where
the runs are read is the part that carries the meaning.

### What the trial varies

Four axes, measured against the objective, on the design split:

| axis | values | what it asks |
|---|---|---|
| graph regularization lambda | 0, 1e-6, 1e-5, 1e-4 | does a STRING prior on attention help at this data size |
| metabolic module | off, `--flux` | does the GEM enter the prediction usefully |
| capacity control | dropout 0.1 / decay 1e-4, dropout 0.3 / decay 1e-2 | is the fit limited by data or by parameters |
| seed | 42 to 44 on the promoted cells | is any gap larger than the split's own noise |

**The lambda ladder was measured before it was chosen, not guessed.** At lambda
1e-3 the KL term contributes 33.3 to the loss while the MSE sits near 1, so
1e-3 is not a weak prior, it is the entire objective. The ladder brackets the
range where the term is a correction.

**Graph regularization is on block 1, and the run fails if it was not.** The KL
term pulls attention head *k* toward the row-normalized adjacency of STRING
channel *k*, one channel per head, six of each. It can only shape attention that
a later block still reads, so block 0 is refused. The run also raises if
`graph_reg_loss` was 0.0 at every epoch, which is how the 031 campaign recorded
378 cells as graph-regularized when the config named a block the stack did not
have ([[graph-reg-and-mask-index-second-block]] in memory).

### The splits, and why there is no cycle-5 arm

| arm | train | select | test |
|---|---|---|---|
| design | 375 records, 0.8 of designs | 0.1 | 44 records |
| cycle | cycles 0 to 4 | cycle 5 | cycle 6, 59 records |

A `cycle5` arm (test on 5, select on 4) was specified and is now deleted. It
leaves cycle 6's 59 records in no split, so the partition check in
`cycle_splits` refuses it, correctly. Cycle 6 is the campaign's last cycle and
belongs in test, which is what the `cycle` arm already does. The only ways to
make a cycle-5 arm valid are to discard 59 released records or to train on a
cycle later than the one being predicted, and neither is worth a card. Job 7612
is the 23-second failure this replaces.

### Cycles 2 and 3 are the same designs on a repaired vector

Cycle 2 and cycle 3 share 50 designs; no other cycle pair shares more than 7.
This is not two plasmids in the biological sense. Carruthers et al. report 40%
DBTL2 failure from low dCas9 expression, a vector repair that removed a
redundant linker, a 25% fidelity improvement, and that DBTL3 repeated DBTL2's
sgRNA arrays. On the 50 shared designs the two cycles agree at Spearman 0.668
while within-cycle split-half reliability is 0.946 and 0.980, so the lost
agreement is a per-construct fidelity defect, not a context effect.

**The authors' own QC flags are in the released file and not in our store.**
`Carruthers_etal_2025.csv` carries `above_dcas9_thresh` (1,210 True / 296
False), `to_keep` (669 / 444) and `to_keep_improved` (717 / 789), plus per-target
proteomics columns `target_N_value` and `target_N_norm`. The loader does not
store any of them, so the `--fidelity-tier` flag approximates the tier by cycle
number (cycles 3 and up) instead of by the authors' filter. Using the released
flags, or the per-target knockdown levels as features, needs either a loader
revision or a read of the raw CSV in the experiment, and is not part of this
trial.

### The compute, and the deadline

One dedicated card, from 2026-11-24 20:00 CST until 2026-11-25 14:00 CST. The
other three GilaHyper cards are held by the 031 program.

Measured on this card: 9.4 s/epoch with the regularizer and the flux layer on,
4.6 s/epoch with both off, about 170 s to build the dataset, embeddings and GEM,
and 17.6M trainable parameters.

`sweep_low_data_trial.py` runs three stages and obeys the deadline rather than
being killed by it, launching nothing whose projected finish passes it:

1. all 16 configurations at seed 42, which ranks them,
2. the top 3 configurations at seeds 43 and 44,
3. the best configuration on the cycle split at 5 seeds.
