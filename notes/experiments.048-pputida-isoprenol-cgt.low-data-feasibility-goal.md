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

## 2026.11.25 - The design-split half, and why its objective was the wrong one

The design pass ran 20:01 to 00:32 CST, 27 cells on one RTX 6000 at about 10
minutes each. Every number below is from the per-cell reports under
`results/low_data_trial/` and the committed scripts that wrote them.

### The finding that reframed the trial

**On the 44-record design-split test set, ranking the strains by DBTL cycle
number alone scores Spearman 0.807.** That beats the boosted trees at 0.773 and
every CGT cell trained all night. Titer rose across the campaign, as a
successful optimization campaign's titer should, and the design split is random
over designs rather than over cycles, so train and test both span all seven
cycles and "a later cycle means a higher titer" is most of the available rank
information.

The model is never handed the cycle. It does not need to be: each cycle used its
own sgRNA sets, so the cycle is recoverable from the guide identity the model
does get.

`cycle_confound_audit.py` reports the corrections:

| ranker | overall test rho | within-cycle rho | cycle-partialed rho |
|---|---|---|---|
| cycle number | +0.807 | undefined by construction | +0.008 |
| guide count | +0.748 | +0.426 | +0.153 |
| control titer | -0.659 | undefined by construction | -0.080 |

Guide count loses most of its apparent skill once the cycle is removed. The test
split's cycle composition is also lopsided: 20 of the 44 records are cycle 0,
the single-sgRNA lowest-titer cycle, so the split is largely a cycle-0 versus
later contrast. Only 24 records across 6 groups support a within-cycle number at
all.

**So the design split cannot carry a clean rank objective at this campaign
size.** The cycle split needs no correction, because every one of its test
records is cycle 6 and the chronology therefore has zero variance inside the
test set. The cycle arm scores much lower, and that is why it is the stronger
evidence of the two, not the weaker.

### What the grid measured

Sixteen configurations at seed 42, then three at two further seeds. Ranked by
VALIDATION Spearman, because ordering the table by the held-out score is how a
test-set selection decision gets made without anyone intending it.

| tag | best ep | val rho | test rho | test r | RMSE | offset |
|---|---|---|---|---|---|---|
| light_flux_reg1e-04 | 123 | 0.816 | 0.818 | 0.866 | 144.9 | 124.9 |
| heavy_pooled_reg1e-05 | 45 | 0.804 | 0.756 | 0.861 | 71.5 | 11.2 |
| heavy_pooled_reg1e-04 | 4 | 0.778 | 0.699 | 0.683 | 102.9 | 13.0 |
| light_pooled_reg0 | 45 | 0.767 | 0.741 | 0.831 | 82.0 | 9.6 |
| heavy_flux_reg0 | 77 | 0.764 | 0.776 | 0.767 | 241.8 | 223.9 |
| heavy_flux_reg1e-05 | 77 | 0.728 | 0.675 | 0.670 | 273.1 | 252.3 |
| light_pooled_reg1e-04 | 3 | 0.717 | 0.788 | 0.766 | 95.0 | 26.3 |
| heavy_flux_reg1e-04 | 60 | 0.717 | 0.707 | 0.668 | 104.6 | 4.1 |
| heavy_flux_reg1e-06 | 41 | 0.686 | 0.735 | 0.363 | 269.4 | 235.4 |

RMSE and the offset are in ug/mL. The eight omitted rows are pooled cells with
offsets between 1.7 and 44.6.

The three-seed replication, and the nulls it is measured against:

| tag | n | test rho | test r | RMSE | offset |
|---|---|---|---|---|---|
| light_flux_reg1e-04 | 3 | +0.775 ± 0.038 | +0.757 ± 0.120 | 159 ± 68 | 108 ± 88 |
| light_pooled_reg1e-04 | 3 | +0.753 ± 0.050 | +0.791 ± 0.022 | 94.7 ± 3.2 | 15.3 ± 9.5 |
| heavy_flux_reg0 | 3 | +0.709 ± 0.077 | +0.692 ± 0.066 | 260 ± 145 | 223 ± 172 |
| boosted trees | | +0.773 | +0.782 | 87.7 | |
| ridge | | +0.731 | +0.751 | 98.0 | |

**No configuration beats the trees.** The best is 0.775 ± 0.038 against 0.773, a
gap of 0.002 on a seed standard deviation of 0.038, and the trees win Pearson
and RMSE outright against all three.

### The error is an offset, and the offset is unstable

MSE decomposes exactly as `bias^2 + (sd_o^2 + sd_p^2 - 2 r sd_o sd_p)`, so the
offset is recoverable from the reported Pearson and the two standard deviations
without saving a single prediction. The share of test MSE that is a constant
offset is 0 to 14 percent for all eight pooled cells and 12 to 86 percent for
all six flux cells, reaching 252 ug/mL on a target whose own spread is 139.

Strip the offset and the best flux cell's scatter is 73.6 ug/mL against the best
pooled cell's 70.6, with a higher Pearson. **The metabolic module is not
predicting worse than the pooled head, it is predicting off-center.** That
matters for the MVP note's `design_flux` arm, whose Pearson 0.745 against
`design` at 0.773 was reading a centering defect rather than the value of
metabolism.

**The first explanation offered for the offset was wrong and is withdrawn.**
Weight decay applied to bias parameters looked like the cause: the two
heavy-decay flux cells were offset by 224 and 235 ug/mL against 37 to 125 for
the four light-decay ones. Then three seeds of ONE light-decay configuration came
back offset by 13, 125 and 186. The within-configuration seed spread is as large
as the between-decay contrast the story rested on, and every stage-1 cell was a
single seed. The flux head's calibration is better described as unstable across
seeds than as a decay effect: its RMSE seed standard deviations are 68 and 145
ug/mL against 3.2 for the pooled cell.

### Two defects in the trial's own method, both found and one fixed mid-flight

**Checkpoint selection on validation Spearman is noisy.** Best epoch across the
grid came out 1, 2, 3, 3, 3, 4, 12, 16, 37, 37, 39, 41, 45, 45, 60, 77, 77 and
123 with no relation to the regularization lambda, and validation Spearman is
computed on 46 records where its standard error is near 0.1. An unregularized
cell selected at epoch 2 and a regularized one at 45, so an early account of
this as a property of the KL term was wrong and is withdrawn. A paired arm tests
the rule directly.

**Stage 2 promoted on the test score.** The driver ranked configurations by
`test_spearman` to choose which to replicate, which is a selection decision made
on the held-out set. Each cell's own test score remains an honest held-out
measurement, but the three promoted tags are test-selected, and they are recorded
that way rather than as "the best three". Concretely it skipped
`heavy_pooled_reg1e-05`, the cell with the best RMSE of all sixteen and the
second-best validation Spearman. Fixed to promote on validation Spearman before
the cycle pass used the same code.

### The first cycle-split evidence

Four seeds of the test-selected winner, on the split the chronology cannot
inflate:

| seed | best ep | val rho | test rho | test r | RMSE | offset | NMSE |
|---|---|---|---|---|---|---|---|
| 42 | 76 | 0.733 | 0.585 | 0.592 | 252.0 | 189.0 | 1.614 |
| 43 | 75 | 0.809 | 0.385 | 0.425 | 192.9 | 66.4 | 0.946 |
| 44 | 94 | 0.798 | 0.523 | 0.492 | 272.4 | 209.6 | 1.885 |
| 45 | 61 | 0.762 | 0.514 | 0.532 | 180.7 | 63.5 | 0.829 |
| mean | | | +0.502 ± 0.084 | +0.510 ± 0.070 | 224.5 | 132.1 | |
| boosted trees | | | +0.510 | | | | |
| ridge | | | +0.415 | | | | |

**A prediction was recorded before these ran and it failed.** If lambda 1e-04's
design-split lead had been a real graph-prior benefit it would have survived
here; instead the configuration that led the design split at 0.818 scores
0.502 ± 0.084 against the trees' 0.510, a tie. The design-split advantage was the
confound.

Validation and test diverge sharply here, 0.73 to 0.81 against 0.39 to 0.59,
where on the design split they tracked each other. Cycle 5 does not tell you
about cycle 6, which is extrapolation failure rather than ordinary overfitting.
Three of the four seeds have NMSE at or above 1, meaning predictions worse than
the test cycle's own mean titer.
