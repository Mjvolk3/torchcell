---
id: 8ae6xl0krg6sdvfiaixc8m8
title: R3_readout
desc: ''
updated: 1791751585261
created: 1791751585261
---

## 2026.10.11 - Every scored arm in two tables, read either way the selection could go

`r3_readout.py` reads every `results/mixture/<sweep>/<config>_scores.csv` and
`_history.csv` on the machine it runs on into `results/r3_readout_arms.csv` (one row per
sweep and arm: median over folds of the seed-mean gene-level held-out score, the wet-lab
readouts under the served call, the three isobole-grid residuals, and the per-source
held-out compound or dose scores) and `results/r3_readout_selection.csv` (one row per
run and seed: the epoch `select: val` chose, the validation and test scores at that step,
the best and last test scores over the history with the epoch of the best, and the epoch
at which each source's held-out loss bottomed).

The selection table is the one to read before trusting an arm row. The gap between the
test score at the selected step and the best over the history is what selecting on four
validation compounds costs, and the last-step score is what `select: last` would have
reported; both come from the history every run writes, so no arm is rerun to be read
another way. The readings as of the first wave's GilaHyper prescreens are in the
experiment note's "Round 3, first readings" section.
