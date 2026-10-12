---
id: vaalx23bcgf905w08784a5c
title: Figure2_gate_sheet
desc: ''
updated: 1791804624332
created: 1791804624332
---

## 2026.10.12 - The dense sheet proposed for Figure 2

Two sheets for `notes-tex/interaction-fitness-figure-gate`, built the way the paper's
Figure 1 is built (one full page, every panel from a committed result file, color = the
input family a method reads at scoring time, status marks beside each letter). Sheet 1,
179 by 168 mm: a the identity composition (idea), b the released score by screen with the
screen means, c and d the interaction ladders on arms R and Q with CGT's validation
window as the last bar, e every rung's within-screen Pearson on both arms, f the fitness
side (ridge over the terms against CGT's fitness head), g measured against predicted for
the boosted trees cut at deciles, h Pearson inside bands of the measured score's
magnitude, i the agreement matrix between the rungs' predictions. Sheet 2: j CGT's
validation curves on both arms with the controls null and the additive ridge as lines, k
the comparators of the earlier draft on their own builds (hatched), l scaling and the
deployment tiers (idea). Inputs: the ladder CSV and summary, the per-triple predictions
parquet (arm R), the term table, the cached fitness-head readout, the 010/006/005/007
comparator files, and W&B (`--wandb` refreshes `results/figure2_gate_curves_wandb.json`).

The user's fairness point is answered in the gate's decision section with two rungs the
sheet does not yet carry: C1, CGT's predicted f_ijk minus the measured null, and C2, tau
derived from CGT's predicted fitness of every term; both wait on E3's per-triple dumps of
the fitness head.
