---
id: 5mgolae6h1hkjzsoa0a3m3x
title: R3_plots
desc: ''
updated: 1791798669826
created: 1791798669826
---

## 2026.10.12 - Every scored arm on one sheet

`r3_plots.py` draws every mixture arm with a score file on either machine as one row:
the public compound-cold score on the left, the growth call on the right, every run a
point and the arm's mean a bar, sorted by the public score so the sheet reads as a
ranking with its noise visible. Rounds 1 and 2 (pool-and-last, 50 epochs, three seeds on
five folds) sit on the same axis as rounds 3 and 4 (held-out protocol, 150 epochs), and
the two machines' copies of a run are drawn alike. The reference lines are nested ridge
on the public side and Loewe on the growth call, read from
`results/figure_gate_numbers.json`. The wide spread of the single-run points on every
arm, against the narrow range of the arm means, is the picture of the noise floor the
experiment note describes; an arm with one run has no spread to show and its row says
(1).

![](./assets/images/040-inhibitor-synergy-wetlab/r3_arms_overview_2026-10-12-04-52-22.svg)
