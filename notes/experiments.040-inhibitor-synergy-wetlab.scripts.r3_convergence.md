---
id: 8pbpnzamt82sovpzv0yjezr
title: R3_convergence
desc: ''
updated: 1791799599968
created: 1791799599968
---

## 2026.10.12 - Is the trainer converged, per source?

`r3_convergence.py` reads every round-3 and round-4 history on both machines (117 runs,
400 run-source rows) and answers, per run and per source: is the training loss still
falling at the end (slope over the last fifth of the epochs, on a three-point median of
the one-step-per-evaluation values, relative to the level there); where does the held-out
loss bottom, what fraction of its first-evaluation value does it remove there, and how
far does it rise afterward; where does the held-out score peak and what does the
validation selector lose against that peak. Tables `results/r3_convergence_runs.csv` and
`results/r3_convergence_arms.csv`; the sheet below draws the mean curves of the headline
arms, both machines pooled.

The readings, means over arm means:

| source | training loss, early to end | held-out loss: fraction removed, epoch of minimum, rise after | note |
|---|---|---|---|
| Vanacloig (target) | 0.59 to 0.09 | 11 percent, epoch 34, +0.14 | overfits in magnitude; the held-out centered Spearman still plateaus near epoch 60 to 75 and barely falls, so the ranking holds while the scale drifts |
| Hoepfner HOP | 0.85 to 0.59 | 13 percent, epoch 53, +0.04 | half learned |
| Hillenmeyer HET | 0.85 to 0.92, RISING | 5 percent, epoch 47, +0.03 | not learned; the training loss goes up over training, 1.12 under auxiliary weight 0.3 |
| Wildenhain | 0.65 to 0.21 | 41 percent, epoch 81 to 100, +0.06 | the one source that is learned, and still falling at 150 in every run |

The learning rate is zero at 150 on every arm (OneCycle), so the end is set by the
schedule, not by convergence. Round 5b ([[experiments.040-inhibitor-synergy-wetlab.scripts.r3_sweep]])
is what this table asked for beyond round 5.

![](./assets/images/040-inhibitor-synergy-wetlab/r3_convergence_2026-10-12-05-04-26.svg)
