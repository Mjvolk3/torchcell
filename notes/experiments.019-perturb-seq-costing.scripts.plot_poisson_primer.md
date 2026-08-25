---
id: wixyiagiexrh0e72hmi60bq
title: Plot_poisson_primer
desc: ''
updated: 1787625658547
created: 1787625658547
---

## 2026.08.24 - A Poisson primer, on the problem it is used for

`experiments/019-perturb-seq-costing/scripts/plot_poisson_primer.py`

Sec. 7.1 goes straight into a zero-truncated Poisson, which is a big step for a
reader who has not met the distribution. This is the step before it, and it sits
BEFORE the arithmetic rather than after.

The panel order is pedagogical, not the order a statistics course would use:

- **(a)** You cannot hand a cell two plasmids. Setting a concentration sets an
  AVERAGE, and cells sort themselves around it.
- **(b)** Selection deletes the zero class, and deleting the smallest value
  RAISES the mean of what survives, 1.59 to 2.0. This is the counterintuitive
  part, and it is why every lambda in table 17 is lower than the
  plasmids-per-cell figure beside it.
- **(c)** Therefore `k` is a spread to steer, not a number to set.

Panels are deliberately bare. At roughly 40 mm per panel every attempt at
in-panel prose collided with something, so the caption carries the teaching. Two
labels still ran off the right edge in review and are now right-anchored.

Calls `lam_for_target_mean` from `scaling_analysis.py`, so the figure and table
17 agree by construction rather than by coincidence. Renders Fig. 14.
