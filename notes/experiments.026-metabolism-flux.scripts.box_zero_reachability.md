---
id: huefusvv7qh2jmh8i8x9ie1
title: Box_zero_reachability
desc: ''
updated: 1788314516065
created: 1788314516065
---

## 2026.09.01 - Why a sigmoid box cannot produce a sparse flux vector

Explains why the mass-balance diagnostic pins at 1.99 in every box arm and never moves. A
ratio of 2.0 is the MAXIMUM the statistic can take, so the median metabolite is completely
unbalanced.

The cause is the parameterization. For the 2,463 irreversible reactions the lower bound is
0, so zero flux needs the logit to reach minus infinity: reaching 1e-6 of the upper bound
needs z = -13.8, and a real flux solution is ~88 % zeros. Measured at initialization, the
box puts only **0.17 %** of fluxes below 1e-6 of scale against the null space's 11.9 %.

Design consequence: an explicit zero (a gate, a hard-concrete mask, a shifted sigmoid with a
flat region at the bound), not a heavier penalty weight.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
