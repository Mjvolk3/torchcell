---
id: 8n6aio7hbpa16ubg8w3gsq7
title: Flux_layer
desc: ''
updated: 1788314317663
created: 1788314317663
---

## 2026.09.01 - The differentiable flux layer

Gene tokens in, a feasible flux vector and its feasibility residuals out. Every hard
constraint is either exact by construction or a smooth penalty, and nothing is enforced by a
binary variable, which is the difference from Thermo-Flux (Smith 2026), whose second law
costs one binary per reaction and a 24 h wall-time budget per model.

Read the module docstring for the term-by-term relaxation. The two facts most likely to bite:

- **Every constraint term must be dimensionless AND commensurate with the data loss.** The
  raw second-law hinge is ~72 at initialization against a data loss of ~2, and the squared
  dissipation excess is ~4e4 and drives the model to NaN in one step. Both are reformulated,
  not merely down-weighted.
- **`parameterization="box"` cannot reach a sparse flux vector.** Zero flux is an asymptote
  of the sigmoid, so the mass-balance residual pins at its maximum. Measured in
  [[experiments.026-metabolism-flux.scripts.box_zero_reachability]].

`coverage_report()` is not optional reporting: a capacity constraint built on a default kcat
is a uniform rescaling of the box, not an enzyme constraint, and a loss curve cannot tell
them apart.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
