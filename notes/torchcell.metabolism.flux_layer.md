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

## 2026.09.30 - No-GPR boxes keep their bounds; each arm builds only its own head

Two fixes from issue #534.

- **Enzyme capacity no longer touches a reaction without a GPR.** `dynamic_box` used to cap a no-GPR reaction at `ub.abs()` on both sides, so an asymmetric box [-10, 5] became [-5, 5] and a reverse-only box [-10, 0] collapsed to the point 0 and carried no flux. Capacity is `|v_j| <= cap_j` and applies only where a GPR names an enzyme; a no-GPR reaction now keeps its own (availability-scaled) `[lb, ub]`. Evidence: `test_dynamic_box_enzyme_capacity` asserts the exact boxes, and `test_second_law_hinge_and_dissipation_closed_form` now runs with capacity on and gets `v_R1 = -5` through the reverse-only no-GPR R1.
- **The nullspace arm no longer builds `reaction_embedding` and `flux_mlp`.** Its forward reads only `latent_mlp`, so those parameters never received a gradient and inflated the parameter count. They are now built only in the box arm. `test_each_arm_builds_only_the_modules_it_reads` asserts the parameter names, the counts (99 nullspace, 138 box on the toy network) and a nonzero gradient on every parameter in each arm. A nullspace checkpoint saved before this change carries the two extra modules and will not load with `strict=True`.
