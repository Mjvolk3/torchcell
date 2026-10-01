---
id: x7bfkwql00e03c3qlj5a9b6
title: Test_flux_layer
desc: ''
updated: 1790769129934
created: 1790769129934
---

## 2026.09.30 - Phase 15: every closed form and the Decision 12 identities

Thirteen to thirty-one tests, 88 to 100 percent. The two-reaction mass-balance residual (c_balance (3/3.500001)^2, parsimony 0.035); a sigmoid box never reaching an exact zero flux (v = ub times sigmoid(z) is positive for every finite z, the unit-level face of the memory note that the box cannot reach a sparse flux); bounds clamped to plus or minus `flux_scale`; softmin 1 - ln 2 / 8 and the clamp at 0; isozyme sums (0.5 exact, 1.5 clamped to 1); capacity 3.6 per unit; default kcat 49320 per hour and 40 kDa; constitutive genes pinned to 1; the second-law hinge (1 uphill, 0 downhill), dissipation 50, the zero-initialized prior 0 then 14/3, budget ratio 2 and hinge 1; the shipped transport term used only on degenerate reactions; exemptions by name; RT at 303.15 K without a thermo table; the stochastic draw in training and the mean in eval with the log-std clamped at 2; seeded determinism and a nonzero gradient on every parameter in ANCHORED and FREE modes; refusals (an unknown constitutive gene, nullspace mode without a basis). The frontmatter lines were added.

Findings: a reaction with no GPR has its lower bound clipped to -|ub|, so a reverse-only box [-10, 0] collapses to the point 0 and carries no flux (lines 631-633); nullspace mode builds `flux_mlp` and `reaction_embedding` but never uses them, so they get no gradient (483-487, 732).

## 2026.09.30 - Issue #534 findings retired

- The dynamic-box Finding is retired: `test_dynamic_box_enzyme_capacity` now asserts that no-GPR reactions keep `[-10, 5]` and `[-10, 0]` (halved at c = 1/2), and the second-law closed form runs with capacity on and carries `v_R1 = -5` through a reverse-only no-GPR reaction.
- `test_nullspace_arm_leaves_the_box_head_untrained` is replaced by `test_each_arm_builds_only_the_modules_it_reads`: exact parameter names and counts (99 nullspace, 138 box) and a nonzero gradient on every parameter in both arms.
