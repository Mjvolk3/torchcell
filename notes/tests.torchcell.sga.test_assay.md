---
id: h1osrmvaio58p5j4oano3p9
title: Test_assay
desc: ''
updated: 1790649949766
created: 1790649949766
---

## 2026.09.28 - Volume assay metrics and the recommendation rule (Phase 9)

8 tests: Z prime closed forms (zero spread is 1.0; SDs 0.1 and 0.1 across a 0.5 gap give 1 - 0.6 / 0.5 = -0.2); the volume-by-position confound on disjoint column spans and, when columns overlap, on rows; `shape_by_volume` excluding gash, missing and blank colonies (medians 0.85 and the below-threshold fractions); the per-volume metrics derived in the module docstring, NaN CV and Z prime for a wild type with a single replicate; `recommend_volume` where reliability dominates (both Z prime negative, 2.5 nL wins 0.50 to 0.35) and a positive Z prime breaks a tie between equal desirabilities, with the rationale strings verbatim. Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
