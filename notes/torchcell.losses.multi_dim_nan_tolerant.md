---
id: 2dt9y2grz6h8ln78b1s27ub
title: Multi_dim_nan_tolerant
desc: ''
updated: 1790546340270
created: 1790546340270
---

## 2026.09.27 - Four fixes from the Phase 6 exact tests

Phase 6 of [[plan.test-suite-buildout.2026.09.25]] pinned the module's behavior in [[tests.torchcell.losses.test_multi_dim_nan_tolerant_exact]] and the independent audit confirmed four defects against the source; this entry records the fixes, each with its flipped test.

- `FastSoftSort.backward` returned the negated Jacobian-vector product. `soft = w - v` with `v = PAV(w - s)`, and the PAV Jacobian is the block-averaging matrix P, so `d soft / d s = +P`; the code emitted `-P`. Now `grad_sorted = grad_v`, `torch.autograd.gradcheck` passes, and the central difference agrees ((3, 1.5, 1.5) for values (1, 1.5, 3) and upstream (1, 2, 3)). Consequence: every gradient that flowed through `WeightedDistLoss` pointed away from the distribution match.
- `WeightedDistLoss.forward` paired the DESCENDING soft sort with ASCENDING theoretical labels, so identical predictions and targets (0, 0, 2, 2) scored 3.25 instead of 0.25. The soft-sorted predictions are now flipped before the reduction; `fast_soft_sort` keeps its descending convention.
- The default `ones(1)` weight was repeated to `(1, 1)` without normalizing, so two dimensions summed where an explicit weight vector averages. The repeated default is now divided by the dimension count. Weights are still not renormalized over valid dimensions when a column is all NaN (an all-NaN column keeps its weight and contributes 0), unlike `WeightedSupCRCell`; pinned, not changed.
- `SupCR.compute_dimension_loss` included the anchor's own `exp(1/T)` in the denominator of every positive whose label tied the anchor's (`searchsorted(side="left")` on a distance of 0 starts the suffix sum at position 0). The paper's denominator sums over `k != i`; the diagonal is now masked out of the suffix sums before sorting, which changes nothing when no labels tie. Labels (0, 0, 1) on the test fixture: 0.8139067324 before, 0.6003624961 after.
- `CombinedRegressionLoss.forward` multiplied a `[D, 1]` loss stack by `[D]` weights; the broadcast to `[D, D]` canceled the weights and the total was the plain sum. The stack is now a `[D]` concatenation, so weights (1, 3) on (2.5, 5.0) give 4.375, not 7.5.

Who was affected: `isomorphic_cell_loss.py` (`ICLoss`, always runs both terms), `mle_dist_supcr.py` (`MleDistSupCR`, the trainer's composite) and `point_dist_graph_reg.py` (imports `WeightedDistLoss` only, for its `type: dist` option) build on these classes. By the configs and scripts (reviewer's table in [[test-campaign.2026.09.25]]): experiments 003, 005 and 006 executed the fixed `dist` and SupCR paths through `ICLoss` and the `MleDistSupCR` configs, so their reported dist and SupCR terms were computed with the pre-fix code; 009, 010, 011 and 025 select `distribution_loss.type: wasserstein` with `supervised_contrastive.lambda: 0.0`, which never calls `WeightedDistLoss` or SupCR; nothing at 016 or later executed them. `CombinedRegressionLoss` was used by the 003 pooling scripts, whose total was the plain sum.
