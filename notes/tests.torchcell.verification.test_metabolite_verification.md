---
id: 5dxhh2yo2f6qxa1fuqgaxwf
title: Test_metabolite_verification
desc: ''
updated: 1790777363386
created: 1790777363386
---

## 2026.09.30 - Phase 17: all seven results and every failure message

Eight to fourteen tests, 94 to 100 percent. All seven results in order with exact messages; a duplicate strain, a nonzero reference (absolute value 0.7) and mixed assays (sorted); `reference_finite` passing a proper subset and failing empty, non-subset and infinite references with `n_values` 2, 2 and 3; a NaN SE dropped and a negative SE failing with `< 0.0`; a `gene_addition` outside both the strain signature and the gene set.
