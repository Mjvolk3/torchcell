---
id: u6td4k8d7il3vd1o7a2x058
title: Test_point_dist_graph_reg
desc: ''
updated: 1790759635489
created: 1790759635489
---

## 2026.09.30 - Phase 12: every term in closed form

Seven to twenty-seven tests, 72 to 99 percent (the geomloss import-time `ImportError` stays). On three samples the buffered dist term is 0.0 (below the 64-sample minimum) with buffer [1, 2, 3]; through a recorder returning 0.25 the total is 2 times 0.5862611926 plus 0.5 times 0.25, with the exact call arguments and the `gather_interval` 2 pattern; the two-rank gather concatenates rank 0 then 1; MSE drops a NaN target (2.0) and weights two dimensions [3, 1] to 1.25 + 1/3; a zero total omits every `norm_*` key; the one-parameter gradient is -2.7832090514.

Findings: the flat keyword path hard-codes 224 as the Wasserstein minimum sample count, ignoring `min_samples_for_dist` (line 125); the default log-cosh loss is not NaN tolerant, one NaN target makes the total NaN and the normalized keys disappear (333, 350).
