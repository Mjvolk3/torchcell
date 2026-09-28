---
id: xbognl98ukfx1lm5d0ir9dm
title: Test_deep_set
desc: ''
updated: 1790549279806
created: 1790549279806
---

## 2026.09.27 - DeepSet by permutation invariance and parameter arithmetic

Ten tests: permutation invariance of the set pooling as an exact identity (`assert_close` between a permuted input's output and the original), the output shape, seeded determinism, the per-layer parameter count written out, and each constructor error. Coverage of the module from this file 80%; the rest is the `main()` smoke script. Findings: with `num_set_layers=1` the `i == 0` branch wins, so the output is `hidden_channels` wide with no Dropout, and with `num_node_layers=1` the node stack maps to `hidden` while the set stack is built for `out_channels`, so the forward fails with a matmul shape error whenever `hidden != out` (deep_set.py lines 82 to 91); `norm="instance"` builds but `InstanceNorm1d` reads an `[N, C]` input as `(C, L)` and raises `ValueError` unless `N` equals the output width (line 59). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
