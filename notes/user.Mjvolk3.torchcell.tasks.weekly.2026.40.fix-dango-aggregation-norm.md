---
id: fq80iyl8ccdv0vlca8slv5t
title: fix-dango-aggregation-norm
desc: ''
updated: 1790872378950
created: 1790872378950
---

## 2026.10.01

- [x] PR: FIX(dango_gi) issue #540 last item, `aggregation_norm` is an explicit argument that refuses any non-null value in [[torchcell.models.hetero_cell_bipartite_dango_gi]] (null builds main's module exactly), the default 006 config now states `aggregation_norm: null`, tests in [[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi]].
