---
id: d18ss7m6bllh2jbceuqya1x
title: fix-equivariant-cgt-sizes
desc: ''
updated: 1790815122704
created: 1790815122704
---

## 2026.09.30

- [x] PR: `FIX(cgt)` issue #523, the CGT batch size comes from `batch.num_graphs` so a trailing wildtype gets its row, HyperSAGNN renumbers set ids, `num_parameters` totals every parameter, observed-label docstrings corrected. [[torchcell.models.equivariant_cell_graph_transformer]], [[torchcell.models.cell_graph_transformer_metabolism]], [[tests.torchcell.models.test_equivariant_cell_graph_transformer]], [[tests.torchcell.models.test_cell_graph_transformer_metabolism]]
