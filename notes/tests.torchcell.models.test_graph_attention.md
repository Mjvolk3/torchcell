---
id: ofag4pwhy8no5kr7vv99duh
title: Test_graph_attention
desc: ''
updated: 1790780604295
created: 1790780604295
---

## 2026.09.30 - Phase 18: the attention sibling, also dead as shipped

New file, twelve functions (13 cases), 14 to 71 percent alone. Findings first: `DeepSet(node_layers=...)` at line 33 raises ("Did you mean 'num_node_layers'?") and line 58 calls `GATConv(..., v2=True)`, which raises `MessagePassing.__init__() got an unexpected keyword argument 'v2'`; `set_layers.insert(0, in_dim)` mutates the caller's list (62). The tests pin both refusals, then swap in a legacy `DeepSet` stand-in and a recording `GATConv` factory returning the real `GATv2Conv`: the keywords per layer including `v2=True`, parameter shapes (`2o(i+2)` giving 158), the star-graph closed form with zeroed `att` (a neighborhood mean in which `lin_r` drops out), the skip flags, equivariance and invariance, gradient reach, seeded determinism.
