---
id: hutmc4bv1tyiepx1ytgr7e0
title: fix-graph-processor-cache
desc: ''
updated: 1790812744664
created: 1790812744664
---

## 2026.09.30

- [x] PR: FIX(graph_processor), issue #527: incidence cache tied to its graph and rebuilt for a different one, one `w_growth` source (stored tensor) in Subgraph/Incidence/Lazy, `Unperturbed` skips a missing statistic, DCell writes no per-sample `perturbation_indices_batch` (`follow_batch` builds it), neighbor phenotype placeholder. [[torchcell.data.graph_processor]], [[tests.torchcell.data.test_graph_processor]], [[tests.torchcell.data.test_graph_processor_subgraph]]
