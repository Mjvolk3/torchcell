---
id: lanqgxzu827jca75vc82sbv
title: Test_gene_graph
desc: ''
updated: 1790415124104
created: 1790415124104
---

## 2026.09.26 - GeneGraph, GeneMultiGraph and the four GO filters without data

Three-gene graphs and a four-term GO DAG with edges child -> parent (the orientation `create_G_go` produces). Pinned exactly: attribute forwarding and the repr strings, sorted iteration of the multigraph, and each filter's resulting node and edge sets. Three findings recorded as pinned behavior: the `GeneGraph.graph` validator runs before `max_gene_set` is parsed (field order), so its "nodes not in max_gene_set" warning never fires and foreign nodes are accepted silently; `filter_redundant_terms` removes the BROADER term whose gene set equals a child's, not the child its docstring names (the walk starts at the roots because leaves are computed as nodes without successors); `filter_by_contained_genes` counts genes over descendants, which is correct only under child -> parent edges. `tests/torchcell/graph/test_graph.py` (SGD-backed) is now `data`-marked. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].
