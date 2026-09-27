---
id: gxa61grg4i89wqpmdmohuc4
title: Test_cell_data_synthetic
desc: ''
updated: 1790415109289
created: 1790415109289
---

## 2026.09.26 - The pure parts of cell_data on hand-built graphs

`to_cell_data` on a three-gene `GeneMultiGraph` whose base graph is built from an unsorted plain list (a `GeneSet` is already sorted, so it could not show the sort): sorted `node_ids`, the `physical_interaction` relation with every missing self loop added (four edges, or one without them), node embeddings from an edgeless graph filling `x`, and the `base` requirement. `compute_strata` on GO edges pointing child -> parent, as `create_G_go` builds them (graph.py line 1082): the root is stratum 0 and children count depth from it; `DCell` iterates the strata in descending order, so the leaves are processed first and the root last. The function's own docstring says leaves are stratum 0, which contradicts its behavior (recorded in [[test-campaign.2026.09.25]]). A cyclic component that never reaches in-degree 0 lands in one stratum after the acyclic part, which flattens whatever depth structure the component had. `tests/torchcell/data/test_cell_data.py` keeps the real sample-batch tests behind `--data`. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.27 - The metabolism paths and the edge naming

Six more tests: the exact self-loop order, a `regulatory` graph renamed to `regulatory_interaction` while an unlisted name is kept as is, self loops on and off as exact `edge_index` tensors, two embedding graphs concatenated into an exact `x`, `metabolism_bipartite` with exact `hyperedge_index`, signed stoichiometry, `w_growth` and the gene-reaction association, and `_process_metabolism_hypergraph` called directly. Findings: `to_cell_data` never dispatches `_process_metabolism_hypergraph` (cell_data.py lines 98 to 105), so a hypergraph under `metabolism_hypergraph` is silently ignored; the hypergraph path reports `num_edges = len(unique) + 1`, so two reactions report three (line 171); `_process_metabolism_bipartite` keeps an edge only when networkx reports it reaction-first (line 514), so a bipartite graph whose metabolite nodes were inserted before its reactions produces no `rmr` edge type at all while the GPR block and `w_growth` are still built (the real build works because yeast_GEM builds an `nx.DiGraph` and stores every edge reaction -> metabolite, yeast_GEM.py lines 257 and 406 to 438, so the stored direction satisfies the check). Phase 6 of [[plan.test-suite-buildout.2026.09.25]].
