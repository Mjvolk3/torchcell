---
id: gxa61grg4i89wqpmdmohuc4
title: Test_cell_data_synthetic
desc: ''
updated: 1790415109289
created: 1790415109289
---

## 2026.09.26 - The pure parts of cell_data on hand-built graphs

`to_cell_data` on a three-gene `GeneMultiGraph` whose base graph is built from an unsorted plain list (a `GeneSet` is already sorted, so it could not show the sort): sorted `node_ids`, the `physical_interaction` relation with every missing self loop added (four edges, or one without them), node embeddings from an edgeless graph filling `x`, and the `base` requirement. `compute_strata` on GO edges pointing child -> parent, as `create_G_go` builds them (graph.py line 1082): the root is stratum 0 and children count depth from it; `DCell` iterates the strata in descending order, so the leaves are processed first and the root last. The function's own docstring says leaves are stratum 0, which contradicts its behavior (recorded in [[test-campaign.2026.09.25]]). A cyclic component that never reaches in-degree 0 lands in one stratum after the acyclic part, which flattens whatever depth structure the component had. `tests/torchcell/data/test_cell_data.py` keeps the real sample-batch tests behind `--data`. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].
