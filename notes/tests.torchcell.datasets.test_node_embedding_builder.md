---
id: 83crh6v4j0pfnpans1me6s1
title: Test_node_embedding_builder
desc: ''
updated: 1791269280237
created: 1791269280237
---

## 2026.10.06 - Phase 21 tests

- `TABLE` restates all 23 `EMBEDDING_CONFIGS` entries by hand (class, root, model name, genome, graph).
- Every class swapped for a recorder: exact kwargs per name (`/root/<root_path>`, genome for genome datasets, `graph.G_gene` for graph datasets, `model_name` only when set).
- Refusals: unknown name (message lists every name; earlier names already built), graph dataset without a graph. `learnable` is skipped.
