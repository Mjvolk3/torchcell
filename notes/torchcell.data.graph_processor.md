---
id: px3gopqs0en6i4dhea8obmu
title: Graph_processor
desc: ''
updated: 1746906455219
created: 1746906455219
---

## 2026.09.30 - Incidence cache tied to its graph, one w_growth source, uniform keys (issue #527)

Previous behavior, pinned by Phase 14 of the test campaign:

- `IncidenceSubgraphRepresentation` and `LazySubgraphRepresentation` built the incidence cache once and returned early on every later call, whatever graph arrived. A second graph through the same processor read the first graph's edge positions and emitted node index -1.
- `SubgraphRepresentation` and `IncidenceSubgraphRepresentation` returned the stored `cell_graph["reaction"].w_growth` subset to the kept reactions, while `LazySubgraphRepresentation` returned a vector recomputed from a `reaction.subsystem` attribute. `to_cell_data` stores `w_growth` (1 for a Growth-subsystem reaction) and writes no `subsystem` attribute, so on every real cell graph Lazy returned all zeros. A tensor `subsystem` compared 0-d tensors with the string "Growth" and never marked anything.
- `Unperturbed` appended `label_statistic_name` unconditionally and raised `TypeError` on a phenotype with none (VisualScorePhenotype).
- `DCellGraphProcessor` wrote `perturbation_indices_batch` only when some perturbed gene was a node.
- `NeighborSubgraphRepresentation._add_phenotype_data` wrote nothing when no label value existed.

Fix:

- The cache records what it was built from: the gene count and the gene-gene `edge_index` tensors, the only inputs it depends on (`_gene_edge_indices`, `_incidence_source_matches`). Every call checks identity per edge type (O(1)); a tensor that is not the recorded object is compared by content and, if equal, the record is re-pointed so the next call is O(1) again. A different graph REBUILDS the cache rather than raising: the cache is a pure function of the graph, and one processor may legitimately serve several graphs. An in-place mutation of the recorded tensor is not detected (same object).
- `w_growth` has one source in all three masking processors: the stored tensor, subset to the kept reactions (Lazy keeps every reaction and returns the stored tensor by reference). The subsystem-derived computation, the only reader of `reaction.subsystem`, is removed, which retires the tensor-subsystem branch.
- `Unperturbed` appends the statistic field only when the class names one.
- DCell always writes `perturbation_indices_batch` (an empty long tensor when no perturbed gene is a node).
- The neighbor processor writes the [NaN] placeholder at type 0, sample 0, plus `phenotype_types`, like the subgraph processors.

Evidence: `test_a_cached_processor_rebuilds_its_cache_for_a_different_graph`, `test_the_cache_is_rebuilt_for_same_size_graphs_with_different_edges`, `test_every_masking_processor_returns_the_stored_w_growth`, `test_a_tensor_subsystem_does_not_touch_the_stored_w_growth`, `test_unperturbed_skips_a_phenotype_without_a_statistic`, `test_dcell_with_a_gene_outside_the_graph_writes_an_empty_batch_vector`, `test_neighbor_phenotypes_write_the_shared_placeholder` in [[tests.torchcell.data.test_graph_processor]], and `test_lazy_marks_invalid_reactions_but_keeps_all_of_them` in [[tests.torchcell.data.test_graph_processor_subgraph]]. The `label_policy.py` items of issue #527 are out of scope here and remain open.
