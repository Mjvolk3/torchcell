---
id: eyzdgu15yd9qnlslw5itkh7
title: Test_graph
desc: ''
updated: 1698172755059
created: 1698172755059
---

## 2026.09.27 - The hermetic half of graph.py

Twenty-two hermetic tests added beside the two SGD-backed ones, which now carry their own `@pytest.mark.data` plus skipif instead of a module-level mark so the hermetic tests run in CI. Covered on hand-built inputs: `GeneGraph` and `GeneMultiGraph` construction and validation, the filters (`filter_by_contained_genes`, `filter_go_IGI`, `filter_redundant_terms`, `filter_by_date`), `create_G_go` and `create_go_subgraph` on a hand-built GO DiGraph with exact child -> parent edges, the STRING 12.0 build path from a pickle, `parse_genome` on a duck-typed genome, and the pickle `save_graph` / `load_graph` round trip (the module has no networkx serialization). Module coverage from this file 73%, 87% together with [[tests.torchcell.graph.test_gene_graph]]; the network bodies of the STRING and TFLink downloads, the 9.1 and 11.0 STRING paths past the pickle load, `check_regulatory_nodes_have_edges` and `main` stay data-gated. Findings: `create_go_subgraph` raises `TypeError: 'NoneType' object is not iterable` for a term with no annotated genes because `gene_set` is `go_to_genes.get(go_id, None)` (graph.py line 1073), and `create_G_go` avoids it only by passing annotated terms; `parse_genome` sets `alias_to_systematic` on the `ParsedGenome` CLASS with `setattr` (line 438), so every later `ParsedGenome`, even from a genome without an alias map, sees the last genome's map (the test cleans up with monkeypatch). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
