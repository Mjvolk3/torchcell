---
id: eyzdgu15yd9qnlslw5itkh7
title: Test_graph
desc: ''
updated: 1698172755059
created: 1698172755059
---

## 2026.09.27 - The hermetic half of graph.py

Twenty-two hermetic tests added beside the two SGD-backed ones, which now carry their own `@pytest.mark.data` plus skipif instead of a module-level mark so the hermetic tests run in CI. Covered on hand-built inputs: `GeneGraph` and `GeneMultiGraph` construction and validation, the filters (`filter_by_contained_genes`, `filter_go_IGI`, `filter_redundant_terms`, `filter_by_date`), `create_G_go` and `create_go_subgraph` on a hand-built GO DiGraph with exact child -> parent edges, the STRING 12.0 build path from a pickle, `parse_genome` on a duck-typed genome, and the pickle `save_graph` / `load_graph` round trip (the module has no networkx serialization). Module coverage from this file 73%, 87% together with [[tests.torchcell.graph.test_gene_graph]]; the network bodies of the STRING and TFLink downloads, the 9.1 and 11.0 STRING paths past the pickle load, `check_regulatory_nodes_have_edges` and `main` stay data-gated. Findings: `create_go_subgraph` raises `TypeError: 'NoneType' object is not iterable` for a term with no annotated genes because `gene_set` is `go_to_genes.get(go_id, None)` (graph.py line 1073), and `create_G_go` avoids it only by passing annotated terms; `parse_genome` sets `alias_to_systematic` on the `ParsedGenome` CLASS with `setattr` (line 438), so every later `ParsedGenome`, even from a genome without an alias map, sees the last genome's map (the test cleans up with monkeypatch). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 15: the date and IGI filters, pickle-first loading, downloads

Twenty-four to thirty-seven tests; alone 73 to 93 percent, the `tests/torchcell/graph/` directory 87 to 98. The 2017-07-19 date filter and the IGI filter on the built GO graph with exact edges, the log line and the surviving child GO:0000002 re-attached to the BP root; the contained-genes filter with no gene set (n = 2 drops the four one-gene leaves); pickle-first loading for `G_gene`, `G_genetic`, `G_regulatory` and `G_go` and every lazy graph keeping its first read; a missing STRING table (9.1, 11.0, 12.0) or TFLink table triggering one download of the exact URL; a TFLink download that writes nothing giving an empty graph and two exact log lines; a genetic partner outside the raw graph dropped; regulatory edges extending a supplied graph; `build_gene_multigraph` omitting a graph that loads as None with a warning; both `check_regulatory_nodes_have_edges` messages.

Findings: only the last evidence row per gene and term is kept, so the IGI filter's result depends on row order (lines 1074-1077); `main` builds the genome with `overwrite=True` (1449).
