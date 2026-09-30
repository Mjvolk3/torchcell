---
id: ine3c9t0djsmt0w2uqya39j
title: test_yeast_GEM
desc: ''
updated: 1790769122378
created: 1790769122378
---

## 2026.09.30 - Phase 15: a five-reaction model by hand

Ten to twenty-five tests; alone 8 to 78 percent (the real-GEM tests skip), with the synthetic sibling 86 to 96. Exact hyperedges and stoichiometric rows (-1, -2, 1, 3), a three-subunit AND as one gene set, irreversible gene-free reactions forward only with the reverse row negated; the exact bipartite edge list, compartments and the isolated EMPTY node; an induced gene set never dropping gene-free reactions; `gene_set` and `bipartite_graph` cached while `reaction_map` is rebuilt on every read; a failed download raising before any file is written; the exact output of `analyze_reactions_without_genes`, the sanity-check lines and the seed-42 attribute report; `main`, `main_bipartite`, `plot_full_network` arguments, the layout arguments. The frontmatter lines were added. The incidence matrix and index maps live in `constraints.build_gem_tensors`, not here.

Findings: a reaction with no metabolites is missing from `reaction_map` (hypernetx drops the empty edge) but present in the bipartite graph (line 138); transport is detected by id suffix, so `s_0001 --> s_0002` in one compartment is classed as transport (1173); `main_with_gene_set` raises `TypeError` on the `gene_set` keyword, the field is `induced_gene_set` (916); an unknown layout name raises `UnboundLocalError` (669-685).
