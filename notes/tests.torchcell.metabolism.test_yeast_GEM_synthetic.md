---
id: p0amfejijo5m0om59bxw2vo
title: test_yeast_GEM_synthetic
desc: ''
updated: 1790562755426
created: 1790562755426
---

## 2026.09.27 - YeastGEM on a three-reaction toy SBML

Twenty-one tests with `requests.get` returning an in-memory zip of a toy SBML (three reactions, three metabolites, a gene rule with AND and OR); no solver runs. Exact bipartite edges (reaction -> metabolite, signed stoichiometry), the reaction hypergraph and the gene set. Module coverage 86%. Findings: `_parse_gene_combinations` strips parentheses before splitting, so `g1 and (g2 or g3)` parses to `[{g1, g2}, {g3}]` instead of `[{g1, g2}, {g1, g3}]` (yeast_GEM.py line 95); measured on the local Yeast9 9.0.2 SBML, 0 of its 2709 gene rules have that nested form, so the served metabolism graph is unaffected and the finding is latent; the sanity check counts nodes by a `bipartite` key that is never set and prints 0 and 0 (lines 1064 to 1065); `plot_bipartite_network` selects `node_type == "gene"`, which no node carries (line 729); `plot_full_network` fails under hypernetx 2.4.0 because `EllipseCollection.set()` rejects `sizes` (line 605). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
