---
id: ia7wramkd66zwzxd9hcn7f3
title: Test_graph_processor_unperturbed_dcell
desc: ''
updated: 1790535420738
created: 1790535420738
---

## 2026.09.27 - Unperturbed and DCellGraphProcessor on hand-built graphs

`Unperturbed`: the cell graph is copied rather than filtered, phenotypes land in per-field tensors with a NaN placeholder for a field a record does not carry, and both `ValueError` paths raise with their exact messages. `DCellGraphProcessor` on a three-term GO DAG: the exact `go_gene_strata_state` rows zeroed for the perturbed genes, the batch indices, the no-GO-graph path and the empty batch. Findings: `perturbation_indices` is node-ordered (graph_processor.py line 2237) while `perturbation_indices_batch` is record-ordered (lines 2258 to 2274), so records {YAL002W} and {YAL001C, YAL003W} give [0, 1, 2] against [0, 1, 1]; `Unperturbed` looks for the legacy `("metabolite", "reactions", "metabolite")` edge type (line 1951), so with the bipartite block it copies the metabolite nodes but no reaction store and no `gpr` or `rmr` edges. Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
