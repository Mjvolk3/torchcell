---
id: 0vwvint9n7fdqtzl1c8vp50
title: test-suite-buildout-p20
desc: ''
updated: 1791257440365
created: 1791257440365
---

## 2026.10.05

- [x] PR-20 of [[plan.test-suite-buildout.2026.09.25]]: the 006 model and data code behind the reported gene-interaction numbers: the lazy Dango GI model with the #596 mechanism ([[tests.torchcell.models.test_hetero_cell_bipartite_dango_gi_lazy]]), the preprocessed stores and the GPU mask generator ([[tests.torchcell.data.test_neo4j_preprocessed_cell]], [[tests.torchcell.data.test_neo4j_preprocessed_cell_full_masks]], [[tests.torchcell.models.test_gpu_edge_mask_generator]]), the diffusion model and decoder ([[tests.torchcell.models.test_hetero_cell_bipartite_dango_diff_gi]], [[tests.torchcell.models.test_diffusion_decoder]]), the 006 cell graph transformer, the NSA retry model and the 003/005 bipartite Dango ([[tests.torchcell.models.test_cell_graph_transformer]], [[tests.torchcell.models.test_hetero_cell_nsa_retry]], [[tests.torchcell.models.test_hetero_cell_bipartite_dango]]); 205 new test functions, 62 pinned findings, two independent Opus 5.5 audits (224 reviewed, 0 rejected, 4 rewritten), TOTAL 62.1% to 64.7%; record in [[test-campaign.2026.09.25]]
