---
id: 3455hhxth3uvpoxpou25kxk
title: Test_cell_graph_transformer_metabolism
desc: ''
updated: 1790562662972
created: 1790562662972
---

## 2026.09.27 - Flux heads on a real FluxLayer

Seven tests added (now fifteen) and the three-line header restored: the exact `perturbed_gene_pool`, the closed forms of `FluxMetaboliteHead` and `FluxScalarHead`, the pooled heads' widths and errors, and the flux wiring on a real `FluxLayer` built on the toy GEM of the flux-layer tests, where the heads read the layer's turnover and every parameter including the flux layer's gets a gradient; both flux head kinds refuse to build without a flux layer. Module coverage 77%. Finding: `num_parameters` leaves out the flux layer, reporting 8063 for a module that holds 8449 (386 trainable flux parameters missing; cell_graph_transformer_metabolism.py lines 574 to 584). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 13: exact counts, sample and gene-order identities, hand-set heads

Fifteen to twenty-five tests, 77 to 78 percent (everything left is `main()`, lines 610-733, Neo4j and the genome). The parameter test now checks exact counts (parent 10529, heads 801, 801, 1107, total 13238). Reordering samples reorders every output row; the order of genes within a sample changes nothing; changing one sample's genotype leaves the others unchanged; a sample with no perturbed genes reads an all-zero pool; seeded determinism, with head order in the config changing a head's initial weights but not the encoder's; a gradient reaches every parameter; `global` heads are left to the parent class; exact errors for a `None` spec or a wrongly capitalized `kind`. With hand-set weights at hidden size 2 the scalar head joins [CLS token, gene mean, perturbed-gene pool] (outputs 35.5 and 0.5), the vector head lays its output out column then parameter, and with the pool off a supplied pool is ignored.

## 2026.09.30 - Flux-layer count finding retired (issue #523)

`test_num_parameters_leaves_out_the_flux_layer` became `test_num_parameters_counts_the_flux_layer`: `flux_layer` is 386, `total` is 8449 and the entries sum to it. The hand-built batches set `batch.num_graphs = 3`, which the model now reads as the genotype count.
