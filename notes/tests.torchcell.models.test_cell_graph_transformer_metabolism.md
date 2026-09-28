---
id: 3455hhxth3uvpoxpou25kxk
title: Test_cell_graph_transformer_metabolism
desc: ''
updated: 1790562662972
created: 1790562662972
---

## 2026.09.27 - Flux heads on a real FluxLayer

Seven tests added (now fifteen) and the three-line header restored: the exact `perturbed_gene_pool`, the closed forms of `FluxMetaboliteHead` and `FluxScalarHead`, the pooled heads' widths and errors, and the flux wiring on a real `FluxLayer` built on the toy GEM of the flux-layer tests, where the heads read the layer's turnover and every parameter including the flux layer's gets a gradient; both flux head kinds refuse to build without a flux layer. Module coverage 77%. Finding: `num_parameters` leaves out the flux layer, reporting 8063 for a module that holds 8449 (386 trainable flux parameters missing; cell_graph_transformer_metabolism.py lines 574 to 584). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
