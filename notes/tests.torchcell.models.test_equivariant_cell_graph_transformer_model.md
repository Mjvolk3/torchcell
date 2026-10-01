---
id: k21s80p651oln0rnr55l2t2
title: Test_equivariant_cell_graph_transformer_model
desc: ''
updated: 1790562655427
created: 1790562655427
---

## 2026.09.27 - The full CGT model by identities

Twenty-eight tests beside the Phase 2 component tests ([[tests.torchcell.models.test_equivariant_cell_graph_transformer_components]]). Relabeling the genes permutes `H_genes`, `H_genes_pert` and the per-gene head and leaves the prediction, `h_CLS`, the global head and the per-metabolite head unchanged; reordering genes within one genotype's perturbed set changes nothing. Exact parameter counts for a 1-layer hidden-16 model (8440) and the preprocessors (288, 192, 278). The graph-regularization loss matches its closed form 8 lambda log 9 with Q and K zeroed, halves at row-sampling rate 0.25, and handles skipped layers, edgeless graphs and the `physical_interaction` alias. The attention mask is exact and masked positions get zero attention. At init the model is bit-identical to the plain one under propagation and Perceiver mixing (closed ReZero gates) and the response basis; Hadamard `add` equals `off`; magnitude matching is an exact rescale by 1/sigmoid(4). With every head on, only the `observed_label_encoder.proj` parameters get no gradient, because that encoder runs only when `observed_values` is passed; with it passed, every parameter does. Every constructor and forward `ValueError` with its message; exact GPR and RMR incidence when node counts come from the edges; exact `MaskedMultitaskLoss` values. Module coverage 67% from this file, 72% with Phase 2; the rest is `main()` and GPU device branches. Phase 8 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Explicit batch size (issue #523)

`EquivariantPerturbationTransform.forward` now takes a required `batch_size`; every direct call passes `B` (or 3 for the [0, 0, 2] assignment). No assertion changed.
