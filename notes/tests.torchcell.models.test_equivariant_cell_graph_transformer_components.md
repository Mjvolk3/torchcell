---
id: cqnziucgf9y7rzs7i1r7u3u
title: Test_equivariant_cell_graph_transformer_components
desc: ''
updated: 1790413695385
created: 1790413695385
---

## 2026.09.26 - The tensor-only CGT components, one class each

Fourteen component classes plus the two helpers, on 8 genes, hidden 16, 4 heads, the conftest batch of 3 genotypes. Exact where a hand value exists: `calculate_weight_l2_norm` 5.0 on a `[[3, 4]]` weight, `compute_smoothness` 2.0, a self-only head mask giving an exact identity attention matrix, ReZero gates (`PerturbationGraphPropagation`, `PerceiverMixing`) returning the input exactly at init and changing it once opened, the FiLM projection of `PerGeneHead` inert at init, `ObservedLabelEncoder` changing only the observed (sample, gene) row. Structural: `EquivariantPerturbationTransform` is equivariant to a gene permutation (output rows permute with the genes), and in `rezero` mode exactly its two post-LN LayerNorms receive no gradient. The assembled model: `[B, 1]` predictions, a gradient on every parameter, seeded determinism, per-layer attention `[1, heads, N, N]` under `return_attention=True`. Kept in its own file so the existing head-level tests stay as they were. Phase 2 of [[plan.test-suite-buildout.2026.09.25]].
