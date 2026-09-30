---
id: mvhc0d5z1ftapnu5v4hgvos
title: Test_equivariant_cell_graph_transformer
desc: ''
updated: 1790762175908
created: 1790762175908
---

## 2026.09.30 - Phase 13: the ten remaining gaps and the identities the sibling files do not state

Five to twenty-two tests (the sibling files `_components.py` and `_model.py` already pin the l2 norm, smoothness, ReZero and gene-permutation identities; nothing is duplicated). New pins: a gene alone in its HyperSAGNN set equals `(x + 0.5*O1.bias + 2*O2.bias - static(x))^2` and changing one set leaves the other bit-identical; with one deleted gene every context row is identical and the Q and K rows of `in_proj` get exactly zero gradient while V does not; propagation features log 5, log 3 and 0 on a 4-gene two-relation graph, `gate_mode="on"` a fixed buffer; the observed-label encoder fully masked equals `H + proj([0, 0])` and its `rezero` gate is a trainable zero; the per-metabolite head equals `mlp(einsum(mr, gpr, h))`; cross-attention head counts 1234 without the FFN, 2338 with it, +16 per output feature; the 82-wide per-gene head input rebuilt from the model's parts; the block order `Perceiver(CrossGene(Observed(Transform)))`; genotype permutation permutes the prediction and every head while `h_CLS` is unchanged; `residual_update_ratios` from forward hooks and the fused-versus-manual attention agreement; every parameter gets a nonzero gradient; the preprocessor with dropout (288 parameters); an edgeless graph adds exactly 0 to the regularization; the cached edge count.

Findings: HyperSAGNN sizes its output by the number of distinct set ids but scatters by id value, so ids {0, 2} raise (line 246); the batch size is `max(batch_assignment) + 1`, so a trailing genotype with no perturbation silently gets no prediction row (668, 1323); `num_parameters["total"]` omits propagation, cross-gene mixing, the Perceiver, the observed-label encoder, the bilinear term and the response basis, reporting 10850 against 16251 real on the test config, and `main()` logs that total (2952-2985, 3382); the claim at lines 980 and 991 that a fully masked forward is an identity is false. Coverage over the three files: 72 to 73 percent; lines 3022 to 3675 are the hydra `main()` (the 006 batch loader, matplotlib, `ASSET_IMAGES_DIR`), and 2775-2776 and 2826 are device moves that cannot happen on CPU.
