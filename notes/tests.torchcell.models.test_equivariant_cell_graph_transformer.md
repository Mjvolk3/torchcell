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

## 2026.09.30 - Findings retired (issue #523)

Retired: the HyperSAGNN non-contiguous id finding, the trailing-wildtype finding and the `num_parameters` total finding; the observed-label test keeps pinning the `proj([0, 0])` offset, now described as the documented behavior. Now asserted: ids {0, 2, 2} give exactly the {0, 1, 1} output and ids {5, 2, 2} give its rows swapped; a batch {1, 2}, {3}, wildtype returns 3 rows, the wildtype row equals the model's own zero-context path (`_apply_residual(H_genes, 0, 0)`, interaction head on `[h_CLS, 0]`) and the row of an all-wildtype batch, and rows 0 and 1 equal the batch without the wildtype; an assignment past `batch_size` raises with an exact message; `num_parameters` lists all 13 blocks, they sum to 16251, which is `sum(model.parameters())`; a seed-0 model's prediction, per-gene and per-metabolite outputs on the fixture equal the pre-fix tensors bit for bit. The `_perturbation_batch` helper takes `num_graphs`. 22 to 24 tests.

## 2026.09.30 - Review: the output pin is 1e-6, not bit for bit

Review of PR #560 ran `test_seeded_model_output_is_unchanged_by_the_batch_size_fix` under `OMP_NUM_THREADS=1` and `4` and it failed by one float32 ulp (max 1.19e-7), passing under 2 and 64: the CPU matmul reduction order depends on the thread count. The three outputs are now compared at atol = rtol = 1e-6 on the same pinned values, and the test also pins the exact list of the 70 `state_dict` keys (what a strict checkpoint load matches) and the three output shapes. An init-order or forward-math change moves the outputs by far more than 1e-6.
