---
id: r2809md0t59zrb86kmmmg3d
title: Test_cell_graph_transformer
desc: ''
updated: 1790979835822
created: 1790979835822
---

## 2026.10.01 - Phase 20 tests for the 006 cell graph transformer

Module under test: `torchcell/models/cell_graph_transformer.py` (the 006 model, driven by `experiments/006-kuzmin-tmi/scripts/cell_graph_transformer.py`). `main` is out of scope and stays uncovered.

### Fixture

The conftest `cell_graph` (8 genes, `physical` edges 0->1, 1->2, 2->3, 3->4, plus `gpr` and a metabolite relation that the adjacency builder must skip) and `batch` (3 genotypes perturbing {1, 2}, {3}, {0, 4, 5}). Model: hidden 16, 4 heads, 1 layer, dropout 0, eval.

### Expected values

- Parameter counts: gene embedding 8 *16 = 128, CLS 16, encoder layer 4* 272 + 2 * 32 + 1088 + 1040 = 3280, cross-attention head 1088 (in_proj 816 + out_proj 272) + readout MLP 545 = 1633, total 5057. HyperSAGNN head 2450 + 545 = 2995.
- Encoder attention on identity projections (D 4, 2 heads): per head `softmax(x_h x_h^T / sqrt 2)`, gene block `[1:, 1:]`, numpy oracle.
- Encoder layer with Q = K = 0, V = out = I, zero last FFN Linear: `LN(LN(x + mean x))`, attention 1/3 everywhere.
- HyperSAGNN masked softmax, one head, identity projections, beta 1: a two-gene set gives `x_i + x_j` exactly, a singleton row (all -inf, NaN, then 0) gives `x_i`, a three-gene set uses a two-key softmax (numpy oracle).
- Cross-attention head, one head, identity projections: `z_b = softmax(q_b . h_j / 4) @ H[1:]` with `q_b` the mean of the perturbed rows; readout MLP from its own weights in numpy.
- Graph regularization with Q = K = 0: each gene row attends 1/9, a degree-1 row contributes `L = -log(1/9 + 1e-8)`, `physical` gives KL 4L, loss `lambda * 4L * scale / (2 / 8)`; lambda 0.5, scale 0.01 gives 0.08 L.
- Permutation: relabeling genes (embedding rows and adjacency) permutes `H_genes` and leaves `h_CLS`, predictions and the reg loss. A batch of 3 equals three single-genotype runs for both heads.

### Findings

- `cell_graph_transformer.py:608-611`: the edge divisor `len(A.nonzero()[0])` is 2 per graph (the length of the first coordinate pair), not the edge count, so the reg loss is scaled by `N / (2 G)` instead of `N / E`. Pinned at 0.08 L (true count would give 0.04 L). A relation with zero edges raises IndexError there.
- `cell_graph_transformer.py:450, 543-547`: the constructor defaults (`graph_reg_scale=0.001`, no regularization config) fail at the first forward with a bare AssertionError.
- `cell_graph_transformer.py:563-564`: an unknown `regularized_heads` key is skipped silently. `to_cell_data` emits `physical_interaction`, the 006 configs key `physical`, so on a suffixed graph that head contributes 0. The successor raises ValueError.
- `cell_graph_transformer.py:360`: the genotype count is `max(batch_assignment) + 1`; `num_graphs` is ignored, a trailing empty genotype gets no prediction, and a middle empty genotype raises in the HyperSAGNN head.
- `cell_graph_transformer.py:73-74`: the encoder layer stores the adjacency and head config but never reads them; attention is unmasked.
- `cell_graph_transformer.py:489-490, 688`: the adaptive loss weights are trainable but absent from `num_parameters`, and they receive no gradient from forward.
- Pinned difference from the successor: the 006 encoder never sees the perturbation; `h_CLS` and `H_genes` are identical for any two batches.

## 2026.10.05 - Audit follow-up: reach and one new finding

### Reach of the findings (from the independent audit)

- Edge divisor (2 per graph): REACHED by every 006 `cell_graph_transformer*.yaml` (gh_001 to gh_004, cabbi_005, cabbi_006, cabbi_008, cabbi_009, mmli_007) through `experiments/006-kuzmin-tmi/scripts/cell_graph_transformer.py` and `int_hetero_cell.RegressionTask`, which adds the term to the loss and logs `{stage}/graph_reg_loss`. With 9 graphs the divisor is 18/N, not nnz/N.
- Unmatched `regularized_heads` key: REACHED by the same configs. The `physical` and `regulatory` heads contribute 0 against suffixed relations, while the tflink and string heads match.
- Encoder ignores the adjacency it stores, and the encoder is perturbation-blind: reached by design in every 006 CGT run.
- Default constructor fails at forward, genotype count is `max + 1`, adaptive weights uncounted: latent (every config passes a reg config, no batch has an empty genotype, `adaptive_loss_weighting: false` everywhere).

### New finding

- `cell_graph_transformer.py:108, 118`: in training mode the KL reads the attention before dropout while the output mixes values with the post-dropout attention. With dropout 0.5 the returned gene attention equals the eval softmax exactly; the tensor the output used is 0 or 2x of it. Pinned in `test_reg_loss_reads_pre_dropout_attention_while_the_output_uses_post_dropout`.

### Test changes

- `test_gradient_reaches_every_parameter_except_the_adaptive_weights`: the docstring states the 1e-7 cut margin (about 5x over the largest key-bias noise, about 14x under the smallest real gradient); an exact `== 0` is not available because the key-bias gradients are float noise, not zero.
- `test_genotype_count_is_max_assignment_plus_one`: the docstring now describes the middle-empty case it asserts and points to the trailing-case test.
- An autouse `torch.random.fork_rng` fixture restores the global RNG after each test.
