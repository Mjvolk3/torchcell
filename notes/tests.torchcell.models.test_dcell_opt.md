---
id: dngx2g3207ne0nafj943syh
title: Test_dcell_opt
desc: ''
updated: 1790820773223
created: 1790820773223
---

## 2026.09.30 - DCellOpt root key and paper loss (issue #554)

New. `DCellOpt` on the conftest hierarchy (seed 0, B = 3) declares `root_key` GO:0; its prediction and GO:0 are equal but distinct tensors. The head values are pinned to 1e-6 and the loss is the paper value 0.4857438 + 0.3 * (0.3546567 + 0.3064786) = 0.6840844, not the 0.8298076 an identity-based root skip gave (issue #554 review).

## 2026.10.01 - Phase 19: the optimized path equals the reference DCell

25 cases (1 before). Three ontologies over the conftest genes: the fixture (dims {0: 5 -> 2, 1: 2 -> 2, 2: 2 -> 2}), an unequal-children variant (`term_gene_counts` [0, 2, 6]: term 2 outputs 3, root input 2 + 3 + 1 = 6) and the three-level DAG of `test_dcell.py` (min 1, ratio 0.3: outputs {0: 2, 1: 2, 2: 1}). The reference `DCell` gets standard-normal parameters and non-trivial running statistics and its state is copied into `DCellOpt` through an explicit name map; the seven DCellOpt index buffers are the only missing keys.

- Root prediction, every `GO:k` head and every subsystem activation agree BITWISE (`torch.equal`) for B = 1 (eval) and B = 3 with different knockouts ({0}, {2, 3}, {1}) in eval and train mode, on all three graphs; train-mode running statistics agree after the step; the DCell objective (`DCellLoss(alpha=0.3, aux_reduction="sum")`) gives bitwise-equal gradients for every parameter. No disagreement was found.
- `all_activations_tensor` holds each term in its own width with zero padding; `_extract_gene_states_parallel` equals the per-term loop (term 1 [[0, 1], [1, 1], [1, 0]], term 2 [[1, 1], [0, 0], [1, 1]], the gene-less root an all-zero row); grouping by (input dim, output dim) in first-seen order is hand-listed per graph; `num_parameters` is out * (in + 3) per subsystem plus out + 1 per head (36 + 9, 43 + 10, 31 + 8) and equals the reference; profiling prints one line per stratum and leaves predictions unchanged; an ontology without stratum 0 is refused with "No root terms found in stratum 0".

Finding: `has_subsystem` is True for every term (dcell_opt.py:105-106, 312), so the skip branches at :474 and :531 cannot fire, and `_group_terms_by_dimensions` and `_extract_gene_states_parallel` are called nowhere in `torchcell/` or `experiments/`.

Coverage of `torchcell/models/dcell_opt.py` from this file: 53% -> 66% (`main`, 800-1148, out of scope; the CUDA-stream branch needs a GPU).

## 2026.10.02 - Audit round 2 corrections

27 cases.

- The `has_subsystem` docstring now states the true consequence: with the flag forced off, only that term is skipped in its stratum, and a full forward then fails at the root's Linear with "mat1 and mat2 shapes cannot be multiplied (3x3 and 5x2)" (the test asserts it). Reach: latent, since every 006 config uses `model_version: dcell`.
- The two helper tests (`_group_terms_by_dimensions`, `_extract_gene_states_parallel`) now say in their docstrings that nothing in `torchcell/` or `experiments/` calls these helpers. Reach: latent.
- New: two roots at stratum 0 (terms 0 and 2). The prediction is the head of `root_terms[0]` only (dcell.py:341, dcell_opt.py:677-678). Root 2's head is computed, differs from the prediction, and reaches the loss only as an auxiliary term, with no warning (a Finding). Not measured whether any served GO hierarchy has two roots. DCellOpt equals the reference on every head.
- New: with `output_size=2`, every head and the prediction are [3, 2] and equal the reference bitwise; 54 parameters (45 + 9).
- Global reseeds wrapped in `torch.random.fork_rng()`. Mutants killed: `root_terms[-1]`, heads built with output 1.

## 2026.10.06 - Phase 21: the plain-int term index

`_extract_gene_states_for_term(1)` equals the tensor index (term 1 states [[0, 1], [1, 1], [1, 0]] for knockouts {0}, {2, 3}, {1}); `_prepare_term_input_optimized(0)` with children 1 and 2 written as 1s and 2s gives [1, 1, 2, 2, 0] per row for int and tensor indices alike. Left uncovered: the CUDA-stream branch of `_process_stratum_parallel` (lines 554-580, CUDA hidden), line 764 (unreachable: the gene-state block always has at least one column) and `main`.

## 2026.10.06 - Phase 21 audit 2

The two int-index ignores use the two-sided `[arg-type, unused-ignore]` form.
