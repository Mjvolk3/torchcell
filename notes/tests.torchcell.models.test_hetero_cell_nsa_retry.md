---
id: k3gcwj7sfzy3edk24fgjkv6
title: Test_hetero_cell_nsa_retry
desc: ''
updated: 1790979843906
created: 1790979843906
---

## 2026.10.01 - Phase 20 tests for the 006 NSA retry model

Module under test: `torchcell/models/hetero_cell_nsa_retry.py` (driven by `experiments/006-kuzmin-tmi/scripts/hetero_cell_nsa_retry.py`). `main` is out of scope. The NSA blocks (`torchcell/nn/hetero_nsa.py`) have their own tests; these pin the model's use of them.

### Fixture

4 genes, 3 reactions, 2 metabolites, `physical_interaction` 0->1, 1->2, 2->3, `regulatory_interaction` 3->0, `gpr` {0, 1} -> r0, {2} -> r1, {3} -> r2, bipartite `(reaction, rmr, metabolite)` as the shipped `metabolism_bipartite` config builds, then `HeteroToDenseMask` with the model's node counts. Perturbed samples drop their genes, relabel, and carry `gene.pert_mask`. Model hidden 8, 2 heads, dropout 0.

### Expected values

- Parameter counts: NodeSelfAttention 970 per edge type, SelfAttentionBlock 872 per node type, NSA layer 4 *970 + 3* 872 = 6496, total 6898 (embeddings 72, preprocessor 160, norms 48, pooling 113, head 9). Nine graph names give an NSA layer of 13286.
- Attentional pooling equals a per-group softmax of the gate times `ReLU(W x + b)` (numpy oracle).
- Wildtype path: `LN_gene(nsa(x)["gene"] + x["gene"])`; forward: `z_p = z_w - z_i`, predictions `head(z_p)`.

### Findings

- Batch dependence (`hetero_cell_nsa_retry.py:249-263, 322`): all genes of a batch go to the NSA blocks as one set. The S block attends across samples; the M block gets the collated adj_mask `[2 * 4, 4]` for 6 tokens and NodeSelfAttention pads and crops it silently. A genotype's eval prediction changes with its batch partner and differs from the single-sample run.
- The gpr gene side receives the gene-by-reaction incidence as a gene-by-gene mask (cropped and padded to square).
- Shipped config passes bare `physical`, `regulatory` as `graph_names`; the data carries suffixed relations, so those M blocks never run (gene states equal those of a graph with no gene-gene edges).
- Only gene paths learn: reaction and metabolite embeddings, their layer norms and S blocks, the `(metabolite, reaction, metabolite)` M block (absent from bipartite data) and every edge-attribute MLP get no gradient.
- `PreProcessor` shares one LayerNorm across its two layers (`hetero_cell_nsa_retry.py:69-77`).
- An unbatched perturbed sample embeds genes `0..n-1` and ignores `pert_mask` (`hetero_cell_nsa_retry.py:299-311`).

## 2026.10.05 - Audit follow-up: hash-seed init, flaky tolerance, reach

### Reach of the findings (from the independent audit)

- Batch dependence, non-square gpr mask, bare graph names, dead parameters, shared preprocessor norm: REACHED by `experiments/006-kuzmin-tmi/conf/hetero_cell_nsa_retry.yaml` (`batch_size: 2`) through `scripts/hetero_cell_nsa_retry.py` and the int_hetero_cell validation metrics.
- Unbatched perturbed sample ignores `pert_mask`: latent, the trainer always collates.

### New finding

- `hetero_cell_nsa_retry.py:140-151` passes Python sets of node and edge types to `HeteroNSA`, which builds its `ModuleDict` by iterating the set (`torchcell/nn/hetero_nsa.py:46`). The RNG draw order, and so the weights, follow `PYTHONHASHSEED`: under `torch.manual_seed(0)` the gpr `q_proj.weight` sum is -0.31147 with hash seed 1 and 1.76596 with hash seed 2 (audit). Seeded runs of `hetero_cell_nsa_retry.yaml` are not reproducible across processes. Pinned in `test_init_depends_on_the_python_hash_seed` (two subprocesses); `test_seeded_construction_is_deterministic` now says it covers one process only.

### Test changes

- The batch-dependence test was flaky under hash seeds 19, 37, 80 and 95 (smallest gaps across 101 seeds: 8.2e-4 S partner, 1.06e-4 M partner, 2.55e-5 M alone, against a 1e-4 threshold). It now asserts `not torch.allclose(..., rtol=0, atol=1e-6)`; float noise is about 1e-7, so a batch-independent model still fails it.
- Module docstring: 552 is the feed-forward MLP alone; SelfAttentionBlock is 872 and NodeSelfAttention 970 with the edge MLPs.
- An autouse `torch.random.fork_rng` fixture restores the global RNG after each test.
