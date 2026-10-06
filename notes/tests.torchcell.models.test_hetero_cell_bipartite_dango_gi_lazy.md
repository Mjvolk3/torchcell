---
id: c4ha07r4375pcz25d5o1ayi
title: Test_hetero_cell_bipartite_dango_gi_lazy
desc: ''
updated: 1790979542737
created: 1790979542737
---

## 2026.10.01 - Phase 20 lane A: first behavioral tests of the 006 lazy Dango model

Test file: `tests/torchcell/models/test_hetero_cell_bipartite_dango_gi_lazy.py` (48 tests). Target: `torchcell/models/hetero_cell_bipartite_dango_gi_lazy.py`, everything except `main` (lines 1508-3302, hydra training entry point, not run). Coverage of the module from this file: 41% overall, 511 of 551 statements outside `main` (92.7%).

### Fixture

- Five genes, two graphs. Physical 0->1, 1->2, 2->3, 3->4; regulatory 4->0, 0->2, 1->3. Genotypes {1}, {0, 3}, {0, 2, 4}.
- Lazy samples follow the `LazySubgraphRepresentation` output contract: full graph, `pert_mask` / `mask`, wildtype `perturbation_indices`, per edge type a mask that is True iff neither endpoint is perturbed. One test checks the fixture masks against `LazySubgraphRepresentation._process_gene_interactions`.
- Collation by the real `lazy_collate_hetero` and, for the pre-#549/#571 history, by PyG `Batch.from_data_list`.
- Kept edge positions by hand: {1}: physical {2, 3}, regulatory {0, 1}; {0, 3}: physical {1}, regulatory {}; {0, 2, 4}: physical {}, regulatory {2}. Batched physical mask [0,0,1,1, 0,1,0,0, 0,0,0,0], regulatory [1,1,0, 0,0,0, 0,0,1]; edge offsets of 5 per genotype.
- Tiny model: d = 8, 1 conv layer, GIN, "sum", norm "layer", local predictor 2 heads x 1 layer, gating. Parameter count 40 + 160 + 322 + 370 + 113 + 81 + 42 = 1128; concat 1086; global-only 716; the shipped pairwise block with `aggregation_norm: "layer"` and `pairwise_hidden_dim` 4 adds 381 (1509).
- Shared helpers (`_multigraph`, `_edge_index`, and the eager `_tiny`, `_batch`, `_cell_graph`) are imported from `tests/torchcell/models/test_hetero_cell_bipartite_dango_gi.py`.

### Issue #596 reproduced in this module

With `follow_batch = ["x", "x_pert"]` the batch carries no `perturbation_indices_ptr` / `_batch`, so `batch_assign` is None (lines 1316-1321). The local predictor pools all six perturbed genes of the three genotypes into one [1, 1] score. Because 1 differs from the batch size 3, lines 1351-1364 allocate `zeros(3, 1)` and skip the scatter at line 1356 (`if batch_assign is not None`). The local term is exactly zeros, and with `combination_method: "concat"` the prediction is exactly `0.5 * global`. The global term is identical to the run that follows `perturbation_indices`. Both collaters give the same tensors.

With `perturbation_indices` followed, local equals `predictor(z_w[[1, 0, 3, 0, 2, 4]], [0, 1, 1, 2, 2, 2])`. Under eval batch norm, every batched prediction equals the genotype run alone (1e-6). A batch of one genotype keeps its nonzero local score under either list. In the inert case, backward leaves all 13 local-predictor parameter tensors (370 values) with grad None. The trainer's `_ensure_no_unused_params_loss` then gives them a zero gradient.

Reached configs:

- Preprocessed script: `hetero_cell_bipartite_dango_gi_lazy_preprocessed.py:366`, slurm 074, 075 and 077 (batch 28, 28, 24).
- Lazy script as committed before 53c257c22: slurm 062, 063, 064, 065 and 069 (batch 36, 36, 38, 28, 86).

All seven configs use concat.

### Findings (all reproduced; each test docstring starts with `Finding:`)

1. Issue #596 mechanism, lines 1316-1321 and 1351-1364 (above).
2. The lazy `PreProcessor` norm is PyG `LayerNorm` in mode "graph", called without a batch vector (line 748 via `get_norm_layer`). One mean and one std are taken over all rows and channels. The eager PreProcessor is per node (`nn.LayerNorm`). With the same weights (strict load) under norm "layer", the lazy and eager z_w and z_i differ.
3. Under eval batch norm with the eager weights loaded, the lazy model equals the eager model on every output to 1e-6. So the masked full-graph path is the subgraph path whenever normalization is per node.
4. Under norm "layer", the deleted gene's own embedding changes the kept genes' z_i: max abs change above 0.1 after adding 1.0 to the deleted gene's embedding, and exactly 0 under eval batch norm. Graph-mode LayerNorm (lines 748 and 853) takes its statistics over deleted rows too.
5. Under norm "layer", a genotype's eval prediction depends on its batch mates, because the wrapper LayerNorm has no batch vector (line 853).
6. An empty genotype at the end of a batch with more perturbed genes than non-empty genotypes raises IndexError at lines 1358-1362. The genotype-length `valid_mask` indexes the group-length local tensor. With one gene per genotype the scores are placed correctly. Unreachable with Kuzmin TMI data.
7. An `AttentionConvWrapper` around a GIN MLP with no `nn.Linear` raises UnboundLocalError (lines 788-807) instead of a named refusal.
8. Stage guards test `isnan` only, so an infinite parameter returns `[inf, inf, inf]` with no error. The eager model has checked finiteness since issue #540.
9. Shared with the eager model: `gin_num_layers: 1` emits `gin_hidden_dim`, not `out_channels` (lines 909-912).

Contract differences from the eager model are pinned without being findings:

- The model dropout and activation replace the aggregation config's own (lines 1026-1029). The 006 configs set both dropouts to 0.0.
- `activation=None` is refused by the wrapper and the PreProcessor.
- The default `encoder_type` "gatv2" is refused for the lazy path.

### Mutation check

11 of 11 mutants were killed (runner in the lane scratch directory):

- size-expansion removed
- ptr counts
- edge mask ignored
- deleted genes pooled into z_i
- HyperSAGNN self mask removed
- concat weight changed
- pairwise identity option removed
- node-mode LayerNorm
- scatter_add for scatter_mean
- z_p sign flipped
- wrapper drops the mask

## 2026.10.02 - Audit 1 corrections: reach by model commit, stronger equivalence, two new pins

The file now has 50 tests. Every test runs inside `torch.random.fork_rng(devices=[])` through an autouse fixture, so manual seeds no longer leak.

### Which model code each run used

The commit is the one that added the slurm file.

- Slurm 062-065: model at 4760f653c.
- Slurm 069: model at 8b68e55d5.
- Slurm 073-077: model at 53c257c22. Today's expansion is unchanged since that commit.

Issue #596 names slurm/config numbers only. It gives no W&B run ids, and none are claimed here.

### Corrected reach

1. **Issue #596, the local term.**
   - Exactly zero, prediction exactly `0.5 * global`: slurm 074, 075, 077 (preprocessed script `:366`, model 53c257c22, concat, batch 28/28/24).
   - Slurm 062-065 and 069 as committed: the older expansion wrote local row i to `batch_assign[i] if not None else 0`. Row 0 of each batch got `0.5 * global + 0.5 * (batch-pooled local)`. The other rows got `0.5 * global`.
   - The audit ran both historical files on this fixture. Local was [-2.545, 0, 0] (4760f653c) and [-0.142, 0, 0] (8b68e55d5); I reproduced both.
   - The test now collates through `LazyCollater(dataset, follow_batch=...)`, as both scripts do, plus PyG `Batch.from_data_list`.
   - The zero-gradient statement holds only for 074/075/077. In the older runs, row 0 carried a gradient.
2. **Lazy and eager differ under "layer" through the PreProcessor.**
   - Applies to slurm 069 and 073 onward. The graph-mode PreProcessor norm arrived with 8b68e55d5.
   - Slurm 062-065 used per-node `nn.LayerNorm` in the PreProcessor (4760f653c `get_norm_layer`).
   - The test now also checks the eager PreProcessor against an exact per-node LayerNorm closed form.
3. **The deleted gene leaks into the kept genes.**
   - Applies to all lazy runs with `norms: "layer"`, including 062-065. The conv-wrapper norm was already PyG graph-mode LayerNorm at 4760f653c.
   - Measured on the historical files with this fixture: max z_i change 0.074 (4760f653c) and 0.575 (8b68e55d5).
   - This corrects the audit's "same as finding 2" reach.
4. **A genotype's prediction depends on its batch mates.**
   - Applies to the same runs as finding 3, for the same reason.
   - Max batched-versus-alone difference: 0.086 (4760f653c) and 0.106 (8b68e55d5).

Measurement script for findings 3 and 4: `old_check.py` in the lane A scratch directory.

### Equivalence test rewrite

Before the transfer, every eager parameter and every BN running statistic is filled from one numpy generator:

- parameters: normal(0, 0.5)
- running mean: normal(0, 0.3)
- running variance: uniform(0.5, 2)

The test then checks every lazy state-dict key against the mapped eager value, `num_batches_tracked` included, and checks that the key sets are equal. The 1e-6 output comparisons are kept.

Maximum output difference against eager:

| Mutant | Default BN state | Filled values |
|---|---|---|
| L7 (wrapper BN skipped) | 1.04e-6 | 1.70 |
| L8 (preprocessor second BN skipped) | 1.04e-6 | 0.40 |
| Two edge types' norms swapped | 0.0 | 0.27 |

The init test now asserts Kaiming fan_out: the std of a seeded Linear(256, 128) is within 3% of sqrt(2/128) = 0.125.

### New pins

- **Train-mode BN statistics include the deleted rows.** The running mean and variance follow `0.9 * (0.1 * m_wt) + 0.1 * m_batch` with the batch statistics over all 15 rows, 6 of them deleted genes. A kept-rows-only mean differs.
- **Finding: the embedding tiling ignores the sample sizes.** `expand(batch_size)` at lines 1154-1155 assumes every sample has `gene_num` nodes.
  - Samples of 4 and 6 nodes (gene_num 5) run without error.
  - The preprocessor input is embedding rows [0,1,2,3,4,0,1,2,3,4] against local ids [0,1,2,3,0,1,2,3,4,5].
  - Latent: the 006 producers always emit the full gene count.

### Mutants, round 2

All 9 were killed: the auditor's L1-L8 and a swapped norm map.

## 2026.10.06 - Phase 21: remaining forward branches and init branches

Expected values are the model's own submodules applied to the subset or index each branch promises.

- Two guards are reachable only through an infinite (not NaN) parameter, found by filling each of the 54 parameters in turn with +inf and -inf under both combination methods: an infinite `global_aggregator.transform_nn.0.bias` makes z_w and z_i both +inf, so "NaN detected in perturbation difference (z_p_global)" (line 1304); an infinite local prediction bias gives two +inf gate logits whose softmax is NaN, "NaN detected in gate weights after softmax" (line 1399).
- A wildtype `pert_mask` restricts the wildtype pool to the kept genes (line 1240); a batch without `pert_mask` pools every row by `gene.batch` (lines 1275-1276).
- `forward_single` skips a relation absent from the data and one whose store holds only a mask (lines 1177, 1188).
- `PairwiseGraphAggregation` skips a missing second graph (options [pp(a, a), a]); the conv wrapper passes kwargs (`edge_weight`) to a non-GIN conv.
- `_init_weights` on any module tree resets nn.LayerNorm and BatchNorm1d to (1, 0) and leaves a GATv2Conv untouched (the eager model's pinned finding; the lazy factory refuses GATv2, so the branch is dead here); a GATv2Conv given the `lin_src` / `lin_dst` / `att_src` / `att_dst` names is re-initialized exactly as the branch says, checked by replaying the draws.

Left uncovered: the remaining NaN guards (each sits after a guard that a NaN or inf reaches first), `HeteroConvAggregator` lines 360 and 380-381 (unreachable through the constructor), and `main`.

## 2026.10.06 - Phase 21 audit 2

The four Phase 21 manual seeds now run inside `torch.random.fork_rng()`, and the two `_init_weights` ignores use the two-sided `[arg-type, unused-ignore]` form.
