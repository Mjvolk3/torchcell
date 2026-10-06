---
id: 1mg9d5m4dqrw2riqifhyjhz
title: Test_hetero_cell_bipartite_dango
desc: ''
updated: 1790979852010
created: 1790979852010
---

## 2026.10.01 - Phase 20 tests for the 003/005 bipartite DANGO model

Module under test: `torchcell/models/hetero_cell_bipartite_dango.py` (driven by `experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango.py`). `main` is out of scope.

### Fixture

The four-gene shape of `test_hetero_cell_bipartite_dango_gi.py`, restated with the relations this model hard-codes: `physical_interaction` 0->1, 1->2, 2->3, `regulatory_interaction` 3->0, 0->2, `gpr` {0, 1} -> r0, {2} -> r1, {3} -> r2, `(reaction, rmr, metabolite)` r0->m0, r1->m1, r2->m1 with stoichiometry [-1, 1, 2]. Samples carry `pert_mask` for all node types, `cell_graph_idx_pert` and `x_pert`, collated with `follow_batch=["x_pert"]`. Model hidden 8, 1 layer, single-head GATv2, dropout 0.

### Expected values

- Parameter counts: embeddings 72, preprocessor 160 (shared norm), convs 3 * 176 + 184 = 712, interaction predictor 298, pooling 113, head 18, total 1373.
- Interaction attention with identity projections: a pair gives `x_i + 0.1 x_j`; three genes use `softmax(x_i . x_j / sqrt 8)` over the other two (numpy oracle).
- Interaction predictor with identity projections: set score `mean_i (w . 0.01 x_j^2 + b)`, empty middle set 0.
- Forward wiring: `fitness = head(z_w - z_i)[:, 0:1]`, interaction from WILDTYPE states at `cell_graph_idx_pert`, grouped by `x_pert_ptr`.

### Findings

- `hetero_cell_bipartite_dango.py:583-587`: predictions are `[(1 - alpha) fitness, alpha interaction]` with `alpha = (current_epoch + 1) / 20`, and nothing advances `current_epoch`, so alpha stays 0.05 (fitness column 0.95 x, interaction column 0.05 x).
- `hetero_cell_bipartite_dango.py:543`: reads `cell_graph_idx_pert`, which `SubgraphRepresentation` renamed to `perturbation_indices` in commit f72cabc7a; current batches raise AttributeError.
- PyG LayerNorm in graph mode without a batch vector (`hetero_cell_bipartite_dango.py:283`) makes the fitness column depend on the batch partner in eval.
- The metabolic branch (gpr, rmr convs, reaction and metabolite embeddings) never receives a gradient, and the head's second output is never read (zero gradient on its row).
- `beta` starts at 0.1 though the comment says 0; scores are scaled by `sqrt(hidden_dim)` and `num_heads` never splits features.
- `out_channels` is ignored (head always 2 outputs); any activation other than "relu" builds SiLU; several shipped config keys are read by no code.

## 2026.10.05 - Audit follow-up: reach

- All findings are historical. The 003 and 005 `hetero_cell_bipartite_dango.py` scripts (via `fit_int_hetero_cell`, which logs `predictions[:, 0]` and `predictions[:, 1]`) ran with 0.95 x fitness and 0.05 x interaction before commit f72cabc7a. At HEAD those scripts crash on the `cell_graph_idx_pert` read. Experiment 004 imports the class but never builds it.
- An autouse `torch.random.fork_rng` fixture restores the global RNG after each test.
