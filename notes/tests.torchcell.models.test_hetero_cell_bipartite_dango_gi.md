---
id: zpwb986jnnjadq6k8lmb1lb
title: Test_hetero_cell_bipartite_dango_gi
desc: ''
updated: 1790716706895
created: 1790716706895
---

## 2026.09.29 - Hand-built four-gene batch with closed-form identities

Fixture: a four-gene, two-graph `HeteroData` batch built by hand from the keys `forward` reads (node and edge types, batch vectors, `perturbation_indices*`, `pert_mask`), and a tiny configuration whose parameter count is derived component by component to 1,120 (1,078 in "concat" mode). Every test pairs shapes with an identity: the norm and rolling-correlation closed forms, the shared-norm `PreProcessor` count (208 for three layers), GATv2 and GIN wrapper shapes, the single-graph attention weight of exactly 1, the two-gene attention closed form, the zero-init mixing identity (`beta` starts at 0.01; at 0 the output equals the input bit for bit), gate weights summing to 1 with the prediction as the gate-weighted sum, invariance under gene relabeling, seeded determinism, a finite gradient on every parameter, and the five named NaN errors in check order.

Findings pinned (Phase 10 of [[test-campaign.2026.09.25]]): graph-mode `LayerNorm` without a batch vector makes an eval-mode prediction depend on its batch mates; SiLU is the silent default for any activation but "relu" and for `activation=None` in the conv wrapper; a one-layer GIN outputs the hidden width; the GATv2 init branch is dead; local scores are misrouted when the pointer fields are absent or the last sample has no perturbations; a batch without `pert_mask` raises `IndexError`. Coverage: 0% to 39.6%; `main()` (lines 1196 to 2891) needs the genome and a served graph and is untested.
