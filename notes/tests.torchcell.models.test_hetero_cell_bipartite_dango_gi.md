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

## 2026.09.30 - Phase 17: exact counts per build, masked-softmax identities, the training script

Thirty-four to forty-nine functions (63 cases, 12 s), 39.6 to 87 percent alone; almost all of the old gap was the 1,700-line `main()` (lines 1196-2891). Exact parameter totals for all eight encoder x aggregation builds (cross_attention +304, pairwise +645, gatv2 with 2 heads +30) with a gradient reaching every parameter; the depth flags (`num_layers=2` 1442, `num_attention_layers=2` 1409, every ReZero beta 0.01); two masked-softmax identities (q/k get exactly zero gradient in two-gene samples, GATv2 `att`/`lin_r` exactly zero at in-degree 1, both nonzero with a third gene or a second in-edge); genotype permutation permuting predictions, gates and z_i; an infinite parameter named at the first stage where it becomes NaN; the wrapper with `norm=None` exactly relu(proj(conv)); `main()` with the loader, genome, graph, `load_dotenv`, `timestamp` and `savefig` faked (the plot schedule, banners and roots at lr 0, the genome built `overwrite=False`, the printed final metrics equal to a same-seed reference model's, the warmup scheduler stepped once per epoch, an unknown loss refused before the plot directory exists, one component line per epoch for icloss and mle_dist_supcr).

Findings: the script reads `loss_dict["weighted_dist"]` while `MleWassSupCR` names that entry `weighted_wasserstein`, so `mle_wass_supcr`, the loss the shipped `hetero_cell_bipartite_dango_gi.yaml` selects, raises `KeyError` in epoch 1 (line 2316); every stage check is `isnan` only, so a single +inf head under concat returns inf with no error; a GIN MLP with no Linear raises `UnboundLocalError` (612-629); with the 006 config `activation: gelu` becomes SiLU everywhere, `aggregation_norm` is never read and the aggregation config's `dropout` is overwritten by the model dropout (571, 645, 814-817); only icloss gets the final loss-components figure (2775).
