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

## 2026.09.30 - Issue #540 findings retired

Retired findings, now asserted as contracts: the Wasserstein loss trains and its epoch-1 component line equals a fresh `MleWassSupCR` on the seed-0 reference (`test_main_with_the_shipped_wasserstein_loss_prints_its_own_components`, replacing the `KeyError` pin); every stage check catches +/-inf with `NaN or inf detected in <stage>` at the first stage it reaches (seven cases, replacing the three "becomes NaN" cases, the opposite-heads test and the concat-returns-inf pin); a GIN MLP with no Linear raises the named `ValueError`, and a nested Sequential still finds its last Linear; "gelu" builds GELU in the preprocessor and every wrapper, "tanh" builds Tanh, an unregistered name is refused with the registered list, and the wrapper's `activation=None` is Identity; the aggregation config's own `dropout` 0.0 wins over the model's 0.25 and a config without it inherits 0.25; `main` seeds from the config (caller RNG 123, config seed 0, metrics equal `_tiny(seed=0)`); `mle_dist_supcr` and `mle_wass_supcr` save `loss_components_evolution` too. Still pinned: `aggregation_norm` is read by nothing (1424 parameters with or without it). 65 cases.

## 2026.10.01 - aggregation_norm finding retired

Retired the last #540 pin ("`aggregation_norm` is read by nothing, 1424 parameters with or without it"). Now asserted: null and absent build main's module exactly, 1120 parameters and 56 state_dict keys for "sum", 1765 parameters and those 56 keys plus 16 listed aggregator keys for "pairwise_interaction", with bitwise-equal weights under one seed; "layer", "batch", the string "none" and the falsy non-null values "", False and 0 are refused with the exact `AggregationNormNotImplementedError` message for sum, cross_attention and pairwise_interaction, from the model and from `HeteroConvAggregator`, without mutating the caller's dict; `main` passes the key through (a "layer" config is refused before the plot directory exists, a null config builds the 1765-parameter model and trains). The shipped-flags test now uses `aggregation_norm: None`. 87 cases (65 before). Mutating the check to truthiness (`if aggregation_norm:`) fails the 9 falsy cases.

## 2026.10.01 - Phase 19: _init_weights by replay, stage guards, the aggregator fallback

94 cases (87 before).

- `_init_weights` is replayed: seed 7, then Kaiming normal (fan_out, relu) and zero bias for every `nn.Linear`, ones/zeros for `nn.LayerNorm`/`nn.BatchNorm1d`, in `nn.Module.apply` order, with `torch.nn.init` as the oracle, reproduces every parameter bitwise for GIN, GATv2, and GATv2 plus a probe carrying the GATConv attribute names (the only way the GATv2 branch runs).
- Built with and without the init at one seed, the parameters that agree are exactly the embedding, the ReZero beta, GIN eps, the PyG LayerNorms, the preprocessor `nn.LayerNorm`, and for GATv2 every GATv2Conv parameter (`lin_l`/`lin_r` are PyG `Linear`); everything else is an `nn.Linear` parameter.
- Stage guards: +inf injected at the batch encoder names "perturbed embeddings (z_i)", at the batch pooling "global perturbed embeddings (z_i_global)", and FINITE pooled values +3e38 / -3e38 overflow z_p to +inf, named "perturbation difference (z_p_global)". The guards at :1049, :1111, :1130, :1143, :1150, :1166 and :1175 only see checked rows, softmax weights of finite logits or convex combinations of finite values, so they cannot fire and stay uncovered.
- `HeteroConvAggregator`'s "fallback to sum" (lines 336-339) is reachable only by rewriting the method after construction (then 2x + 3x = 5x, no weights); the empty-output skip (line 318) cannot run.

Coverage of `torchcell/models/hetero_cell_bipartite_dango_gi.py` from this file: 86.9% -> 88% (`main` out of scope).

## 2026.10.02 - Audit round 2 corrections

94 cases.

- The GATv2 probe rewrite: a plain `GATv2Conv(8, 8)` gets `torch_geometric.nn.Linear(8, 4)` `lin_src`/`lin_dst` with biases filled to 7.0, plus 7-filled `att_src`/`att_dst`. `_init_weights` sets both biases to exactly 0, and only the GATv2 branch can do that, because PyG `Linear` is not `nn.Linear`. Mutants deleting either bias zeroing die. The subclass and its `misc` ignore are gone.
- The guard test's last assertion now checks what the module's pooling handed on: the wildtype [1, 8] all 3e38 and the batch [2, 8] all -3e38, both finite. The inf therefore arises in the module's own subtraction. The two `assignment` ignores are replaced by `monkeypatch.setattr`.
- My new tests run inside `torch.random.fork_rng()` (fixture `restore_rng`), because `_tiny` reseeds globally.
- Reach: `lin_l`, `lin_r` and `att` keep PyG's own init on real GATv2Conv layers (the existing finding); the aggregator fallback-sum branch is unreachable from a constructed layer.
