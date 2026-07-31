---
id: vgwyoe0qsy8rj7jcxyf2ct3
title: Perturbation_selector_degeneracy
desc: ''
updated: 1785531995102
created: 1785531995102
---

## 2026.07.31 - Proving there is no (perturbation, gene) pair term to lose at |S|=1

The whole decoder-rank ladder of this round is only worth GPU time if the baseline genuinely
cannot express a term that depends on the pair (perturbed gene `p`, measured gene `i`). This
script settles that by measurement on the REAL `EquivariantPerturbationTransform` rather than
by reading the algebra: at `|S| = 1` -- 95.4% of the expression build -- the perturbation
cross-attention is a degenerate softmax over a one-element key set, and half its parameters
are provably dead.

- **Configuration is _006's measured winner**, so the numbers describe what is actually being
  swept: `hidden_dim 90`, `num_heads 9`, `num_genes 6607`, `seed 0`. Script
  `experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy.py` ->
  `experiments/019-simb-multimodal/results/perturbation_selector_degeneracy.json`.

- **The degeneracy, three regimes.** The selector is not broken -- it is starved of keys. It
  comes alive the instant there are two of them.

  | regime | attention weight set / range | across-query spread of the context |
  | --- | --- | --- |
  | `S = 1` (Kemmeren) | `{1.0}` exactly | `3.31e-09` |
  | `S = 1` + one null key | `0.0525 - 0.9581` | `0.3202` |
  | `S = 2` (Sameith) | `0.0218 - 0.9782` | `0.5744` |

  The `|S|=1` spread is `1.10e-08` RELATIVE to the attended vector's own magnitude (`0.3014`),
  i.e. float32 epsilon: every one of the 6607 gene tokens receives a bit-identical context.

- **The decisive evidence is an ablation, not a small number.** Re-drawing `W_Q` and `W_K`
  at `std=10` changes the output by `0.0` EXACTLY -- not approximately. `16,200` of `32,760`
  attention parameters receive no gradient at `|S|=1`. (The script counts the two `d x d`
  projection blocks, `2 * 90 * 90`; the in-code comment in
  `torchcell/models/equivariant_cell_graph_transformer.py` quotes `16,380`, which also counts
  the equally-dead `b_Q` / `b_K`. Same claim, two accounting conventions.)

- **What that means for the model class.** With `alpha == 1` the context collapses to an affine
  map of the deleted gene alone, `c_b = W_O(W_V h_p + b_V) + b_O`, one `d`-vector shared by all
  `N` genes, and the post-perturbation token is `H_pert_i = g(h_i + c_b)`. Gene identity and
  strain identity meet exactly ONCE, by addition, and the next op is a LayerNorm that discards
  the magnitude of the sum -- which is where `<h_i, c_b>` lives. So no term anywhere in the
  forward pass is a function of the pair `(p, i)`. On a single-deletion build this is the
  entire task, not an edge case. The module is a `d x d` linear layer wearing an attention
  costume.

- **The null-key probe bounds the cheapest possible repair.** Prepending one learned token to
  K/V gives the softmax a second denominator term, so the real-key weight becomes
  `sigmoid(q_i . k_p / sqrt(d) - b)` and is finally a function of the QUERYING gene: query
  spread `0.3202`, or `9.67e7x` the `|S|=1` spread, for `+90` parameters. Note this bounds
  EXPRESSIVITY only -- the wave-4b arm that trained it (job 1413, 8 runs x 300 epochs,
  `cgt_expr_010`, 4 paired seeds) came back indistinguishable from the reference
  (paired within-seed delta `+0.0024`, sd `0.0090`), which is what pushed the round from "add
  a pair term inside the attention" to the explicit rank ladder in the decoder.

- **Run it with `PYTHONPATH` pinned to the worktree** -- the primary checkout carries a
  different model and will silently measure the wrong module:

  ```bash
  PYTHONPATH=<worktree> ~/miniconda3/envs/torchcell/bin/python \
      experiments/019-simb-multimodal/scripts/perturbation_selector_degeneracy.py
  ```
