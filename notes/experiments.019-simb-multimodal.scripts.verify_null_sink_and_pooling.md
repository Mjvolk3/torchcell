---
id: 2a2jhnda6dnu8cbqoc77z1c
title: Verify_null_sink_and_pooling
desc: ''
updated: 1785532016826
created: 1785532016826
---

## 2026.07.31 - Make a null result on the null sink interpretable: six contracts that must hold before the wave gets a GPU

The first null-sink attempt was uninformative because a flat arm had two readings that no
measurement could separate -- "the mechanism was active and bought nothing" versus "the
mechanism was never active." This script closes that hole on CPU, in seconds, before the
wave launches: six assertions that each pin down one way the arm could produce an
uninterpretable number, written to
`experiments/019-simb-multimodal/results/verify_null_sink_and_pooling.json` and gating the
launch with a nonzero exit (`FAILED: ... -- do NOT launch the wave`).

All six pass on the committed run (`hidden_dim=90`, `num_heads=9`, 6607 genes, seed 0, both
modules in `.eval()` with dropout 0, so two builds are bit-comparable):

| contract | what it rules out | measured |
| --- | --- | --- |
| C1 sink off unchanged | the new code path silently perturbing the arms that do not use it (A1_ref, A2_prop_sparse, A3_deep2) | single-perturbation context is query-independent, per-gene spread 2.98e-09 |
| C2 sham is identity | the "paired null" A5_sham measuring something, in which case it is not a null | `bias_init=-20`, `trainable=false` matches sink=off to 5.96e-08 |
| C3 sink restores query dependence | the arm being scored without its claimed mechanism ever engaging | spread 2.98e-09 -> 0.0624, a 2.09e+07x change |
| C4 gradient revives W_Q/W_K | the dead attention projections staying dead, so the arm trains the same function | `W_Q/W_K` grad 0.0 -> 56645.7; `null_bias` grad 418.7 |
| C5 sum inert at one perturbation | the pooling default quietly changing the 95.4% of the build that is a single perturbation | sum 0.1022434 == mean 0.1022434 |
| C6 sum is cardinality-aware | shipping a pooling change that does nothing at any cardinality | S={A,A} vs S={A}: mean 0.1022434 vs 0.1022434 (blind), sum 0.1868488 vs 0.1022434 (aware) |

- **C2 is the load-bearing one.** A5_sham is the empirical paired null for the whole
  null-sink comparison, and it runs the sink code path with `p_null ~ 2e-9`. If it were not a
  numerical identity to sink=off, the paired difference would contain the sham's own effect
  and the arm's estimate would be biased by an unknown amount. 5.96e-08 is float32 noise.
- **C4 is the direct measurement of the round's degeneracy claim, in gradient form.** With
  the sink off at `|S|=1` the query/key block of `cross_attn_layers[0].in_proj_weight`
  receives *exactly* 0.0 -- not small, zero -- because softmax over a one-element key set is
  the constant 1. Adding one zero-valued K/V column plus a learned scalar logit makes
  `alpha_i = sigmoid(q_i . k_p / sqrt(d_k) - null_bias)`, which depends on the measured gene
  `i`, and the gradient becomes nonzero. That is the pair-`(p, i)` term the round says the
  model cannot express.
- **C5/C6 are why sum pooling landed as a default rather than as an arm.** Inert on 95.4% of
  the expression build, so it costs nothing to adopt now, and it removes the
  cardinality-blindness that would otherwise have to be discovered later during
  mixed-cardinality joint training.
- Run it from the repo root: `PYTHONPATH=<worktree> ~/miniconda3/envs/torchcell/bin/python
  experiments/019-simb-multimodal/scripts/verify_null_sink_and_pooling.py`. It touches no
  dataset and no GPU; the config surface it guards is `perturbation_head.{pooling, null_sink,
  null_sink_bias_init, null_sink_trainable}` in `conf/cgt_expr_008.yaml`.
