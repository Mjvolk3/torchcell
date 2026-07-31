---
id: sizye5rq6565dbgr151i6d6
title: Gh_expr_008_arm
desc: ''
updated: 1785531937112
created: 1785531937112
---

## 2026.07.31 - Make the Arm Name the Complete Statement of What Is Being Compared

An arm comparison only means something if the arm name accounts for every difference between
two runs, and this round already lost a wave to the opposite arrangement: `score_decoder_arms.py`
inferred the arm *after the fact* from two config keys (`free_gene_dim` + `perturbation_propagation`),
so eleven distinct arms all resolved to the literal string `A0_baseline` and were averaged into
the baseline. This launcher is the fix -- one arm x one seed, every other hyperparameter frozen in
the config, and the arm's identity, its overrides, and its W&B tags all declared in the SAME `case`
branch so they cannot drift apart.

**The contract.** `bash gh_expr_008_arm.sh <arm> <seed>`; the base config comes from
`TORCHCELL_CONFIG` (`cgt_expr_008` -> `_010` wave 4 -> `_011` wave 5 -> `_012` wave 6 ->
`cgt_expr_v9_mask` for the masked objective), and the arm contributes only its own override list.
Two things are deliberately NOT per-arm:

- **Wave-wide overrides (`TORCHCELL_EXTRA_OVERRIDES`) apply identically to every arm.** An epoch
  budget that differs between arms scores "how long did the plateau look flat" alongside "how good
  is the mechanism". This matters more than it looks: the round found best-by-metric checkpoints at
  epochs 933-1906 against best-by-loss at 147-484, so a short budget does not measure a smaller
  version of the same thing -- it measures a different model.
- **`PYTHONPATH="$PROJECT_ROOT"` and an explicit interpreter.** Without the first, a script run from
  a worktree silently imports the PRIMARY checkout's `torchcell`, where none of the `_008` model code
  exists, and the run *looks like a baseline*. Without the second, bare `python` resolves to base
  conda, which has no hydra.

### A-series -- is the win "identify the deleted gene" or "route along the graphs"?

`A2_self` (propagation hops=0) is the cheapest possible pair term: a hop-0 indicator of "is gene i
itself the deleted gene", which the additive baseline cannot mark at all. `A3_prop_h1` / `A4_prop_h2`
add graph routing on top, so **A2 vs A3/A4 is the attribution** -- if A2 captures most of the gain,
graph propagation is not what is buying it. `A1_free16` is the capacity-only control (unties the
output-gene identity from `h_k`, adds no information).

`A6_prop_sparse` restricts propagation to `regulatory_interaction`, `tflink`, `physical_interaction`
because the dense graphs are informationless at hop 2 by counting: coexpression (avg degree 151)
reaches 22,742 genes and experimental (124) reaches 15,475, both larger than N = 6,607, so their
hop-2 features are near-constant across genes and carry no pair-(p,i) information. They are also 75%
of all 2.44M edges, so they dominate hop 1 with a diffuse 1/151 signal. regulatory (avg degree 6.0,
reaching 36 genes at two hops) is the sharp, directed channel.

### Wave 6 V-series -- a monotone pair-rank ladder

Every V arm differs ONLY in how much (perturbation, gene) interaction the architecture can express,
so the axis is ordered and the question is where it saturates:

| arm | mechanism | pair rank |
|---|---|---|
| `V_ref` | additive `g(h_i + c_b)` | 0 |
| `V_sink` | gated attention, one bounded scalar per head | 9 |
| `V_basis16/32/64` | explicit factor model `sum_j b_ij(h_i) a_j(z_S)` | r |
| `V_film` | diagonal multiplicative at the readout | d = 90 |
| `V_hadamard`, `V_hadamard_add` | diagonal multiplicative at the OPERATOR | d = 90 |

- **Headroom is not the constraint, so a null here is about the mechanism.**
  `lowrank_output_ceiling.py -> results/lowrank_output_ceiling.json` measures the rank-r
  reconstruction ceiling at **0.7265 (r=32)** and **0.7799 (r=64)** `pearson_per_feature`, against
  `reference.replicate_noise_ceiling = 0.7746` and `reference.observed_model_pearson_per_feature =
  0.109`. Rank is not what is stopping the model.
- **Two Hadamard inits because they ask different questions.** `replace` drops the additive path, so
  at init (gamma=0) the model sees NO perturbation and must learn the whole pathway -- a strictly
  harder start. `add` keeps the additive context and is bit-identical to the reference at init
  (verified max|diff| = 0.000e+00), so a null there reads as "the multiplicative term does not help"
  rather than "the arm never got off the ground".
- **`V_sink` is the null-sink retried STARTING OPEN.** The first attempt (`bias_init=-4.0`,
  `N_sink0`) was unfalsifiable in practice: read from four `last.ckpt` at epoch ~187, `null_bias` had
  moved only -4.0 -> -3.92/-3.95, so the sink carried ~1.9% of attention mass versus ~1.8% at init,
  and the ~0 paired delta meant "the mechanism was never in the forward pass". `bias_init=0.0` puts
  p_null at 0.5, so the model must ACTIVELY CLOSE the sink for it to be useless.
  `null_sink_magnitude_match=true` (`N_sink_mm`, `V_sink`) removes the confound that came with it:
  unmatched, ||c||/||c_ref|| = 0.5129 (a permanent ~2x attenuation); matched, 1.0258 -- while the
  pair term itself (per-query CV 0.0971) is identical in both, versus 3.24e-07 for the reference.
- **Regularization arms in the same wave** (`V_drop2/3`, `V_wd1e4/1e2`) because train 0.64-0.72 vs
  val 0.20 is a generalization gap while `weight_decay` sits at 1e-8, i.e. effectively off, and 86%
  of parameters (369,639 + 393,480 of 887,879) live in the encoder + embedding preprocessor.

### v9 M-series -- three questions and two controls

Launched with `TORCHCELL_CONFIG=cgt_expr_v9_mask` into a **separate W&B project**
(`torchcell_019_expr_v9`), because these runs SEE revealed labels and coincide with v8 only at k=0.
The arms are ordered by question:

- **A. Does the objective work at all -- does pearson@k rise with k?** `M_sched`.
- **B. Is cross-gene mixing NECESSARY?** `M_nomix` disables post-perturbation mixing and is a
  NEGATIVE control that should come back inert: with no routing between gene tokens after the
  perturbation, a revealed value at gene j cannot reach gene i. If it moves anyway, the mechanism is
  not what we think it is.
- **C. Does the objective help or hurt the genotype-only score at k=0?** `M_sched` vs `M_off`.
  `M_off` (`~multitask.mask_schedule`) builds the observed-label encoder and the mixing channel but
  never reveals anything, so it **isolates the OBJECTIVE from the PARAMETERS** -- any difference is
  the objective and not the extra modules.

The schedule ladder is sized by a measurement, not by taste. `residual_covariance_diagnostic.json`
puts the **effective rank of the reproducible residual gene-gene structure at 32.78** (split-half
r = 0.8687 against a permutation null of 8.45e-05), so the interesting region is |M| ~ 33:

| arm | schedule | position relative to rank 32.78 |
|---|---|---|
| `M_fine` | `[0,10,30,100,300,1000]` | samples densely across it |
| `M_coarse` | `[0,100,1000]` | skips it |
| `M_lo` | `[0,5,10,30]` | under-determined throughout |
| `M_hi` | `[0,1000,3000]` | already saturated at the first step |

The ridge oracle on this same grid (`masked_conditioning_oracle.json`, 5 draws, seed 0) gives
val Pearson **0.4084 +/- 0.0310 at m=10**, **0.6756 +/- 0.0116 at m=100**, **0.7932 +/- 0.0014 at
m=1000** -- so the m=1000 end sits at the replicate noise ceiling (0.7746) and there is a real,
large signal for the objective to chase. `M_gate_rezero` starts the conditioning gate CLOSED so
"the objective did not help" stays separable from "the pathway was never opened" -- the exact
ambiguity that made the first null-sink attempt uninformative.

Note what this deliberately does NOT claim: `conditioning_gain_after_genotype.json` shows the masked
gain retains **97.5% / 99.2% / 100.6%** (m = 10 / 100 / 1000) after removing a genotype predictor,
i.e. the conditioning signal is essentially orthogonal to genotype. So the masked objective cannot
move the m=0 metric and is an **imputation capability**, which is why it is a separate project rather
than another arm in the wave-6 pair-rank ladder.

### Tags are declared, not derived

```bash
V_basis32)  OVERRIDES=(multitask.response_basis_rank=32)
            ARM_TAGS=(mech-basis pair-rank32 xfer-yes stage-wave6) ;;
```

`TAGS` is always `<config>,<arm>,seed<N>` plus whatever the branch declared, so a wave is filterable
in W&B by `stage-*`, a mechanism family by `mech-*`, and the pair-rank axis by `pair-rank*`.
`xfer-no` marks a mechanism that needs curated yeast graphs and therefore cannot be part of the
organism-general story no matter how well it scores. Arms predating wave 4 set no `ARM_TAGS` and
keep exactly their old three tags, so they stay comparable. A post-hoc lookup table keyed on arm name
would have repeated the original scorer bug one level up; declaring the tags in the branch that sets
the overrides is what makes that impossible.
