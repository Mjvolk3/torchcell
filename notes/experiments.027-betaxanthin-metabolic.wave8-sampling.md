---
id: uk37i2gq356weu4vlttxnlo
title: Wave8 Sampling
desc: ''
updated: 1789246455730
created: 1789246455730
---

# Wave 8: can the flux module sample its own polytope?

## 2026.09.12 - The question, and the guidepost

Constraint-based methods read a genotype's phenotype off its flux polytope by SAMPLING it.
Flux Cone Learning (Merzbacher 2025, mirror
`/scratch/projects/torchcell-scratch/torchcell-library/merzbacherAccuratePredictionGene2025/paper.md`,
Methods L96-108, SI Table S2) is the closest published analogue to what our flux layer does,
and its yeast numbers are the guidepost this wave is measured against.

| Quantity | Flux Cone Learning, Yeast9 | Our flux layer, waves 5 to 7 |
|---|---|---|
| Sampler | OptGPSampler, thinning k = 5,000 | one Gaussian latent draw, no chain |
| Samples per genotype cone | 124 | 1 (`_latent` draws once per forward pass) |
| Genotypes sampled | 1,121 single deletions | every genotype in the batch, at forward time |
| Sample matrix | 143,716 rows x 4,130 fluxes | never materialized |
| What carries the phenotype signal | the cone's GEOMETRY, which fluxes a deletion forces | one point in the polytope |

Their released Yeast9 sample set covers SINGLE deletions. Our genotypes include doubles,
triples and combinatorial designs for which no sampled set exists or ever will, so a design
that depends on pre-sampled flux data cannot serve this project. Whatever wave 8 builds has
to get its signal from the genotype's own bounds at forward time.

## 2026.09.12 - Step 0: the latent spread, measured before any GPU

Script: `experiments/027-betaxanthin-metabolic/scripts/audit_latent_spread.py`.
Results: `experiments/027-betaxanthin-metabolic/results/wave8_latent_spread.json`.
Four trained checkpoints from `$DATA_ROOT/checkpoints-027/`, rebuilt exactly as
`analyze_projection.py` rebuilds a cell, measured on one real batch of 128 genotypes with
64 draws and 256 rays, on CPU.

![](./assets/images/027-betaxanthin-metabolic/wave8_latent_spread_2026-09-12-15-53-48.svg)

**Figure 1. The latent posterior of four trained 027 cells, and what a draw from it moves.**
All four at seed 2700. **a**, the posterior standard deviation per latent dimension, p10 to
p90, against the value the spread head is initialized at, `exp(latent_log_std_init) =
exp(-3) = 0.0498`. Three of the four sit exactly on the initialization line because their
arm never turned the latent draw on, so the spread head received no gradient; the fourth is
the only 027 cell trained with `latent_stochastic`, and its posterior is 130x narrower than
the initialization. **b**, the across-draw flux standard deviation as a fraction of each
reaction's own box width, over all 128 genotypes and 4,131 reactions, against the
`1/sqrt(12) = 0.289` a uniform sample of that interval would give. **c**, the supervised
readout's prediction standard deviation across draws divided by its standard deviation
across genotypes; the amino-acid concentration readout has no bar because its prediction is
not a function of the flux at all.

### What the arms actually ran

`latent_stochastic` is set by exactly ONE entry of 026's `ARMS` registry
(`flux_projection_stochastic`, `experiments/026-metabolism-flux/scripts/train_flux.py`
L235). Every other flux arm, including every wave-6 and wave-7 projection cell, ran with
`latent_stochastic: false`, which makes `_latent` return `mu` deterministically and `c_kl`
exactly zero. The KL weight is therefore not the interesting number for those arms: the
spread head's zero-initialized weights and constant bias were never moved, so sigma is
`exp(-3) = 0.049787` for every dimension and every genotype (measured across-genotype sd of
sigma: 1.86e-8, i.e. float32 noise).

| Cell | arm | `latent_stochastic` | `kl_weight` x scale | `c_kl` first -> last | test Spearman |
|---|---|---|---|---|---|
| `wave6_proj_flux_projection_s2700` | flux_projection | false | 1e-3 x 0.0 = 0 | 0 -> 0 | 0.1170 |
| `wave6_proj_..._tdg_stoch_ue_s2700` | flux_projection_anchored_tdg_stochastic | **true** | 1e-3 x 0.1 = 1e-4 | 2.527 -> 7.870 | 0.0380 |
| `wave7_aa_turnover_e200_s2700` | flux_projection_anchored | false | 1e-3 x 0.1 = 1e-4 | 0 -> 0 | 0.0023 |
| `wave7_aa_conc_e200_s2700` | flux_projection_anchored | false | 1e-3 x 0.1 = 1e-4 | 0 -> 0 | 0.0648 |

The KL sits INSIDE the ramped constraint sum (`FluxLayer.constraint_loss` multiplies the
whole sum by `constraint_scale`), so the effective weight on `c_kl` is
`kl_weight * constraint_scale`.

### Finding 1: sigma is not collapsed on the deterministic arms, it is frozen at its init

Three of the four cells carry sigma = 0.049787 exactly, on every dimension. That is not a
collapse and it is not a learned value: it is `latent_log_std_init` untouched. Post-hoc
sampling from those checkpoints, which is what `analyze_projection.py --draws` and
`audit_top_strain_feasibility.py` do, therefore draws from an UNTRAINED spread. Every draw
audit in waves 6 and 7 was a draw from `exp(-3)`, not from anything the model learned.

### Finding 2: where sigma WAS trained, the objective collapsed it by 130x

The one cell trained with the latent draw on (`flux_projection_anchored_tdg_stochastic`,
seed 2700) ended with a median sigma of 3.857e-4 against the 4.979e-2 it started at, a
factor of 129. The KL decomposition attributes it: of the final `c_kl` of 7.870, the sigma
half `0.5 * mean(sigma^2 - 1 - 2 ln sigma)` is 7.387 and the mu half `0.5 * mean(mu^2)` is
0.0017. The KL is the only term that pushes sigma UP and at an effective weight of 1e-4 it
did not hold: `c_kl` rose monotonically from 2.527 at epoch 0 to 7.870 at epoch 79. So the
data and physics terms shrink the posterior, and the prior as weighted does not resist.

Caveat, stated because the number invites over-reading: that cell also never left the
betaxanthin mean plateau (`val_pred_sd_ratio` 0.0000 from epoch 5 to epoch 79, test Spearman
0.0380). Its sigma collapse is measured; whether the collapse causes the prediction collapse
or accompanies it is NOT measured. Hypothesis (untested): a deterministic head is the
easiest way to reduce a per-draw data loss, so any sampled arm without a spread-preserving
term will walk to sigma = 0.

### Finding 3: the draws do not explore the polytope, they overshoot it

At the frozen sigma the draw is not small relative to the polytope; it is LARGER than it.
Along random admissible latent directions from the box center, the box chord and one draw's
displacement `sigma_rms` compare as:

| Cell | admissible dim / 1,538 | zero-width reactions | chord p50 | `sigma_rms` | `sigma_rms` / chord p50 | rays where sigma exceeds the chord |
|---|---|---|---|---|---|---|
| betaxanthin projection | 1,351 | 211 | 2.69e-2 | 4.98e-2 | 1.85 | 1.000 |
| betaxanthin, sigma trained | 1,263 | 344 | 0.00 | 5.06e-4 | -- | 1.000 |
| aa turnover | 1,529 | 10 | 1.48e-7 | 4.98e-2 | 3.4e5 | 1.000 |
| aa concentration | 1,529 | 10 | 4.26e-7 | 4.98e-2 | 1.2e5 | 1.000 |

In flux space the same thing reads as the across-draw sd per reaction against that
reaction's box width, with `1/sqrt(12) = 0.289` the value a uniform sample of the interval
would give:

| Cell | sd / box width p50 | p90 | as a multiple of uniform | reactions moved |
|---|---|---|---|---|
| betaxanthin projection | 0.145 | 1.300 | 0.50 | 0.877 |
| betaxanthin, sigma trained | 0.000 | 0.0005 | 0.00 | 0.106 |
| aa turnover | 4.398 | 161.4 | 15.2 | 0.877 |
| aa concentration | 0.692 | 39.9 | 2.40 | 0.874 |

Two regimes, and neither is sampling. On the plain betaxanthin projection arm, whose box has
no enzyme-capacity bound, the draws land at half the spread of a uniform sample of the box:
that is the one arm where the latent draw is in the right ballpark. On the enzyme-constrained
amino-acid arms the capacity bound makes the box four to seven orders of magnitude narrower
than the draw, so every draw leaves the box by a wide margin and the alternating projection
clips it back onto a face. That is why 52 to 72 percent of the head's own reactions sit
exactly ON a bound.

### Finding 4: per-coordinate chords are the wrong instrument, and why that matters for Step 2

For all four cells, the fraction of the 1,538 basis directions that are BLOCKED, meaning
they move some reaction whose box has zero width, is exactly 1.000 -- with as few as 10
zero-width reactions. The polytope is not degenerate: the admissible subspace, the latent
moves that keep every blocked reaction pinned, has dimension 1,263 to 1,529 of 1,538. The
null basis is built from the stoichiometry alone and knows nothing about the bounds, so no
single latent coordinate is a feasible direction. Any sampler that works in these
coordinates has to respect that; the diagnostic gets it from an SVD of the zero-width block
`N[Z, :]`.

### Finding 5: the amino-acid concentration readout cannot respond to a draw at all

`FluxConcentrationReadout` (`train_bx.py` L1443) emits
`mixture_log10_molar(flux["thermo_log_c"], ...)`, and `thermo_log_c` is
`lo + (hi - lo) sigmoid(base + W context)`, a function of the genotype context alone. The
flux vector never enters. So a K-draw version of the wave-7 concentration cell is null by
construction, whatever K is, and the (b) arm of the planned K-draw experiment has to move to
a readout that reads the flux. The candidate, the wave-7 amino-acid TURNOVER readout, scores
0.0023 mean test Spearman against the pooled control's 0.0500, so it is a dead readout to
build a sampling experiment on.

### Step 0 conclusion

The K-draw experiment is NOT moot, but only one of the two planned arms survives contact
with the measurement.

- Arm (a), betaxanthin plain projection: alive. The readout moves with the draw (across-draw
  prediction sd is 4.75 percent of the across-genotype sd) and the draws cover half a uniform
  sample of the box. K draws will average over a real distribution.
- Arm (b), the amino-acid concentration head: dead by construction, the readout does not read
  the flux. Its turnover alternative reads the flux but scores below its own pooled control.

Two things must be fixed in the K-draw implementation, both consequences of the findings
above rather than preferences. First, K > 1 has to turn `latent_stochastic` ON, or the K
draws are K copies of `mu`; that also switches the KL on. Second, the spread has to be held
up against a data term that shrinks it, so the arm reports the posterior sd every epoch and
the KL weight is a knob the cell records rather than an inherited default.
