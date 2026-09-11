---
id: 8n6aio7hbpa16ubg8w3gsq7
title: Flux_layer
desc: ''
updated: 1788314317663
created: 1788314317663
---

## 2026.09.01 - The differentiable flux layer

Gene tokens in, a feasible flux vector and its feasibility residuals out. Every hard
constraint is either exact by construction or a smooth penalty, and nothing is enforced by a
binary variable, which is the difference from Thermo-Flux (Smith 2026), whose second law
costs one binary per reaction and a 24 h wall-time budget per model.

Read the module docstring for the term-by-term relaxation. The two facts most likely to bite:

- **Every constraint term must be dimensionless AND commensurate with the data loss.** The
  raw second-law hinge is ~72 at initialization against a data loss of ~2, and the squared
  dissipation excess is ~4e4 and drives the model to NaN in one step. Both are reformulated,
  not merely down-weighted.
- **`parameterization="box"` cannot reach a sparse flux vector.** Zero flux is an asymptote
  of the sigmoid, so the mass-balance residual pins at its maximum. Measured in
  [[experiments.026-metabolism-flux.scripts.box_zero_reachability]].

`coverage_report()` is not optional reporting: a capacity constraint built on a default kcat
is a uniform rescaling of the box, not an enzyme constraint, and a loss curve cannot tell
them apart.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]

## 2026.09.10 - The projection parameterization, a latent sampler, and the eQuilibrator covariance

Three additions, all behind new config values, so `box` and `nullspace` and every arm
configured before today run byte for byte as they did. Verified by
`tests/torchcell/metabolism/test_flux_layer.py` (48 passed, including the two real-GEM tests
when `DATA_ROOT` is set) and by `scripts/run-mypy.sh` on the layer, `constraints.py` and the
test file (strict, clean).

### `parameterization="projection"`

`box` and `nullspace` each buy one exact constraint by giving up the other. `projection`
starts at the null-space point `v0 = N z` and then alternates two projections `K` times:

- `P_bal(v) = N (N^T v)`, the orthogonal projector onto `ker S`, exact because
  `constraints.null_space_basis` returns an orthonormal basis;
- `P_box(v) = clip(v, lb, ub)`, with `lb, ub` the genotype-dependent box from `dynamic_box`,
  so a deletion's `lb = ub = 0` pins its reaction to zero through every step.

`K` counts the clips (`projection_steps`, default 5). Each is followed by the balance
projection except, under `projection_end="box"`, the last one.

**Which constraint is exact is decided by which projection runs last, and that is a config
choice rather than a detail.** `projection_end="balance"` (the default) gives `S v = 0` to
machine precision and reports the leftover box overshoot; `projection_end="box"` gives the
bounds to machine precision and reports the leftover balance residual. Finite `K` does not
reach the intersection, it shrinks whichever residual is left. `projection_method="dykstra"`
runs Dykstra's correction, whose limit is the NEAREST point of the intersection to `v0`
rather than merely a point of it.

Measured on yeast-GEM 9.0.2 at initialization, `flux_projection_anchored`, batch 4, one CPU
forward (the script is the build harness quoted in
[[experiments.027-betaxanthin-metabolic.projection-head]]):

| quantity | K = 5 | K = 20 |
|---|---|---|
| balance ratio, median over independent rows | 4.8e-9 | 5.3e-9 |
| balance ratio, max over independent rows | 7.2e-5 | 1.5e-4 |
| `c_box`, mean overshoot / flux_scale | 3.75e-3 | 7.90e-4 |
| box violation fraction (tol 1e-4) | 0.647 | 0.647 |
| mean absolute flux | 0.94 | 0.25 |

So mass balance is exact, and the leftover box overshoot falls 4.7x from K=5 to K=20 in
MAGNITUDE while the FRACTION of reactions outside their bounds does not move. At
initialization the latent is random and roughly half the reactions are irreversible with
`lb = 0`, so a sign is wrong on about half of them by construction; the fraction is a
statement about the untrained latent, not about the projection. The `c_box` penalty is still
in the loss at weight 0.5, so the leftover is attacked structurally by `K` and by learning at
the same time.

Cost: `2K` matmuls of `[B, 4131]` against the `4131 x 1538` basis per forward. The basis is
25 MB and stays on the GPU.

`_validate_null_basis` measures rather than assumes the two properties the projection rests
on, and refuses a basis that is not orthonormal: `N (N^T v)` is a projector only for an
orthonormal `N`, and an oblique map still lands in `ker S` but no longer contracts the
distance to the box, so the alternation need not converge. On yeast-GEM the measured
orthonormality error is 2.4e-6 and the balance error 3.5e-6, both reported in
`coverage_report()`.

### `latent_stochastic` and `FluxLayer.sample`

For `projection` and `nullspace`, `z = mu + sigma eps` from an amortized Gaussian, with
`KL(N(mu, sigma) || N(0, I))` returned as `out["c_kl"]` (the per-sample mean over latent
dimensions, so one weight serves a 1,538-dimensional yeast basis and a 2-dimensional toy).
The spread head is a zero-weight linear whose bias is `latent_log_std_init = -3.0`, so every
genotype starts at sigma 0.050 against a latent of order 1.

`sample(h_genes, context, perturbation_indices, perturbation_indices_batch, n_samples)`
returns a pydantic `FluxSample` with `v` of shape `[S, B, r]` and five `[S, B]` feasibility
fields. The genotype chain, the box and the reaction energies are computed once and shared;
only the latent is redrawn, and the draw is taken regardless of `training`. **Every draw is
projected**, which is the difference from the 026 stochastic head: that one drew `z` per
reaction and projected nothing, and its draws had a median balance ratio of 1.98 with 30
percent thermodynamic violations (`experiments/026-metabolism-flux/results/flux_sampling_demo.json`).
Drawn here at initialization on the real GEM, the median balance ratio over 8 draws is
5.5e-9.

`latent_stochastic` is a runtime PROPERTY, not an `__init__` flag, so a cell trained
deterministically can be switched into a sampler after its weights are loaded, which is what
`audit_top_strain_feasibility.py` does by setting `config.stochastic`. The spread head is
therefore built on every projection layer whether or not the run samples, so the flip needs
no parameter the checkpoint lacks. On `nullspace`, `stochastic` stays the no-op it has always
been.

### `feasibility(out)`, the canonical per-sample definitions

`[B]` tensors for `balance_ratio_median`, `balance_ratio_max`, `box_violation_frac`,
`thermo_violation_frac` and `thermo_uphill_frac`. `forward` averages these into scalars for
the training log, and a batch mean can meet a 95 percent target while a third of the
individual strains do not, which is the form the Genesis metabolic-grounding target actually
takes. `forward`'s `feas_balance_median` is the median over the FLATTENED batch and is
therefore related to, but not the mean of, the per-sample medians.

`thermo_violation_frac` counts active reactions with `v dG > 0`; `thermo_uphill_frac` counts
only those CARRYING flux (`|v| > 1e-6`), because a reaction at zero flux is not running
uphill, it is not running.

### The eQuilibrator square-root covariance is available, and the docstring said it was not

The module docstring claimed `Q` was unavailable. It is not. Our recomputation writes
`$DATA_ROOT/data/torchcell/thermo_equilibrator/formation_sigma.npz`, which holds the
metabolite square-root factor of the component-contribution covariance as a dense
`2099 x 590` array. The reaction factor is `Q = S^T Sigma^(1/2)`, `4131 x 590`, and
`parameters.resolve_reaction_delta_g` already forms exactly this product and keeps only its
row norms as `ReactionThermoTable.sigma`.

`reaction_sqrt_covariance(thermo_dir, s, met_ids)` keeps the matrix, with its sha256 and its
coverage mask. `test_real_reaction_covariance_row_norms_equal_the_tabulated_sigma` checks
`||Q_j|| == table.sigma[j]` on all 1,755 covered reactions of the union route and passes, so
the covariance path fits the same uncertainty every coverage report prints.

`thermo_uncertainty="covariance"` then replaces the free per-reaction offset with
`delta_g_error = Q m`, `m` a learned latent of dimension 590 with an `N(0, I)` prior, which
is Thermo-Flux Eq. (5) as written. The 2,376 reactions with an incompletely covered
participant set take no offset at all, because a reaction whose error model is unknown should
not be given a free parameter that looks like one. `"isotropic"` remains the default and is
what every 026 and 027 run to date fit.

**Unverified:** nothing has been TRAINED on the covariance path yet. Its shape, its prior and its
agreement with the tabulated sigma are tested; whether it fits differently from the isotropic
offset is not measured, and reaching it from an 026 arm needs the `build_flux_layer` patch
recorded in [[experiments.027-betaxanthin-metabolic.projection-head]].

## 2026.09.11 - The second law as a box, not a penalty

`thermo_box`, off by default. Reachable from an 026 arm as the `tdg` token of the
parameterization string, since that string is the only field `build_flux_layer` forwards.

### Why a penalty could not do it

Wave 6's first four cells made mass balance exact and left the second law as the hinge
`c_thermo`. On job 1708's top-100 predicted strains, at a constraint scale of 0.1, every
strain met the balance criterion at every tolerance and `thermo_model_violation_frac` still
read a median of 0.405. A weight that has to be traded against the data term cannot deliver
a criterion of the form "95 percent of the top 100 violate nothing", because the trade is
what the weight is for.

### The observation

For a reaction whose standard energy has a known sign, `v_j dG_j <= 0` is not a penalty at
all. It is a sign restriction on `v_j`, and a sign restriction is a bound:

- `dG_j > +thermo_epsilon` forces `ub_j <- min(ub_j, 0)`;
- `dG_j < -thermo_epsilon` forces `lb_j <- max(lb_j, 0)`;
- `|dG_j| <= thermo_epsilon` leaves the box alone, because near equilibrium the sign is not
  resolved and forcing one would assert a direction the table does not have.

Reactions outside `dg_mask`, and the ones Thermo-Flux's Box 2 exempts, are untouched.

**The box is per sample**, because `dG` carries the learned concentration term. That is what
makes it a constraint the model can satisfy rather than one it must fight: to license a flux
in some direction, the model moves concentrations inside the physiological window until the
reaction energy has the sign it needs, which is the same degree of freedom Thermo-Flux gives
its own solver.

Then the alternating projection runs against the tightened box and **ends on the clip**, so
bounds, enzyme capacity and direction are all exact and only the balance residual is
reported. `thermo_box` refuses to build on any head that does not end on the box, because a
head ending on the balance projection would report the second law as exact while the last
step moved the flux back off it.

The tightening can empty a box, when a reaction carries a strictly positive lower bound and
an energy that forbids forward flux. The second law wins and the lower bound is dropped to
meet the upper one, but the count comes back as `feas_thermo_box_emptied_frac` rather than
being absorbed.

### Measured on yeast-GEM 9.0.2 at initialization, untrained, four samples

| head | balance median (max over samples) | balance max | box violated | second law violated | mean abs flux | reactions carrying flux |
|---|---|---|---|---|---|---|
| `projection` (K=5, end balance) | 5e-9 | 4.5e-5 | 0.643 | 0.511 | 1.076 | 0.881 |
| `projection_k20_box` | 1.4e-2 | 2.000 | 0.000 | 0.130 | 0.194 | 0.237 |
| `projection_k20_box_tdg` | 0.000 | 2.000 | 0.000 | 0.000 | 0.046 | 0.147 |

All three constraints at once, exactly, with no training and no penalty. The test
`test_thermo_box_on_the_real_gem_is_exact_and_nearly_balanced` pins it: violation fractions
0, and the per-sample balance ratio median under 1e-2 on every sample.

### Two things to read carefully

**`c_thermo` runs HIGH on a `tdg` head while it violates nothing.** The hinge is
`relu(v dG + eps) / (|v||dG| + eps)`, and at `v = 0` both numerator and denominator are
`eps`, so a reaction the directional box pinned to zero scores exactly 1. Measured on the
toy network: violation fraction 0, `c_thermo` 0.84. `c_thermo` is kept unchanged only
because every arm before 2026-09-11 trained against it;
`feasibility`'s `thermo_violation_frac` and `thermo_uphill_frac` are the readouts. Pinned by
`test_c_thermo_is_not_a_readout_once_the_second_law_is_a_box`.

**The median balance ratio is passed partly by zeros.** The `tdg` head's balance MAXIMUM is
2.0, the largest that ratio can take, and only 14.7 percent of reactions carry any flux. A
median balance ratio of 0 on a flux vector that is 85 percent zeros is not evidence of a
balanced flux, and the Genesis criterion as currently written does not separate the two.
Recommendation for the trainer's `criteria_flags`: add a max-based balance tolerance beside
the median one, and report the fraction of reactions carrying flux next to every feasibility
number.

### Also landed: the shipped reaction column is two units

`constraints._read_delta_g_csv` multiplied the whole of yeast-GEM 9.0.2's
`model_rxnDeltaG.csv` by 4.184. The curation script fills that column from yETFL in kcal/mol
and then fills an index block from dGPredictor in **kJ/mol**, so 33 of the 3,210 priced
reactions were coming back 4.184x too large while looking entirely ordinary.
`dgpredictor_reaction_ids` reads the audited id list and `_read_delta_g_csv` now exempts
those rows at parse time; a missing id list raises rather than skipping the correction.
`ThermoTable.n_reaction_rows_already_kj` records the count (33 on this GEM) and the sidecar's
path and sha256 join `source_paths` / `sha256`. Coverage is unchanged at 2,389 metabolites
and 3,210 reactions. Pinned on a synthetic CSV by
`test_the_dgpredictor_rows_of_the_shipped_column_are_not_converted`.
