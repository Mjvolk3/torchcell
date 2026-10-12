---
id: 9lufq6dv0wr3bglthxu0ncy
title: Train_bx
desc: ''
updated: 1789099154960
created: 1789099154960
---

Source: `experiments/027-betaxanthin-metabolic/scripts/train_bx.py`

## 2026.09.10 - Wave 6 additions: feasibility block, checkpoints, and the amino-acid head

Three additions, all of which leave the wave-1 to wave-5 defaults alone. A result file
written before tonight deserializes unchanged, and a cell launched with the wave-5 command
line trains exactly what it trained before.

### New command-line flags

| flag | default | what it does |
|---|---|---|
| `--head {betaxanthin,mulleder19}` | `betaxanthin` | which phenotype the cell trains, selects and scores on |
| `--no-feasibility` | off (so the audit runs) | skip the two feasibility passes at the selected epoch |
| `--no-checkpoint` | off (so the checkpoint is written) | do not save the selected-epoch parameters |

### The feasibility block

The Genesis "Metabolic Grounding" endpoint is a property of the model's output, not of its
training curve: of the top 100 strains the model ranks highest, how many carry a flux vector
that respects mass balance and the second law. Nothing measured that before tonight.

After the selected-epoch snapshot is restored, the trainer runs two extra forward passes,
one over the pinned test split and one over the whole 4,930-genotype dataset, and computes
per genotype:

- `balance_median_ratio` and `balance_max_ratio`: the median and maximum over the
  independent rows of S of `|(Sv)_i| / (omega_i + balance_epsilon)`, with omega the layer's
  own turnover. This is `FluxLayer.forward`'s `feas_balance_median` per sample rather than
  batch-reduced, so the numbers reduce to the `c_balance` the training loop already logs.
- `box_violation_frac`: the fraction of the 4,131 reactions outside `dynamic_box` at that
  genotype's availability and abundance. The box parameterization satisfies this by
  construction, so a nonzero value is a bug and not an infeasibility.
- `thermo_model_violation_frac` and `thermo_model_uphill_frac`: the second law using the
  layer's own `reaction_potential`, which carries the learned concentration term. Present
  only when the arm's `thermo_mode` is not `OFF`.
- `thermo_standard_state_violation_frac` and `thermo_standard_state_uphill_frac`: the same
  quantities evaluated at standard state, `v (delta_r_G0 + delta_r_G_transport)`. This is an
  AUDIT REFERENCE and is labeled as one in the JSON. The flagship parity arm is
  `flux_capacity`, whose `thermo_mode` is `OFF`, so it learns no concentrations; the
  concentration term is absent rather than zero, and without the standard-state read there
  would be no second-law number for the arm the experiment is actually about.

Four groups are reported: `all` and `pinned_test` (the populations), and `all_top100` and
`pinned_test_top100`, the 100 genotypes with the highest predicted betaxanthin. The
population number is what makes the top-100 number readable: a top-100 fraction is only
evidence about the nominated strains if it differs from the base rate.

Tolerances are written into the JSON beside the numbers rather than fixed in prose: balance
median ratio at 1e-1, 1e-2 and 1e-3; thermo violation fraction at 0 and 0.01; and the
combined criterion, balance within 1 percent and no uphill active reaction.

An arm with no flux layer writes `available: false` with a reason. Zeros there would read as
perfect feasibility for a model that never made a feasibility claim.

### Checkpoints

The selected-epoch TRAINABLE parameters (not `state_dict`, which would carry ~46 MB of
constant stoichiometric matrix), the cell record and the target standardization go to
`$DATA_ROOT/checkpoints-027/<out-stem>.pt`. The target mean and SD are saved with the
weights because without them the weights cannot be turned back into a prediction in the
phenotype's own units. `audit_top_strain_feasibility.py` reads that file, rebuilds the model
through the shared `build_model_for_cell`, and recomputes the block; the trainer's number
and the rebuild's number must agree, and the audit reports the maximum absolute difference.

### The amino-acid head

`--head mulleder19` makes the 19-column Mulleder amino-acid profile the only supervised
head. Selection is the mean per-feature validation Pearson over the 19 columns; the test
metrics are the mean per-feature Spearman and Pearson with the per-feature values and the
per-feature n reported beside them, because the columns are not measured on the same strains
and a mean over features whose sample sizes differ is not a mean over a common set.

On a flux arm the head is 026's `FluxMetaboliteHead`, which reads `log1p(omega)` at the 19
named GEM metabolites off the predicted flux vector, so the arm contrast stays on the
readout rather than on the trunk alone. On the pooled arm it is `MetabolomeVectorHead`.
`--head-normalize` does not apply: it is a switch on `FluxScalarHead`'s dense `w^T v` term,
which is what carried the measured +7.03 output offset, and `FluxMetaboliteHead` has no such
term.

### The amino-acid head fix, measured

`--head-normalize` means something different on each head, and the difference is worth
stating because the flag name is the same. On `FluxScalarHead` it divides the dense `w^T v`
input by the layer's working flux magnitude. `FluxMetaboliteHead` has no dense term and no
such switch, so `train_bx.py` divides its INITIAL slope by `log1p(flux_scale)` instead,
which is the identical forward pass at initialization and leaves the slope free to learn.

Measured on this dataset (seed 9001, two epochs, same recipe), epoch-0 data loss against a
target standardized to mean 0 and sd 1:

| head | epoch-0 data loss |
|---|---|
| pooled `MetabolomeVectorHead` | 0.4872 |
| `FluxMetaboliteHead`, no fix | 2.4478 |
| `FluxMetaboliteHead`, fix | 0.6028 |

### What the audit round trip actually verified

`audit_top_strain_feasibility.py` rebuilt `wave5_g7_flux_capacity_nocstr_s2705` (job 1713,
80 epochs) from its cell record and its saved weights and recomputed the whole feasibility
block: **maximum absolute difference from the trainer's own numbers 0.000e+00**
(`results/audit_wave5_g7_flux_capacity_nocstr_s2705.json`). The checkpoint is 4.1 MB, which
is what excluding buffers buys; a full `state_dict` would have carried the constant
stoichiometric matrix.

`--stochastic` refuses on this arm, and the refusal is the correct behavior rather than a
gap. On the `box` parameterization the flux MLP is built with `z_out = 2 if cfg.stochastic
else 1`, so a deterministically trained cell has no spread channel and setting
`config.stochastic` afterwards would index a dimension of size 1. The script checks the
width and exits with a message naming the `projection` parameterization, which builds the
spread head unconditionally and is the arm the draw audit is for. Nothing is faked: a spread
that was never supervised is not a posterior.

### Two shape traps fixed on the way

- The CRPS branch of `loss_fn` squeezed the target's feature axis and restored it after,
  which is correct for a width-1 head and would have broadcast `[B, 19, 1]` against a
  `[B, 19]` mask into `[B, 19, 19]` for the amino-acid head, averaging a loss over pairs of
  amino acids. A separate branch handles a distributional vector head.
- `_predict` now returns `[N, W]` for every head width. The betaxanthin callers take column
  0, which is the same array they had before.
