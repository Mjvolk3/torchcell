---
id: 71sxnyony5fbh8c6nhrq32m
title: Wave6
desc: ''
updated: 1789099162567
created: 1789099162567
---

## 2026.09.10 - Wave 6 design: what is measured and how

Wave 6 does three things. It adds a per-genotype FEASIBILITY audit to every cell that runs a
flux layer, it saves the selected-epoch weights so a finished cell can be re-audited, and it
moves the whole protocol onto a second phenotype, the Mulleder 19 amino acids, so the
flux-versus-pooled contrast is measured on a target the flux vector has a mechanistic route
to.

Generating scripts:

- `experiments/027-betaxanthin-metabolic/scripts/train_bx.py` (the trainer, feasibility
  block, checkpoints, `--head`)
- `experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py`
  (re-audit a finished cell from its checkpoint, optionally with the flux sampler on)
- `experiments/027-betaxanthin-metabolic/scripts/analyze_wave6.py` (aggregation, the
  paired contrasts, the feasibility table, the figure)

### The feasibility metric, defined

The Genesis "Metabolic Grounding" endpoint asks what fraction of the top 100 strains a model
nominates carries a flux vector that does not violate mass balance or thermodynamic
feasibility. Every quantity below is computed from the flux layer's own forward output and
its own buffers, so it reduces to the `c_balance`, `c_thermo` and `feas_*` scalars the
training loop already logs, per genotype instead of batch-reduced.

| quantity | definition |
|---|---|
| balance median ratio | median over the independent rows i of S of `abs((Sv)_i) / (omega_i + balance_epsilon)`, omega the layer's turnover |
| balance max ratio | the same ratio maximized over the independent rows |
| box violation fraction | fraction of the 4,131 reactions outside `dynamic_box` at that genotype's availability and abundance |
| thermo model violation fraction | fraction of active reactions with `v dG > thermo_epsilon`, dG from the layer's own `reaction_potential` |
| thermo standard-state violation fraction | the same with `dG = delta_r_G0 + delta_r_G_transport`, an audit reference for arms whose `thermo_mode` is OFF |
| uphill fraction | `relu(v dG + eps) / (abs(v) abs(dG) + eps)` averaged over active reactions, the per-sample form of `c_thermo` |

Tolerance ladder, recorded in the JSON beside the numbers: balance median ratio at 1e-1,
1e-2 and 1e-3; thermo violation fraction at 0 and 0.01; and the combined Genesis criterion,
balance within 1 percent AND no uphill active reaction.

Four groups per cell: `all` and `pinned_test` are the populations, `all_top100` and
`pinned_test_top100` are the 100 genotypes with the highest predicted betaxanthin. The
population fraction is reported so the top-100 fraction can be read against its base rate; a
top-100 number alone cannot distinguish a model that nominates feasible strains from one
whose whole flux space happens to be feasible.

A pooled arm writes `available: false` with a reason. It has no flux vector, so neither
constraint is defined for it, and zeros would read as perfect feasibility.

### The frozen physics snapshot, and why the cells needed one

`torchcell/metabolism/flux_layer.py` and 026's `train_flux.py` are being edited by another
agent in this same worktree tonight, and a queued slurm cell imports them at the moment it
STARTS, not at the moment it is submitted. This is not a hypothetical: a smoke run at
23:03:0X on 2026-09-10 read `flux_layer.py` mid-write and died with
`AttributeError: 'FluxLayer' object has no attribute 'latent_stochastic'` inside
`coverage_report`. The file's mtime was 23:03:07, so the process caught the write in
progress; the version on disk one minute later imports cleanly.

The crash is the benign failure. The damaging one would have been silent: six seeds pooled
into one paired contrast with half of them run on a different flux layer than the other
half. So the whole torchcell package plus 026's scripts were copied at 23:05 on 2026-09-10
to `/scratch/projects/torchcell-scratch/frozen-027-wave6/`, and every wave-6 cell runs with
`PYTHONPATH` and `TC_FLUX_SCRIPTS_DIR` pointed at that copy. Snapshot hashes:

| file | sha256 |
|---|---|
| `frozen-027-wave6/torchcell/metabolism/flux_layer.py` | `5cfa3d01b59168086806c204c13fd96ae31d59e744916aea8b4fa51cb91515cf` |
| `frozen-027-wave6/026-scripts/train_flux.py` | `fa516c3e15b85a9c336bf49d36787dc727a8690210de62215d05bf9bc705f39c` |

`train_bx.py` records the sha256 of the `train_bx.py` and `flux_layer.py` it actually
imported into every result file, under `code_provenance`, so a cell can be attributed to a
code state without a commit hash. Git is not the mechanism here on purpose: the worktree is
uncommitted and several agents are writing to it, so a commit hash would name a tree that
never existed on disk.

**The frozen snapshot reproduces the pre-edit box path.** Measured by replaying the exact
wave-6 nocstr recipe at seed 2702 for three epochs against the frozen tree and comparing
with job 1681, which ran the same seed and recipe on the pre-edit layer:

| epoch | job 1681 train loss / data | frozen replay train loss / data |
|---|---|---|
| 0 | 1.5547 / 0.5822 | 1.5546 / 0.5821 |
| 1 | 1.4810 / 0.5371 | 1.4803 / 0.5363 |
| 2 | 1.3922 / 0.5385 | 1.3950 / 0.5351 |

Agreement to 1e-4 at epoch 0 and to 3e-3 by epoch 2. A change in the layer's parameter set
would change the initialization RNG draw and put epoch 0 somewhere else entirely, so this
says the box path and its RNG consumption are unchanged. Interpretation, labeled as such:
the drift across epochs is consistent with GPU reduction nondeterminism accumulating, which
has NOT been separately confirmed. Validation Pearson diverges much faster (+0.0716 against
+0.0152 at epoch 1) because the head is on the mean plateau there with a prediction spread
of 1e-4 to 7e-4 of the target's, so its correlation is jitter and is not a stable quantity
to compare on.

### The freeze earned its keep: the kcat table changed under the queue

At 00:22 on 2026-09-11 the live `experiments/026-metabolism-flux/scripts/train_flux.py` had
gained a `kcat_homolog_tier` switch, `arm.get("kcat_homolog_tier", True)`, which is ON by
default and lets a cross-organism EC-plus-substrate measurement fill a catalytic unit the
organism's own assays do not reach. That changes the resolved kcat table, which sets the
enzyme-capacity ceiling, which IS the flux box. A cell built that way is not a replicate of
one built before it.

`torchcell/metabolism/flux_layer.py` is byte-identical across the change
(`5cfa3d01b591...` in both the 23:05 freeze and the 00:22 job snapshot), so a hash of the
flux layer does NOT catch this. What catches it is the tier counts each run records for
itself in `flux_meta`, and `analyze_wave6.py` now keys on exactly that: it takes the modal
parameter table across the paired flux cells and drops any seed built from a different one
BY NAME, into `excluded_physics_mismatch`, rather than averaging two different models into
one contrast.

The wave-6 cells pin `TC_FLUX_SCRIPTS_DIR` and `PYTHONPATH` at the 23:05 freeze, so all
eleven of them run the pre-homolog table and match seeds 2700, 2701 and 2703. The one cell
that does not is the resubmitted seed-2702 flux cell (job 1707), which was relaunched from
the worktree snapshot without that pin, so it carries the homolog tier. **Hypothesis
(untested): the homolog tier changes the score as well as the table.** The cell will be
reported on its own rather than pooled, and the difference between it and the other five
flux seeds is not attributable to the seed.

### The amino-acid phenotype has plenty of pinned test genes

The concern going in was that the pinned test split is defined by Merzbacher's BETAXANTHIN
gene list, so it might carry few amino-acid measurements. Measured on the smoke pass of
2026-09-10 (`results/smoke_w6_aa_flux.json`, seed 9001, 2 epochs, so the metrics are
meaningless but the counts are not):

| split | genotypes in split | genotypes with amino-acid data | measurements |
|---|---|---|---|
| pinned Merzbacher test | 639 | 564 | 10,716 |
| whole test split | 933 | 842 | 15,998 |
| validation (selection) | 294 | 270 | 5,130 |

`n_per_feature` is 564 for all 19 columns on the pinned split, so every genotype that has an
amino-acid profile has the complete profile. The pinned test set is therefore a usable
comparison set for this head and no fallback to the validation split is needed; the
validation numbers are still written into each result file under
`validation_selection_split`, labeled as the split the epoch was selected on and therefore
not held out.

### The amino-acid flux head carries the same offset defect, and the fix is an init

`FluxMetaboliteHead` emits `a_k log(1 + omega_k) + b_k` with `a_k` initialized to 1, so
against a target standardized to mean 0 and sd 1 it starts wherever `log1p(omega)` happens
to be. Measured (smoke, seed 9001, same recipe): epoch-0 data loss 2.4478 for the flux arm.
This is the same class of defect wave 5 measured on `FluxScalarHead` (+7.030 output mean),
and wave 5 measured that defect holding the betaxanthin head on the mean plateau for the
whole 80-epoch budget.

`FluxScalarHead` takes `normalize_inputs`; `FluxMetaboliteHead` has no such switch. Since
its slope is learnable and multiplies exactly the input that would be divided, `train_bx.py`
divides the INITIAL slope by `log1p(flux_scale)` when `--head-normalize` is passed with
`--head mulleder19`. The forward pass at initialization is identical to normalizing the
input, and the slope is free to move from there. No shared model file is touched.

### Why the standard-state read exists

The arm the parity contrast is about is `flux_capacity`, whose `thermo_mode` is `OFF`. It
learns no metabolite concentrations, so the concentration term `RT S^T ln c` of the in-cell
reaction energy is ABSENT rather than zero, and the layer emits no `delta_r_g`. Reporting
only the model's own second-law number would therefore leave the flagship arm with no
thermodynamic number at all. The standard-state read uses the tabulated
`delta_r_G0 + delta_r_G_transport` that the layer already holds as buffers (the union route
covers 3,722 of 4,131 reactions), which is a real and checkable criterion, and it is labeled
as an audit reference rather than as the layer's own term everywhere it appears. On the
wave-6 build the mask leaves 3,466 reactions under the second law and 2,593 independent
rows under mass balance (`results/smoke_w6_bx_flux.json`, `feasibility` block).

### The cells

Launched 2026-09-10 on GilaHyper, partition `main`, one RTX 6000 Ada each, 80 epochs.
Seeds 2700 and 2701 of the parity pair are wave 5 and predate the feasibility block; 2702
and 2703 were launched at 22:35 on 2026-09-10, before tonight's edits, so they carry no
feasibility block either. 2704 and 2705 and every amino-acid cell carry it.

| cell | result stem | job | feasibility | checkpoint |
|---|---|---|---|---|
| flux nocstr s2700 | `wave5_g7_flux_capacity_nocstr_s2700.json` | 1673 | no | no |
| pooled s2700 | `wave5_g8_pooled_s2700.json` | 1674 | no | no |
| flux nocstr s2701 | `wave5_g7_flux_capacity_nocstr_s2701.json` | 1679 | no | no |
| pooled s2701 | `wave5_g8_pooled_s2701.json` | 1680 | no | no |
| flux nocstr s2702 | `wave5_g7_flux_capacity_nocstr_s2702.json` | 1681 | no | no |
| pooled s2702 | `wave5_g8_pooled_s2702.json` | 1682 | no | no |
| flux nocstr s2703 | `wave5_g7_flux_capacity_nocstr_s2703.json` | 1683 | no | no |
| pooled s2703 | `wave5_g8_pooled_s2703.json` | 1684 | no | no |
| flux nocstr s2704 | `wave5_g7_flux_capacity_nocstr_s2704.json` | 1694 | yes | yes |
| pooled s2704 | `wave5_g8_pooled_s2704.json` | 1695 | null, no flux layer | yes |
| flux nocstr s2705 | `wave5_g7_flux_capacity_nocstr_s2705.json` | 1696 | yes | yes |
| pooled s2705 | `wave5_g8_pooled_s2705.json` | 1697 | null, no flux layer | yes |
| amino acid, flux nocstr s2700 | `wave6_aa_nocstr_s2700.json` | 1698 | yes | yes |
| amino acid, pooled s2700 | `wave6_aa_pooled_s2700.json` | 1699 | null, no flux layer | yes |
| amino acid, flux nocstr s2701 | `wave6_aa_nocstr_s2701.json` | 1700 | yes | yes |
| amino acid, pooled s2701 | `wave6_aa_pooled_s2701.json` | 1701 | null, no flux layer | yes |
| amino acid, flux nocstr s2702 | `wave6_aa_nocstr_s2702.json` | 1702 | yes | yes |
| amino acid, pooled s2702 | `wave6_aa_pooled_s2702.json` | 1703 | null, no flux layer | yes |
| amino acid, flux all-on s2700 | `wave6_aa_allon_s2700.json` | 1704 | yes | yes |

### Preflight, all four code paths

Every launch was preceded by a two-epoch smoke against the frozen tree. The epoch-0 data
loss is the head-offset diagnostic, on a target standardized to mean 0 and sd 1.

| smoke | result file | epoch-0 data loss | outcome |
|---|---|---|---|
| betaxanthin, flux, head fix | `smoke_w6_bx_flux.json` | 0.5400 | full feasibility block and checkpoint written |
| betaxanthin, pooled | `smoke_w6_bx_pooled.json` | 0.4872 | feasibility null with a reason |
| amino acid, flux, no head fix | `smoke_w6_aa_flux.json` | 2.4478 | the offset defect, measured |
| amino acid, flux, head fix | `smoke_w6_aa_flux_headfix.json` | 0.6028 | the fix, measured |
| amino acid, flux, all on | `smoke_w6_aa_allon.json` | 0.6022 | constraint ramp reaches the head |
| amino acid, pooled | `smoke_w6_aa_pooled.json` | 0.4872 | feasibility null with a reason |
| amino acid, flux, on CPU | `smoke_cpu_aa_flux.json` | 2.4687 | the CPU path completes end to end |

`audit_top_strain_feasibility.py` was run against `smoke_w6_aa_flux.json`, rebuilding the
model from the cell record and loading the saved weights: the recomputed feasibility
fractions match the trainer's to a maximum absolute difference of 0.000e+00
(`results/audit_smoke_w6_aa_flux.json`).

## 2026.09.11 - Results

### The Genesis metabolic-grounding number is 0 percent, not 95

First fully trained cell carrying the feasibility block: `wave5_g7_flux_capacity_nocstr_s2702.json`,
job 1707, 80 epochs, selected epoch 62, pinned test Spearman 0.0972 on 629 genes, final
validation prediction spread 0.3601 of the target's, so the head had left the mean plateau
and this is a working model rather than a collapsed one.

| group | n | balance median ratio (max is 2) | uphill fraction of active reactions | fraction meeting the Genesis criterion |
|---|---|---|---|---|
| top 100 predicted, all genotypes | 100 | 1.9977 | 0.3468 | 0.000 |
| all genotypes | 4,930 | 1.9961 | 0.3335 | 0.000 |
| top 100 predicted, pinned test | 100 | 1.9964 | 0.3402 | 0.000 |
| pinned test | 639 | 1.9961 | 0.3344 | 0.000 |

Every tolerance in the ladder is missed by every genotype: balance median ratio under 1e-1,
1e-2 and 1e-3 all read 0.000, and the standard-state second law is violated on about a third
of the 3,466 active reactions in every strain. The only criterion that passes is the box, at
1.000, and it passes by construction rather than by learning, since the sigmoid
parameterization cannot leave the box.

The balance ratio is worth reading twice. Its theoretical maximum is 2, because
`abs(sum_j S_ij v_j)` can at most equal `sum_j abs(S_ij v_j)` and omega is half of that. A
median of 1.9977 means the typical metabolite in the typical nominated strain has
essentially nothing it produces being consumed. The flux vector is not a nearly-balanced
flux vector; it is about as far from balanced as the stoichiometry allows.

**This is not a surprise and it is not a bug.** The parity recipe these cells run is
`--constraint-scale 0.0`, chosen in wave 5 because the constraint penalties collapsed the
escaped solution. At that scale the loss contains no mass-balance term and no second-law
term, so nothing in training ever asks the flux vector to be a flux. What the number
establishes is the size of the gap: the arm that currently scores best on the phenotype
carries a "flux" that is a point in an enzyme-capacity box and nothing more, and the
Genesis target of 95 percent is 95 points away rather than a few.

**The top 100 are indistinguishable from the population** (1.9977 against 1.9961; 0.3468
against 0.3335), so the model's nominated strains are neither more nor less feasible than
any other genotype. Reporting the top-100 fraction without the population fraction beside it
would have hidden that.

### Reruns

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
    experiments/027-betaxanthin-metabolic/scripts/analyze_wave6.py

# Re-audit one finished cell from its saved weights, optionally with the flux sampler on.
PYTHONPATH=/scratch/projects/torchcell-scratch/frozen-027-wave6:$PWD \
TC_FLUX_SCRIPTS_DIR=/scratch/projects/torchcell-scratch/frozen-027-wave6/026-scripts \
~/miniconda3/envs/torchcell/bin/python \
    experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py \
    --result wave5_g7_flux_capacity_nocstr_s2704.json --stochastic --draws 16
```

`analyze_wave6.py` reads only finished result files, so it can be rerun at any point while
the queue drains; a cell with no result file is listed by name under `missing_files` rather
than dropped.
