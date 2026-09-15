---
id: w2gap3siaa8n120cd1xv979
title: Kcat Uncertainty 2026 09 14
desc: ''
updated: 1789432478588
created: 1789432478588
---

## 2026.09.14 - Turnover-number uncertainty reaches the polytope, and the 1.5e-7 box does not reproduce

Experiments E1, E2, E3 and E6 of the parameter review, run on CPU under slurm at
`--nice=1000` behind the database rebuild. Every number below names the job that produced it
and the result file it was read from. Scripts:

- `experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py` (E1 and E6)
- `experiments/026-metabolism-flux/scripts/kcat_sigma_table.py` (E3)
- `experiments/027-betaxanthin-metabolic/scripts/audit_capacity_checkpoint.py` (E2)

### Jobs

| job | script | state | elapsed | result file |
|---|---|---|---|---|
| 1914 | `gh_cpu_kcat_sigma_table.slurm` | COMPLETED | 00:00:22 | `results/kcat_sigma_ladder.json` |
| 1915 | `gh_cpu_fva_kcat_primary_sbml.slurm` | COMPLETED | 00:54:10 | `results/fva_kcat_primary_sbml.json` |
| 1916 | `gh_cpu_fva_kcat_primary_ypd.slurm` | COMPLETED | 00:57:58 | `results/fva_kcat_primary_ypd.json` |
| 1917 | `gh_cpu_fva_kcat_sweep_a.slurm` | COMPLETED | 01:11:24 | `results/fva_kcat_sweep_a.json` |
| 1918 | `gh_cpu_fva_kcat_sweep_b.slurm` | COMPLETED | 02:06:13 | `results/fva_kcat_sweep_b.json` |
| 1919 | `gh_cpu_fva_kcat_pool.slurm` | COMPLETED | 00:01:13 | `results/fva_kcat_pool.json` |
| 1921 | `gh_cpu_audit_capacity_checkpoint.slurm` | COMPLETED | 00:01:17 | `results/capacity_checkpoint_audit.json` |
| 1944 | `gh_cpu_kcat_sigma_table.slurm` rerun, after the two additive options landed, with the regression suites | COMPLETED | 00:00:41 | `results/kcat_sigma_ladder.json` |

Job 1944 is the one that matters for the options: it reruns E3 with `complex_aggregation` and
`kcat_ceiling_sigma_k` present in `parameters.py`, resolves the same 97 / 782 / 2,831 / 18,
reports the same element-wise equality on all four shipped fields, and runs
`tests/torchcell/metabolism/test_parameters.py`,
`tests/torchcell/metabolism/test_kinetics_pipeline.py` and
`tests/torchcell/metabolism/test_enzyme_kinetics.py` (**36 passed**) followed by
`tests/torchcell/metabolism/test_kcat_sigma.py` (**15 passed**, 8 of them the ladder and the
protein-pool record, 7 the two new options).

### Headline

1. **rho is small: 0.219 over the three admitted predictors, so the correlation between two
   units in one draw is 0.048** (job 1915, `rho_admitted` in `fva_kcat_primary_sbml.json`).
   Predictor error is therefore almost entirely per unit, not a common-mode offset.
2. **The kill rule does not fire, and not narrowly.** At the estimated rho the wild-type
   growth spread is 165 percent of the median and 69.2 percent of reactions move their FVA
   width by more than 2x (job 1915). The rule kills at under 10 percent and under 5 percent.
3. **The 1.5e-7 median box width does not reproduce.** On a trained wave-8 amino-acid
   checkpoint the measured median box width is 0.0928 mmol gDW-1 h-1, a factor of 6.2e5
   larger (job 1921).
4. **ATP synthase `r_0226` is the single most frequent bottleneck** in both media: at its
   capacity bound in 50.0 percent of draws on the SBML default bounds (job 1915) and 67.5
   percent on YPD-approx (job 1916), the top reaction in both lists.
5. **The budget-consistent abundance of 2.7e-6 mmol/gDW leaves the GEM no feasible point at
   all** under the capacity bound (jobs 1917, 1918, 1919). The smallest uniform multiplier
   that restores one is 5.457, that is E = 1.47e-5 mmol/gDW.

### rho, estimated rather than assumed (job 1915)

The draw is `log10 k_cat_u = mu_u + sigma_u eps_u` with
`eps_u = rho z_global + sqrt(1 - rho^2) z_u`, one `z_global` per draw, so
`corr(eps_u, eps_u') = rho^2`. rho is estimated by a two-way random-effects split of the
per-unit predictor values into a per-predictor offset and a per-unit residual, on the
balanced table of units every admitted predictor covers.

| predictor set | units | tau (offset sd, decades) | residual sd | rho | rho^2 |
|---|---|---|---|---|---|
| admitted 3 (EZKinetics, EITLEM, UniKP) | 3,709 | 0.1072 | 0.4769 | **0.2193** | 0.0481 |
| all 7 mirrored | 3,709 | 0.2068 | 0.5223 | 0.3680 | 0.1354 |

What it is a correlation OF: the offsets are per predictor, so this describes the predicted
tier's error. It is not evidence about a measurement error, and the measured and homolog
rungs of the ladder are not covered by it. With three predictors tau carries two degrees of
freedom, so the estimate is coarse; the seven-predictor value is recorded beside it.

### E1, the primary setting (jobs 1915 and 1916)

200 draws, the shipped min aggregation, nominal E_bar = 1e-3, FVA at 90 percent of each
draw's own optimum on all 4,131 reactions.

| read | SBML default (1915) | YPD-approx (1916) |
|---|---|---|
| point-estimate growth under the capacity bound | 0.07406 /h | 0.17314 /h |
| growth p05 / p50 / p95 across draws | 0.01558 / 0.04251 / 0.08584 | 0.04425 / 0.13927 / 0.48062 |
| relative growth spread (p95 - p05) / p50 | 1.653 | 3.133 |
| per-reaction FVA width ratio, p50 | 84.4 | 53.9 |
| per-reaction FVA width ratio, p90 | 1,048 | 1,520 |
| reactions moving width more than 2x | 0.6916 | 0.6928 |
| infeasible draws | 0 | 0 |

Both media agree on the conclusion and disagree on the level, which is why the medium is
named on every number. `fva_wildtype.csv` is on the SBML default bounds with no cassette;
YPD-approx is reached through `torchcell.metabolism.media.media_to_bounds`.

**Where the sensitivity sits** (job 1915, `fva_kcat_primary_sbml_*_per_reaction.csv`), as the
fraction of reactions moving width more than 2x:

| genes per unit (max over the reaction's units) | reactions | moved > 2x |
|---|---|---|
| 0 (no GPR) | 1,422 | 0.658 |
| 1 | 2,500 | 0.697 |
| 2 | 160 | 0.850 |
| 3 to 5 | 37 | 0.919 |
| 11 or more | 5 | 0.800 |

| EC class | reactions | moved > 2x |
|---|---|---|
| 1 oxidoreductase | 345 | 0.716 |
| 2 transferase | 1,159 | 0.753 |
| 3 hydrolase | 623 | 0.639 |
| 4 lyase | 118 | 0.847 |
| 5 isomerase | 31 | 0.806 |
| 6 ligase | 99 | 0.697 |
| none | 1,756 | 0.652 |

| substrate heavy atoms | reactions | moved > 2x |
|---|---|---|
| under 12 | 287 | 0.652 |
| 12 to 29 | 379 | 0.625 |
| 30 to 60 | 1,214 | 0.759 |
| over 60 | 809 | 0.707 |
| unknown | 1,442 | 0.652 |

The hypothesis the review offered, that the sensitive reactions are "a small, identifiable
minority concentrated in multi-subunit complexes and large substrates", is **rejected as
stated**: the sensitive set is not small (69 percent of the model) and the concentration in
complexes is real but modest (0.85 at two genes against 0.70 at one). The clause about large
substrates is not supported: the 30-to-60 bin is the highest at 0.759 and the over-60 bin is
0.707, below it.

**The bottleneck list** (job 1915, `top_bottleneck_reactions`), as the fraction of the 200
draws in which the reaction sat at its own capacity bound at the FBA optimum: `r_0226` 0.500,
`r_4262` 0.405, `r_2704` 0.315, `r_2634` 0.310, `r_2644` 0.310, `r_2738` 0.300, `r_2632`
0.300, `r_2668` 0.280. On YPD (job 1916): `r_0226` 0.675, `r_2764` 0.640, `r_2964` 0.625,
`r_2644` 0.610. ATP synthase heads both lists, which is the review's finding reproduced under
draws rather than at a single point.

### E1 controls (job 1915)

| control | measured |
|---|---|
| no capacity, SBML default bounds, growth | 0.0858440 /h against the committed 0.085844 |
| same, max abs difference in FVA minimum over 4,131 reactions | 1.216e-5 |
| same, max abs difference in FVA maximum | 1.216e-5 |
| sigma = 0 draw against the point estimate | identical, bit for bit |
| sigma = 0 at rho 0.5 against rho 0.9 | identical, bit for bit |

The growth reproduces to seven digits and the intervals to 1.22e-5 mmol gDW-1 h-1, which is
solver tolerance on a bound of order 1,000. The script's `reproduces_shipped` flag reads
False only because the threshold written into it was 1e-5, tighter than the solver; the
number, not the flag, is the result. The threshold is the thing that was wrong.

### E1 sweep, 17 settings (jobs 1917 and 1918)

200 draws per setting on the 1,583 reactions the point-estimate capacity narrows under either
abundance, so every setting is measured on the same reaction list.

| setting | point growth /h | relative growth spread | moved > 2x | width ratio p50 |
|---|---|---|---|---|
| `rho0_min_e_bar_1e-3` | 0.07406 | 1.662 | 1.000 | 119.0 |
| `rho0_max_e_bar_1e-3` | 0.08584 | 0.821 | 1.000 | 78.6 |
| `rho0_median_e_bar_1e-3` | 0.08584 | 0.860 | 1.000 | 93.0 |
| `rhoest_max_e_bar_1e-3` | 0.08584 | 0.817 | 0.999 | 66.3 |
| `rhoest_median_e_bar_1e-3` | 0.08584 | 0.797 | 1.000 | 90.5 |
| `rho09_min_e_bar_1e-3` | 0.07406 | 0.845 | 1.000 | 16.0 |
| `rho09_max_e_bar_1e-3` | 0.08584 | 0.782 | 1.000 | 54.5 |
| `rho09_median_e_bar_1e-3` | 0.08584 | 0.788 | 1.000 | 59.6 |

Three readings, and the third is the one that matters for the model.

- **The aggregation decides whether the bound binds at the point estimate at all.** Under the
  shipped min, capacity costs 13.7 percent of wild-type growth, 0.07406 against 0.08584.
  Under max and under median it costs nothing measurable: growth is 0.08584, the
  unconstrained value, at the point estimate. GECKO 3's convention removes our capacity
  constraint's only bite on growth.
- **rho buys less than it looks like it should.** Going from independent draws to rho = 0.9
  halves the growth spread on the min aggregation, 1.662 to 0.845, and barely moves the other
  two. Every setting still moves essentially every reaction past 2x.
- **The width ratio is enormous everywhere.** The smallest p50 in the sweep is 16.0, at
  rho = 0.9 with the min rule. Half the reactions in the narrowed set change their feasible
  interval by at least an order of magnitude between draws.

**Nine of the 17 settings have no result, and the absence is the finding.** Every setting at
the budget-consistent abundance 2.7e-6 mmol/gDW is INFEASIBLE at its own point estimate: the
capacity bound at that abundance leaves the GEM no feasible point. The smallest uniform
multiplier on the abundance that restores one, by bisection:

| aggregation | min feasible abundance multiplier | implied E, mmol/gDW |
|---|---|---|
| min | 5.457 | 1.47e-5 |
| median | 3.025 | 8.17e-6 |
| max | 1.997 | 5.39e-6 |

The multiplier depends on the aggregation and not on rho, which is what it should do: the
point estimate is what has to be feasible before a draw around it means anything.

### E2, the capacity bound at a trained checkpoint (job 1921)

Two cells, one forward pass each on a fixed batch, the genotype chain recomputed and checked
against the box the forward pass returned (it matches exactly on both).

| read | `wave8_aa_conccyto_e200_s2700` | `wave6_proj_flux_projection_s2700` |
|---|---|---|
| arm | `flux_projection_anchored` | `flux_projection` |
| median box width, mmol gDW-1 h-1 | **0.09277** | **0.5595** |
| same with `use_enzyme_capacity=False` | 100.0 | 0.5595 |
| GEM base box width | 100.0 | 100.0 |
| availability factor `c_j`, p50 | 1.000 | 0.004074 |
| capacity factor, p50 | 0.000730 | 1.000 |
| `E_g` p50, mmol/gDW | 1.55e-6 | no abundance head |
| `E_u` p50, mmol/gDW | 1.445e-6 | no abundance head |
| capacity p50, mmol gDW-1 h-1 | 0.02358 | no capacity bound |
| softmin over arithmetic min, p50 | 1.000 (0.9202 on multi-gene units) | not applicable |
| `feas_capacity_active_frac` | 0.8236 | not applicable |
| `protein_used / p_avail` | 0.3084 | not applicable |
| k_cat table matches the checkpoint's buffer | yes, 0 of 3,728 units differ | yes |

**The quoted 1.5e-7 is not the box width of these arms.** The measured median is 0.0928, a
ratio of 6.2e5, so the review's failure to reproduce it was correct and the number needs a
different origin than "the capacity bound and its abundance prior". What the capacity bound
actually does is still large and is now attributed: it takes the box from 100 to 0.0928, a
factor of 7.3e-4, and the availability gate contributes nothing on this arm because an
abundance-head cell has no learned gate, so `c_j` is a hard 1 on every undeleted reaction.
The softmin's own undershoot is a 0.92 factor on multi-gene units and 1.000 overall, which is
three orders of magnitude too small to be the missing term.

The plain projection is the mirror image: its box is narrowed by availability (`c_j` p50
0.0041) and not at all by capacity, which that arm does not use.

**The promiscuity overcount is real and large.** 491 genes serve more than one reaction. On
the amino-acid checkpoint the implied `sum_j |v_j| / (k_cat E_g)` per gene has median 50.98
and 76.7 percent of gene-genotype pairs exceed 1, so the flux the model carries implies about
fifty times more of the median enzyme than the abundance head assigns it. The protein budget
is the only term that would object and it sits at `constraint_final_scale` 0.0 in this recipe.
On the projection cell, which reads a nominal E, the same statistic is 0.0132.

### E3, the sigma ladder (jobs 1914 and 1944)

Each rung is recomputed from the mirror on every run and asserted against the constant that
ships. Job 1914:

| tier | shipped sigma | recomputed | groups | rows | statistic |
|---|---|---|---|---|---|
| measured | 0.50 | 0.4968 | 492 | 1,157 | pooled within-group sd over (accession, substrate) groups reduced to one value per paper, 25 to 35 C, all organisms |
| homolog | 0.73 | 0.7331 | 2,789 | 16,588 | pooled within-group sd over (EC, substrate) groups across organisms |
| predicted | 0.47 (a floor) | 0.4687 | 58 | 58 | sd of the ensemble's signed residual against the measured yeast pairs, signed median -0.1094 |
| default | 1.23 | 1.2300 | | 826 | sd of log10 k_cat over the yeast wild-type rows |

The tiers are a provenance ordering and not an accuracy ordering: the homolog rung is the
widest of the three evidence tiers. 40.2 percent of predicted units have an across-predictor
sd above the 0.47 floor and take their own width instead; the predicted rung's p90 is 0.758
and its maximum 1.618.

**Neutrality.** The same run resolves the table through `ParameterTable.build` with and
without the width and compares element-wise: `values`, `provenance`, `known_mask` and
`experimental_mask` are all equal, the tier counts are the committed 97 / 782 / 2,831 / 18,
and all three admitted predictor parquets match their sha256 in
`results/kinetic_table_versions.json`.

New versioned files, written beside the shipped ones and never over them:
`results/kcat_unit_sigma_v2026.09.14.parquet` (3,728 rows: unit, reaction, tier, value,
across-predictor sd, sigma) and `results/kcat_unit_shipped_v2026.09.14.parquet` (the same
without the two width columns).

### E6, the protein pool (job 1919)

Two enforcements that are not the same constraint. The LP pool prices every flux against the
cheapest enzyme that can carry it, `sum_j c_j (v_j+ + v_j-) <= sigma p_avail` with
`c_j = min_u MW_u / (k_cat_u 3600)`, and lets the solver allocate protein optimally. The
equal-molar path gives every gene `sigma p_avail / sum_g MW_g` and allocates nothing, which is
the flux layer's own prior. The first is the most growth a pool permits and the second the
least.

| p_avail g/gDW | saturation | LP growth /h | equal-molar E, mmol/gDW | equal-molar growth | min feasible multiplier |
|---|---|---|---|---|---|
| 0.125 | 0.5 | 0.01454 | 9.65e-7 | infeasible | 15.3 |
| 0.125 | 1.0 | 0.02041 | 1.93e-6 | infeasible | 7.64 |
| 0.25 | 0.5 | 0.02041 | 1.93e-6 | infeasible | 7.64 |
| 0.25 | 1.0 | 0.03044 | 3.86e-6 | infeasible | 3.82 |
| 0.35 | 0.5 | 0.02442 | 2.70e-6 | infeasible | 5.45 |
| 0.35 | 1.0 | 0.03844 | 5.40e-6 | infeasible | 2.73 |

Unconstrained growth on the same medium is 0.08584 /h and the pool binds exactly in every
row (`pool_used` equals the pool to four digits).

Three readings.

- **Our `p_avail` of 0.35 with no saturation factor is the loosest setting in the table and
  still costs 55 percent of growth** under optimal allocation, 0.03844 against 0.08584.
  GECKO's own yeast pool of 0.125 costs 76 percent.
- **The equal-molar prior at p_avail 0.35 and saturation 0.5 gives E = 2.70e-6 mmol/gDW,
  which is the 2.7e-6 the wave-8 note quotes.** The number's origin is settled: it is
  `abundance_prior_budget_fraction` 0.5 times `p_avail` 0.35 divided by the summed molecular
  weight, and it is the layer's default prior rather than a measurement.
- **No setting reproduces a measured growth rate, and none is close.** The min feasible
  multiplier column shows why: the equal-molar allocation needs E near 1.47e-5 mmol/gDW just
  to be feasible, which is 5.4x the prior the abundance head starts from. The honest
  statement is that the pool cannot be satisfied by an equal-molar proteome at any pool value
  in this table. Comparing any of these to GECKO's `gR_exp` of 0.41 /h would be comparing two
  media and is not done here.

`p_avail` is now a sourced record. `GECKO3_YEAST_PROTEIN_POOL` in `parameters.py` carries
0.125 g/gDCW as `sigma f P_tot = 0.5 x 0.5 x 0.5`, the citation key
`chenReconstructionSimulationAnalysis2024`, the sha256
`3c340a8d1d4f57b86620b5f091b2f622272837943e96d571a5f0c9219b98a2c4` of that key's `paper.md`
in the mirror, the lines (L132 and L964) and the verbatim quote. The shipped 0.35 is recorded
as `TORCHCELL_DEFAULT_PROTEIN_POOL` with the plain statement that no derivation for it exists
anywhere in the repository. **The default is not changed.**

### A reproducibility defect found on the way, and its fix

Resolving the k_cat table in two processes with a free hash seed gives **2 of 3,728 units a
different value, by up to 0.313 decades**: the resolver breaks a tie between equally
acceptable database rows through a container whose iteration order depends on string hashing.
Within one process it is exactly reproducible. Measured by resolving the table in four
processes, two with a free seed and two with `PYTHONHASHSEED=0`; the pinned pair is identical
and the free pair is not.

Every script in this block therefore refuses to start unless `PYTHONHASHSEED=0` is set, and
every slurm script exports it. Without the pin the sweep shards would have been compared
against each other on slightly different tables.

### Two additive options, for the warm-started residual cell

Both live in `parameters.py`, both default to the behavior that shipped, and both are
reachable through the table route rather than by overwriting a layer buffer.

| keyword | on | default | what it is |
|---|---|---|---|
| `complex_aggregation` | `resolve_kcat_table`, `load_predicted_unit_table`, `_database_tiers`, `_per_unit_from_predictor` | `"min"` | `min` as shipped, `max` as GECKO 3, `median`; applied to the measured, homolog and predicted tiers alike, and to the in-process registry path. `K_M` keeps its `max` whatever is passed. |
| `kcat_ceiling_sigma_k` | `resolve_kcat_table` | `0.0` | widens every value to `10^(mu_u + k sigma_u)` with sigma from `KCAT_SIGMA_LADDER`, and attaches a `KcatCeilingWidening` record naming k, the ladder version, the median shift and the per-tier sigma |

A caller that wants the aggregation applied end to end passes it to BOTH
`load_predicted_unit_table` and `resolve_kcat_table`, because the predicted values arrive at
the resolver already reduced. `resolve_kcat_table` now also attaches `log10_sigma` to every
table it returns, widened or not.

The two answer different questions and are not interchangeable. The aggregation is a
CONVENTION change on the same predictions, which moves complexes and leaves the table's
center alone; the sigma widening is an UNCERTAINTY statement about the committed convention,
which moves everything by its own width.
