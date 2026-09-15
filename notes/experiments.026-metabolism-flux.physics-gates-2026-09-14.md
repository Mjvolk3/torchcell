---
id: m0f2oygwoe6ixamf1c19gqp
title: Physics Gates 2026 09 14
desc: ''
updated: 1789374159781
created: 1789374159781
---

## 2026.09.14 - Rank-0 physics gates, and the warm-started residual head

Every number below comes from a committed script under
`experiments/026-metabolism-flux/scripts/` and carries the slurm job that produced it. The
design these gates test is the 2026-09-14 physics review; its probe scripts were promoted
here rather than re-derived.

### Headline

1. **X0a passes decisively and it is the one that mattered.** A trained head's proposal sits
   at a relative distance of **0.990** from the feasible set, so the projection discards 99
   percent of what the head emits (job 1904). The report's read was "if rel > 1, option A is
   the main event".
2. **E2 passes.** Removing the 47 reactions the slack LP puts its violation on drops the
   total L1 slack from **2,641.5 kJ to 0.000 kJ** over the remaining 378 constraints, so the
   thermodynamic infeasibility is concentrated, not distributed (job 1906).
3. **E3 splits, and the count half fails.** The transport term frees its named offender:
   ATP synthase `r_0226` goes from a slack of **9.90 kJ to 0.00 kJ** on the shipped route
   (job 1920). But the violated count RISES on every route and both tables (shipped 46 to
   52; union_external 47 to 51), so by the review's own kill rule E3 is a correctness fix
   for the writeup and does not get a GPU.
4. **E1's head is implemented and its initialization check FAILS, so no GPU cell was
   submitted.** On the real GEM, untrained, `warmstart_k100_boxforce` reads
   `balance_active_max` **1.955**, p95 **1.824**, **20** active rows, `flux_active_frac`
   **0.208**, against the target 0.00118 / 0.00038 / 82 / 0.21 (job 1922).
5. **The init check names the blocker, and it is not the warm start.** The genotype box does
   not contain the reference flux. The enzyme-capacity ceiling has a median of **0.0485
   mmol/gDW/h** over GPR reactions against a pFBA reference whose largest flux is **9.12**,
   so 82 of the reference's 495 flux-carrying reactions sit above their own ceiling by a
   median factor of **4.31**, and clipping costs **0.761** of the reference's norm with no
   directional tightening at all. The directional tightening then forbids the direction of a
   further **145** of those 495 reactions.

### X0c. The probes are committed scripts now

`gh_cpu_physics_probes.slurm` reruns all seven in 89 seconds (job 1900, COMPLETED,
MaxRSS 1.1 GB). The merged `probe_tfa_lp.py --route` carries the old `probe_tfa_union` and
`probe_offenders` in one file. Every number in the review reproduces:

| probe | key numbers, reproduced |
|---|---|
| `probe_fba_rule.py` | FBA 83 active rows / 0.1075 density / `bal_act_max` 2.1e-8, all-rows max **0.098**; pFBA 84 / 0.1029 / 2.1e-8; random vertex 296 / 0.1104 |
| `probe_dirbox.py` | max growth and max downhill throughput both exactly **0.0** at margins 1e-3 and 5 kJ/mol (1,071 / 1,031 reactions pinned), against an unconstrained growth of 0.0858; the LP is infeasible at 20 kJ/mol |
| `probe_tfa_lp.py` | shipped 46 violated / 1,239 kJ; equilibrator 24 / 1,551; union 48 / 3,157; union_external 47 / 2,642 / max 313; verified 46 / 2,877. Widening to 1e-12..10 M leaves 14 to 25 |
| `probe_projrate.py` | `bal_act_max` 2.0 at every K from 20 to 5,000, alternating and Dykstra alike; active rows 1,808 to 557 |
| `probe_warmstart.py` | rel 0.01 at K=100 gives 0.00118 / 0.00038 / 82 rows / 0.210; rel 1.0 gives 1.80; uniform random gives 2.0 |
| `probe_dual.py` | float32 dual ascent leaves the max absolute balance residual at **42.0** after 1,000 L-BFGS iterations, 8.1 s at batch 2 |
| `probe_massbal.py` | 212 of 3,857 internal reactions unbalanced, **46** with a charge imbalance, 35 with a formula-less participant |

`probe_warmstart.py` additionally records `projection_displacement_rel` beside each `rel`,
which is what makes the trained-head measurement placeable in that table at all: rel 0.01
displaces 0.0080, rel 0.1 displaces 0.0685, rel 1.0 displaces 0.602, and a uniform random
point displaces 0.948.

### X0b. The audit has a reference column, and one rule is called Genesis

`audit_top_strain_feasibility.py` now solves pFBA on the GEM the audited layer was built
from, scores it through the layer's own `_s_matmul` and `turnover`, and prints it as a fixed
column beside every group's table. It is recomputed, never transcribed, and it refuses a GEM
whose reaction order differs from the layer's.

The rename, and nothing about how a number is computed moved:

| before | after | why |
|---|---|---|
| `genesis_median_rule_*` | `median_rule_*` | a mostly-zero flux passes the balance half on its zeros |
| `genesis_max_rule_*` | `solver_noise_max_rule_*` | the reference pFBA flux FAILS it at 0.098, so it cannot be a feasibility criterion |
| `genesis_active_max_1e-03_rule_*` | `genesis_turnover_rule_1e-03_*` | the only rule now called Genesis |
| `genesis_active_{p95,other thresholds}_*` | `turnover_rule_<stat>_<t>_*` | sensitivity, reported beside the criterion rather than as one |

Audits written before today keep the old names and still parse: `StrictFeasibility.reference`
is optional, and `analyze_wave8_density.py` reads each rule as a preference order over the
new name and the old one.

What the column reads, measured through a built layer (float32, the layer's own `_s_matmul`
and `turnover`, union_external table, predicted kcat tier): biomass 0.08584 on `r_2111`,
`flux_active_frac` **0.1029**, `balance_active_rows_1e-03` **84**,
`balance_active_max_ratio_1e-03` **1.50e-7**, `balance_active_p95_ratio_1e-03` 9.01e-8, and
`balance_max_ratio` **0.0980**, which is the number that fails the retired rule. No reaction
of the reference exceeds the layer's flux scale, so the column is a flux the head could in
principle emit.

**Correction to the brief.** It asked for the reference row as "83 active rows, 0.107
density, bal_act_max 1.1e-7". Measured, those three numbers are not one row: the FBA flux
reads 83 rows and 0.1075, pFBA reads **84 rows and 0.1029**, and 1.1e-7 is the float32 torch
reading of the FBA flux in `probe_projrate.py` against the float64 cobra reading of 2.1e-8
in `probe_fba_rule.py`. The audit's column is pFBA, as instructed, and carries pFBA's own
numbers.

### X0a gate. A trained head's proposal distance

`probe_trained_proposal_distance.py` rebuilt job 1784 (density 2.0, seed 2700,
`fluxforce_k100_boxforce`), forwarded 32 genotypes and read the layer's own
`v_pre_projection` (job 1904, 30 s):

| statistic | median | min | max |
|---|---|---|---|
| `proposal_distance_rel` | **0.99004** | 0.98989 | 0.99042 |
| `balance_active_max` | 2.0 | 2.0 | 2.0 |
| `balance_active_rows` | 820 | 796 | 824 |
| `flux_active_frac` | 0.3325 | 0.3186 | 0.3343 |
| `v_pre_active_frac` | 0.8267 | 0.8250 | 0.8303 |

The nearest band in `probe_warmstart.json` is the uniform-random-in-box row (displacement
0.948), not the rel-1.0 row (0.602). The trained head's proposal is as far from the feasible
set as a random point of its own box, which is the strongest form the review's hypothesis
could have taken.

### E2 gate. The infeasibility is concentrated

`thermo_exempt_gate.py` (job 1906), route `union_external`, window 1e-7 to 1e-2 M, epsilon
1e-3 kJ/mol, flux fixed at the cobra FBA optimum. Base LP: 425 constrained reactions, 47
violated, total L1 slack 2,641.5 kJ, max 312.75 kJ. Removing exactly those 47 leaves
**0.000 kJ over 378 constraints with 0 violated**, so the greedy minimum-cardinality pass
never had to run and the method recorded is `l1_violated`.

`results/thermo_exempt_candidates.json` carries the 47 ids with the route, the thermodynamic
table's per-file sha256 set, the window, the epsilon and the method. The list:

```
r_0014 r_0361 r_0563 r_1025 r_1115 r_1129 r_1166 r_1194 r_1244 r_1266 r_1697 r_1738
r_1748 r_2032 r_2125 r_2194 r_2308 r_2317 r_2432 r_2434 r_2437 r_2446 r_2447 r_2451
r_2455 r_2456 r_2481 r_2909 r_2972 r_3008 r_3515 r_3518 r_3520 r_3528 r_3536 r_3540
r_3563 r_3564 r_3575 r_3578 r_3580 r_3581 r_3587 r_3653 r_3765 r_3770 r_3854
```

**Caveat, in the JSON as well as here.** The LP minimizes the TOTAL slack, so this is where
it chose to place a joint infeasibility. A residual of exactly 0.000 says the set is
sufficient, not that it is minimal.

### E3 gate. The mechanism is confirmed and the count is not

`probe_transport_tfa.py` builds the transport table with
`torchcell.metabolism.transport.write_transport_table` under two condition sets and re-solves
the same LP. `three_compartment` is what `FluxLayerConfig`'s `CompartmentParameters` actually
tabulates (pH for c, m, e only); `pipeline` is `transport.pipeline_conditions()`, textbook pH
for all 14 compartments.

| route | transport | violated | L1 slack (kJ) | `r_0226` slack (kJ) | `r_0226` transport (kJ/mol) |
|---|---|---|---|---|---|
| union_external (job 1907) | off | 47 | 2,641.5 | 0.00 | 0.00 |
| union_external | three_compartment | 51 | 2,828.1 | 0.00 | -35.52 |
| union_external | pipeline | 50 | 2,827.1 | 0.00 | -37.84 |
| shipped (job 1920) | off | 46 | 1,239.1 | **9.90** | 0.00 |
| shipped | three_compartment | 52 | 1,319.1 | **0.00** | -35.52 |
| shipped | pipeline | 52 | 1,318.3 | **0.00** | -37.84 |

So ATP synthase leaves the violated list exactly where the review said it should, on the
shipped table where it was named, and the count does not fall anywhere. Under
`union_external`, the route every wave-7 and wave-8 cell ran, `r_0226` was never violated in
the first place.

**The labeled assumption.** Both tables give terms for all 1,468 membrane-crossing reactions.
Under `three_compartment`, **709** of them cross a compartment with no tabulated pH and
therefore take the reference compartment's pH, which zeroes their pH term; that is an
assumption, not a measurement, and the count is recorded in the JSON beside the result. Under
`pipeline` that count is 0 because textbook pH is assumed for all 14 compartments, which
moves the assumption rather than removing it. Nonzero terms: 264 and 326 respectively.

### E1. The warm-started residual head

**Landed in `torchcell/metabolism/flux_layer.py`.** A new parameterization
`warmstart[_k<N>][_balance|_box|_force|_boxforce][_dykstra]`:

```
v_pre = clip(v_ref, lb_g, ub_g) + alpha * ||clip(v_ref)|| * (N z) / ||N z||
```

then the existing clip and alternation. Four config fields with docstrings
(`warm_start_reference`, `warm_start_init_rel`, `warm_start_growth_fraction`,
`warm_start_cache_path`), a module-level `parsimonious_reference_flux` that solves the two
pFBA linear programs from `S`, the bounds and the objective column with scipy HiGHS, and a
`ReferenceFluxRecord` that travels into `coverage_report`. Three properties are deliberate:

- **the reference is solved inside the build step**, never passed in, so a different GEM
  brings its own reference and no organism constant enters the module;
- **the split in the parsimonious LP preserves the original box exactly** rather than
  relaxing it, which the lazy `p in [0, max(ub,0)]` form silently does not;
- **alpha IS the proposal's relative distance**, because the null-space direction is
  normalized before it is scaled, which is what makes the one learned scalar comparable to
  the `rel` of `probe_warmstart.json`.

Off is bit-identical: no buffer, no parameter and no `state_dict` entry is added to any other
channel, and 110 tests pass in `tests/torchcell/metabolism/test_flux_layer.py` (13 new).

**Init check, job 1922, untrained, batch 4, built through 026's own `build_flux_layer` on
`flux_projection_anchored_k20_boxend` with the predicted kcat tier and union_external
energies:**

| statistic | `fluxforce_k100_boxforce` (control) | `warmstart_k100_boxforce` | target |
|---|---|---|---|
| `balance_active_max` | 2.000 | **1.955** | 0.00118 |
| `balance_active_p95` | 2.000 | **1.824** | 0.00038 |
| `balance_active_rows` | 143 | **20** | 82 |
| `flux_active_frac` | 0.2954 | **0.2076** | 0.21 |
| `proposal_distance_rel` | 0.9988 | **0.9294** | ~0.008 |

**The check fails, so the pre-approved GPU submission was not made.** The cells are written
and ready.

**Why it fails, measured in the same JSON.** The reference carries flux on 495 reactions with
a maximum magnitude of 9.12 and a norm of 19.56. The genotype box cannot hold it:

| box | clip cost | sign-forbidden reference reactions | over the ceiling | median excess | after clip | after clip and K=100 |
|---|---|---|---|---|---|---|
| `box` (no directional tightening) | 0.761 | **0** | 82 | 4.31x | max 1.998, 72 rows | max 1.083, p95 0.608, 103 rows, density 0.457 |
| `boxforce` | 0.793 | **145** | 58 | 4.17x | max 2.000, 58 rows | max 1.955, p95 1.824, 20 rows, density 0.206 |

Two causes, and they have different remedies:

- **The enzyme-capacity ceiling.** Median 0.0485 mmol/gDW/h over GPR reactions, minimum
  2.2e-6, at the nominal prior abundance. With zero reactions sign-forbidden, clipping still
  costs 0.761 of the reference's norm and takes `bal_act_max` from 0 to 1.998, so the
  capacity bound alone is enough to destroy a feasible flux. The remedy is the abundance
  prior, which this session does not own.
- **The directional tightening.** It forbids the direction of 145 of the reference's 495
  flux-carrying reactions, which is the per-genotype form of `probe_dirbox.json`: at standard
  state the maximum downhill throughput over the whole tightened polytope is exactly 0.0. The
  E2 exemption list is the first candidate fix.

**Hypothesis (untested).** These two also explain X0a without any appeal to the head: if the
box contains no flux of physiological magnitude, then no proposal inside it can be both
balanced and dense, whatever emits it. That would make the box, not the proposal and not the
projection, the thing four waves were fighting. It is not measured, and the measurement that
would settle it is whether any point of the genotype box at the trained abundance satisfies
`Sv = 0` with a turnover above the rule's threshold on more than a handful of rows.

### E4. Parsimony outside the ramp

`balance_active_weight` was already outside the ramp, so only parsimony needed adding.
`FluxLayerConfig.parsimony_outside_ramp_weight` defaults to 0.0 and `constraint_loss` adds it
outside the `constraint_scale` product beside the density and active-row balance terms. Two
tests cover it, including that the default leaves the loss bit-identical.

The reason it is worth a cell: `constraint_loss` multiplies `c_parsimony` by
`constraint_scale`, every wave-7 and wave-8 cell runs that scale at exactly 0.0, so the
parsimony weight those cells record has been multiplied by zero in every one of them.
Parsimony has never been applied in this experiment, and parsimony is what makes an FBA flux
sparse: the pFBA reference carries flux on 10.3 percent of reactions against the boxforce
head's 28 to 32.

**The two cells are blocked on a five-line patch to `train_bx.py`, which this session does
not own.** Verbatim, with the anchors as they stand today:

1. Beside the `flux_density_weight` field of `BxCell` (near line 746):

```python
    flux_parsimony_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the flux layer's parsimony term, added OUTSIDE the constraint "
            "scale. The `weights['parsimony']` entry is inside the ramp, and this recipe "
            "holds that ramp at 0.0, so parsimony has never actually been applied."
        ),
    )
```

2. In `build_cell`'s signature, beside `flux_balance_active_weight: float = 0.0` (near line
   1528):

```python
    flux_parsimony_weight: float = 0.0,
```

3. In the `BxCell(...)` construction inside `build_cell`, beside
   `flux_balance_active_weight=flux_balance_active_weight,` (near line 1634):

```python
        flux_parsimony_weight=flux_parsimony_weight,
```

4. Beside `flux_layer.config.balance_active_weight = cell.flux_balance_active_weight` (near
   line 3152):

```python
        flux_layer.config.parsimony_outside_ramp_weight = cell.flux_parsimony_weight
```

5. Beside the `--flux-density-weight` argument (near line 4941), and its call site beside
   `flux_balance_active_weight=args.flux_balance_active_weight,` (near line 5019):

```python
    parser.add_argument(
        "--flux-parsimony-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the flux layer's parsimony term, added outside the constraint "
            "scale. Parsimony is what makes an FBA flux sparse and it has never been "
            "applied: the recipe these cells run holds the ramp at 0.0."
        ),
    )
```

```python
        flux_parsimony_weight=args.flux_parsimony_weight,
```

### Pre-registered reads and kill rules, verbatim from the review

E1, rank 1:

> **Pre-registered read:** (a) `genesis_turnover_rule_1e-03` on the top 100, success >= 0.95
> on >= 2 of 3 seeds; (b) `flux_active_frac` and `balance_active_rows_1e-03`, which must move
> TOWARD 0.107 and 83, not away; (c) pinned test Spearman, which must not fall more than the
> control's seed sd (0.076) below the control mean (+0.056).
>
> **Kill:** rule not met on any seed -> the warm start is not the lever; go to E2/E3.
>
> **Contingency if it passes but the score collapses:** the head is pinned to `v_ref`.
> Diagnose with across-genotype flux sd; if near zero, ramp `alpha` up over training and
> rerun one seed.

E2, rank 2:

> **Kill:** if removing the 47 does not drop the TFA L1 slack below ~10 kJ on CPU, the
> infeasibility is distributed rather than concentrated and E2 is dead before any GPU runs.

E3, rank 3:

> **Kill:** if the CPU LP does not improve, E3 is a correctness fix for the writeup, not a
> Genesis lever, and it does not get a GPU.
>
> **Caveat to record:** pH exists for 3 of 14 compartments. The 1,056 reactions crossing an
> untabulated compartment get zero transport term either way, and that is a labelled
> assumption, not a measurement.

E4, rank 4:

> **Pre-registered read:** `flux_active_frac` must fall toward 0.107 and
> `balance_active_rows_1e-03` toward 83; `bal_act_p95` is the outcome.
>
> **Kill:** if `flux_active_frac` does not fall below 0.20 at the strong parsimony weight,
> the head cannot be made sparse by a penalty and the box parameterization is the blocker
> (which the 026 `box_zero_reachability` note already predicted for the sigmoid box; the
> fluxforce magnitude is also a sigmoid, so this is a real risk).

On E3's caveat: the review's 1,056 is the count against `CompartmentParameters`' three
compartments over multi-compartment reactions; the number this run measures from the built
table is **709** of the 1,468 membrane-crossing reactions, and that is the one to quote,
since it comes from the transport table the LP actually consumed.

### Ready cells, none submitted

| cell | state |
|---|---|
| `gh_wave9_warmstart_e200_s2700.slurm` | ready, NOT submitted: the init check failed |
| `gh_wave9_warmstart_e200_s2701.slurm` | ready, NOT submitted |
| `gh_wave9_warmstart_e200_s2702.slurm` | ready, NOT submitted |
| `gh_wave9_parsimony_1e-3_e200_s2700.slurm` | ready, blocked on the `train_bx.py` flag above |
| `gh_wave9_parsimony_1e-2_e200_s2700.slurm` | ready, blocked on the same flag |

All five run through `freeze_and_submit.sh`. The three warm-start cells are wired end to end:
`build_cell('flux_warmstart_boxforce', ...)` resolves to
`flux_projection_anchored_k20_boxend` with channel `warmstart_k100_boxforce`, density 0.0 and
active-row balance 0.0.

### 2026.09.14 (follow-up) - the 2x2, and the box was the enemy

The init check above failed for two measured causes, so both were made switchable and the
check was rerun as a 2x2 with the control, plus an LP that settles the hypothesis the first
run could only state. Job 1925 (`gh_cpu_warmstart_init.slurm`, 96 s; jobs 1923 and 1924 are
its two earlier passes, superseded by the objective fix described under the LP).

**Nothing wins, and nothing was submitted.** The rule was `balance_active_max` and p95 both
below 1e-2 with at least 40 active rows.

| setting | max | p95 | rows | density | proposal rel |
|---|---|---|---|---|---|
| control `fluxforce` / prior / exempt off | 2.000 | 2.000 | 143.0 | 0.2954 | 0.9988 |
| warmstart / prior / exempt off | 1.955 | 1.824 | 20.0 | 0.2076 | 0.9294 |
| warmstart / prior / exempt **on** | 1.911 | 1.791 | 28.5 | 0.2266 | 0.9085 |
| warmstart / **nominal** / exempt off | 2.000 | 1.833 | 44.5 | 0.2326 | 0.8829 |
| warmstart / **nominal** / exempt **on** | 2.000 | 1.616 | 59.2 | 0.2473 | 0.8493 |
| target (`probe_warmstart.json`, rel 0.01) | 0.00118 | 0.00038 | 82.0 | 0.21 | 0.008 |

Both axes move every statistic the right way and monotonically, and the best cell is still
three orders of magnitude from the balance target. The proposal distance falls from 0.929 to
0.849 against a target of 0.008.

**The clip-cost table says why, and it separates the two causes cleanly.**

| setting | box | clip cost | sign-forbidden | over ceiling | median excess |
|---|---|---|---|---|---|
| prior / exempt off | `box` | 0.761 | 0 | 82 | 4.31x |
| prior / exempt off | `boxforce` | 0.793 | 144.8 | 58 | 4.17x |
| prior / exempt on | `boxforce` | 0.791 | 119.8 | 61 | 4.20x |
| nominal / exempt off | `box` | **0.316** | 0 | **3** | 2.65x |
| nominal / exempt off | `boxforce` | 0.495 | 150.0 | 2 | 1.76x |
| nominal / exempt on | `boxforce` | 0.477 | **124.2** | 2 | 1.76x |

The nominal abundance path fixes the capacity axis outright: the ceiling median goes from
0.0485 to **8.22** mmol/gDW/h, the reference's over-ceiling reactions from 82 to **3**, and
the clip cost into the untightened box from 0.761 to 0.316. The exemption list moves the
other axis by the amount it can: 47 exempted reactions take the sign-forbidden count from
144.8 to 119.8 and from 150.0 to 124.2 of the reference's 495 flux-carrying reactions.

**The LP, and it settles the hypothesis the first run labeled as untested.** For each
setting, over the tightened box of one genotype, maximize total turnover subject to
:math:`Sv=0`. Turnover is convex so this is not an LP; what is solved is the split
:math:`v=p-n` with the original box preserved, the objective scored ONLY on the reactions
whose box lies on one side of zero (where :math:`p_j+n_j=|v_j|` exactly, which is 4,014 to
4,045 of the 4,131) and the two-sided remainder carried as an additive headroom on the upper
bound.

| setting | box as built | forced-bound relaxed | one-sided turnover at the optimum | rows at the witness | density at the witness |
|---|---|---|---|---|---|
| control / prior / exempt off | **infeasible** | optimal | **0.00** | **0** | 0.0000 |
| warmstart / prior / exempt off | **infeasible** | optimal | **0.00** | **0** | 0.0000 |
| warmstart / prior / exempt on | **infeasible** | optimal | 1.90 | 14 | 0.0019 |
| warmstart / nominal / exempt off | **infeasible** | optimal | **0.00** | **0** | 0.0000 |
| warmstart / nominal / exempt on | **infeasible** | optimal | 76.76 | 14 | 0.0019 |

Two findings, and the second is the one that matters.

1. **The tightened box holds no balanced point at all as it is built.** The LP is infeasible
   in every setting, and the cause is one reaction: `r_4046`, the maintenance reaction, whose
   bounds the GEM fixes at 0.7 so its box excludes zero. Inside the directionally tightened
   box the network cannot balance it. Relaxing that single bound to admit zero makes the
   program feasible in every setting, which is how the second finding is read at all.
2. **Once it is feasible, the maximum turnover any balanced point can carry on the one-sided
   reactions is exactly 0.00 in three of the five settings.** The objective is a sum of
   non-negative terms over 4,045 reactions, so a maximum of zero means every one of them is
   zero at every balanced point of that box. The witness has **0 active rows**; with the
   exemption on it reaches 14 rows and a density of 0.0019, against the pFBA reference's 82
   rows and 0.1029.

So the box, not the proposal and not the projection, is what four waves were fighting. This
is the per-genotype form of `probe_dirbox.json`, where the standard-state directional
polytope has a maximum downhill throughput of exactly 0.0, and it explains the whole record
without appealing to the head: `bal_act_max` = 2.0 in every `boxforce` cell, the density term
raising the row count without moving the ratio, and a trained proposal 0.990 away from a set
that is essentially the origin. It also retires the first run's framing: the warm start is
not the lever, because no proposal inside that box can be both balanced and dense.

Also measured, and worth recording on its own: after the directional tightening only **86**
of 4,131 reactions can still take either sign, rising to **117** with the exemption list
applied. The second law as a per-reaction sign clamp at the learned concentrations is a far
harder constraint than it reads.

**What landed for this check.** `FluxLayerConfig.thermo_exempt_path` plus
`read_thermo_exempt_list` and `FluxLayer.apply_thermo_exempt_list`, which extend the Box 2
exempt set from the provenance-carrying JSON and record the file's sha256, route, window and
table hashes in `coverage_report`. Off is bit-identical; 4 new tests, 114 pass. The abundance
axis needed no new code: `use_abundance_head` already selects it, and the check confirms that
`flux_nullspace` is the registry twin of `flux_projection_anchored_k20_boxend`, differing in
`use_abundance_head` and `parameterization` alone, so a 027 arm file can reach the nominal
path by naming it as `base_arm`.

**The remaining cause, stated plainly.** The directional tightening (`boxforce`), evaluated
at the layer's own learned concentrations, leaves an intersection with `ker S` that is
essentially the origin. The 47-reaction exemption is a real but partial remedy: it buys 14
active rows where there were 0. The next thing worth measuring is not another head but the
same LP run against the `_box` end with no tightening at all, and then a decision about
whether the second law belongs in the box on this table.

### 2026.09.14 (follow-up 2) - direction from consistent concentrations

The previous section concluded that the directional box's intersection with `ker S` is
essentially the origin, and stopped one step short: the tightening is evaluated at the
layer's OWN concentrations, and at initialization those are a zero-centered parameter plus a
random linear head, so the box was being tightened against a concentration vector that
licenses nothing in particular. The E2 gate's LP already produced the vector that DOES
license the reference flux. Starting the layer there changes the result substantially.
Job 1933 (`gh_cpu_warmstart_init.slurm`, 117 s; 1929 is the same run before the offenders
were named, 1927 rebuilt the exemption gate to emit the witness).

**Still no winner, so nothing was submitted.** The rule remains both balance statistics
below 1e-2 with at least 40 active rows.

| setting | max | p95 | rows | density | proposal rel | sign-forbidden |
|---|---|---|---|---|---|---|
| control `fluxforce` / prior / off | 2.0000 | 2.0000 | 143.0 | 0.2954 | 0.9988 | n/a |
| warm / prior / off | 1.9550 | 1.8241 | 20.0 | 0.2076 | 0.9294 | 144.8 |
| warm / prior / exempt | 1.9106 | 1.7910 | 28.5 | 0.2264 | 0.9086 | 119.8 |
| warm / nominal / off | 2.0000 | 1.8331 | 44.5 | 0.2326 | 0.8829 | 150.0 |
| warm / nominal / exempt | 2.0000 | 1.6164 | 59.2 | 0.2473 | 0.8493 | 124.2 |
| warm / prior / exempt / **ln c** | 1.6801 | 0.9404 | 67.2 | 0.2844 | 0.8507 | **14.0** |
| warm / **nominal** / exempt / **ln c** | **0.6568** | **0.2757** | **120.5** | 0.2722 | 0.7097 | **14.0** |
| target | 0.00118 | 0.00038 | 82.0 | 0.21 | 0.008 | 0 |

The concentration init does exactly what it was predicted to do on the axis it addresses:
sign-forbidden reference reactions fall from 124 to **14**, and every remaining one carries a
flux of 0.027 or less, a squared-norm share of 0.0000. The balance statistics move with it,
`bal_act_max` from 2.000 to **0.657** and p95 from 1.616 to **0.276**, and the active-row
count passes the target for the first time at **120.5** against 82. It is still two orders of
magnitude from 1e-2.

**The LP on the untightened box, which is the ceiling the tightening is measured against.**

| setting | box | LP as built | rows at the witness | density | one-sided turnover | two-sided reactions |
|---|---|---|---|---|---|---|
| nominal / exempt / ln c | `boxforce` | **optimal** | **98** | 0.0211 | 180.9 | 351 |
| nominal / exempt / ln c | `box` | optimal | **105** | 0.0220 | 707.5 | 1,668 |
| nominal / exempt (no ln c) | `boxforce` | infeasible | 14 (relaxed) | 0.0019 | 76.8 | 117 |
| prior / exempt / ln c | `boxforce` | infeasible | 8 (relaxed) | 0.0235 | 1.5 | 351 |
| prior / exempt / ln c | `box` | **infeasible even relaxed** | n/a | n/a | n/a | 1,668 |

Three things follow and they change the previous section's reading.

1. **With consistent concentrations the tightened box is feasible as built**, for the first
   time in this whole investigation. Every earlier setting was infeasible because the GEM's
   forced maintenance reaction could not be balanced inside it.
2. **The sign constraint in the box now costs almost nothing.** The untightened box reaches
   105 turning-over rows at the LP optimum and the tightened one reaches **98**. The earlier
   conclusion that the second law cannot live in the box was a property of the concentrations
   it was evaluated at, not of putting it in the box. That correction stands against the
   previous section.
3. **At the prior abundance the box is infeasible with or without the tightening.** The
   untightened LP does not solve even with the forced bound relaxed, so the capacity ceiling
   at that abundance admits no balanced point at all, independently of thermodynamics.

**What remains binding, named.** At the best setting the clip still costs 0.316 of the
reference's norm, and it is three reactions:

| reaction | reference flux | ceiling | excess | share of the reference's squared norm |
|---|---|---|---|---|
| `r_0226` ATP synthase | 5.950 | 0.1996 | **29.8x** | 0.0925 |
| `r_0439` cytochrome c | 4.558 | 1.7476 | 2.61x | 0.0543 |
| `r_0961` | 0.786 | 0.2573 | 3.06x | 0.0016 |

So of the three candidate causes, **capacity is what remains binding, and it is concentrated
in three reactions carrying 14.8 percent of the reference's squared norm, dominated by ATP
synthase at 29.8x over its enzyme-capacity ceiling**. The sign axis is effectively closed (14
reactions, none carrying meaningful flux). The 351 two-sided reactions are not binding: they
are what the tightening now leaves alone, and the LP shows the box reaching 98 of a possible
105 rows with them.

The remedy for the remaining cause is a kcat or an abundance for those three reactions, which
is the kinetics stream's, not the head's. `r_0439` is also the cytochrome c pair carrying the
fractional charge imbalance `probe_massbal.json` reports, so its capacity and its
thermodynamics are both suspect for the same underlying reason.

**What landed.** `FluxLayerConfig.log_c_init_path`, `read_log_c_init`, a
`LogConcentrationInit` record and `FluxLayer.apply_log_c_init`, which set `log_c_base` by
inverting the window sigmoid on the witness and zero the genotype head so the initial
concentration IS the witness; `thermo_exempt_gate.py` now writes that witness as
`results/thermo_consistent_log_c.json` with its route, window, exemption-file sha256 and the
count of constraints it satisfies. Off is bit-identical. 3 new tests, **117 pass**.

**Two properties to keep stated, since both are easy to overclaim.** The vector is a per-GEM
quantity: it is solved from this GEM's stoichiometry, the `union_external` table and this
exemption list, all named in the file and recorded in `coverage_report`, so another GEM
brings its own and no organism constant enters the module. And it is an INITIAL VALUE of
learned parameters, not a constraint: `log_c_base` and `log_c_from_context` both keep their
gradients, so training is free to move the concentrations anywhere in the window, which is
exactly why `feas_thermo_violation_frac` is logged every epoch rather than assumed.

**One caveat on the witness itself.** It is an LP vertex: 2,604 of the 2,806 metabolites sit
at the window ceiling of 1e-2 M and 124 at the floor, because the program has no preference
for a metabolite no constraint touches. It is a valid witness for the directions and it is
not a physiological concentration profile. A centered witness (an interior point of the same
feasible set) would be a better initialization and is not what this run used.

### 2026.09.14 (follow-up 3) - the capacity axis on principled rules

The capacity blocker is three reactions, dominated by ATP synthase at 29.8x its ceiling, and
the parameter review names the convention that produces it: our per-unit k_cat takes the
MINIMUM over a complex's subunits while GECKO 3 takes the maximum, and `r_0226` is a
17-subunit complex whose predictors disagree by 0.993 decades (agent-2-parameters.md, 1.2d
and 3.4). So the capacity axis is varied here on rules, never on a hand-edited number.

**The rules, all computed in `probe_warmstart_init.py` from the shipped predictor parquets
and the shipped tier ladder, with `parameters.py` untouched.**

| rule | what it changes |
|---|---|
| `shipped` | the committed table: median over admitted predictors, MINIMUM over subunits |
| `complex_max` | the same predictions, MAXIMUM over subunits, which is GECKO 3's reading |
| `complex_median` | the same predictions, MEDIAN over subunits |
| `sigma_plus_1` | the shipped table times `10^sigma_u`, about 3x |
| `sigma_plus_2` | the shipped table times `10^(2 sigma_u)`, about 10x |

`sigma_u` is `parameters.KCAT_SIGMA_LADDER`, whose rungs the review measured: 0.50 decades
measured, 0.73 homolog, `max(0.47, per-unit across-predictor sd)` predicted, 1.23 default.
The two families answer different questions and are not interchangeable: the aggregation
rules are a CONVENTION change on the same predictions, the sigma rules are an UNCERTAINTY
statement about the committed convention.

**The reimplementation is checked, not trusted.** The probe reproduces
`parameters._per_unit_from_predictor` with one line parameterized, and asserts that at
`min` it returns exactly the shipped helper's values on every admitted predictor's parquet
before any rule is evaluated. Without that check every number in this section would be about
a different table.

**What the kinetics agent has to land before any of this can reach a GPU cell.** The layer
reads its k_cat through `parameters.py`, which this session does not own, so the probe swaps
`layer.kcat_per_h` after construction and a real cell cannot. The change is one option with
a default that reproduces today's behavior exactly:

1. Beside `SigmaTier` (near line 793), the type:

```python
#: How a complex's per-subunit values become one value for the complex. ``min`` is what
#: every table before 2026-09-14 used, "a complex cannot turn over faster than its slowest
#: subunit"; ``max`` is GECKO 3's reading, "the highest predicted k_cat across its
#: subunits", which treats the per-subunit spread as noise around one complex rate. The
#: convention is contested and the reaction it decides is ATP synthase, a 17-subunit complex
#: whose predictors disagree by 0.993 decades.
ComplexAggregation = Literal["min", "max", "median"]
```

2. In `_per_unit_from_predictor` (near line 663), the signature gains
   `complex_aggregation: ComplexAggregation = "min"` and the one line becomes:

```python
    grouped = per_gene.groupby("unit_id")[column]
    per_unit = {
        "min": grouped.min, "max": grouped.max, "median": grouped.median
    }[complex_aggregation]()
```

3. `load_predicted_unit_table` gains the same keyword with the same default and forwards it
   to every `_per_unit_from_predictor` call, and `PredictedUnitTable` gains
   `complex_aggregation: ComplexAggregation = "min"` so the table records which rule built
   it.

4. In `_database_tiers` (near line 555) the measured and homolog tiers reduce over member
   genes with the same rule, since a convention that applies to predictions and not to
   measurements would make the tiers incomparable:

```python
    reduce = {"min": min, "max": max, "median": _median_by}[complex_aggregation]
    take = reduce if kind is KineticKind.KCAT else max
```

   with `complex_aggregation` threaded from `resolve_kcat_table`'s new keyword. `K_M` keeps
   its `max` unchanged, since the affinity argument ("its weakest binding step sets where it
   saturates") is a different one and the review does not contest it.

5. `resolve_kcat_table` gains `complex_aggregation: ComplexAggregation = "min"`, passes it to
   `_database_tiers`, and applies it to the in-process registry path (near line 1136) in
   place of `min(scored)`.

Default `"min"` everywhere makes the change bit-identical for every existing caller,
including 026's `build_flux_layer`, which names no keyword. The arm then reaches it the way
it reaches `kcat_tier`, through the ARMS registry entry.

**What the rules do to the three watched reactions**, k_cat in 1/s summed over each
reaction's catalytic units, against the excess each one has to cover:

| rule | median k_cat | ratio | `r_0226` (needs 29.8x) | `r_0439` (2.61x) | `r_0961` (3.06x) |
|---|---|---|---|---|---|
| shipped | 4.27 | 1.0 | 1.24 | 2.81 | 0.48 |
| `complex_max` | 4.38 | 1.0 | **81.1** (65x) | **90.2** (32x) | 0.48 (1x) |
| `complex_median` | 4.37 | 1.0 | 10.4 (8.4x) | 11.6 (4.1x) | 0.48 (1x) |
| `sigma_plus_1` | 15.3 | 3.16 | 12.2 (9.9x) | 20.8 (7.4x) | 2.58 (5.4x) |
| `sigma_plus_2` | 53.0 | 10.0 | **120.3** (97x) | **153.9** (55x) | **13.8** (29x) |

The median k_cat is UNCHANGED by the aggregation rules (4.27 to 4.38) and moves by the
expected 3.16x and 10x under the sigma rules. That is the point: min-versus-max over subunits
is not a global rescaling, it is a change to complexes only, and it moves ATP synthase by 65x
while leaving the table's center alone. `r_0961` does not move under either aggregation rule,
so it is a single-gene unit and only the sigma rules reach it.

### 2026.09.14 (follow-up 4) - the capacity grid, and what the last 8x is

Job 1935 (`gh_cpu_warmstart_init.slurm`, 4 min 28 s), 15 settings: the five capacity rules
crossed with the two concentration witnesses at nominal abundance with the exemption on, plus
the five earlier settings. Job 1934 (7 s) produced the centered witness.

**No cell wins, and the best one misses on the maximum alone.**

| setting (all nominal / exempt / ln c) | max | p95 | rows | density | prop rel | forbidden | over |
|---|---|---|---|---|---|---|---|
| `shipped`, vertex | 0.6568 | 0.2757 | 120.5 | 0.2722 | 0.710 | 14 | 3.0 |
| `shipped`, centered | 1.1556 | 0.2822 | 116.8 | 0.2862 | 0.702 | 16 | 3.0 |
| `complex_max`, vertex | 0.2486 | 0.0806 | 97.0 | 0.2690 | **0.132** | 14 | 1.8 |
| `complex_max`, centered | 0.3079 | 0.0771 | 95.8 | 0.2763 | 0.147 | 16 | 1.8 |
| `complex_median`, vertex | 0.2726 | 0.1128 | 113.5 | 0.2536 | 0.480 | 14 | 2.0 |
| `sigma_plus_1`, vertex | 0.1946 | 0.0809 | 112.5 | 0.2545 | 0.424 | 14 | 1.0 |
| **`sigma_plus_2`, vertex** | **0.0847** | **0.0082** | **90.5** | **0.2098** | **0.109** | 14 | **0.8** |
| `sigma_plus_2`, centered | 0.1318 | 0.0140 | 88.0 | 0.2153 | 0.133 | 16 | 0.8 |
| target | 0.00118 | 0.00038 | 82.0 | 0.21 | 0.008 | 0 | 0 |

`sigma_plus_2` at the vertex witness **passes the p95 half of the rule** (0.0082 against 1e-2)
with 90.5 active rows and a density of 0.2098 against the target's 0.21, and misses the
maximum by 8.5x. Every statistic has moved monotonically with the capacity rule, and the
projection now discards 11 percent of the proposal rather than 99.

**The centered witness does not help.** It is a genuinely different point (mean
`|ln c - mid|` 0.1409 and 30 of 2,806 metabolites at a bound, against the vertex's 2,728) and
it makes the maximum WORSE in every pairing (0.66 to 1.16, 0.25 to 0.31, 0.085 to 0.132),
the p95 a wash, and the sign-forbidden count slightly worse (14 to 16). Measured and
negative: the vertex is the better initialization of the two, and the reason is not
established here.

**What the last 8x is, and it is no longer k_cat.** Under `complex_max` and `sigma_plus_2`
the enzyme capacity for `r_0226` is **13.06** and **19.37** mmol/gDW/h against a reference
flux of 5.95, so the capacity bound clears it. Its box ceiling is nevertheless **4.4707** in
both, because the box is `min(ub * c_j, capacity)` and the GEM's own upper bound for
`r_0226`, 1000, is clamped by the layer to `flux_scale` = 100 and then multiplied by the
availability gate: **`c_j` = 0.0447**.

So the residual 1.33x excess on ATP synthase is the AVAILABILITY GATE at initialization, not
the turnover number. And the gate is the same shape as the k_cat rule that was just fixed:
`reaction_availability` folds the gene gates up the GPR with a **softmin over the unit's
subunits**, and a softmin over 17 untrained sigmoids sits far below 0.5. The two "minimum
over 17 subunits" rules were stacked on the same reaction, and swapping one of them exposes
the other.

That gate is LEARNED, unlike the k_cat table: `availability` is a linear head with a live
gradient, so this particular residue is expected to move during training in a way an
initialization check cannot see. Whether it does is a training question, and stating that it
will would be exactly the unlabeled-hypothesis failure this note is trying to avoid.

**Which reactions remain over the ceiling.** Under `complex_max`: `r_0226` at **1.33x** and
`r_0961` at **3.06x**, since no aggregation rule reaches a single-gene unit. Under
`sigma_plus_2`: `r_0226` alone at **1.33x**, with `r_0439` (0.26x of its ceiling) and
`r_0961` (0.11x) both comfortably held. The clip cost falls from 0.3159 at the shipped table
to **0.0506** at `sigma_plus_2` and 0.0612 at `complex_max`.

**The reachability LP at the best rules** confirms the box is no longer the constraint: under
`complex_max` the tightened box reaches 88 turning-over rows against the untightened box's
91, and under `sigma_plus_1` 88 against 101. The sign constraint costs 3 to 13 rows out of
about 100, and both boxes are feasible as built.

**Stopping here, as instructed.** No GPU cell is submitted: no setting met the rule, and the
route that would carry the winning capacity rule into a real cell (the `complex_aggregation`
option in `parameters.py`, written out above) is not landed. The probe reaches these tables by
swapping `layer.kcat_per_h` after construction, which a training cell cannot do.

### 2026.09.14 (follow-up 5) - the availability gate, the second minimum over subunits

The capacity grid left ATP synthase over its box ceiling at 1.33x with the enzyme capacity
already raised past the reference flux, because the ceiling is `min(ub * c_j, capacity)` and
`c_j` was 0.0447. That is the availability gate, and it is the same shape as the k_cat rule
the grid had just fixed: `reaction_availability` folds the per-gene gates up the GPR with a
**softmin over the unit's subunits**.

**The gate is biased on an intact complex, and the bias is exactly `log(n_u)/beta`.**
Evaluated with every gate at 1, the softmin returns `1 - log(n_u)/beta` rather than 1, so at
the shipped `beta` of 8 a 2-subunit complex reads 0.913 and a 17-subunit complex **0.646** on
a wild type where nothing is deleted. It charges a complex for having subunits.

**`availability_fold='normalized_softmin'`** divides the fold by that same all-ones value.
Default `'softmin'` is bit-identical. The alternative was considered and rejected on a
measurement: SHIFTING by `log(n_u)/beta`, which makes the fold a log-mean-exp, also returns 1
on the intact complex and **destroys the deletion**, returning 0.354 for one gate at 0 among
sixteen at 1. Dividing is a positive per-unit constant, so it preserves the ordering and the
gradient, returns exactly 1 on the intact complex, and leaves a deleted subunit at 0 because
the raw softmin goes non-positive the moment any gate is 0 and the existing clamp catches it.
Four tests cover those properties; 121 pass. The per-unit gene count is cached as a plain
attribute rather than a registered buffer, so no already-trained arm's `state_dict` changes.

**Job 1937** (`gh_cpu_warmstart_init.slurm`, 5 min 29 s), the two best grid cells rerun with
the fixed gate:

| setting (nominal / exempt / ln c vertex) | max | p95 | rows | density | prop rel | over |
|---|---|---|---|---|---|---|
| `complex_max` | 0.2486 | 0.0806 | 97.0 | 0.2690 | 0.132 | 1.8 |
| `complex_max` + gate fix | 0.1867 | 0.0754 | 88.2 | 0.2643 | **0.040** | 1.2 |
| `sigma_plus_2` | 0.0847 | 0.0082 | 90.5 | 0.2098 | 0.109 | 0.8 |
| **`sigma_plus_2` + gate fix** | **0.0306** | **0.0019** | **81.8** | 0.1774 | **0.018** | **0.2** |
| target | 0.00118 | 0.00038 | 82.0 | 0.21 | 0.008 | 0 |

**Still no winner.** `sigma_plus_2` with the gate fix clears the p95 half at **0.0019**, sits
at **81.8 active rows against the target's 82**, and proposes at **0.018** against the
target's 0.008, and it misses the maximum by **3.1x** (0.0306 against 1e-2). Its density,
0.1774, now undershoots the target's 0.21 rather than overshooting it.

**The three watched reactions, and the gate fix is what held two of them.**

| cell | `r_0226` | `r_0439` | `r_0961` |
|---|---|---|---|
| `complex_max` + gate fix | capacity 20.22, ceiling **6.92**, excess 0.86, HELD | HELD (0.185) | capacity 0.322, ceiling 0.322, excess **2.44, NOT held** |
| `sigma_plus_2` + gate fix | capacity 29.99, ceiling **6.92**, excess 0.86, HELD | HELD (0.185) | capacity 9.29, excess 0.085, HELD |

The gate fix raises ATP synthase's ceiling from 4.4707 to **6.9222**, a factor of 1.548,
which is exactly `1/(1 - log(17)/8) = 1.548`: the whole of the residual 1.33x excess was the
subunit-count bias, and removing it holds the reaction. The clip cost falls to **0.0073**
under `sigma_plus_2` and **0.0190** under `complex_max`, from 0.3159 at the shipped table.

**Answering the question that was asked: yes, `r_0961` blocks under `complex_max` and not
under `sigma_plus_2`.** No aggregation rule reaches it, because it is a single-gene unit and
min, max and median over one subunit are the same number. Only the sigma widening moves it,
from a ceiling of 0.322 to 9.29 against a reference flux of 0.786. So the two rules are not
interchangeable here: the widening is the one that covers all three reactions, and it is the
sourced rule rather than a convention swap.

**The reachability LP at the gate fix.** Under `complex_max` the tightened box reaches **101**
turning-over rows against the untightened box's 102, so the sign constraint now costs one
row; under `sigma_plus_2` it reaches 66 against 102. Both are feasible as built.

**What remains, and it is no longer capacity.** With the clip cost at 0.0073 and every
watched reaction held, the residual maximum of 0.0306 is not the reference failing to fit the
box. The two things still measurably wrong at initialization are the **14 sign-forbidden
reference reactions** that survive the 47-reaction exemption, and the per-genotype spread in
`c_j` that leaves ATP synthase clipped on a minority of genotypes (`over` 0.2 of 4). Which of
those dominates the remaining 3.1x is NOT measured here, and attributing it would be exactly
the unlabeled-mechanism failure this note has been avoiding.

**Nothing written into a cell, nothing submitted.** No configuration met the rule, so the
three warm-start cells keep their current flags. The configuration that came closest, for the
record and not as an approved flag set, is: nominal abundance (`base_arm: flux_nullspace`),
`thermo_exempt_path` at the 47-reaction list, `log_c_init_path` at the vertex witness,
`availability_fold='normalized_softmin'`, and a k_cat table widened by 2 sigma from the
committed ladder. The last of those has no route into a training cell until `parameters.py`
gains the option written out above, and the k_cat widening would need its own option there
(`kcat_ceiling_sigma_k`, consuming the same `KCAT_SIGMA_LADDER`) since the probe applies it
by overwriting `layer.kcat_per_h` after construction.

### 2026.09.14 (follow-up 6) - attributing the residual 3.1x

Job 1938 (`gh_cpu_warmstart_init.slurm` with `TC_INIT_ONLY=attrib`, 4 of 23 settings), the
best cell rerun with the two candidate causes removed separately and together.

| cell | max | p95 | rows | density | prop rel | forbidden | over |
|---|---|---|---|---|---|---|---|
| baseline | 0.0306 | 0.0019 | 81.8 | 0.1774 | 0.018 | 14.0 | 0.2 |
| + the 14 exempted | 0.0272 | 0.0014 | 82.0 | 0.1740 | 0.017 | **0.0** | 0.2 |
| + gate opened on the wild type | **0.0045** | **0.0004** | 81.2 | 0.1687 | **0.007** | 14.0 | **0.0** |
| + both | **0.0022** | **0.0002** | 81.8 | 0.1651 | 0.007 | 0.0 | 0.0 |
| target | 0.00118 | 0.00038 | 82.0 | 0.21 | 0.008 | 0 | 0 |

**The residual is the availability spread, not the sign set, and it is not close.** Exempting
the 14 reactions the tightening still forbids removes them completely (forbidden 14 to 0) and
moves the maximum by **1.1x**, from 0.0306 to 0.0272. Forcing the gate open moves it by
**6.8x**, from 0.0306 to 0.0045, and takes the proposal distance to 0.007 against the
target's 0.008. Together they reach **0.0022 / 0.0002**, which is the target row of
`probe_warmstart.json` to within a factor of two on both statistics.

So the geometry question that has run through this whole note is answered: with the nominal
capacity path, the exemption list, consistent concentrations, the normalized fold and a
2-sigma k_cat widening, what is left between the head and the honest rule is the spread in
the UNTRAINED availability gate across genotypes.

**Two things this does NOT license, and both matter.**

The gate-open cells are a **probe intervention, not a configuration**. They zero the
availability head's weight and push its bias to +20 so every gate reads 1, which answers what
the box would be if the gate were open and is not something a training cell can adopt: the
gate is the layer's dosage, allele and over-expression channel, and pinning it to 1 would
delete that channel rather than train it. The probe's own win rule flags those two rows as
winners because it compares numbers and does not know what produced them. **They are not
submittable and no cell was written from them.**

And the gate is LEARNED, with a live gradient through `availability`. Whether training moves
it toward the open end, and therefore whether the 6.8x is a real obstacle or an
initialization transient, is **not measured here**. Saying it will resolve itself would be
precisely the unlabeled hypothesis this note keeps refusing to make.

**The augmented exemption list is written and labeled as the weaker claim it is.**
`results/thermo_exempt_candidates_augmented.json` carries the base 47 plus the 14, with the
base file's sha256, how the 14 were found, and the sentence that keeps the two apart: the
base list is an LP result, no concentration in the window makes those reactions downhill; the
added 14 are forbidden at ONE concentration vector under ONE capacity rule and one
availability fold, so the sets must never be quoted as one. Given the measurement above they
buy 1.1x, which is an argument for leaving them out rather than in.

**Density is now the statistic drifting.** Every attribution cell sits between 0.165 and
0.177 against the reference's 0.107 and the target row's 0.21, so the arm now approaches the
target from the sparse side. Pre-registered read (b) reads in both directions.

### 2026.09.14 (follow-up 7) - the prepared cell, and what it is not

The three warm-start cells now carry the full flag set. **None is submitted**, and the
header of each says why in the same words used here: at initialization the configuration
reads `bal_act_max` **0.0306** against a rule of 1e-2, so it is **3.1x short of the maximum
half**, while already meeting the p95 half at 0.0019 and sitting at 81.8 active rows against
a target of 82. The cell is an experiment on whether the PRICED balance term closes that
remaining factor now that the geometry admits a balanced flux at all. It is not a
pre-approved pass, and nothing in the init check licenses calling it one.

**The flag set, and the measurement behind each flag.**

| flag | value | what measured it |
|---|---|---|
| `--arm` | `flux_warmstart_boxforce_nominal` | new arm file: `base_arm: flux_nullspace` (the nominal E_bar capacity path) plus the channel and the frozen row set |
| `--flux-thermo-exempt-path` | the 47-reaction list | L1 slack 2,641.5 kJ to 0.000 over 378 constraints (job 1906) |
| `--flux-log-c-init-path` | the vertex witness | forbidden reference reactions 124 to 14; box feasible as built (job 1929) |
| `--flux-availability-fold` | `normalized_softmin` | ATP synthase ceiling 4.4707 to 6.9222, exactly the predicted 1.548 (job 1937) |
| `--flux-kcat-ceiling-sigma-k` | 2 | the only rule covering all three over-ceiling reactions; `r_0961` is a single-gene unit no aggregation rule reaches |
| `--flux-balance-active-weight` | 0.5, frozen set | the term under test; frozen because job 1795 left the dynamic set by killing rows |
| `--flux-parsimony-weight` | 1e-3 | parsimony has never been applied: it sits inside a ramp held at 0.0 |
| `--flux-density-weight` | 0.0 | the density term pushed away from the reference, which is 83 rows and 0.107 |
| `--constraint-scale` | 0.0, 200 epochs | the recipe the control arms (jobs 1767/1770/1771) ran |

The pre-registered read and the kill rule are in each cell's header verbatim from the
review, with the job-1937 initialization values as the starting point. Read (b) has a sign
worth noting: `flux_active_frac` starts at **0.1774**, BELOW the target of 0.21, so this
arm can fail that read by becoming denser, which is the opposite of every wave-8 failure.

**Four trainer flags do not exist yet.** `--flux-parsimony-weight` landed at 03:50. The
other four are one plumbing site each in `train_bx.py`, following the pattern that flag
already established: a `BxCell` field, a `build_cell` parameter and its pass-through, an
assignment onto the built layer, and an argparse entry with its call site.

```python
# BxCell, beside flux_parsimony_weight
    flux_thermo_exempt_path: str | None = None
    flux_log_c_init_path: str | None = None
    flux_availability_fold: str = "softmin"
    flux_kcat_ceiling_sigma_k: float = 0.0
```

```python
# where the built layer is configured, beside flux_layer.config.parsimony_outside_ramp_weight
        flux_layer.config.availability_fold = cell.flux_availability_fold
        # ORDER MATTERS for these two: apply_log_c_init reads conc_lower_m / conc_upper_m,
        # which are assigned a few lines above, and it refuses a window that disagrees with
        # the file it was solved in.
        if cell.flux_thermo_exempt_path is not None:
            flux_layer.apply_thermo_exempt_list(cell.flux_thermo_exempt_path)
        if cell.flux_log_c_init_path is not None:
            flux_layer.apply_log_c_init(cell.flux_log_c_init_path)
```

The fifth, `--flux-kcat-ceiling-sigma-k`, cannot be assigned onto a built layer: the k_cat
table is resolved inside 026's `build_flux_layer`, so the widening has to happen in
`parameters.py` beside the `complex_aggregation` option already written out above, reading
the same `KCAT_SIGMA_LADDER`. **The keyword name in the cells is a PLACEHOLDER** and is
marked as one in each header; it must be confirmed against whatever the kinetics owner
lands before any of the three cells runs.

### 2026.09.14 (follow-up 8) - the real build path, and the cell is submitted

**The k_cat rule reaches the cell through `base_arm`, not through a trainer flag, and that
is a correction to the plan.** `ArmConfig` carries `extra="forbid"` and four fields, so an
027 arm file cannot name `kcat_ceiling_sigma_k` directly: adding it raises a validation
error, checked rather than assumed. The registry entry is the only channel an 027 cell has,
and it is the better one anyway, since `base_arm` is serialized into the result file with
the cell. `flux_nullspace_kcat_sigma2` is `flux_nullspace` with that one field set to 2.0,
verified by a key-diff against its twin: the two differ in `kcat_ceiling_sigma_k` alone, and
against `flux_projection_anchored_k20_boxend` in that field plus `parameterization` and
`use_abundance_head`. The placeholder trainer flag is gone from all three cells.

`build_flux_layer` now forwards two arm keys, both defaulted so every existing arm resolves a
bit-identical table: `complex_aggregation` to `load_predicted_unit_table` (k_cat only, K_M
keeps its own maximum) and to `resolve_kcat_table`, and `kcat_ceiling_sigma_k` to the latter.
Both are recorded in `flux_meta`.

**`PYTHONHASHSEED=0` is exported by all three cells.** The kinetics owner measured that the
resolved table depends on it, 2 of 3,728 values moving by up to 0.313 decades, so a cell
that omitted it would not be reproducible from its own record.

**Verification through the path the cell runs (job 1945, 27 s).**
`probe_cell_build_verify.py` builds the layer the way the trainer does, from the arm file's
registry entry through `parameters.resolve_kcat_table`, rather than by overwriting
`layer.kcat_per_h` the way the init probe does. It refuses to run without
`PYTHONHASHSEED=0`.

| statistic | measured, real build | job 1937 | relative delta |
|---|---|---|---|
| `balance_active_max` | 0.030578 | 0.0306 | **0.0007** |
| `balance_active_p95` | 0.0018757 | 0.0019 | 0.0128 |
| `balance_active_rows` | 81.75 | 81.8 | 0.0006 |
| `flux_active_frac` | 0.17738 | 0.1774 | 0.0001 |
| `proposal_distance_rel` | 0.017725 | 0.018 | 0.0153 |

All three watched reactions are held through the real path: `r_0226` capacity 29.99 with a
box ceiling of 6.9222 against a reference flux of 5.95 (excess 0.860), `r_0439` 0.185,
`r_0961` capacity 9.2888, excess 0.085. The two code paths agree to well inside the 5 percent
tolerance, so the cell will train on the table it was designed on.

**Submitted: job 1946**, seed 2700, through `freeze_and_submit.sh` (snapshot
`20260914-194105-gh_wave9_warmstart_e200_s2700`), preflight green on all 37 call-site
keywords. Seeds 2701 and 2702 stay unsubmitted until seed 2700 shows its epoch-20 read.

Restating what this cell is, because a submitted job is the easiest thing in this note to
misread: at initialization it meets the p95 half of the rule and is **3.1x short of the
maximum half**, and it is an experiment on whether the priced balance term closes that factor
now that the geometry admits a balanced flux. It is not a pre-approved pass, and the kill
rule in its header is the review's.

### 2026.09.14 (follow-up 9) - job 1946 failed, and the reference has to carry the genotype

**Job 1946, read at epoch 143 of 200 (W&B run `6s8r3aov`), never left the betaxanthin mean
plateau.** Validation prediction spread 0.0004 to 0.012, validation Pearson flat at -0.052
after epoch 5, training data loss flat at 0.527 from epoch 0. What moved instead was the
residual scale: the learned alpha grew from 0.0099 to 0.087 (`feas_warm_proposal_rel` 0.0087
to 0.116), and as it grew the training-batch active-row balance max went from about 1.0 to
2.0 by epoch 45 with `c_balance_active` rising 0.001 to 0.17, p95 swinging 0.005 to 0.9 after
epoch 110, and density 0.175 to 0.25. The box held at zero violation throughout and `g_diss`
stayed between 1,314 and 1,568 J/gDW/h, under a 3,700 limit it was never penalized against.

**A second number from that read matters as much.** The training-batch active-row max was
already about 1.0 at epoch 0 while the init probe read 0.03. The init probe forwarded four
WILD TYPES, so deletion genotypes fail the rule from the start and no wild-type-only check
could see it. Every init number in this note before today carries that caveat.

**The diagnosis under test, and it is a hypothesis.** With one wild-type reference the
proposal is `v_ref + alpha N z`, so the flux is very nearly the same vector for every
genotype, the head reading it is close to a constant predictor, and the only channel the data
term has for genotype information is alpha. Growing alpha is what it did, and feasibility was
the price. The reference has to carry the genotype, or the residual does, and only one of
those is cheap.

**What the dataset's genotypes are, and it settles a design choice.** Measured by the
builder on the 027 dataset: **4,928 distinct deleted-gene sets over 4,930 records**, every
record perturbed, and the sizes are **4,190 quadruple deletions, 479 triples, 213 doubles,
48 singles**. So a table keyed on single genes with a fallback for everything else would have
put about 99 percent of the dataset on the fallback and changed nothing. The table is keyed
on the sorted gene SET and each set is solved as itself, because a double deletion's flux is
not a composition of two single-deletion fluxes.

**What landed in the layer.** `warm_start_reference_table_path` plus `ReferenceFluxTable`,
`genotype_key`, `read_reference_flux_table` and `FluxLayer.genotype_reference`, which looks
each batch row's reference up by its gene-set key; a table built for another GEM is refused
on a reaction-list mismatch rather than applied by position. The fallback is the wild-type
reference and `coverage_report` carries the fallback RATE, so a table that misses the dataset
would say so instead of training as the single-reference head did.
`warm_start_alpha_learnable` freezes alpha with no gradient while keeping it in the state
dict, so the two arms' checkpoints interchange. Off is bit-identical: one reference,
repeated, exactly as job 1946 ran it. Five new tests, 126 pass.

**The builder** (`probe_reference_flux_table.py`) solves pFBA per gene set with the genes
knocked out through the GPR by `cobra.manipulation.knock_out_model_genes`, inside a
`with model:` scope so each genotype starts from the shipped bounds. A knockout whose growth
LP is infeasible is recorded as the ZERO flux rather than dropped: a lethal genotype's
reference is a cell that does not grow. The table is cached under
`$DATA_ROOT/data/torchcell/reference_flux/` with the GEM version, the medium statement, the
growth fraction, the flux sha256 and the dataset's own record counts.

### 2026.09.14 (follow-up 10) - the flux layer has been deleting the betaxanthin pathway

**The defect, stated plainly.** Every Cachera betaxanthin strain carries a four-gene
betaxanthin cassette as GENE ADDITIONS: the heterologous `CYP76AD1` and `DOD`, plus the
feedback-resistant native alleles **ARO4^K229L (`YBR249C`)** and **ARO7^G141S (`YPR060C`)**
integrated ectopically at locus XII-5, on top of the intact native loci
(`torchcell/datasets/scerevisiae/cachera2023.py`, lines 67 to 113, `GeneAdditionPerturbation`
with `variant=`). `GraphProcessor._process_gene_info`
(`torchcell/data/graph_processor.py`, line 158) builds `perturbed_names` from
`systematic_gene_name` over EVERY perturbation regardless of type, so those genes are removed
from the cell graph and arrive at the model as perturbation indices. The flux layer's
`gene_availability` then hard-zeroes them.

**So the layer has been holding ARO4 and ARO7 at availability 0 in every betaxanthin
genotype since wave 5.** `YBR249C` carries DAHP synthase (`r_0042`, `r_0020`) and `YPR060C`
chorismate mutase (`r_0278`). Those strains overproduce betaxanthin THROUGH the shikimate
pathway, and the precursor readout reads that pathway. The layer was deleting the thing it
was measuring.

**Blast radius, as measured rather than estimated.** The perturbation-type census over the
dataset's own records: **31,894 `gene_addition` against 13,753 `kanmx_deletion`**, with the
non-deletion genes and their record counts

| gene | records | cassette |
|---|---|---|
| `CYP76AD1`, `DOD`, `YBR249C` (ARO4), `YPR060C` (ARO7) | 4,669 each | Cachera betaxanthin |
| `crtYB`, `crtI`, `YPL069C` (BTS1) | 4,406 each | Ozaydin beta-carotene |

Every betaxanthin flux cell since wave 5 gated ARO4 and ARO7 off, and every beta-carotene
cell gated `BTS1` off, which is an essential gene. The pooled backbone removed those nodes
from every Cachera cell graph, so it is a constant across strains there rather than a
per-strain error, but in the flux layer it is a per-strain false knockout.

**A correction to my own numbers from three sections ago.** I reported the dataset as "4,190
quadruple deletions, 479 triples, 213 doubles, 48 singles". That was the count of
PERTURBATIONS, not deletions. Read through the records with the deletion rule, the dataset is
**13,753 records, every one of them a SINGLE deletion** (`deletion_sizes: {'1': 13753}`) over
**4,930 distinct deleted genes**. The "quadruples" were one deletion plus three cassette
additions.

**And it explains the 0.0 growth better than the medium does.** The lethal knockouts I built
were deleting ARO4 plus ARO7 plus, in the beta-carotene records, BTS1, which is essential.
The medium finding stands on its own and YPD is right for Cachera regardless, but the false
knockout is the larger term, and the minimal-medium table is being built to separate them.

**What changed.** The builder now keys on the DELETION-only gene set, reusing
`DeletionKeyedGenotypeAggregator`'s rule (`perturbation_type.endswith(DELETION_TYPE_SUFFIX)`)
rather than restating it, and reads it from the raw LMDB records, since the processed graph
has already lost the type. It also writes the engineered background genes into the table's
own provenance, and the layer reads that list from the artifact and drops those genes from
the lookup key. Without it the key would name a cassette the strain HAS, miss every row, and
fall back to the wild type on 100 percent of the dataset. One new test covers exactly that;
127 pass.

**This is a stopgap and it is marked as one.** The right fix is the `deletion_indices` field
the targets builder is adding, which will carry the deletion-only set to the layer for the
availability gate as well as for the reference lookup. The background-gene list fixes the
REFERENCE key only. **The availability gate is still zeroing ARO4, ARO7 and BTS1**, so every
number measured before `deletion_indices` lands, including the ones below, is measured on a
layer that still believes the cassette genes are deleted.

**One incidental defect found and worked around.** `Neo4jCellDataset.len()` opens its own
LMDB handle and closes it, so a handle opened before that call is closed underneath the
caller and the next read fails on a `None` env. The builder takes the length first, then
opens its handle.

### CPU jobs this session

| job | cell | state |
|---|---|---|
| 1900 | `gh_cpu_physics_probes.slurm` | COMPLETED, 89 s |
| 1904 | `gh_cpu_proposal_distance.slurm` | COMPLETED, 30 s |
| 1906 | `gh_cpu_thermo_exempt_gate.slurm` | COMPLETED |
| 1907 | `gh_cpu_transport_tfa.slurm` (union_external) | COMPLETED |
| 1920 | `gh_cpu_transport_tfa.slurm` (`TC_THERMO_ROUTE=shipped`) | COMPLETED |
| 1922 | `gh_cpu_warmstart_init.slurm` | COMPLETED, 36 s |
| 1923 | `gh_cpu_warmstart_init.slurm`, the 2x2 | COMPLETED, 93 s |
| 1924 | the same, with the two-stage reachability LP | COMPLETED, 91 s |
| 1925 | the same, with the objective scored where the split is exact | COMPLETED, 96 s |
| 1927 | `gh_cpu_thermo_exempt_gate.slurm`, rerun to emit the concentration witness | COMPLETED, 9 s |
| 1929 | `gh_cpu_warmstart_init.slurm`, seven settings with the concentration init | COMPLETED, 117 s |
| 1933 | the same, with the over-ceiling reactions named | COMPLETED, 117 s |
| 1934 | `gh_cpu_thermo_exempt_gate.slurm`, rerun to emit the CENTERED witness | COMPLETED, 7 s |
| 1935 | `gh_cpu_warmstart_init.slurm`, the 15-setting capacity grid | COMPLETED, 4 min 28 s |
| 1937 | the same, with the two availability-gate-fix cells | COMPLETED, 5 min 29 s |
| 1938 | the same, `TC_INIT_ONLY=attrib`, the four attribution cells | COMPLETED, 2 min 1 s |
| 1945 | `gh_cpu_cell_build_verify.slurm`, the real build path | COMPLETED, 27 s |

One GPU job: **1946**, `gh_wave9_warmstart_e200_s2700.slurm` via `freeze_and_submit.sh`.

All ran at `--nice=1000` on at most 8 cores and 24 GB with a time bound, behind the database
rebuild's queue.
