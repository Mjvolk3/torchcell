# Reviewer 4: statistics, power, and the shape of a provable claim

Slice: the registered statistic, its error control, collapse handling, and the minimal
14 day design. Sources read in full: `experiments/019-simb-multimodal/scripts/v13_split_readout.py`
(510 lines), `notes/experiments.019-simb-multimodal.expression-fit-review.md`,
`notes/experiments.019-simb-multimodal.scripts.split_gene_overlap_audit.md`, all eleven
readout JSONs, `scripts/igb_expr_wave5.slurm`, and the W&B `_runtime` / `pred_sd_ratio`
histories of 118 first-segment runs in `_v13`, `_v16`, `_v17`, `_v18`. Coordinator addendum
`ADDENDUM_0207.md` taken as given (collapse taxonomy, J_expr confound, pooled paired sd
0.0167, continuation state).

**Provenance caveat on every number below.** `results/v13_split_readout.json` and
`v14_proteome_readout.json` were REGENERATED at 2026-09-27 01:00:31 and 01:00:44 while I was
mid-read (v13 matched epoch changed 5034 -> 5179, v14 1308 -> 1352, and both gained the
`window`/`registered_statistic` keys they lacked). I copied all eleven JSONs to
`joint_review/snap_r4/` at 01:01:20 and every number here is computed from that snapshot.
Scripts: `joint_review/an01_variance.py`, `an02_test_and_collapse.py`, `an04_power.py`,
`an05_clustering.py`, `an06_conj_and_wb.py`, `an07_detector.py`.

## 1. The registered statistic has five defects, and three of them change a call

The registered rule (expression-fit review, "Plan for the rounds after the queued 1 to 3"):
four split seeds, three init seeds, 1,200 epochs, decision statistic the mean of
`val/<pheno>/pearson_per_feature` over epochs 1,000 to 1,200, paired t on the 12
(split, seed) differences, adopt only if the paired mean exceeds +0.02 with a
Bonferroni-corrected CI excluding 0, at least 3 of 4 partitions positive, and the test-side
sign agrees.

**(a) The code does not implement the documented window.** The docstring at
`v13_split_readout.py:182-183` says the window "defaults to the last sixth of the round's
budget"; the code is `win_lo = max(1, matched - 200)` (line 264), always 200 epochs. At a
1,200 epoch budget these coincide; at v16's 500 they do not (last sixth is 83 epochs, the
code used 299 to 499, 40% of the run). Not a bug that changed a number yet, but the
registered statistic is whatever the code does, and the two disagree in writing.

**(b) The collapse rule is applied to the wrong column, and that flipped v17.** `PLATEAU
= 0.05` is tested against `roll_max_matched` (lines 311, 390), never against the registered
`window_mean`. Consequence, measured on the snapshot: v17's `plateau_runs` list is EMPTY and
`prop2_minus_ref_excluding_plateau` is byte-identical to the all-pairs block (n = 12 both).
`sylu3gsw` has `roll_max_matched` 0.0618 (above 0.05) and `window_mean` 0.0230 (below), so the
declared detector does not see it. Applying the same 0.05 to the registered column:

| v17 L_prop2 minus L_ref, fixed window | n | mean | sd | t | p | positive |
|---|---|---|---|---|---|---|
| as the script reports (all 12) | 12 | +0.0122 | 0.0330 | +1.28 | 0.226 | 9/12 |
| 0.05 applied to `window_mean` | 11 | +0.0204 | 0.0178 | +3.79 | 0.004 | 9/11 |

The brief's "+0.020 on 11 pairs, +0.012 on 12" is therefore reproduced by the coherent rule,
not by the committed one. One line decides whether v17 reads null or p = 0.004.

**(c) One epoch window cannot serve both heads.** Measured `roll_max` epoch, median (range):

| round | head | arm | median peak epoch | range | registered window |
|---|---|---|---|---|---|
| v14 | proteome | P_ref | 140 | 74-418 | 1152-1352 |
| v14 | proteome | P_concat | 174 | 66-1358 | 1152-1352 |
| v16 | proteome | J_ref | 191 | 93-463 | 299-499 |
| v16 | proteome | J_joint | 259 | 186-497 | 299-499 |
| v16 | expression | J_joint | 494 | 486-529 | 299-499 |
| v17 | expression | L_ref | 971 | 804-1182 | 999-1199 |

The proteome head peaks by epoch 150 to 260 and then decays; the expression head peaks at
the end of whatever budget it is given. `roll_max_matched` minus `window_mean` is +0.0179 on
v16's proteome runs and +0.0204 on v14's, against +0.0095 to +0.0099 on v17/v18 expression,
i.e. the proteome numbers are scored roughly 1,000 epochs (v14) or 200 epochs (v16) past
their own plateau. 8 of 12 v16 proteome runs end below 0.79 of their own maximum
(`dnjggnwl` 0.594, `nrjiq4az` 0.704). A single fixed window scores one head in its overfit
regime and the other before it has arrived, which is exactly the regime where an arm that
merely trains FASTER looks better.

**(d) The paired t over 12 (split, seed) cells treats seeds as independent replicates of the
arm effect, and they are not.** One-way random-effects decomposition of the 12 fixed-window
differences into an arm-by-partition term and a seed-pair term (`an04_power.py` part 1):

| contrast | total sd | sd_partition | sd_seed | ICC | SE ratio (naive/honest) |
|---|---|---|---|---|---|
| L_self minus L_ref (v17) | 0.0230 | 0.0171 | 0.0170 | 0.50 | 1.48 |
| Y_ctx minus Y_ref (v18) | 0.0154 | 0.0141 | 0.0087 | 0.72 | 1.68 |
| Y_k0 minus Y_ref (v18) | 0.0173 | 0.0043 | 0.0168 | 0.06 | 1.07 |
| L_prop2 minus L_ref (v17) | 0.0330 | 0.000 | 0.0352 | 0.00 | 0.80 |
| J_joint minus J_ref (v16 prot) | 0.0133 | 0.0109 (P=3) | -- | -- | 1.16 |
| P_concat minus P_ref (v14) | 0.0152 | 0.000 | 0.0187 | 0.00 | 0.55 |

"SE ratio" is the SE from the 4 partition means divided by the naive SE from 12 cells. Where
the arm effect really does differ by partition (ICC 0.50 to 0.72) the registered test
understates its own standard error by 1.5x to 1.7x, so its p is anticonservative by about
that factor. This is not hypothetical: it is the same quantity the "3 of 4 partitions
positive" clause is groping for.

**The hard constraint that follows.** If the partition is the exchangeable unit, a
partition-level sign-flip test has minimum attainable two-sided p of `2 / 2^P`: 0.125 at
P = 4, 0.0625 at P = 5, 0.031 at P = 6. **With four partitions no partition-level
randomization test can ever reach p < 0.05, whatever the effect size.** Six partitions is the
floor. Measured sign-flip p on the existing rounds: L_self 0.500, L_prop2 0.375, Y_ctx 0.625,
Y_k0 0.125, J_joint-prot 0.500, J_joint-expr 0.250. Every one of them is at or near its own
resolution limit.

**(e) "mean exceeds +0.02 AND the CI excludes 0" is a hybrid rule whose size is not the
stated alpha.** It is an intersection of a point-estimate threshold and a test; at 12 pairs
and sd 0.0167 the CI clause binds first (MDE 0.0148), so the +0.02 clause is doing the work
of an effect-size floor, silently. Say so explicitly, or replace the pair with a one-sided
test against the +0.02 boundary.

## 2. Test versus validation agreement: the ordering is partition-level noise

`test_at_best_val` exists only in v16 (12 of 12, 8 finite), v17 (36 of 36) and v18 (34 of 36);
v13, v14 and v15 have none (v13 died on the wall before `trainer.test`, per the review).

| round | raw r(val window, test) | within-partition r | per-partition test minus val |
|---|---|---|---|
| v17 | +0.078 | **+0.683** | s0 -0.064, s1 +0.039, s2 +0.032, s3 +0.021 |
| v18 | -0.236 | +0.317 | s0 -0.056, s1 +0.019, s2 +0.036, s3 +0.010 |
| v16 expr | +0.933 | +0.960 | s1 +0.010, s2 +0.033 |
| v16 prot | +0.307 | -0.296 | s1 -0.014, s2 +0.002 |

The raw correlation is near zero (v17 +0.078, Spearman -0.222) purely because split 0 has the
highest validation draw and the lowest test draw, which is the review's own "split 0 is the
luckiest 155-strain validation draw" finding. Within a partition the agreement is real but
loose (r +0.32 to +0.68). Paired test-side contrasts (`an02` part B):

| contrast | val window | test | sign agreement |
|---|---|---|---|
| L_self minus L_ref | +0.0095 (t +1.43) | +0.0043 (t +0.79) | 7/12 |
| L_prop2 minus L_ref | +0.0122 (t +1.28) | +0.0027 (t +0.23) | 6/12 |
| Y_k0 minus Y_ref | -0.0122 (t -2.44) | -0.0150 (t -2.53) | 8/11 |
| Y_ctx minus Y_ref | +0.0050 (t +1.12) | +0.0107 (t +1.30) | 6/11 |
| J_joint minus J_ref (prot) | -0.0076 (t -1.41) | +0.0067 (t +0.55) | 1/4 |
| J_joint minus J_expr (expr) | +0.0605 (t +3.43) | +0.0646 (t +1.69) | 4/4 |

**Per-pair sign agreement is 6/12 to 8/12, i.e. at or barely above chance.** So "the test
sign agrees" cannot be a per-pair clause; it is only meaningful on the MEAN, where it is one
binary draw and adds essentially no error control. Treat the test read as a confirmatory
direction check on the aggregate, state that it is one draw, and do not report per-pair test
agreement as support.

## 3. The claim: a conjunction needs a power budget, not a multiplicity correction

"Joint >= single-head on the proteome head AND on the expression head" is an
intersection-union test: H0 is the union of the two nulls (at least one head fails), H1 is
their intersection. Rejecting only when BOTH one-sided tests reject has size at most alpha
with **no multiplicity correction** (Berger 1982); the Bonferroni in the current template is
aimed at the opposite structure, a union-intersection over two candidate ARMS, where it is
correct. So:

- **Two heads, conjunction: no correction.** One-sided alpha 0.05 per head.
- **Two candidate arms, either may win: Bonferroni.** alpha 0.025 per arm, as registered.
- Do not apply both, which is what a literal reading of the current rule does and which
  costs about 3 extra pairs at sd 0.0167 and delta +0.02 (n 6 -> 8).

The price of the conjunction is power. Simulation at the measured cross-head correlation
(rho = +0.18, from the 6 shared v16 cells; sensitivity run at rho 0, 0.5, 0.8 changes the
conjunction power by less than 0.002 because the proteome arm is far from its boundary):

| n pairs | true prot | true expr | power prot | power expr | power BOTH |
|---|---|---|---|---|---|
| 12 | +0.020 | +0.020 | 0.999 | 0.880 | 0.880 |
| 12 | +0.010 | +0.020 | 0.785 | 0.880 | 0.701 |
| 12 | +0.010 | +0.010 | 0.784 | 0.409 | 0.338 |
| 12 | 0.000 | +0.020 | 0.050 | 0.879 | 0.047 |
| 24 | +0.010 | +0.010 | 0.973 | 0.662 | 0.649 |
| 36 | +0.010 | +0.010 | 0.997 | 0.819 | 0.817 |

Read the fourth row. **If the true proteome effect is exactly zero, a superiority conjunction
has 5% power no matter how many pairs are run.** Joint training that is genuinely neutral on
the proteome and helpful on expression can never be declared "helps on both" under a
superiority-and-superiority rule. That is the decisive argument for the asymmetric claim.

### The claim to pre-register instead

**Co-primary intersection-union test, one-sided alpha 0.05 on each, no correction:**

1. **Expression head, superiority.** H0: delta_e <= 0 against H1: delta_e > 0.
2. **Proteome head, non-inferiority at margin m.** H0: delta_p <= -m against H1: delta_p > -m.

Declare "joint training helps" only if both reject. Assign the superiority side to the
expression head because that is where joint training has a measured positive point estimate
and where the single-head arm is the weaker baseline (expression ceiling 0.21 against the
ProtT5 ridge 0.13, versus proteome 0.11 against 0.07); assign non-inferiority to the proteome
head because its measured effect is -0.008 (fixed window, t -1.41, 3/6 positive), which no
feasible design will turn into a superiority win.

**Margin choice, and it must be justified before launch.** Candidate anchors, all measured:
the strain-bootstrap sd of the metric on 155 validation strains is 0.026 (test 0.018, review
report 10); the between-partition sd of the proteome score is 0.0085 to 0.0162; the paired sd
of the proteome contrast is 0.0133. A margin below the instrument's own bootstrap sd cannot
be defended as "no loss", and a margin above the between-partition spread gives away more
than the quantity being claimed. **m = 0.015 is the defensible middle**: above the
between-partition sd (0.0162 in v14 on `roll_max`, 0.0085 to 0.0095 on the window statistic),
below the single-partition bootstrap sd. Sample size, one-sided alpha 0.05, 80% power:

| paired sd | m = 0.010 | m = 0.015 | m = 0.020 | m = 0.025 |
|---|---|---|---|---|
| 0.0133 (v16 proteome), true effect 0 | 13 | 7 | 5 | 4 |
| 0.0133, true effect -0.005 | 46 | 13 | 7 | 5 |
| 0.0133, true effect -0.008 (the v16 point estimate) | 275 | 24 | 10 | 6 |
| 0.0167 (pooled), true effect 0 | 19 | 10 | 6 | 5 |
| 0.0167, true effect -0.008 | 433 | 37 | 14 | 8 |
| 0.0241 (v17 expression sd), true effect 0 | 38 | 18 | 11 | 8 |

**The load-bearing row is the third.** If joint training really costs 0.008 on the proteome,
non-inferiority at a 0.010 margin needs 275 pairs and is unreachable; at a 0.015 margin it
needs 24, which fits only the maximal design below; at 0.020 it needs 10. So the margin has
to be fixed BEFORE the round, and the PI has to accept 0.015 or 0.020 as "no meaningful
loss". Choosing the margin after seeing the point estimate is the single largest source of
optimism available in this design and it must be closed by pre-registration.

## 4. Collapse: intent-to-treat, per-protocol, and a detector that actually fires

Rates as given: 4/6 v16 expression-only, 2/36 v17, 1/24 v13, 0/36 v18. Jeffreys 95% CIs:
0.667 [0.286, 0.923]; 0.056 [0.012, 0.166]; 0.042 [0.005, 0.179]; 0.000 [0.000, 0.067].
Pooled 7/102 = 0.069 [0.031, 0.130]. Expression-only against every other arm pooled (4/6
against 3/90) is Fisher p = 1.5e-4, odds ratio 58. Collapse is not a background rate, it is
an ARM property, which is why an intent-to-treat analysis that keeps collapsed runs measures
the arm's optimization fragility and NOT its representational quality, and a per-protocol
analysis that drops them measures the second while hiding the first.

What each choice does to the v16 expression headline (fixed window):

| analysis | n | mean | sd | t | p |
|---|---|---|---|---|---|
| ITT, all pairs | 6 | +0.0605 | 0.0433 | +3.43 | 0.019 |
| PP, script plateau rule | 2 | +0.0069 | 0.0121 | +0.81 | 0.568 |

So the published +0.060 is an artifact of comparing against four references that never
trained, and the per-protocol read is n = 2. Neither is a measurement of joint synergy.

### Pre-declared detector, calibrated on 118 first-segment runs

Two rules, both on quantities already logged (`an07_detector.py`):

- **Launch gate at epoch 200.** Fail if `max(val/<head>/pred_sd_ratio)` over epochs <= 200 is
  <= 0.05. Measured separation: the 7 declared failures have 0.0009, 0.0021, 0.0008, 0.0193,
  0.0379, 0.0622, 0.2247; the 110 healthy runs have minimum 0.0658 and 5th percentile 0.1006.
  At 0.05 the gate has **0 false positives in 110** and catches 5 of the 7 failures plus
  `tnv3dbe4`, the late launch, which is correctly excluded as not comparable.
- **Decay gate at the end, expression head only.** Fail if the fixed-window mean is below
  0.70 of that run's own `roll_max`. Measured: declared failures 0.000 to 0.270 of own max;
  healthy expression runs 0.745 at the 5th percentile. The two rules together catch **7 of 7**
  declared failures and flag 3 others: `hzca60i6` (0.644) and `vbwd3w6f` (0.518), which are
  the only two J_expr runs that survived at all, and `dnjggnwl` (0.594), a PROTEOME run.
  **Every one of the six J_expr runs fails the pre-declared detector**, so the v16 expression
  contrast has zero usable reference runs and is a non-measurement, not a null.
- **Do NOT apply the decay gate to the proteome head.** 8 of 12 healthy v16 proteome runs sit
  below 0.79 of their own max because the proteome head genuinely decays after epoch 250.
  Its window must be placed at its own measured plateau instead (section 1c).

### Pre-registered handling and restart budget

1. Every run is scored ITT at the registered window and ALSO under the detector. Both
   numbers are reported. The primary analysis is **per-protocol with restart**: a run that
   fails the launch gate at epoch 200 is relaunched with a new init seed inside the same
   (partition, arm) cell, and the relaunch is the run that enters the analysis. This is a
   pre-declared protocol deviation, not a post-hoc exclusion, because the gate fires at epoch
   200 before any outcome is visible.
2. A run that fails the DECAY gate cannot be relaunched honestly (the failure is only visible
   at the end), so it is handled by dropping the whole PAIR and reporting the drop, with the
   ITT number beside it. If more than 2 of 12 pairs are dropped this way the round is declared
   underpowered rather than read.
3. Restart budget. At the measured non-J_expr rate 3/108 = 0.028 per run, a 36 run design has
   E[failures] 1.0 and P(at least one) 0.64; at the pooled 0.069, E = 2.5 and P = 0.92. To end
   with 12 clean pairs at per-pair failure rate q the expected launches are 12/(1-q): 12.4 at
   q = 0.033, 12.8 at 0.06, 13.3 at 0.10. **Reserve 2 spare card-slots (about 4 card-days).**
   A restart caught at epoch 200 costs 11.9 h of detection plus a full rerun of that one slot.

## 5. Remaining sources of optimism, ranked by size

1. **Margin or statistic chosen after the fact** (section 3). Unbounded. Close by
   pre-registration in the config header before the first sbatch.
2. **`roll_max` instead of the window.** +0.0095 to +0.0225 per run, measured per round above.
   It cancels in an epoch-matched pair but not in any absolute claim, and it is still the
   column the `PLATEAU` rule reads.
3. **Checkpoint selection for the test read.** `test_at_best_val` is scored at the checkpoint
   selected on the same validation split whose maximum is an upward-biased order statistic
   over about 1,200 epochs. The review records that minimum-loss checkpoints do not exist for
   v13 and that val loss bottoms 500 to 900 epochs before the Pearson peak. Pre-declare ONE
   selection rule (best-metric, loss-minimum, or fixed epoch) per head, save all three, and
   report the other two as sensitivity. Per-head selection is required anyway given 1c, and it
   must then be declared as per-head, not slipped in.
4. **The gene-disjoint subset is not logged.** `pearson_per_feature` is computed on all 155
   held-out rows, 17 to 21 of which share a deleted gene with train
   (`split_gene_overlap_audit`), and those score 0.398 against 0.178 for the 134 gene-disjoint
   rows on v13 seed 0. A paired contrast is only biased by this if an arm helps differentially
   on the shared subset, which is precisely what a graph-routing or a cross-modality arm might
   do. Logging `pearson_per_feature` on the gene-disjoint rows is a metrics-only change with
   zero GPU cost and it should land before the round, not after.
5. **Per-instance versus per-feature.** `pearson_per_instance` is logged as a diagnostic and
   `nmse` beside it; every v16/v17/v18 run has `nmse` at or above 1.0 at its peak (1.00 to
   1.19), meaning the model loses to the per-gene mean in squared error while winning in
   correlation. A claim that says only "Pearson improved" while NMSE stays above 1 is true and
   incomplete. Report both, and pre-declare that NMSE is descriptive, not co-primary, or the
   conjunction grows a third arm and loses power again.
6. **The J_expr confound** (addendum item 2). The current expression-only control differs from
   the joint arm in store iteration (117 against 40 steps per epoch) and in head type (masked
   `per_gene` against unmasked `per_gene_aux`). Any comparison built on it measures those two
   things as well as joint training.

## Proposed experiments

Throughput is measured, not assumed: W&B `_runtime` divided by epochs at 3 runs per card.
v16 J_joint 3.56 min/epoch (29.7 h at 500, 35.6 h at 600, 47.5 h at 800, 71.2 h at 1,200);
J_ref 3.14; J_expr 2.91; v17 expression arms 1.18 to 1.51; v18 on cabbi 0.76 to 1.25.
Budget: `gpu` 3 cards x 14 d = 42 card-days (shared with other users, the addendum records
tasks 2 to 5 of 2410399 pending behind them), cabbi 1 to 3 cards x 14 d = 14 to 42 card-days
with every job <= 5 d. A 12 cell x 3 arm task of 800 epochs is 47.5 h = 2.0 d, inside the
cabbi limit; 1,200 epochs is 71.2 h = 3.0 d, also inside it.

Allocation rule derived in section 1d: with the arm effect clustered by partition, sd of the
contrast mean is `sqrt(sd_a^2/P + sd_e^2/(P*S))`, so at a fixed run count MORE PARTITIONS
always beats more seeds. At L_self's components (sd_a 0.0171, sd_e 0.0170), 36 runs spent as
12 partitions x 1 seed gives SE 0.0070 against 0.0098 for 4 x 3, a 1.4x tighter estimate for
the same GPU-days, and it raises the sign-flip resolution from p >= 0.125 to p >= 0.0005.
One seed per partition is safe only with the epoch-200 launch gate and the restart budget.

### E1. Head-matched joint contrast, 12 partitions x 1 init seed, 800 epochs (RANK 1)

- **Arms, all three derived from ONE config by zeroing loss weights** so iteration order,
  step count, head type and reveal schedule are identical: `K_joint` (both losses),
  `K_prot` (expression loss weight 0), `K_expr` (proteome loss weight 0). This replaces
  J_ref/J_expr and removes both confounds in addendum item 2.
- **Partitions x seeds:** split seeds 0 to 11, one init seed each, on `fig3_proteome`.
  12 pairs per contrast. Three arms co-resident on one card per cell, which is what makes the
  pairing a valid denominator (launcher note in `igb_expr_wave5.slurm`).
- **Epochs:** 800. **Per-head scoring windows, pre-declared from v14 and v16 curves, not from
  this round:** proteome = mean over epochs 200 to 400; expression = mean over epochs 667 to
  800 (the last sixth). Log `pearson_per_feature` on the gene-disjoint rows and
  `pearson_per_instance`/`nmse` as declared descriptives. Save best-metric, loss-minimum and
  last checkpoints; the test read uses best-metric per head, declared in advance.
- **Wall:** 12 tasks x 47.5 h = 570 card-hours = 23.8 card-days, plus 2 spare slots for
  restarts (4 card-days) = 27.8 card-days. On 3 `gpu` cards plus 2 cabbi cards: 12 cells over
  5 cards = 3 rounds x 47.5 h = 143 h = **6.0 days**. On 3 cards alone: 9.3 days. Fits.
- **Pre-registered statistic and decision rule.** Intersection-union, one-sided alpha 0.05 per
  head, no multiplicity correction: expression superiority (H0 delta_e <= 0) AND proteome
  non-inferiority at margin **m = 0.015** (H0 delta_p <= -0.015). Both tests are paired t on
  the 12 partition-level differences of the per-head window mean. Supporting, reported always
  and never substituted for the primary: a partition-level sign-flip p (attainable minimum
  0.0005 at P = 12), the ITT analysis, the detector-flagged runs, and the test-side mean with
  its direction.
- **Resolution.** MDE at 12 pairs, 80% power, one-sided alpha 0.05: 0.0102 at the proteome
  sd 0.0133, 0.0128 at the pooled 0.0167, 0.0185 at the v17 expression sd 0.0241. Conjunction
  power 0.88 if both true effects are +0.02, 0.70 if proteome is +0.01 and expression +0.02.
  Non-inferiority power at m = 0.015 is 0.80 with n = 7 if the true proteome effect is 0 and
  n = 24 if it is -0.008, so **E1 can declare non-inferiority only if the true proteome cost
  is smaller than about 0.005.** State that limit in the pre-registration.
- **What lets the PI say "provable".** "On 12 independently drawn strain partitions of the
  Messner proteome panel, joint training raises held-out expression `pearson_per_feature` by
  D_e (95% CI lower bound above 0, one-sided p = ...) and is non-inferior on proteome
  `pearson_per_feature` within a pre-registered 0.015 margin (one-sided p = ...), with the
  same direction on the held-out test split; the two arms differ only in which loss terms are
  active." That is a conjunction claim at level 0.05 with no correction owed.

### E2. Maximal version: 24 partitions x 1 init seed, 600 epochs (RANK 2, only if cabbi gives 3 cards)

- Same three arms, same per-head windows scaled to a 600 epoch budget (proteome 200 to 400,
  expression 500 to 600).
- **Wall:** 24 tasks x 35.6 h = 854 card-hours = 35.6 card-days, plus 4 spare slots
  (6 card-days) = 41.6. On 6 cards: 24 cells over 6 cards = 4 rounds x 35.6 h = 142 h =
  **5.9 days**. On 5 cards: 7.1 days. On 3 cards: 11.9 days, no slack, do not attempt.
- **Resolution.** 24 pairs: MDE 0.0070 (prot sd), 0.0087 (pooled), 0.0126 (expr sd), one-sided.
  Non-inferiority at m = 0.015 has 80% power even at a true proteome cost of -0.008 (n = 24
  exactly). Conjunction power 0.65 at (+0.010, +0.010) and 0.97 at (+0.010, +0.020).
- **Cost of the shorter budget, stated as a risk, not hidden.** The expression head has never
  plateaued by epoch 600 in any round; v16's joint expression head was still rising at 499 at
  +0.003 to +0.023 per 100 epochs. An early read favors the faster-converging arm. Head
  matching should make the rates symmetric, but that is **Hypothesis (untested)**. Mitigation
  at zero GPU cost: log the window mean at 400, 500 and 600 and pre-declare that the primary
  is 500 to 600 while the other two are reported as a convergence check; if the three disagree
  in sign the round is reported as unresolved.

### E3. Zero-GPU prerequisites, run first, land before any sbatch (RANK 1 for value per GPU-day: infinite)

1. Fix `v13_split_readout.py` so `PLATEAU` reads the registered `window_mean` column, and make
   the window the documented last sixth rather than a hardcoded 200 epochs. This alone changes
   v17 L_prop2 from +0.0122 (t 1.28) to +0.0204 (t 3.79) on 11 pairs.
2. Add the epoch-200 launch gate and the expression-head decay gate as columns in the readout,
   with the 0.05 and 0.70 thresholds and the calibration above (0 false positives in 110).
3. Add a killed-run guard: `finished` with epoch below `max_epochs` is a killed run. Six of the
   v16 continuation runs are marked `finished` at epochs 531 to 558 against `max_epochs` 1200.
4. Log `pearson_per_feature` on the gene-disjoint held-out rows, and per-head scoring windows
   with the per-head checkpoint monitors.
5. Write the pre-registration into the config header: arms, partitions, seeds, epochs, per-head
   windows, the two one-sided tests, the margin m = 0.015 with its justification, the detector
   thresholds, the restart protocol, and the ITT-plus-PP reporting pair.

### E4. Fallback statement if the expression effect is +0.01 (not an experiment, a pre-declared wording)

At 12 pairs the one-sided MDE on the expression head is 0.0128 (pooled sd) to 0.0185 (v17 sd),
so +0.010 will NOT clear the superiority test in E1, and at 24 pairs (E2) it clears only at
the pooled sd, not the v17 sd. Pre-declare the fallback now so it is not written after the
fact:

> "On 12 (24) independent strain partitions, joint proteome-plus-expression training changed
> held-out expression `pearson_per_feature` by +0.010 (95% CI a to b) and proteome
> `pearson_per_feature` by D_p (95% CI c to d). The expression gain is positive on k of 12
> partitions but below the design's minimum detectable effect of 0.013 (0.009), so it is
> measured and unresolved, not null. The proteome head is non-inferior within the
> pre-registered 0.015 margin. The design that would resolve +0.010 on the expression head
> needs 19 to 38 pairs at the measured paired sd of 0.0167 to 0.0241, which is 38 to 76
> card-days at 3.56 min per epoch and does not fit 14 days on three cards."

That sentence is defensible, and it is the one the PI should be prepared to say. The one
sentence that is NOT available from any 14 day design is "joint training improves both heads
by a small but significant amount", because a genuinely neutral proteome head gives a
superiority conjunction 5% power regardless of n.

### Explicitly NOT recommended

- Reading the existing v16 round as evidence either way. All six J_expr runs fail the
  pre-declared detector; the +0.060 is ITT against four never-launched references and the
  per-protocol read is n = 2 at +0.0069.
- Continuing v16 to 1,200 epochs as the answer. It carries the confounded control, only 2 init
  seeds, 3 partitions, and a single window for two heads with opposite trajectories. At 3.0 d
  per task it costs about 6 card-days to extend and cannot support a conjunction claim at any
  length.
- Four partitions with three seeds. It is 1.4x less precise than 12 x 1 at the same cost and
  its partition-level randomization test cannot reach p < 0.05 by construction.
