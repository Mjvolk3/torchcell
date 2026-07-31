---
id: 7hnv0x0j9eu5jzjufq3z4am
title: Expression round Retrospective
desc: ''
updated: 1785458877957
created: 1785458877957
---

## 2026.07.30 - Expression round: the structural diagnosis, what was measured, what was corrected

Round-level record for the Fig-3 **expression** task on branch `019-decoder-pair-routing`.
The per-wave launch log lives in [[experiments.019-simb-multimodal.fig3-expression-experiments]];
the mechanism design record is [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]].
This note is the *round* view: why the round was stuck, what the round can and cannot
distinguish, and which claims changed.

**Provenance.** Three classes of number appear below and are marked inline:

- **(A)** read or recomputed from a committed artifact under
  `experiments/019-simb-multimodal/results/` -- the recomputation is shown where it matters.
- **(W)** read from W&B (`zhao-group/torchcell_019_expr_v8`) for runs whose scores are not
  yet dumped to a committed CSV. These are **partial** where the job is still in flight.
- **(H)** hypothesis / mechanism story, **not measured**. Labeled every time.

### 0. W&B projects and runs

Every claim below that is marked (**W**) resolves to a run here. Two projects carry this
round; the earlier `_v2`..`_v7` projects are prior rounds (`_v7` = the `_007` round).

- **v8 -- waves 1-6** (163 runs):
  <https://wandb.ai/zhao-group/torchcell_019_expr_v8>
- **v9 -- teacher-forced masked-label objective** (16 runs):
  <https://wandb.ai/zhao-group/torchcell_019_expr_v9>
- Fig-3 origin project (the pre-round sniff sweep):
  <https://wandb.ai/zhao-group/torchcell_019-simb-multimodal_cgt_multitask>

**v8 by stage tag** (**W**, epochs are last-logged):

| stage tag | runs | epochs | states |
|---|---|---|---|
| `stage-arch` | 21 | 49-249 | 8 finished, 12 crashed, 1 failed |
| `stage-ladder` | 12 | 18-57 | 3 finished, 9 crashed |
| `stage-wave4` | 8 | 76-80 | 8 crashed |
| `stage-wave4b` | 8 | 299 | 8 finished |
| `stage-wave5` | 75 | 2-1625 | 63 finished, 12 crashed |
| `stage-wave6` | 12 | 105-174 | 12 finished (offline-synced; **still training**) |
| untagged | 27 | 0-275 | 25 finished, 2 failed |

**The long runs** -- the source of every convergence claim in §3.3 (**W**, `stage-wave5`):

| arm | seed | run | last epoch | peak val pf | traineval pf (last) |
|---|---|---|---|---|---|
| `H1_ref` | 1 | [1vhu95lc](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/1vhu95lc) | 1620 | **0.2044 @ ep 1597** | 0.6520 |
| `H1_ref` | 0 | [famj38u0](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/famj38u0) | 1620 | 0.1890 @ ep 1609 | 0.6535 |
| `H1_nodrop` | 0 | [l6a0cd91](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/l6a0cd91) | 1625 | 0.1698 @ ep 1515 | 0.7248 |
| `H1_nodrop` | 1 | [tc33mjlf](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/tc33mjlf) | 1625 | 0.1545 @ ep 77 | 0.7242 |

(Peaks from `run.history()`, which downsamples to 500 points -- exact epochs are approximate
to that resolution. `traineval/mean/pearson_per_feature` is the eval-mode train metric.)

**v9 live arms** (**W**, GilaHyper job 1443, all seed 0, 8 running). An earlier launch of the
same 8 arms failed at epoch 0 (7 runs) plus one crash at epoch 21 -- those ids are dead:

| arm | run | epoch | val pf |
|---|---|---|---|
| `M_off` | [tow1z48n](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/tow1z48n) | 485 | 0.1287 |
| `M_nomix` | [rb3bhryq](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/rb3bhryq) | 341 | 0.1203 |
| `M_hi` | [da5g4o9v](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/da5g4o9v) | 348 | 0.1129 |
| `M_fine` | [hx8pxdic](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic) | 311 | 0.1047 |
| `M_lo` | [f2wf23oy](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/f2wf23oy) | 342 | 0.0908 |
| `M_coarse` | [ebkzn1ao](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/ebkzn1ao) | 350 | 0.0838 |
| `M_gate_rezero` | [u1vuznme](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/u1vuznme) | 344 | 0.0689 |
| `M_sched` | [8r5ewoaq](https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/8r5ewoaq) | 350 | 0.0629 |

Per §3.0 these epochs are **far below saturation** and are not an arm comparison.
Wave-6 run links are in §7.

### 1. The starting position

The CGT sat at val `pearson_per_feature` ~0.17 against a replicate ceiling of **0.7746**
(**A**, `expression_ceiling_replicate.json` -> `primary_ceiling_mean_sqrt_r.ceiling`,
n = 82 paired strains, mean sqrt test-retest). The round could not distinguish "we explored
the space and found nothing" from "we never engaged the mechanism."

### 2. Structural diagnosis -- the perturbation enters additively, exactly once

At `|S| = 1` the perturbation operator is **degenerate**: softmax over a one-element key set
gives `alpha == 1`, so re-drawing `W_Q`, `W_K` changes the output by **exactly 0.0**, and
**16,200 of 32,760** attention parameters are dead (**A**,
`perturbation_selector_degeneracy.json`, via `perturbation_selector_degeneracy.py`).
**95.4%** of the expression build is `|S| = 1` (**A**, 1484 singles vs 72 Sameith doubles in
`fig3_overlap_census.json` -> 1484/1556 = 95.4%).

The model therefore reduces to `g(h_i + c_b)`: gene identity and strain identity meet once,
by addition, and **no term depends on the pair `(p, i)`**. Full derivation + line refs in
[[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]].

This is what framed the rest of the round: **how much pair-rank can the architecture
express?**

| mechanism | form | pair rank |
|---|---|---|
| `V_ref` additive | `g(h_i + c_b)` | **0** |
| `V_sink` gated attention | one bounded scalar per head | 9 |
| `V_basis{16,32,64}` factor model | `sum_j b_ij(h_i) a_j(z_S)` | r |
| `V_film`, `V_hadamard(_add)` | diagonal multiplicative | d = 90 |

Headroom is **not** the constraint: the measured rank-r reconstruction ceiling is **0.7265**
at r = 32 and **0.7799** at r = 64 (**A**, `lowrank_output_ceiling.json`, train-basis arm)
against a 0.198 best. So a null on this axis is a statement about the **mechanism**, not
about capacity.

### 3. What was measured

#### 3.0 The governing fact: NO arm in this round was scored after saturation

**Not one of the 67 runs in the v8 arm table reached past 300 epochs** (**A**,
`decoder_arms_torchcell_019_expr_v8.csv`):

| wave | runs | epoch range |
|---|---|---|
| wave 1 | 10 | 70-140 |
| wave 2 | 5 | 76-140 |
| wave 3 | 36 | 18-276 |
| wave 4a | 8 | 77-81 |
| wave 4b | 8 | 300 |
| **all** | **67** | **18-300, zero above 300** |

Against §3.2: val Pearson **dips at epoch 200-300** and then climbs past its early peak,
reaching a project best at epoch **1367** and still rising; eval-mode train Pearson is still
climbing at **1500**. So every arm in this round was halted **inside the dip, before the
curve turns**.

**Consequence -- and this governs every arm delta below.** An arm comparison at 300 epochs is
not a small effect measured noisily; it is **not a measurement of the mechanism at all**.
Both arms were stopped before either had the chance to separate, so the delta reflects
early-training transients, not the mechanism's contribution. Seeds do not fix this: replicate
count buys precision on the wrong estimand. **Do not read any per-arm delta in this round as
evidence for or against a mechanism -- including the deltas recorded below, which are kept
only because they are what the round actually produced.**

#### 3.1 Null sink (wave 4b) -- truncated at 300 epochs, not interpretable

Recomputed from `wave_convergence_stage-wave4b.csv` (**A**), scoring column `roll_max` (max
of a centered 5-epoch rolling mean of val `pearson_per_feature` -- an upward-biased order
statistic, naming the rule because it is biased):

```
N_sink_mm - R_ref, seeds {0,1,2,42}: [-0.0011, -0.0081, +0.0064, +0.0125]
mean = +0.0024   sd = 0.0090 (ddof=1)   SE = 0.0045   n = 4 pairs
```

Both arms ran exactly 300 epochs, so per §3.0 **this is a non-measurement**: the null sink was
never given the epochs in which the arms could diverge. The secondary point -- that at n = 4
pairs an exact paired sign test floors at p = 0.125 two-sided (2/2^4), so no experiment of
this *shape* can reach p < 0.05 regardless of effect size -- is true but subordinate. Adding
seeds at 300 epochs would still measure nothing. **The fix is epochs, not replicates.**

**On the `+-0.0076` in the session prose (W, untraced).** The `+0.0024` above reproduces
exactly from the committed artifact at `wave4b_convergence.py`'s default `--window 5`
(line 81). The `+-0.0076` does **not**: sweeping windows 1-50 over
`wave_history_stage-wave4b.csv`, and every summary column in
`wave_convergence_stage-wave4b.csv`, no `(window, ddof)` pair yields 0.0076 *and* the
`+0.0024` mean -- window 15 gives sd 0.0076 but a mean of `+0.0002`. Two facts suggest it is
a **shared noise floor** rather than this comparison's own sd: the session prose quotes the
identical `+-0.0076` for both the null-sink and the mixing delta (one error bar reused across
two arms is a pooled floor), and commit `866312a6` states the floor is "carried from wave 5's
measured across-init sd of ~0.006 (n = 8, fixed split)". Treat `+-0.0076` as a carried-over
wave-5 pooled figure, not a wave-4b statistic, until the wave-5 scoring is dumped to
`results/` (Next Steps). The convention-consistent sd for this comparison is **0.0090**
(ddof=1), matching `score_decoder_arms.py:324` (`deltas.std()`).

#### 3.2 Post-perturbation mixing (wave 5) -- truncated, and untestable as posed

-0.0023 +- 0.0076 (**W**, wave-5 scoring; not yet dumped to a committed CSV -- dumping it is
a Next Step). Two independent reasons this says nothing about mixing: the runs are truncated
per §3.0, **and** mixing was tested with **nothing in the objective that requires it**. The
second is why v9 pairs mixing with an objective that does.

#### 3.3 Training was never converged -- the finding that invalidates §3.1 and §3.2

Eval-mode train `pearson_per_feature` (`traineval/mean/pearson_per_feature`) -- a metric that
did not exist before this round -- reaches **0.652-0.725 and is still climbing** at the point
the runs die, versus the ~0.30 the biased in-training series showed. Val dips around epoch
200-300, then climbs past its early peak to a project best of **0.2044 at epoch 1597**
(**W**, `1vhu95lc`; see the §0 long-run table for all four).

**None of these runs converged -- they were WALLTIME-KILLED while still climbing.** All four
died at **12.60-12.61 h of wallclock** at 1620/1625 epochs (**W**): identical wall time,
different epoch counts, which is a SLURM limit, not a failure mode and not convergence. Their
val peaks sit at **93-99% of the way through the run** (0.2044 @ 1597 of 1620; 0.1890 @ 1609
of 1620). The one apparent exception, `H1_nodrop` seed 1 peaking at epoch 77, is a single
early spike on a curve that is otherwise still rising.

**Consequence: we have never observed a peak.** ">750 epochs" is a *lower bound* on where max
performance sits, established only because that is as far as any run has been allowed to go.
The saturation epoch is **unknown and greater than ~1600**.

**This is the actual result of the round.** It is not a caveat attached to the arm
comparisons -- it is the finding that voids them (§3.0). Everything else here is bookkeeping
on numbers produced before it was known.

#### 3.4 Loss and metric move in opposite directions

In the committed wave-4b history (**A**,
`wave_history_stage-wave4b.csv`), val `expression/loss` reaches its minimum long before val
Pearson peaks, then rises while Pearson keeps climbing:

| run | val loss min | val Pearson max |
|---|---|---|
| `N_sink_mm` s1 | 0.2640 @ ep 53 | 0.0640 @ ep 257 |
| `N_sink_mm` s42 | 0.2653 @ ep 217 | 0.0824 @ ep 296 |
| `R_ref` s1 | 0.2639 @ ep 42 | 0.0632 @ ep 293 |
| `R_ref` s42 | 0.2662 @ ep 202 | 0.0701 @ ep 297 |

(Seed 0 is the exception in wave 4b -- loss min ep 132/94 and Pearson peak ep 93/94 nearly
coincide.) In the long runs the separation is much wider (**W**: loss bottoms ~ep 103-136 and
rises for the remaining ~1200 epochs while Pearson climbs). 010 does **not** show this.
**Consequence: every "best" checkpoint 019 ever saved for expression was the early-dip
model**, ~1300 epochs before the good one. Fixed by best-by-metric checkpointing (see §6).

### 4. The masked-objective oracle -- the largest signal, and why it does not count

A ridge oracle that **observes m random genes of a held-out strain and predicts the rest**
(**A**, `masked_conditioning_oracle.json`, 5 draws, tuned lambda, 155 val strains):

| m observed | val pearson_per_feature | sd |
|---|---|---|
| 0 | 0.0 | 0.0 |
| 10 | **0.4084** | 0.031 |
| 100 | **0.6756** | 0.012 |
| 1000 | **0.7932** | 0.001 |

That is **2-4x the entire model**. Two follow-ups made it honest:

**Cross-study (A, `cross_study_conditioning_oracle.json`).** Train on one study, evaluate on
the 82 deletions shared with the other:

| m | within-Kemmeren | within-Sameith | Kem -> Sam | Sam -> Kem |
|---|---|---|---|---|
| 10 | 0.4562 | 0.3649 | 0.2335 | 0.2122 |
| 100 | 0.6693 | 0.6417 | 0.3815 | 0.3922 |
| 1000 | 0.7832 | 0.7779 | **0.4838** | **0.4803** |

The within-study 0.793 exceeded the 0.775 replicate ceiling; **the cross-study 0.48 resolves
that violation exactly** -- same-array conditioning was predicting the target's own
measurement noise.

**Orthogonality (A, `conditioning_gain_after_genotype.json`).** Retained fraction of the
conditioning gain after removing a genotype predictor (kNN on `prot_T5_all`, k = 25):
**0.9753 / 0.9924 / 1.0058** at m = 10/100/1000. So **97.5-100.6% of the gain survives** --
the objective optimizes a channel that is switched **off at m = 0**, which is where we score.

**Verdict: it is imputation.** A real capability, and a defensible product surface -- but it
is **not** a better genotype -> expression number. The v9 arms below test whether the
objective's *representation* transfers to m = 0; the oracle says the *conditioning* does not.

### 5. Corrections to the record

- **`A2_prop_sparse` was never a win** -- and the replacement number is not a result either.
  It was on record as the best result anywhere at +0.0231, which was a **cross-wave**
  comparison (a scoring-hygiene defect: see the three pooling bugs in §6). Against a valid
  in-wave reference (wave 3, `A1_ref`, same 4 seeds) it is **+0.0030, 95% CI +-0.0321**
  (**A**, recomputed from `decoder_arms_torchcell_019_expr_v8.csv`: diffs
  `[-0.0271, +0.0132, +0.0158, +0.0100]`, sd 0.0202, SE 0.0101, t_{3,.975} = 3.182). But
  those wave-3 runs ran **18-276 epochs**, so per §3.0 the +0.0030 is a non-measurement too.
  What is genuinely corrected here is the **comparison hygiene**, not the arm's standing:
  the arm has never been tested to saturation in either direction. Killing it on
  transferability cost nothing.
- **The project's "~0.17" is a seed-0 artifact of the validation draw.** Best non-seed-0
  score anywhere in the v8 table is **0.0901** (`A6_prop_sparse`, wave 2, seed 2) (**A**).
  Seed 0 tops nearly every wave; that is a property of the split, not of the arms.
- **Checkpoints were never missing** -- the earlier search looked in the wrong tree.
- **Low-rank output is not a regularizer.** `per_gene_head` is 9,919 of 887,879 params
  (1.1%), and `ResponseBasisHead` *adds* ~2dr rather than bottlenecking. It is a **pair
  term** -- which is why the rank arms moved along the mechanism axis, not the capacity axis.
- **The ceiling bounds generalization, not fitting.** Train 0.72 is evidence of
  noise-fitting capacity, not a wall being hit.

### 6. Measurement infrastructure rebuilt

The round's scoring apparatus was the actual blocker, so it was rebuilt:

- wave / pool / budget **partitioning in the scorer** -- **three pooling bugs** found, each
  of which had silently voided numbers (cross-wave references, as in the `A2_prop_sparse`
  correction above).
- `perf/epoch_seconds`; **eval-mode train metric** (the unbiased train curve); `mse`/`nmse`.
- **best-by-metric checkpoints** in addition to best-by-loss and last -- the fix for §3's
  early-dip checkpoint problem.
- **source-hash logging** resolved at submit time on the login node (IGB compute nodes have
  no `git`), so an unattributable run is never pooled with attributable ones.
- declared tags (`stage-*`, `mech-*`, `pair-rank*`, `seed*`, `xfer-*`) so a wave is
  filterable in W&B rather than reconstructed by hand.
- Throughput **27.7 -> 17.05 s/epoch** via batch 32: the encoder runs on the wildtype graph,
  so its cost is **per-step**, not per-sample.

### 7. Running now

Both waves report `val/mean/pearson_per_feature`, so they are directly comparable.
One seed, **arms not replicates**, 10000-epoch caps, 48 h walltime -- the runs are *expected*
to hit the wall, which is fine because best-by-loss / best-by-metric / last are all retained.

**IGB `mmli` job 2324896** -- 12 wave-6 arms, `cgt_expr_012`, 3 runs/GPU x 4 A100
(`igb_expr_wave5.slurm`, `W5_STAGE=wave6`). Pair-rank ladder (0/9/16/32/64/90/90) plus
regularization (`weight_decay` was **1e-8**, i.e. effectively off; dropout arms alongside).

**Wave-6 W&B runs** (project `zhao-group/torchcell_019_expr_v8`, all tagged `stage-wave6`).
Synced from the IGB **login** node at 2026.07.30 19:47 -- compute nodes have no internet, so
runs are `WANDB_MODE=offline` under
`$DATA_ROOT/wandb-experiments/compute-5-7-<task-job-id>_*/wandb/offline-run-*` and are pushed
with `wandb sync --include-offline --include-synced`. The four task-job ids
(2324896/2324946/2324948/2324978) are the per-array-task `SLURM_JOB_ID`s, 3 co-resident runs
each.

| arm | pair rank | run | epoch at sync | val pf at sync |
|---|---|---|---|---|
| `V_ref` | 0 | [sm6efleg](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/sm6efleg) | 174 | 0.0986 |
| `V_sink` | 9 | [fqk5mbr3](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/fqk5mbr3) | 173 | 0.0984 |
| `V_basis16` | 16 | [hoheq7rz](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/hoheq7rz) | 174 | 0.1043 |
| `V_basis32` | 32 | [gykwmhro](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/gykwmhro) | 142 | 0.1547 |
| `V_basis64` | 64 | [b50f93ju](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/b50f93ju) | 141 | 0.1424 |
| `V_film` | 90 | [1j0tn0p7](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/1j0tn0p7) | 141 | 0.1507 |
| `V_hadamard` | 90 | [y69altye](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/y69altye) | 105 | 0.1463 |
| `V_hadamard_add` | 90 | [9ahs039r](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/9ahs039r) | 105 | 0.1530 |
| `V_drop2` | -- | [ylykn8v6](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/ylykn8v6) | 105 | 0.1164 |
| `V_drop3` | -- | [tcifmwri](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/tcifmwri) | 107 | 0.0968 |
| `V_wd1e4` | -- | [0k9ae8co](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/0k9ae8co) | 107 | 0.1355 |
| `V_wd1e2` | -- | [jyl2i2d0](https://wandb.ai/zhao-group/torchcell_019_expr_v8/runs/jyl2i2d0) | 107 | 0.1388 |

**These columns are NOT an arm comparison.** The runs are at different epochs (105-174), and
per §3 that entire range sits **inside the val dip** -- the regime in which every previously
mis-scored arm was scored. They are a liveness check only. Two offline-sync artifacts to know
about: a synced run shows state `finished` on W&B even while it is still training on IGB, and
its history only advances when the login-node sync is re-run.

**GilaHyper job 1443** -- 8 v9 arms: teacher-forced masked-label objective with the loss
restricted to the still-hidden set `U_k`, revealed genes taking **exactly zero gradient**
(contract-verified, **A**, `verify_masked_objective.json`), plus `M_nomix` as a negative
control and `M_off` to isolate the objective from its parameters.

### 8. The honest bottom line

**No mechanism has been tested to a conclusion.** Not "no mechanism helped" -- every arm in
the round was scored before saturation (§3.0: 67/67 runs stopped at or below 300 epochs,
against a curve that is still climbing at 1367), so the round contains **no valid arm
comparison at all**. The per-arm deltas recorded above are kept as a record of what was
produced, not as evidence.

What the round did produce: (a) the finding that training never converged, and that val loss
and val Pearson diverge -- which is what voids the arm table and what invalidated every
checkpoint previously saved; (b) a structural reason the additive operator cannot express a
pair term at all (§2); (c) one large signal -- cross-study 0.48 -- **measured to be
orthogonal** to the task being scored (§4); and (d) an apparatus that will make the *next*
round's comparisons readable, provided they are run past saturation.

### 9. Next steps

- Dump the wave-5 scoring (mixing arm, null-sink pooled) to a committed CSV under
  `results/` so the `-0.0023 +- 0.0076` and `+-0.0076` figures stop being **W** and become
  **A**.
- Land the init-time probe used in
  [[experiments.019-simb-multimodal.multiplicative-perturbation-conditioning]] as a committed
  script (repo artifact rule).
- **Re-run, do not re-score, the historical arms.** Re-scoring cannot recover a comparison
  from runs that stopped at <=300 epochs -- the post-dip epochs were never trained. Any arm
  worth a verdict has to be re-run past saturation. Establish where saturation actually is
  from the `long` stage first, so the next round's epoch budget is set by measurement rather
  than by walltime.
- Decide whether masked-label imputation becomes its own deliverable, given §4.
