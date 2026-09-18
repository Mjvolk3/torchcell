# Agent 10: methodology and experimental design, devil's advocate

Read-only review, 2026-09-17. Every number below traces to a file I read or a computation I
ran; the analysis scripts I wrote are in the scratchpad work directory listed at the end and
would need to be promoted into `experiments/019-simb-multimodal/scripts/` before any of these
numbers enters a note or the paper (repo rule: artifacts come from committed scripts).

---

## 0. The four findings that matter most

1. **No round in the expression or proteome strand has ever produced a test number.** Zero
   runs with any `test/*` key across 410 runs in `torchcell_019_expr_v8/v9/v10/v11/v12/v13/v15`
   and `torchcell_019_prot_v14` (checked through the W&B API). v13 asked for one
   (`conf/cgt_expr_v13_split.yaml:60`, `trainer.run_test: true`) and did not get one, because
   the runs were killed by the wall at epoch ~4,080 of 6,000 after 111.5 h, so
   `trainer.test(...)` at `train_cgt_multitask.py:3457` never executed. All 24 say
   `state = finished` because they are offline runs and `wandb sync` marks a synced directory
   finished regardless of how the process died.

2. **The first held-out read that exists reverses the arm ordering the campaign has been
   chasing.** From the checkpoint dumps written today by `gh_eval_ckpt_predictions.slurm`:
   V_ref_s0 seed0 scores **val 0.2227 / test 0.1095**; V_concat_s0 seed0 scores
   **val 0.1926 / test 0.1391**. On validation ref beats concat by +0.030; on test concat
   beats ref by +0.030. Paired strain bootstrap (400 resamples, my `paired2.py`):
   val concat-ref = -0.0302, 95% [-0.0598, -0.0007], P(concat>ref) = 0.02; test concat-ref =
   +0.0296, 95% [-0.0039, +0.0621], P(concat>ref) = 0.95. Two draws of 155 strains from the
   same pool give opposite, nominally-significant answers about the same pair of checkpoints.

3. **The measurement instrument has a draw-to-draw standard deviation of 0.017 to 0.027, for a
   model with no training stochasticity at all.** The fixed bilinear ridge baseline B2 on
   ProtT5 scores between 0.0762 and 0.1476 across the 24 (12 val + 12 test) 155-strain draws in
   `results/expression_baselines_split/seed0..11.json`: sd 0.0173, range 0.071. B3 kNN ranges
   0.0456 to 0.1623, sd 0.0274, range 0.117. Every design effect the campaign has acted on
   (+0.0135 head, +0.0056 concat, +0.027 mech per-gene) is inside one standard deviation of the
   instrument.

4. **"Split 0 is +0.089 easier" is wrong as stated; split 0's VALIDATION draw is lucky and its
   TEST draw is unlucky.** For B2, split-0 val is z = +1.72 above the 12-split val mean while
   split-0 test is z = -1.38 below the test mean. For B3, split-0 test (0.0456) is the minimum
   of all 24 draws. The campaign pinned `split_seed 0` and reads the val half, which is the
   single most favorable 155-strain draw available to it.

---

## 1. The scoring rule

### What the rule is

`roll_max` = the maximum of a centered 5-epoch rolling mean of
`val/expression/pearson_per_feature` (`pull_round_leaderboards.py:165,269-274`, re-implemented
at `v13_split_readout.py:105-107`, imported by `budget_rank_preservation.py`,
`mech_round_readout.py`, `head_round_readout.py`, `short_budget_spread.py`). The
`build_retrospective_tables.py` docstring already names it "an upward-biased order statistic"
and flags that it grows with epochs run, so the problem is known; what has not been done is
quantifying it.

### How big the bias is, measured

I pulled the full per-epoch histories of all 24 v13 runs (every epoch, no W&B downsampling) and
estimated the local trend as a centered 201-epoch rolling median, then measured the excess of
each statistic over that trend (`rules.py`). On the 23 non-plateau runs, inside the matched
window (epochs <= 3,754):

| quantity | value |
|---|---|
| late-training residual sd around the trend | **0.00500** (range 0.00356 to 0.01716) |
| lag-1 autocorrelation of residuals | +0.264 |
| integrated autocorrelation time tau | 3.72 |
| effective independent looks in 3,754 epochs | **~1,009** |
| Blom E[max] of 1,009 iid normals | 3.23 sd = +0.0161 |
| measured **raw-max** excess over trend | **+0.0126** (sd 0.0021 across runs) |
| measured **roll5-max** excess over trend | **+0.0071** (sd 0.0018) |
| roll5-max minus fixed-epoch mean of the last 200 epochs | +0.0118 (sd 0.0079) |

Theory and measurement agree: 3,754 epochs is about 1,000 independent looks, and the 5-epoch
smoothing buys back roughly half the raw-max bias. On the two 10,000-epoch curves in
`results/expression_curve_{hx8pxdic,b50f93ju}.csv` the same computation gives raw bias +0.0171
and +0.0117, roll5 bias +0.0077 and +0.0051, and the bias grows monotonically with the epoch
cap (for hx8pxdic, roll5-minus-trend is +0.0023 at cap 2,000 and +0.0076 at cap 10,000).

Headline run of the strand, the max-of-8 v9 arm:

https://wandb.ai/zhao-group/torchcell_019_expr_v9/runs/hx8pxdic

### The honest verdict, which is not the one you might expect

**The mean bias is not what breaks the campaign's arm conclusions.** Because the readouts pair
arms within split and seed at a matched epoch, the +0.0071 roll5 bias cancels to first order.
I recomputed the v13 concat-minus-ref contrast under three rules on the 9 clean 80/10/10 pairs:

| rule | mean | sd | t | positive |
|---|---|---|---|---|
| roll5 max at matched epoch (the campaign's rule) | +0.0015 | 0.0127 | +0.36 | 4/9 |
| raw max at matched epoch | +0.0120 | 0.0314 | +1.32 | 6/12 |
| fixed-epoch mean of epochs 3,554 to 3,754 | +0.0072 | 0.0188 | +1.60 | 6/9 |

(The published `v13_split_readout.json` reports +0.0056, sd 0.0197, t 0.94 over 11 pairs; the
difference is that it folds in the two 90/10 pairs, one of which is a +0.053 outlier.) Under
every rule the contrast is inside the noise. The rule change moves the point estimate by less
than the noise it sits in. Three things the max rule DOES break:

- **Absolute capability claims.** "0.1965 +/- 0.0222", "0.2382", "model 0.1965 vs B2 0.1040"
  are all roll_max numbers, and the max rule adds +0.007 to +0.017 on top of a 155-strain draw
  whose own sd is 0.026 on a partition whose val draw is z = +1.7. The stacked optimism is
  roughly +0.05 before any selection across configs.
- **Checkpoint selection.** `trainer.checkpoint.monitor` is set to the metric itself, so the
  saved "best" checkpoint sits at the **raw** max, which is +0.0126 above trend by construction.
  That is the checkpoint whose held-out score is 0.1095.
- **Any comparison across unmatched budgets.** The bias grows with epochs run, so a 9,900-epoch
  arm carries ~+0.017 of raw-max bias against a 1,400-epoch arm's ~+0.013. `mech_round_readout`
  and `head_round_readout` match budgets correctly; the cross-round comparisons in the
  retrospective tables do not, and cannot.

### What survives an unbiased rule

- **v10 embedding content, +0.068 (ProtT5 minus random_1024), survives comfortably.** 16 runs
  vs 16 at a matched 990 epochs, pooled within-cell sd 0.0246 (0.0122 excluding the one run
  that never learned), se 0.0087, t 7.8 (17.3 healthy) (`results/v10_grid_factorial.json`). It
  is 8 times the roll5 bias and 2.6 times the draw sd. Note what it is, though: a contrast
  against RANDOM embeddings, a sanity check, not a design choice. The note already records that
  `calm` vs ProtT5, the contrast that would change anything, is unmeasured.
- **v12 H_concat, +0.0135, survives as a split-0 result and dies as a general one.** All four
  seeds positive at 1,399 epochs (+0.0174/+0.0143/+0.0107/+0.0116) and all four positive at
  1,000 too, which is genuinely more consistent than anything else in the round. But the same
  contrast across four partitions is +0.0015 to +0.0056 (t 0.36 to 0.94).
- **Everything measured at n <= 2 does not survive anything.** The mechanism round's per-gene
  arms are `n_pairs: 2` (`results/mech_round_readout.json`: R_pergene diffs +0.0051 and +0.0598,
  mean +0.0276). A two-pair paired t has a minimum detectable effect of 0.15, so that round
  could not have resolved anything it found.
- **The "+0.041 per-gene head" quoted in the campaign summary has no source I could find** in
  the results JSONs. The closest measured numbers are +0.0319 (v12 H_pergene at 500 epochs,
  n=4, sd 0.0193) and +0.0374 (the same arm, seed 0 alone). At the matched 1,399 epochs the same
  arm reads **-0.0085**. A single-seed +0.04 is one draw from a distribution with paired sd
  0.019, which is uninformative.

---

## 2. Sample sizes and power

### Variance components, measured from v13 (`rules.py`)

| component | roll5 rule | fixed-epoch rule |
|---|---|---|
| pooled within (arm, split) init-seed sd | **0.0108** (13 df) | 0.0129 |
| paired difference sd, per (split, seed) pair | **0.0127** (n=9) | 0.0188 |
| same, including the two 90/10 pairs | 0.0197 (n=11) | -- |
| sd of the four split-mean differences | 0.0131 | 0.0176 |
| implied split-by-arm interaction sd tau | ~0.000 | 0.0046 |
| 155-strain draw sd (strain bootstrap, `paired2.py`) | 0.026 (val) / 0.018 (test) | same |

The split-by-arm interaction is essentially zero, which is good news: the arm effect is the
same on every partition, the seed noise dominates, and pairing within split works (the paired
sd 0.0127 is below sqrt(2) x 0.0108 = 0.0153, so the shared val draw does cancel).

### Minimum detectable effect, paired t, alpha 0.05 two-sided, 80% power

| n pairs | sd 0.0108 | sd 0.0127 | sd 0.0153 | sd 0.0197 | sd 0.0262 |
|---|---|---|---|---|---|
| 2 | 0.125 | 0.147 | 0.177 | 0.228 | 0.303 |
| 3 | 0.035 | 0.042 | 0.050 | 0.064 | 0.086 |
| 4 | 0.023 | 0.027 | 0.033 | 0.042 | 0.056 |
| 6 | 0.016 | 0.018 | 0.022 | 0.028 | 0.038 |
| 8 | 0.013 | 0.015 | 0.018 | 0.023 | 0.030 |
| 12 | 0.010 | 0.011 | 0.014 | 0.018 | 0.023 |
| 16 | 0.008 | 0.010 | 0.012 | 0.015 | 0.020 |
| 24 | 0.006 | 0.008 | 0.009 | 0.012 | 0.016 |

Power to detect **0.02** at the realistic paired sd of 0.0153: n=2 gives **0.12**, n=4 gives
0.44, n=8 gives 0.88, n=12 gives 0.98.

### Is the campaign powered to see 0.02?

- **v13 (the best-designed round): marginally no.** 9 to 11 clean pairs, MDE 0.027 to 0.042.
  It could not have resolved 0.02 and did not claim to.
- **v14 proteome: borderline.** 8 pairs, sd 0.0058, MDE 0.007 within split 0-3 on the val
  statistic, which looks strong until you notice the proteome contrast is -0.0002 and the val
  draw sd from the strain bootstrap is 0.017.
- **v12: no, despite t = 8.92.** 4 pairs on ONE split. The 4-seed paired sd of 0.0030 is 4x
  smaller than v13's 0.0127 for the same contrast, purely because it holds the partition fixed
  and repeats only the init seed. That is a variance estimate for the wrong inference target.
- **v15 as it stands: no, by an order of magnitude.** 4 runs, n=1 per contrast, no sd, no t
  (`results/v15_wd_readout.json`). And the two replicates of the SAME reference arm differ by
  **0.0176** while the effect being claimed is **+0.0185** from one pair:

https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/dabvh5po

https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/mv5zd8bu

https://wandb.ai/zhao-group/torchcell_019_expr_v15/runs/223ce8px

- **The mechanism round: no, catastrophically.** n=2, MDE 0.15.

### A correction the campaign should make to its own sizing rule

Memory `019-expr-long-budget-replicate-spread` says "0.02 needs 19.2 per arm" from a
**two-sample** power calculation at the arm spread sd 0.0222. The design actually run is
**paired within split and seed**, sd 0.0127 to 0.0153, which needs **6 to 10 pairs** for 0.02.
The campaign has simultaneously been too pessimistic about what a well-paired round can resolve
and too optimistic about what the rounds it actually ran did resolve.

---

## 3. The validation set

### What the metric is

`per_feature_pearson` (`train_cgt_multitask.py:275-311`) correlates each of 6,127 gene columns
across the 155 validation strains and averages the 6,127 correlations. The gene columns are
strongly co-varying, so the 6,127-fold averaging buys far less than sqrt(6127); the honest
sample size is the 155 strains.

### Sampling sd of the metric on 155 strains, measured by strain bootstrap

On the actual best-val checkpoint dumps (`paired2.py`, 400 resamples):

| checkpoint | val | bootstrap sd | test | bootstrap sd |
|---|---|---|---|---|
| V_ref_s0 seed0 | 0.2227 | **0.0262** | 0.1095 | 0.0189 |
| V_concat_s0 seed0 | 0.1926 | **0.0258** | 0.1391 | 0.0170 |
| P_ref_s0 seed0 (proteome, 448/447 records) | 0.1164 | 0.0170 | 0.0672 | 0.0174 |
| P_concat_s0 seed0 | 0.1137 | 0.0195 | 0.0818 | 0.0150 |

The 95% bootstrap interval for the headline 0.2227 is **[0.173, 0.272]**. A random split-half
of the val set gives 0.2355 vs 0.2136 (ref) and 0.2067 vs 0.1817 (concat), consistent.

**The permutation null is not zero.** Shuffling strain labels 100 times gives mean +0.0024,
sd 0.0166, max +0.0506. So on 155 strains, a score up to ~0.05 is reachable by chance
alignment; the v13 readout's `PLATEAU = 0.05` threshold is, by accident, exactly the right
chance band.

### How much of the epoch-to-epoch wobble is the val draw?

None of it. The val set is fixed within a run, so the epoch-to-epoch residual sd of **0.0050**
is model-state jitter, not resampling. It is **19% of the 0.0262 draw sd**. The right reading:
the wobble you can see in the curve is five times smaller than the uncertainty in where that
whole curve sits relative to the population, and the second uncertainty is invisible in every
plot the campaign has made.

### Is the split-seed effect bigger than any arm effect ever measured?

**Yes, by a wide margin.** v13 partition means at the matched epoch are 0.1970 / 0.1647 /
0.1410 / 0.1387, between-partition sd **0.0271**, range **0.0584**, against a pooled
within-partition sd of 0.0092. The same ordering holds under the fixed-epoch rule (0.1863 /
0.1470 / 0.1279 / 0.1276, between sd 0.0276). The largest design-choice effect ever measured
with more than 2 runs per arm is +0.0135 on one split, which is 0.23 of the partition range.
The only effect that exceeds the partition spread is ProtT5 vs random embeddings (+0.068).

And, per finding 4 above, the partition spread is itself mostly draw noise, not partition
difficulty: the same partitions rank differently on their val and test halves for the
deterministic baselines.

---

## 4. Test reads

**Rounds with a test number: zero, until today.**

| project | runs | runs with any `test/*` summary key |
|---|---|---|
| torchcell_019_expr_v8 | 163 | 0 |
| torchcell_019_expr_v9 | 81 | 0 |
| torchcell_019_expr_v10 | 68 | 0 |
| torchcell_019_expr_v11 | 12 | 0 |
| torchcell_019_expr_v12 | 42 | 0 |
| torchcell_019_expr_v13 | 24 | 0 |
| torchcell_019_expr_v15 | 4 | 0 |
| torchcell_019_prot_v14 | 16 | 0 |

`results/v13_split_readout.json` carries `test_at_best_val: null` for all 24 runs, and
`v14_proteome_readout.json` for all 16. The v13 mechanism of loss is worth naming precisely
because it will recur: `max_epochs: 6000` was not reachable inside the wall (the runs died at
~4,080 epochs after **111.5 h** of a 5-day allocation), so the `trainer.test` call that was
correctly configured never ran, and the offline sync stamped every dead run `finished`. **A
`finished` state with `epoch < max_epochs` is a killed run, and nothing in the readout checks
for that.**

Example of a run that reads `finished` at 4,073 of 6,000 epochs:

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/bn37i9vs

**The first four test numbers in the strand** (computed here from the dumps that
`gh_eval_ckpt_predictions.slurm` wrote to `$DATA_ROOT/test-predictions/` today), against the
split-matched baselines in `results/expression_baselines_split/seed0.json` and
`results/baselines_split_fig3_proteome/seed0.json`:

| checkpoint | val | test | B2 test | B3 test |
|---|---|---|---|---|
| expression V_ref_s0 seed0 | 0.2227 | **0.1095** | 0.0847 | 0.0456 |
| expression V_concat_s0 seed0 | 0.1926 | **0.1391** | 0.0847 | 0.0456 |
| proteome P_ref_s0 seed0 | 0.1164 | **0.0672** | 0.0586 | 0.0921 |
| proteome P_concat_s0 seed0 | 0.1137 | **0.0818** | 0.0586 | 0.0921 |

Two readings follow, and they point in opposite directions:

- **The expression model is genuinely better than the linear baselines on held-out data**, by
  +0.025 (ref) to +0.054 (concat) over B2, with a paired-strain bootstrap sd around 0.017. That
  is the one capability claim the evidence supports, and it should replace the 0.1965-vs-0.1040
  val comparison everywhere it is quoted.
- **The proteome model does not beat its kNN baseline on held-out data at split 0**
  (0.067 and 0.082 against B3's 0.092). The v14 note's "arms lead B2/B3 by 0.03 to 0.05" is a
  validation-side statement that does not survive the first test read.

**Has any conclusion ever been confirmed on test? No.** The first one attempted was confirmed
backwards: concat, which lost on val, won on test.

---

## 5. Model selection on the same validation draw

`results/grid_manifest.json` records **288 configs** ("n_configs": 288, 5 primary levers, 2
secondary profiles, 3 seeds) for the `gh_expr_grid_000..287` sweep.
`results/round_leaderboards.csv` records **1,514 expression runs with a score** across 15
projects, and every score since `cgt_expr_011` is read on the same pinned split-0 validation
draw.

- The Blom expected maximum of 1,514 iid draws is 3.35 sd. At the measured replicate sd of
  0.0117 (1,500 epochs) to 0.0222 (9,900 epochs, `results/short_budget_spread.json`), the
  pure-noise component of "the best score we ever got" is **+0.039 to +0.074**. That is an
  upper bound, since genuinely better configs do exist in the pile, but it is the same size as
  the whole distance from the headline to the baseline.
- An empirical anchor that needs no theory: the eight v9 arms at 9,900 epochs have mean 0.1965
  and max 0.2382, so max-of-8 alone is **+0.0417**. The memory note already records this
  correctly. Max-of-1,514 is not smaller.
- The anchor that settles it: the one config selected this way that now has a held-out read
  scored **val 0.2227, test 0.1095**. The val-minus-test gap of **+0.113** is 3.6 times the
  combined draw sd of two independent 155-strain draws (sqrt(0.0262^2 + 0.0189^2) = 0.032), so
  it is not sampling alone; the rest is selection, both of the epoch and of the config lineage.

**Multiplicity within rounds is also unmanaged.** v12 tested 7 arms at 3 budgets = 21 paired
t-tests and reported the ones that cleared. At budget 500, `H_state` read **+0.0356 with
t = +9.29** and 4/4 seeds positive; at 1,399 the same arm reads **-0.0007**. The same statistic,
on the same runs, at the same n, certified an arm that evaporated. That is not an argument
against `H_concat` specifically (it is positive at 500, 1,000 and 1,399), it is a demonstration
that this round's inference rule produces t ~ 9 false positives:

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/mfgjmq57

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/vahi7g0w

https://wandb.ai/zhao-group/torchcell_019_expr_v12/runs/zigh98ds

---

## 6. Compute allocation: long runs versus paired short runs

### What v13 cost and bought

24 runs x 111.5 h, 4 per card = 6 cards x 111.5 h = **669 card-hours = 27.9 card-days**. It
bought one paired contrast at n = 9 to 11 (MDE 0.027 to 0.042), a partition spread, and zero
test reads.

### What the long tail is actually worth

Across the 23 clean v13 runs (`rules.py`):

| budget gain | mean | sd | max |
|---|---|---|---|
| 500 -> 3,754 | +0.0637 | 0.0154 | +0.0924 |
| 1,000 -> 3,754 | **+0.0197** | 0.0116 | +0.0369 |
| 1,500 -> 3,754 | +0.0115 | 0.0107 | +0.0329 |
| 2,000 -> 3,754 | +0.0071 | 0.0079 | +0.0258 |

**88.6% of the epoch-3,754 score is already reached at epoch 1,000.** Going from 1,000 to 3,754
epochs costs 3.75x the compute and buys +0.0197 on a statistic whose draw sd is 0.026. Going
from 2,000 to 3,754 buys +0.0071.

Concretely, the run with the held-out read reaches roll5 0.1952 at epoch 1,000 and 0.2070 at
3,754; the extra 2,754 epochs bought +0.012 of a quantity whose 95% interval is +/- 0.05.

### The "longer beats more replicates" conclusion is contradicted past 1,500 epochs

Memory `019-expr-long-budget-replicate-spread` concludes "LONGER BEATS MORE REPLICATES" from
the 700 -> 1,500 segment (sd 0.0166 -> 0.0117). The full series in
`results/short_budget_spread.json` for the same 8 runs is:

| epochs | 350 | 500 | 700 | 1,000 | 1,500 | 2,000 | 2,800 | 4,000 | 9,900 |
|---|---|---|---|---|---|---|---|---|---|
| mean | 0.1198 | 0.1223 | 0.1411 | 0.1609 | 0.1670 | 0.1745 | 0.1821 | 0.1883 | 0.1965 |
| sd | 0.0096 | 0.0058 | 0.0166 | 0.0099 | 0.0117 | 0.0151 | 0.0175 | 0.0171 | 0.0222 |

Past 1,000 epochs the spread grows monotonically: 0.0099 -> 0.0222. Ten times the compute buys
2.2 times the noise and 1.22 times the mean. The 700 -> 1,500 descent the note generalized from
is the one descending segment in the series, and the note itself warns against exactly this
kind of generalization ("never assume shorter is quieter") in the opposite direction.

### Does a short screen rank correctly?

Measured on the v13 curves, two different questions with two different answers:

- **Run-level ranking is preserved early.** Spearman against the 3,754 score: +0.864 at 300
  epochs, +0.874 at 500, **+0.907 at 1,000**, +0.933 at 1,500, +0.961 at 2,000. (Most of this
  is the partition, which dominates run-level variance.)
- **The paired arm contrast is NOT stable at short budgets.** concat-minus-ref by budget over
  the same 9 pairs: 300 -> +0.0030 (t +1.40); **500 -> +0.0089 (t +2.30, 6/9 positive)**;
  1,000 -> +0.0042 (t +0.61); 1,500 -> -0.0025; 2,000 -> -0.0014; 3,754 -> +0.0015. A
  500-epoch screen would have declared concat resolved at p < 0.05. The split-level mean
  differences, however, ARE stable from 1,000 onward (s0 -0.0115 -> -0.0096, s2 +0.0255 ->
  +0.0211), which is what a design with enough pairs would read.

### The allocation answer

**Shorten the runs and buy pairs.** The same 669 card-hours at 1,200 epochs buys 80 runs = 40
pairs (MDE 0.0058 to 0.0070) instead of 24 runs = 9 pairs (MDE 0.027 to 0.042). Nothing in the
data supports spending 4x the compute per run for a +0.02 shift in a biased statistic that has
never been shown to transfer to a held-out draw.

**One caveat, stated as a hypothesis because it is unmeasured.** Hypothesis (untested): the
late val rise from epoch 1,000 to 4,000 is uncalibrated ordering that does not transfer to a
held-out draw. The evidence pointing that way is that NMSE at the selected peak is **1.036
(s0) to 1.133 (s3)**, meaning the reported checkpoint has a larger squared error than
predicting each gene's training mean, and NMSE worsens monotonically with partition difficulty.
The experiment that settles it is cheap and should be run first: score the **existing** v13
checkpoints at epochs 500, 1,000, 2,000 and best-val on the test partition. Two runs, a few GPU
hours on the checkpoints already copied to GilaHyper, and it decides whether any future round
should ever exceed 1,200 epochs.

---

## 7. Proposed decision rule and design template for the rounds after the queued 1 to 3

### Design template (one contrast family per round)

| parameter | value | why |
|---|---|---|
| arms | 1 reference + at most 2 candidates | multiplicity: 2 tests per round, Bonferroni alpha 0.025 |
| split seeds | **4** (0, 1, 2, 3), always all four | the partition moves the score by 0.058, more than any arm effect |
| init seeds | **3** per (arm, split) | 12 pairs per contrast |
| epochs | **1,200 fixed**, no early stop, no max_epochs the wall cannot reach | 88.6% of the 3,754 score is at 1,000; the wall killed v13's test read |
| scoring rule | **mean of `val/<pheno>/pearson_per_feature` over epochs 1,000 to 1,200**, fixed window, no max | removes the +0.0071 to +0.0126 order-statistic bias and the epochs-run confound; carry roll_max as a secondary column only |
| primary analysis | paired t on the 12 (split, seed) differences | the split-by-arm interaction is ~0, so pairing is valid and efficient |
| secondary analysis | sign of the 4 split-level means | guards against an effect carried by one partition |
| mandatory closing step | score every arm's best-val checkpoint on that partition's test split, report the arm mean over 4 partitions with a strain-bootstrap CI | no round without a test read |

**Power of this template.** With paired sd 0.0127 to 0.0153, n = 12 gives MDE **0.0113 to
0.0136** at 80% power, so a true effect of 0.02 is detected with power > 0.98 and a claimed
0.02 carries a 95% CI of about +/- 0.008 to +/- 0.010. That satisfies the brief.

**Cost.** 3 arms x 4 splits x 3 seeds = 36 runs at 1,200 epochs = 33 h per run at 36 epochs/h,
4 per card, 9 cards x 33 h = **297 card-hours = 12.4 card-days**, 44% of what v13 cost, with
4x the resolving power and a test read at the end.

### Go/no-go rule

Adopt an arm into the reference stack only when **all four** hold:

1. the 12-pair paired mean difference on the fixed-epoch statistic exceeds **+0.02** and its
   95% CI excludes 0 at the Bonferroni-corrected level for the round;
2. at least **3 of the 4** split-level mean differences are positive;
3. the arm's **test** score at the best-val checkpoint, averaged over the 4 partitions, moves
   in the same direction, with a strain-bootstrap CI that does not contradict it;
4. no run in either arm sits below the 0.05 chance band (the permutation null max is 0.0506, so
   a plateau run is a training failure and its pair is reported and set aside, as the v13
   readout already does).

Report a failing arm as **"measured, not resolved, MDE 0.013"**, never as "did not help".

### Three specific things to change immediately, independent of the next round

1. **Re-quote every headline.** The strand's capability number should become the mean test
   score at the best-val checkpoint over partitions, with a strain-bootstrap CI. Today that is
   a single-partition 0.1095 / 0.1391 against B2 0.0847, not 0.1965 against 0.1040.
2. **Add a killed-run guard to the readouts.** `state == "finished" and epoch < max_epochs`
   means the wall killed it and `trainer.test` never ran; this cost the campaign its entire
   v13 test read and is invisible in W&B for offline runs.
3. **Retire the max rule from decisions.** Keep `roll_max` as a descriptive column so old
   rounds remain comparable, make the fixed-epoch window mean the decision statistic, and state
   the scoring rule in every table caption as the repo's evidence discipline already requires.

### What to do with the queued rounds 1 to 3

Do not re-litigate them, but **add the closing test read to each** (the checkpoint eval job
already exists and works), **and add seeds to v15**: as configured it is n=1 per contrast and
cannot produce a result at any effect size, while its two same-arm replicates already differ by
more than the effect it is measuring.

---

## Summary (600 words)

The 019 expression campaign has a measurement problem that is larger than every effect it has
been steering on, and it has never once checked its answers against held-out data.

**The scoring rule is biased, but it is not the main problem.** Measured on all 24 v13
per-epoch curves, the late-training residual sd around the trend is 0.0050 with integrated
autocorrelation time 3.72, giving about 1,009 independent looks in the matched 3,754-epoch
window. The raw max therefore sits +0.0126 above the local trend and the 5-epoch rolling max
+0.0071, matching the Blom prediction of +0.0161 for 1,000 iid draws. Because arms are paired
at a matched epoch, that bias largely cancels: recomputing v13's concat-minus-ref under raw
max, rolling max and a fixed-epoch window gives +0.0120, +0.0015 and +0.0072, all inside the
noise. The bias does corrupt three things: absolute capability claims, cross-budget
comparisons, and checkpoint selection, which is monitored on the raw max by construction.

**The real problem is the 155-strain validation draw.** The metric averages 6,127 gene-wise
correlations over 155 strains, and a strain bootstrap on the actual checkpoint dumps gives a
standard deviation of 0.026 on validation and 0.018 on test; the 95% interval for the reported
0.2227 is [0.173, 0.272]. A permutation null reaches 0.0506 by chance. The fixed, deterministic
bilinear baseline scores anywhere from 0.076 to 0.148 across 24 such draws (sd 0.017); the kNN
baseline from 0.046 to 0.162 (sd 0.027). Split 0, the pinned partition, is the luckiest
validation draw available: z = +1.72 for the baseline on val and z = -1.38 on test, so the
"+0.089 split-0 offset" is a draw, not a partition property. The epoch-to-epoch wobble everyone
watches is 0.0050, five times smaller than this invisible uncertainty.

**No round has ever produced a test number.** Zero `test/*` keys across 410 runs in eight
projects. v13 configured `run_test: true`, then died on the wall at epoch 4,080 of 6,000 after
111.5 hours, and the offline sync stamped all 24 dead runs `finished`. The first held-out reads
in the strand, from the checkpoint dumps written today, reverse the campaign's ordering: on
validation ref beats concat by 0.030, on test concat beats ref by 0.030, both with paired
bootstrap intervals that nearly exclude zero. Expression does beat the linear baselines on test
(0.1095 and 0.1391 against 0.0847), which is the one capability claim the evidence supports.
Proteome does not beat its kNN baseline on test (0.067 and 0.082 against 0.092).

**Power.** The paired difference sd is 0.0127 to 0.0153, so 6 to 10 pairs resolve 0.02 and 2
pairs resolve only 0.15. v13 had 9 to 11 pairs (MDE 0.027 to 0.042), v12 had 4 pairs on one
split, the mechanism round had 2, and v15 has 1 while its same-arm replicates differ by 0.0176
against a claimed effect of 0.0185. v12's t = 8.92 for H_concat came from the same rule that
gave t = 9.29 to H_state at 500 epochs, an arm that reads -0.0007 at 1,399.

**Compute.** 88.6% of the 3,754-epoch score is reached by epoch 1,000; the remaining 2,754
epochs buy +0.0197 on a statistic with draw sd 0.026, while the replicate spread grows from
0.0099 to 0.0222 between 1,000 and 9,900 epochs. The 669 card-hours v13 spent on 9 pairs would
buy 40 pairs at 1,200 epochs.

**Template.** Three arms, four split seeds, three init seeds, 1,200 fixed epochs, score as the
mean over epochs 1,000 to 1,200, paired t on 12 differences, adopt only if the effect exceeds
0.02 with a Bonferroni-corrected CI excluding zero, at least three of four partitions positive,
and the test-side difference agreeing in sign. That is 36 runs, about 12 card-days, MDE 0.011
to 0.014.

---

## Files

Scripts I wrote for this review (scratchpad, not committed; promote them into
`experiments/019-simb-multimodal/scripts/` before any number here is quoted in a note):

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/bias1.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/rules.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/paired2.py

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/work/v13_rules.csv

Repository files the numbers came from:

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/pull_round_leaderboards.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/v13_split_readout.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/train_cgt_multitask.py

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/scripts/gh_eval_ckpt_predictions.slurm

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/conf/cgt_expr_v13_split.yaml

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v13_split_readout.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v14_proteome_readout.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v15_wd_readout.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/head_round_readout.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/mech_round_readout.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/v10_grid_factorial.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/short_budget_spread.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/grid_manifest.json

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/round_leaderboards.csv

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/expression_baselines_split/

/home/michaelvolk/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective/experiments/019-simb-multimodal/results/baselines_split_fig3_proteome/

/scratch/projects/torchcell-scratch/val-predictions/

/scratch/projects/torchcell-scratch/test-predictions/

The report:

/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/f1a47094-2dfb-4a64-98b6-de1092dca661/scratchpad/review/agent-10-methodology.md
