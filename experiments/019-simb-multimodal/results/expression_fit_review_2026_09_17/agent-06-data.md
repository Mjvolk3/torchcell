# agent-06: is the target the problem? An audit of the expression label and its representation

Scope: the expression label (`fig3_core`, Kemmeren 2014 + Sameith 2015 log2 ratios), its
per-gene reliability, the metric's weighting, the rank structure of what the model actually
predicts, the split, and the strains. Everything below traces to a file I read or a number I
computed in this session; scripts are in
`/scratch/tmp/claude-1000/.../scratchpad/review/a06_stage{1,2,3,4}.py` (scratch, read-only
session, nothing written to the repo).

## 0. Verdict up front

**The target is NOT the problem, and the two target-side fixes the question proposes are both
measured dead ends.** The label is far more reliable than the framing assumes (79% of the
6,169 genes have cross-study test-retest reliability above 0.5; only 0.3% below 0.1), so the
0.20 / 0.775 headline is not diluted by thousands of dead genes. Reliability-weighting the
metric moves it by +0.006; keeping only the 28% most reliable genes moves it from 0.2227 to
0.2564. And the model's output basis is already effectively rank-4 and already 95% aligned with
the train PCA basis, so predicting in a rank-64 PCA basis reparameterizes something that is
already correct.

Three things I did find that are real, measured, and actionable:

1. **21 of the 155 seed-0 validation strains share a deleted gene with a training strain, and
   they score per-feature 0.398 against 0.178 for the 134 gene-disjoint strains** (permutation
   z = 3.5 over 400 draws of 21 disjoint strains; per-strain r 0.620 vs 0.468, t = 4.53). All
   15 Sameith doubles are in that 21. The headline 0.2227 is a mixture; the generalizing number
   is 0.178.
2. **The model's output directions are right and its per-strain coefficients are wrong.**
   Oracle coefficients in the model's OWN top-1 direction already score 0.2425, above the
   model's entire prediction (0.2227); in its own top-64 directions, 0.7270, i.e. the replicate
   ceiling. Nothing about the output parameterization is limiting.
3. **An oracle per-strain magnitude rescale is worth +0.071 (0.2227 to 0.2941), and the model
   does not currently have it** (a recalibration using only the model's own predicted norm is
   *worse* than doing nothing, 0.2110). Response magnitude is the one target-side quantity with
   real measured headroom.

Plus two defects worth fixing before any further analysis: the per-gene prediction dumps carry
a gene-name list of the wrong length, and 42 high-variance reporter genes are silently dropped
from the head.

---

## 1. The per-gene reliability distribution (question 1)

Recomputed from the raw LMDBs (`microarray_kemmeren2014`, `sm_microarray_sameith2015`), the
same 82 shared single deletions `expression_ceiling_replicate.py` uses, per-gene Pearson across
those 82 strains (`a06_stage2.py`, `a06_stage3.py`). Reproduces
`results/expression_ceiling_replicate.json` exactly (mean rho 0.6110, ceiling 0.7746).

| threshold | fraction of the 6,169 genes with rho above it |
|---|---|
| rho > 0.7 | 27.9% |
| **rho > 0.5** | **79.3%** |
| **rho > 0.3** | **97.0%** |
| **rho > 0.1** | **99.7%** |
| rho > 0.0 | 99.9% |

Percentiles of rho: p1 0.198, p5 0.350, p10 0.420, p25 0.521, p50 0.620, p75 0.710, p90 0.794,
p99 0.940.

**The premise behind the question is wrong.** "Most Kemmeren deletions change almost nothing"
is true in *magnitude* (median per-gene train sd 0.1487 log2, and the top 10% of genes carry
44% of the total variance) but it is not true in *reproducibility*. A gene with a small
across-strain spread still moves in the same direction in both studies: the bottom half of genes
by train sd has mean rho 0.5794 and mean ceiling 0.7553, against 0.7221 / 0.8416 for the top
decile. Reliability rises with variance only weakly (Pearson corr(rho, sd_train) = 0.239,
Spearman 0.289).

The estimator is not manufacturing this. Permuting the strain order of the Sameith matrix and
re-running the identical estimator 20 times gives a null ceiling of **0.1283 +/- 0.0226** with
null mean rho -0.0036. So 0.775 sits 29 null-sd above its own null. (The permutation null is
*not* zero, because `np.clip(r, 0, 1)` then `sqrt` is a positively biased functional of a
noisy r; anyone quoting a per-gene ceiling near 0.13 is quoting noise.)

### Ceiling and score restricted to responsive genes

Computed on the seed-0 `fig3_core` partition, genes stratified by percentile of per-gene sd
over the 1,244 TRAIN strains, scored against the best-val V_ref_s0 dump (`a06_stage4.py`):

| gene stratum (train-sd pct) | n | model val r | replicate ceiling | fraction realized |
|---|---|---|---|---|
| 0-50 | 3,085 | 0.1829 | 0.7553 | 0.242 |
| 50-80 | 1,850 | 0.2439 | 0.7749 | 0.315 |
| 80-90 | 617 | 0.2780 | 0.8035 | 0.346 |
| 90-95 | 309 | 0.2883 | 0.8433 | 0.342 |
| 95-99 | 246 | 0.3251 | 0.8445 | 0.385 |
| 99-100 | 62 | 0.3271 | 0.8220 | 0.398 |
| **all** | **6,169** | **0.2227** | **0.7746** | **0.288** |
| top 10% | 617 | 0.3064 | 0.8416 | 0.364 |
| top 1% | 62 | 0.3271 | 0.8220 | 0.398 |

There *is* dilution, and it is modest: restricting to the top decile raises the score by +0.084
(38% relative) and the realized fraction from 0.288 to 0.364 (26% relative). It does not
rescue the campaign. The honest statement is "the model realizes a quarter of the ceiling on
quiet genes and two fifths on loud ones", not "the headline is an artifact of dead genes".

(Note the dump's own score, 0.2227, is above the W&B rolling statistic 0.2097 for the same run
`wq8y8nd5`; the dump is the single best-val epoch, the readout's number is a 5-epoch centered
rolling mean, so they are different statistics of the same run and both are correct.)

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/wq8y8nd5

---

## 2. Is per-gene z-scoring on the train split the right normalization? (question 2)

**Mostly yes, and the failure mode the question names does not exist here.**

Mechanics, read from the code. `multitask.standardize_per_feature_target` puts `per_gene` on a
forced plain z-score fit on the TRAIN indices only
(`train_cgt_multitask.py:3013-3044`, `compute_per_feature_target_stats` at :657). The realized
stats for the seed-0 partition are in
`experiments/019-simb-multimodal/results/calmorph_train_target_norm_per_gene.json`
(n_train 1244, method zscore, 6,127 keys): per-gene std min 0.0449, p1 0.0583, median 0.1509,
max 1.3176, `dropped_features` 0, `degenerate_features` 0. So the loss reweights genes by at
most 29x in sd (860x in variance) between the quietest and loudest gene.

Three points settle it:

- **The metric is scale-free, so z-scoring cannot change the reported number at all.**
  `pearson_per_feature` is a per-gene Pearson across strains; it is invariant to any per-gene
  affine map of prediction and target. Z-scoring changes only what the pinball/CRPS loss
  optimizes. It does not "amplify noise-only genes into the metric".
- **Equal weight in the loss is the correct match to an equal-weight metric.** Without
  z-scoring, the loss would be dominated by the 62 loudest genes (train sd up to 1.32) while the
  metric averages 6,127 equally, which is exactly the mismatch the code comment at :3020 was
  written to prevent.
- **The genes it upweights are not noise.** The bottom-half-by-variance genes have reliability
  0.579. Z-scoring a gene with rho 0.58 to unit variance makes its irreducible loss floor
  0.42 of the variance, not 1.0. No gene in this panel is a noise-only target: only 0.3% have
  rho below 0.1.

### What a reliability-weighted or filtered metric actually says

Measured on the same dump, weights from the per-gene cross-study rho:

| scoring rule | value |
|---|---|
| unweighted mean over 6,127 genes (the leaderboard metric) | **0.2227** |
| weight = rho | 0.2286 |
| weight = rho^2 | 0.2335 |
| restricted to the 4,850 genes with rho > 0.5 (79%) | 0.2310 |
| restricted to the 1,700 genes with rho > 0.7 (28%) | 0.2564 |
| mean of r/ceiling over genes with ceiling > 0.3 | 0.2886 |

Reliability weighting is worth +0.006 to +0.011. Any conclusion the campaign has drawn from the
unweighted metric survives every one of these reweightings. **A reliability-weighted metric is
not a lever; it is a sanity check that has now been run and has come back null.** (The third
queued item in the plan, "reliability-weighted per-protein loss", is a different object -- it is
about the proteome, whose reliability really is low, route D 0.21 in
`proteome_ceiling_replicate.json` -- so this finding does not bear on it.)

---

## 3. The low-rank structure: what rank is the model's prediction? (question 3)

From the same dump (155 val strains x 6,127 genes, targets verified identical to the LMDB rows
to 5e-7 after fixing the column map, see section 7).

| | cumulative variance at rank k |
|---|---|
| **prediction** | k=1 0.416, k=2 0.533, k=4 0.704, k=8 0.824, k=16 0.923, k=32 0.984 |
| **target** | k=1 0.241, k=2 0.312, k=4 0.421, k=8 0.547, k=16 0.679, k=32 0.808 |

Participation-ratio effective rank: **prediction 4.80**, target 13.01 (entropy rank 10.4 vs
38.0). Truncating the model's own prediction to rank k and rescoring:

| rank | 1 | 2 | 4 | 8 | 16 | 32 | full |
|---|---|---|---|---|---|---|---|
| score | 0.1303 | 0.1646 | **0.2263** | 0.2242 | 0.2268 | 0.2225 | 0.2227 |

**The model is already a rank-4 predictor.** Rank-4 truncation costs nothing (it is fractionally
*better* than the full prediction). So the "6,000 independent per-gene outputs" framing does not
describe what the head is doing, and constraining it to a rank-64 PCA basis removes degrees of
freedom the model is not using.

The decisive measurement is the basis-versus-coefficient split. Replace the model's coefficients
with ORACLE coefficients (project the val residuals onto a basis) and rescore:

| k | oracle coefficients in the MODEL's own basis | oracle coefficients in the TRAIN PCA basis |
|---|---|---|
| 1 | **0.2425** | 0.2962 |
| 2 | 0.3520 | 0.3978 |
| 4 | 0.4651 | 0.5051 |
| 8 | 0.5351 | 0.5666 |
| 16 | 0.6161 | 0.6423 |
| 32 | 0.6969 | 0.7146 |
| 64 | **0.7270** | 0.7755 |

(The train-basis column reproduces `results/lowrank_output_ceiling.json` to within 0.03 at every
rank; the small gap is that I fit mu and the basis on the 1,244-strain train split rather than
the 1,327-strain fit split that script uses.)

Read this line by line. Perfect coefficients on **one** of the model's own learned directions
outscore the model's entire 6,127-dimensional prediction. Perfect coefficients on 64 of its own
directions reach 0.727, i.e. 94% of the 0.775 replicate ceiling. And the model's directions live
inside the train PCA subspace already: the energy of the model's top-1 right singular vector
inside the train top-4 is **0.945**, top-4 inside train top-16 is 0.856, top-16 inside train
top-64 is 0.901.

**Conclusion: the output basis is solved and the coefficient map from genotype to response is
not.** A PCA-target arm is predicted (Hypothesis, but the evidence is strong) to return a null,
and a null there would be uninterpretable because the capacity it removes was never in use.

### The one thing the rank analysis does hand you

The prediction's total spread is 0.334 of the target's (`pred_sd_ratio`), consistent with the
known uncalibration (`nmse_at_peak` 1.030 for `wq8y8nd5`). Per-gene Pearson is invariant to a
global rescale (verified: x1/x2/x3/x4 all give 0.2227), but **not** to a per-strain rescale:

| treatment | score |
|---|---|
| unit-normalize every predicted profile (direction only) | 0.1902 |
| as predicted | 0.2227 |
| oracle per-strain scale | **0.2941** |
| linear-in-(predicted norm) rescale | 0.2110 |

So the model's own magnitude variation is already worth +0.032 over equal magnitudes, and a
perfect response-magnitude predictor would be worth a further **+0.071 (+32% relative)**. The
model does not have that signal latent: corr(||pred||, ||target||) is 0.375 Pearson / 0.373
Spearman, and a calibration that uses only the predicted norm makes things worse. On the
gene-disjoint 134 strains the oracle rescale is worth +0.082 (0.1779 to 0.2595).

---

## 4. Split seed 0 (question 4)

The v13 round's per-split means at the matched epoch 3,754
(`results/v13_split_readout.json`): s0 **0.1970** (n=7), s1 0.1647 (n=4), s2 0.1410 (n=4), s3
0.1387 (n=4). sd between partitions 0.0271, pooled sd within a partition 0.0092. The split is
about 9x the model seed.

**What is it about that draw?** I can answer this without the model, because
`results/expression_baselines_split/seed{0..11}.json` holds B0-B3 on **twelve** split seeds
(only four of which ever got GPU arms). For the B2 bilinear ridge on ProtT5:

| | mean over 12 seeds | sd | seed 0 | z |
|---|---|---|---|---|
| val | 0.1035 | 0.0183 | 0.1349 | **+1.72** |
| test | 0.1078 | 0.0167 | 0.0847 | **-1.38** |

B3 kNN on ProtT5 shows the same signature (val z +0.41, test z **-2.12**). So seed 0's
*validation* draw is an easy 155 strains and its *test* draw is a hard one, for a
parameter-free baseline that has no interaction with the model. The +0.089 is at least in part
a property of the partition, not of the architecture.

Two further facts from the same table that matter more than the seed-0 question:

- **The val-to-val spread of a fixed method is sd 0.018-0.020 on 155 strains** (B2 ranges 0.076
  to 0.135 across the 12 seeds, a factor of 1.8). Several headline architectural effects the
  campaign has banked, +0.041 for the per-gene head and +0.020 for graph routing, are inside
  one sd of this. They were measured on a single val draw.
- **Val is not predictive of test at this size.** Seed 0 is the best val draw of twelve and a
  below-average test draw.

There is also a model-seed hazard that K-fold will not fix: of the four `V_ref_s0` runs, one
collapsed outright (`825on260`, matched 0.0260, against 0.2070 / 0.1934 / 0.2127 for its
siblings), and `v13_split_readout.json` has to carry a `concat_minus_ref_excluding_plateau`
variant because that single run flips the sign of the readout's headline comparison (mean diff
+0.0181, t 1.33 with it; +0.0056, t 0.94 without).

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/825on260

https://wandb.ai/zhao-group/torchcell_019_expr_v13/runs/8i75d8h1

**Should selection move to K-fold? Yes, but with a specific shape.** `make_split_indices.py`'s
note is explicit that the four draws are "random over records within each index key (80/10/10),
not disjoint folds, so the four draws measure the spread of the absolute number, not a K-fold
estimate". Because the baselines are already computed on twelve partitions, the cheap version
is: define 10 disjoint folds once, score every arm as the mean over folds, and require an
effect to clear the *between-fold* sd, not the within-partition seed sd. See arm A1.

---

## 5. Which strains are hard (question 5)

`stratified_responsiveness_eval.md` only ever ran its DATA-level pass; its own note says the
model-level analysis "is not implemented in the landed script and no model-level number was
measured". I ran the model-level pass here.

- Per-instance (per-strain, across genes) Pearson: mean **0.4887**, median 0.5415. That is more
  than twice the per-feature number, and it is the axis on which the model looks competent.
- **Val Pearson is NOT driven by the most responsive strains.** corr(per-strain r, per-strain
  target sd) = **-0.140** (Spearman -0.132). The 39 most responsive strains average r 0.4402
  against 0.5051 for the rest. The model does slightly *worse* on the loudest strains.
- No single strain carries the metric. Leave-one-strain-out over all 155: sd of the metric 0.0022,
  largest single drop 0.0151 (YGR157W / CHO2, per-strain r 0.814), next 0.0097. Dropping the 8
  most influential would cost about 0.04 of 0.2227.
- The per-*feature* metric does rise when computed on responsive strains only, because the
  correlation needs across-strain spread: 155 strains 0.2227, top 78 0.2459, top 40 0.2700,
  top 20 0.2976. That is a property of the estimator, not evidence the model is better there
  (the per-strain numbers say the opposite).

This also rules the responsiveness axis out as a curriculum lever on its own terms: the
data-level pass already showed the random split is responsiveness-balanced to within 3 points in
every seed, and "non-responsive" strains are 60% as dispersed, not flat.

---

## 6. Sameith doubles, and the split-hygiene finding (question 6)

`fig3_core` is 1,554 expression records: the 1,484 Kemmeren singles (with the 82 Sameith singles
averaged into them by `MeanExperimentDeduplicator`, they are the *same* genotypes) plus the 72
Sameith double deletions. Seed 0 puts 54 doubles in train, 15 in val, 3 in test
(`expression_baselines_split/seed0.json::split.n_double_deletion`).

Measured on the val dump:

| subset | n | per-feature | per-strain r | target sd |
|---|---|---|---|---|
| all | 155 | 0.2227 | 0.4887 | 0.196 |
| singles only | 140 | 0.1797 | 0.4780 | 0.192 |
| **doubles only** | **15** | **0.4134** | 0.5890 | 0.248 |
| 15 random singles (50 draws) | 15 | 0.1794 +/- 0.0838 | | |

So the doubles help, by a lot, and the reason is not that they are doubles.

**All 15 val doubles have at least one of their two deleted genes also deleted in a training
strain. Only 6 of the 140 val singles do.** Splitting the 155 val strains on that criterion
instead:

| subset | n | per-feature | per-strain r |
|---|---|---|---|
| shares a deleted gene with train | 21 | **0.3977** | 0.6199 |
| gene-disjoint from train | 134 | **0.1779** | 0.4682 |

Significance: against 400 random 21-strain subsets of the 134 disjoint strains (mean 0.1783,
sd 0.0623) the overlap set is z = 3.52, p_emp = 0/400; on the n-independent per-strain axis,
0.6199 vs 0.4682, se 0.0335, t = 4.53.

I checked and rejected the obvious alternative explanation: this is not a Sameith-vs-Kemmeren
batch offset. Removing each group's own mean profile before scoring leaves both subset scores
unchanged (doubles 0.4134, singles 0.1797), and removes only 0.011 from the all-strain number.
The advantage is per-strain, so it is a lookup: for a double, the model has seen one of the two
deleted genes in training and can carry that strain's profile across.

This is the same failure mode as memory `010-review-round-1-corrections` ("pair-disjoint
collapses the ladder 0.400 -> 0.127"). It is also exactly what `knn_embedding_probe.md` predicts:
"only 4.8% of validation genes are ever perturbed in training, because every strain is a single
deletion, so splitting by strain IS splitting by gene". True for the singles; false for the
doubles, and nothing in the split enforces it. 13 of the 155 test strains have the same overlap.

**Do the doubles help or hurt?** They help the number and they hurt its meaning. As training
data, 54 doubles among 1,244 strains are the only records in the panel that carry any
combinatorial signal, so I would keep them. As validation data they should be scored separately,
because an arm that improves gene-lookup and an arm that improves genotype-to-response
generalization currently land in the same 0.22.

---

## 7. Two defects found while doing this

**(a) The per-gene prediction dumps cannot be joined to gene names.** Every dump under
`$DATA_ROOT/{val,test}-predictions/` writes `head_keys["per_gene"]` with **6,169** entries while
every `pred`/`target` vector has **6,127**. The writer at `train_cgt_multitask.py:2359` takes
`task.head_align[h]["keys"]`, which is the raw key vocabulary *before* `keep_mask` is applied
(:599, :609-617); the vectors are post-mask. Consequence:
`variance_stratified_pearson.py::_load_dump` raises `IndexError` on the first column index past
6,126, which is why `results/variance_stratified_pearson.json` does not exist and why
`notes/experiments.019-simb-multimodal.scripts.variance_stratified_pearson.md` is an empty
frontmatter stub. The one-line fix is
`[k for k, f in zip(keys, align["keep_mask"]) if f]` (with the `keep_mask is None` case
falling through to `keys`). Any downstream join that "worked" by truncating or by
`dump_keys.index(k)` was silently misaligned by up to 42 positions. I recovered the true map
here by matching dump targets to LMDB val columns (monotone, injective, max abs diff 4.8e-7)
and every number in this report uses it.

**(b) 42 reporter genes are dropped from the head, and they are the loud ones.** The dropped set
is `SNR10` plus 41 Ty-element / dubious ORFs (`YAR009C`, `YDR210C-C`, `YDR210W-B`, `YGR109W-A/B`,
`YPR158C-D`, ...). Their median train sd is **0.2949** against 0.1486 for the panel. Dropping
cross-hybridizing repeat probes is defensible, but it is happening implicitly (they are absent
from the model's gene set), it is not recorded (`dropped_features` is `[]` in the norm JSON), and
it makes the metric's denominator 6,127 while the published ceiling's is 6,169.

---

## 8. Proposed arms, each with cost and the measured evidence for it

Ordered by (measured evidence) x (size of the headroom it addresses). Everything here comes
*after* the three queued items; none of it re-litigates them.

### A1. Gene-disjoint evaluation, and a fixed 10-fold selection protocol. CPU only, ~2 h.
**Evidence:** 21/155 val strains score 0.398 vs 0.178 (t 4.53, p_emp < 0.0025); 12-seed baseline
val sd 0.0183 with a 1.8x range; seed 0 is +1.72 sd on val and -1.38 sd on test for a
parameter-free baseline; `make_split_indices.md` states the four draws are not folds.
**What it is:** (i) report `val/expression/pearson_per_feature` additionally on the gene-disjoint
subset, as a logged metric, not a post hoc script; (ii) build 10 disjoint folds once with
`make_split_indices.py`, hold out by *deleted-gene set* rather than by record so a double whose
partner gene is in train cannot land in val, and make the selection statistic the fold mean with
the between-fold sd as the noise bar. **Why first:** it is free, it is the only thing that makes
every later arm interpretable, and without it the campaign cannot distinguish a +0.02
architecture effect from a +0.02 draw.
**Risk:** the gene-disjoint number will be lower than the published one (0.178 vs 0.223 on seed
0). That is a correction, not a regression.

### A2. A per-strain response-magnitude head. 1 GPU arm x 4 seeds, ~1 day at 36 epochs/h.
**Evidence:** oracle per-strain rescale 0.2227 -> **0.2941** (+0.071, +32%); on the gene-disjoint
134, 0.1779 -> 0.2595 (+0.082). The model does not have it latent (corr(||pred||, ||target||)
0.375; a calibrator built from the predicted norm alone *loses* 0.012). `pred_sd_ratio` 0.334
and `nmse_at_peak` 1.030 are the same deficiency seen from the loss side.
**What it is:** a scalar head predicting `log ||y_i - mu||` from the genotype, trained with its
own loss, and used multiplicatively on the normalized profile at readout:
`y_hat_i = mu + m_hat_i * (r_hat_i / ||r_hat_i||)`. This is the arm the +0.071 oracle bounds.
**Why it is a target-side arm:** it changes the factorization of the target into direction x
magnitude, which is where the metric's per-strain sensitivity actually lives.
**Caveat (untested):** whether magnitude is predictable at all from the deletion genotype is
open. A cheap CPU pre-check exists: regress `||y_i - mu||` on the deleted gene's ProtT5 embedding
with ridge, on the same partition, and report held-out Spearman. If that is near zero, do not
spend the GPU.

### A3. Score the loud-gene stratum as a second reported metric. CPU only, ~1 h.
**Evidence:** the stratum table in section 1 (realized fraction 0.242 in the quiet half vs 0.398
in the top percentile); `stratified_responsiveness_eval.md`'s data-level finding that the bottom
half of genes carries 15-17% of the variance.
**What it is:** log `pearson_per_feature` restricted to the top-decile-by-train-sd genes
alongside the full metric. Not a loss change and not a filter on training data, just a second
readout, so it costs nothing and it separates "the model got better at the genes that move" from
"the model got better at the mean".
**Do NOT** turn this into a training filter: the quiet genes have reliability 0.579 and carry
real signal; dropping them would discard 3,085 legitimate targets to buy +0.08 of a
scale-free metric.

### A4. Reliability-weighted expression loss: SKIP for expression, keep for the proteome.
**Evidence:** weight = rho gives 0.2286 vs 0.2227; weight = rho^2 gives 0.2335; rho > 0.5 subset
0.2310. The reliability spread on expression is 0.58 to 0.72 between the extreme strata, so the
weights span a factor of ~1.25 and cannot move anything. Contrast the proteome, where route D
reliability is 0.21 (`proteome_ceiling_replicate.json`) and the weights would span a real range.
**Cost of skipping:** zero. **Cost of running it on expression:** one GPU arm for a predicted
+0.01, which is half the split noise.

### A5. Rank-64 PCA output basis: SKIP, and say why in the document.
**Evidence:** the model's prediction already has participation-ratio effective rank 4.80 and
rank-4 truncation of its own output loses nothing (0.2263 vs 0.2227); its top-1 direction sits
94.5% inside the train top-4 subspace; oracle coefficients in its *own* top-1 direction beat its
whole prediction. The bottleneck is the coefficient map, not the basis.
**What to do instead:** if the low-rank idea is to be tested at all, test it as a *coefficient*
arm, not a basis arm: freeze the train PCA basis at rank 64 and train only the 64 coefficients
from the genotype, which is a 64-output regression problem and trains in a fraction of the time.
That arm has a stated ceiling (0.7755 at rank 64, `lowrank_output_ceiling.json`) and a stated
floor (the model's current 0.2227), and it isolates exactly the map that is failing. ~1 GPU
arm, or arguably CPU with a ridge/MLP on the embeddings.

### A6. Fix the dump writer and rerun `variance_stratified_pearson.py`. ~1 h, CPU.
**Evidence:** section 7(a); the empty result JSON and the empty note.
Also record the 42 dropped genes explicitly, and state the metric's denominator (6,127) next to
the ceiling's (6,169) wherever both appear.

### A7. Target transform (Yeo-Johnson or rank-Gaussian per gene): low priority.
**Evidence, such as it is:** the panel is heavy-tailed (1.1% of responsive-strain measurements
exceed |y| > 1 against 0.19% for non-responsive, `stratified_responsiveness_eval.md`), so a
per-gene power transform would stabilize the pinball loss. **Against it:** the metric is
Pearson on the *raw* scale, so a monotone per-gene transform changes the loss and then has to be
inverted for scoring, and the transform is already implemented and used for morphology
(`vector_norm_method: yeo_johnson`). Hypothesis (untested): worth at most a few thousandths,
and it costs an inversion path. Park it.

### Not proposed, and why
- **Strain-level curriculum on responsiveness.** The split is already responsiveness-balanced to
  within 3 points in every seed, non-responsive strains are 60% as dispersed rather than flat,
  and the model is *worse* on the loudest strains (corr -0.140). There is no gradient to ride.
- **Dropping the Sameith doubles.** They are 54 of 1,244 training strains and the only
  combinatorial records in the panel. Fix the split (A1), do not discard the data.
- **Pooling Nadal-Ribelles.** Settled and measured: no within-study replication
  (memory `nadal-ribelles-assignment-impure`).

---

## 9. What I did not measure, stated as such

- Every model-side number comes from **one** dump, the best-val checkpoint of `V_ref_s0` seed 0
  (`wq8y8nd5`, split seed 0). The other two v13 expression dumps named in
  `variance_stratified_pearson.py::RUNS` (`V_concat_s0`, `V_ref_s2`) are **not present** in
  `$DATA_ROOT/val-predictions/`; only four dumps exist there, three of which are v14 proteome.
  So the rank, magnitude and gene-overlap findings are n = 1 run on n = 1 split. The gene-overlap
  finding is a property of the *partition* and would replicate for any model; the rank and
  magnitude findings are properties of this checkpoint and should be confirmed on a second dump
  before anything is built on them.
- The oracle numbers (coefficient oracle, magnitude oracle) use validation targets. They are
  upper bounds on what a perfect predictor of that quantity would score, not achievable scores.
- The cross-study reliability rests on Kemmeren and Sameith having independent noise. Both are
  Holstege-lab microarray with a shared normalization lineage; `expression_ceiling_replicate.md`
  flags this as assumption A1, unverified, with the error direction "ceiling too high". My
  permutation null bounds the *estimator's* noise (0.128) but not that shared-bias term.
- I did not test whether the low-variance genes' reproducible signal is a single global program
  (a slow-growth / ESR common mode) rather than deletion-specific. If it is, the reliability is
  real but the predictable part of it is low-dimensional, which would reinforce section 3's
  conclusion rather than overturn it. The test is cheap: recompute rho after projecting out the
  top 1-5 strain-space components from both studies.
- No training run, no GPU job, nothing written to the repo.

---

## Summary (600 words)

The expression label is not the bottleneck. Recomputing the cross-study test-retest reliability
gene by gene on the 82 shared Kemmeren/Sameith deletions gives 79.3% of the 6,169 genes above
rho 0.5, 97.0% above 0.3 and 99.7% above 0.1 (median 0.620, p10 0.420). The permutation null of
that same estimator is 0.128 +/- 0.023, so 0.775 is 29 null-sd above chance. Quiet genes are
quiet, not unreliable: the bottom half by train variance still has mean rho 0.579 against 0.722
for the top decile, and reliability tracks variance only weakly (Spearman 0.29). The headline
"0.20 of a 0.775 ceiling" is therefore not an artifact of thousands of dead genes. Restricting
to the top decile moves the model from 0.2227 to 0.3064 and the realized fraction from 0.288 to
0.364, real but not decisive.

Per-gene z-scoring is the right normalization and cannot affect the metric at all, which is
scale-free per gene; it only aligns the loss with the metric's equal weighting. Reliability
weighting is now measured and null: weight rho gives 0.2286, rho^2 0.2335, the rho > 0.5 subset
0.2310, against 0.2227 unweighted.

The low-rank question resolves against the PCA-target arm. The model's val prediction has
participation-ratio effective rank 4.8 (target 13.0), and truncating its own prediction to rank
4 costs nothing (0.2263 vs 0.2227). Its directions are already right: the top-1 predicted
singular vector sits 94.5% inside the train top-4 subspace, and oracle coefficients in the
model's own top-1 direction score 0.2425, beating its entire prediction; in its own top-64, 0.727,
essentially the replicate ceiling. The failure is the genotype-to-coefficient map, not the output
basis, so a rank-64 PCA head reparameterizes something already solved and a null there would be
uninterpretable.

What does have headroom is per-strain magnitude. Predictions carry 0.334 of the target spread;
an oracle per-strain rescale lifts 0.2227 to 0.2941 (+0.071, +32%), and to 0.2595 from 0.1779 on
the gene-disjoint strains. The model does not have this latent (corr(||pred||,||target||) 0.375;
a calibrator from its own norm scores 0.2110, worse than nothing).

The split finding is the one that changes how results should be read. Of the 155 seed-0
validation strains, 21 share a deleted gene with a training strain, and they score per-feature
0.398 against 0.178 for the 134 gene-disjoint strains (permutation z 3.5; per-strain r 0.620 vs
0.468, t 4.53). All 15 Sameith doubles are in that 21, and it is not a study batch offset
(removing group means changes nothing). Seed 0 is also an easy val draw independently of the
model: across twelve baseline partitions the B2 ridge is +1.72 sd on seed-0 val and -1.38 sd on
seed-0 test, with a val-to-val sd of 0.018 that swallows several banked architectural effects.
Val Pearson is not carried by responsive strains (corr of per-strain r with strain sd is -0.140)
nor by any single strain (max leave-one-out drop 0.015).

Two defects: the per-gene dumps write a 6,169-name key list against 6,127-long vectors (the
pre-`keep_mask` vocabulary at `train_cgt_multitask.py:2359`), which is why
`variance_stratified_pearson.json` does not exist and its note is empty; and 42 reporter genes,
median train sd 0.295 against 0.149, are silently dropped from the head.

Arms, in order: gene-disjoint evaluation plus fixed 10-fold selection (CPU, ~2 h); a per-strain
magnitude head (1 GPU arm, oracle-bounded at +0.071, with a CPU pre-check); a loud-gene stratum
as a second logged metric (CPU); fix the dump writer (CPU). Skip reliability-weighted expression
loss and the PCA output basis; if low rank is to be tested, test it as a 64-coefficient
regression against the frozen train basis.
