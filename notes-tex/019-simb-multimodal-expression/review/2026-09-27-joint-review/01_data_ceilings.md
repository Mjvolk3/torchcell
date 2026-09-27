# Reviewer 1: the data. What the proteome and expression panels permit, and where a shared trunk could gain

Scope: the `fig3_proteome` store, the two labels in it, their reliability, and the ceiling on
cross-modal synergy. Every number below is either read out of a committed results file (cited
by path) or computed by me in this session from the built LMDB; scripts are in
`/scratch/tmp/claude-1000/.../scratchpad/joint_review/work/` (`extract.py`, `analyze.py`,
`analyze2.py`, `analyze3.py`, `analyze5.py`, `analyze6.py`, outputs `part1..6.json`). I read
the LMDB and W&B only; nothing in the repo was touched.

## 1. What is actually in the store (verified by reading it)

`build_fig3_proteome.py` unions Kemmeren 2014, Sameith 2015 Sm and Dm, and Messner 2023
(`experiments/019-simb-multimodal/queries/fig3_proteome.cql`). I enumerated all 4,681 records
of `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig3_proteome/processed/lmdb`
and counted the experiments per record:

| record type | count |
|---|---|
| proteome only | 3,127 |
| **both labels** | **1,349** |
| expression only | 205 (of which 72 are the Sameith double deletions) |
| total | 4,681 |

This reproduces `results/fig3_proteome_build_census.json` exactly (4,476 `protein_abundance`,
1,554 `expression_log2_ratio`, 1,349 with both). Provenance inside the records (my count):
the proteome side is 4,332 single Messner strains plus 140 two-strain and 4 three-strain
averages (the duplicated-origin ORFs), and the expression side is 1,400 Kemmeren-only records,
82 Kemmeren + Sameith-Sm averages, and 72 Sameith-Dm doubles. **All 1,349 both-label records
are single deletions**, since Messner is singles only.

Splits (`data_module_cache/index_details_seed_{0..3}.json`, read directly). Proteome
3,581 / 448 / 447 and expression 1,244 / 155 / 155 on every seed. Records carrying BOTH labels:

| split seed | train | val | test |
|---|---|---|---|
| 0 | 1,103 | 125 | 121 |
| 1 | 1,099 | 126 | 124 |
| 2 | 1,102 | 119 | 128 |
| 3 | 1,108 | 122 | 119 |

This confirms the addendum's "about 1,099" and adds the val/test counts: **a paired
cross-modal contrast on held-out strains has only 119 to 128 strains to work with per seed.**

Target construction. The proteome loader
(`torchcell/datasets/scerevisiae/messner2023.py`) stores the LINEAR batch-corrected MaxLFQ
quantity plus a 388-replicate HIS3 reference; `ProteinAbundanceLog2RatioConverter`
(`torchcell/datamodels/protein_abundance_log2_ratio_conversion.py`) rewrites it at build time
to `log2(strain / HIS3 mean)` over a FIXED 1,850-protein union, `NaN` and `n_replicates 0`
where the strain did not quantify, reference exactly 0, reference SE carried by the delta
method. The census's first record confirms it: `measurement_type =
log2_ratio_to_reference(swath_ms_maxlfq_batch_corrected_quantity)`, 1,850 keys, 1,819
measured, `reference_all_zero: true`. Expression is stored as `log2(deletion / WT)` natively.
Both heads then get a plain per-feature z-score fitted on TRAIN only
(`standardize_per_feature_target: [per_gene, per_gene_aux]`, `train_cgt_multitask.py` l.812,
`np.nanmean`/`np.nanstd` so NaN does not enter the stats).

Missingness and NaN scoring. Proteome missing fraction 2.58 / 2.59 / 2.51 percent on
train/val/test (`results/baselines_split_fig3_proteome/seed0.json`); 2.74 percent on the 1,349
shared rows (mine); 1,657 of 1,850 proteins are quantified in at least 90 percent of the
proteome rows. Expression is fully dense on the shared rows (0 NaN, and 0 cells with
`|log2| >= 20`, so no sentinels survive into this build). NaN is handled honestly at all three
places: the loss ANDs `isfinite(target)` into the feature mask and zeroes the target
(`MaskedMultitaskLoss.forward`), the metric drops to `_per_feature_pearson_sparse` which
correlates each column over its own finite pairs and needs at least 3 of them, and the reveal
schedule assigns `+inf` to NaN entries so an unmeasured protein can never be teacher-forced.

## 2. Label reliability, and the cross-check the coordinator asked for

The literature helper's Messner per-protein ceiling (median achievable r 0.67, IQR 0.50 to
0.79) is **numerically identical to what is already in the repo**:
`results/proteome_ceiling_replicate.json` -> `route_w_his3_replicate` gives median ceiling
0.6651, IQR [0.5061, 0.7907], mean reliability 0.4336, from 388 HIS3 columns against 4,699 KO
columns. A second WT-CV route in `results/expression_ceiling_all.json`
(`messner2023.decomposition_wt_replicates`) gives median 0.6983, IQR [0.610, 0.794].

Cheap cross-check against the built store (mine): the median per-protein SD of the *stored
log2 ratio* over the build's 4,476 proteome records is 0.2354, against
`median_sd_log2_ko = 0.2367` in that JSON (the small gap is the build averaging 144 duplicated
ORFs). Feeding the build's own SDs through route W's decomposition with `sd_wt = 0.1659` gives
median reliability 0.503 and median ceiling 0.710. So the helper's number survives contact
with the build. Two caveats the number does not carry:

- **28 percent of the 1,850 proteins have a KO-side SD at or below the HIS3 replicate SD**,
  i.e. clipped reliability 0: no detectable knockout-driven variance. Only 50 percent reach
  reliability 0.5. The per-feature z-score inflates every one of those columns to unit
  variance, so both the pinball loss and the reported `pearson_per_feature` average spend
  about a quarter of their budget on columns with no signal to find.
- **The WT-CV route is the optimistic one.** The route that uses an *independently
  constructed strain of the same deletion* (`route_d_duplicate_strains`, 149 pairs over 145
  ORFs, 98.7 percent on different plates) gives mean ceiling 0.4169, median 0.4437, and a
  per-strain median r of 0.0283. For a model that must predict a deletion's proteome from its
  genotype, that is the right ceiling, and the build cannot even reproduce it because it
  averages the duplicate strains away.

Expression side, for contrast: cross-study test-retest on the 82 Kemmeren/Sameith shared
deletions gives ceiling 0.7746 with per-feature reliability 0.611 and 98.7 percent of genes
above reliability 0.5 (`results/expression_ceiling_replicate.json`). So **expression is the
reliable label and the proteome is the noisy one**, by a factor of about two in achievable r
(0.775 vs 0.417 on the honest routes).

Structure: rank 32 of a train-basis SVD captures 67.2 percent of expression variance and only
38.8 percent of proteome variance (mine, on the build's matrices). The proteome is the harder,
flatter, noisier target.

## 3. Cross-modal agreement, measured on the build

All on the 1,349 both-label records over the 1,832 gene keys the two panels share
(`part1.json`). I reproduce the EDA (`notes/experiments.019-simb-multimodal.proteome-expression-eda.md`)
on the build's own log2 values rather than the DB pull:

| statistic | this session | EDA (DB pull) |
|---|---|---|
| per-gene r across strains, median | 0.075 (IQR 0.033 to 0.135; 9.8 % negative; max 0.510) | 0.079 |
| per-strain r across genes, median | 0.037 z-scored / 0.028 raw log2 (33.7 % negative) | 0.038 |

**One EDA claim does not hold and should be corrected.** The EDA says "the few high-r genes
are dominated by the strain in which that gene is itself deleted". Only 343 of the 1,349
shared strains have their own deleted gene among the 1,832 shared keys, and blanking every one
of those entries moves the per-gene median from 0.0748 to 0.0749 and the per-strain median
from 0.0365 to 0.0357. Of the top 12 genes by r, exactly one (YOR007C, 0.359 -> 0.190) is
carried by its self-KO point; YJL026W, the EDA's named example, has no self-KO strain in the
shared set at all, so its r = 0.510 is not a self-KO artifact. The low agreement is a property
of the panels, not of a handful of self-deletion points.

## 4. The cross-modal channel is large, and orthogonal to the genotype channel

This is the measurement that decides the question, and it was not in the repo. Out-of-fold
ridge, 5 folds on the 1,349 shared strains, penalty tuned on an inner split, scored with the
training metric `per_feature_pearson` (`part2.json`; a fixed-penalty rerun with a
strain-permuted null is `part6.json`):

| source -> target | per-feature Pearson | uniform-average R^2 |
|---|---|---|
| observed proteome -> expression | **0.226** | 0.058 |
| genotype (ProtT5 of the deleted gene) -> expression | 0.101 | 0.009 |
| genotype + observed proteome -> expression | 0.236 | 0.061 |
| observed expression -> proteome (1,657 dense) | **0.218** | 0.046 |
| genotype -> proteome | 0.056 | -0.003 |
| genotype + observed expression -> proteome | 0.214 | 0.040 |

Strain-permuted null: -0.014 (proteome -> expression) and -0.009 (expression -> proteome), so
both reads are far outside chance. Calibration of my pipeline against the committed baselines:
on the full label sets my genotype ridge reads 0.110 for expression and 0.076 for the dense
proteome (`part3.json`), against `expression_baselines_split/summary.json` B2/prot_T5 val mean
0.1035 over 12 draws and `baselines_split_fig3_proteome` B3/prot_T5 val 0.071 to 0.083. Same
place.

Two consequences, and they point in opposite directions:

- **Observing the other modality is worth two to four times more than knowing the genotype.**
  0.226 from the proteome against 0.101 from the genotype on the same strains; 0.218 against
  0.056 the other way. The observed proteome predicts expression about as well as the best
  trained CGT expression arm does (0.21 on split 0), from a plain ridge.
- **Almost none of that is reachable from the genotype.** Adding the genotype to the proteome
  buys +0.009 on expression and -0.004 on the proteome; adding the proteome to the genotype
  buys +0.135. The two channels barely overlap. This is the same shape as
  `results/conditioning_gain_after_genotype.json`, where 97.5 to 100.6 percent of the
  within-modality conditioning gain survived removing a genotype predictor.

**A shared trunk sees only the genotype at validation time.** So the large cross-modal channel
is not available to it, and joint training cannot collect it. Whatever a shared trunk gains has
to come through the representation, which is what section 5 measures.

## 5. The linear analogue of a shared trunk: transfer is real, small, and one-directional

A shared trunk helps a head when the genotype directions the other label's task uses are also
useful for this head. Linear version (`part3.json`): fit ridge genotype -> label A on A's rows
excluding the held-out fold, take the top r left singular vectors of the coefficient matrix
(the r directions of ProtT5 space that task uses), then fit genotype -> label B **restricted to
that r-dimensional projection**, out of fold. Controls: B's own top-r subspace (the ceiling for
a rank-r representation) and a random orthonormal r-dim subspace (the floor).

Target = expression (full-dimensional baseline 0.110):

| r | from the proteome task | expression's own | random |
|---|---|---|---|
| 8 | 0.044 | 0.110 | 0.030 |
| 32 | 0.071 | 0.111 | 0.054 |
| 128 | 0.105 | 0.110 | 0.098 |
| 256 | 0.112 | 0.110 | 0.105 |

Target = proteome (full-dimensional baseline 0.076):

| r | from the expression task | proteome's own | random |
|---|---|---|---|
| 8 | 0.056 | 0.087 | 0.015 |
| 16 | 0.071 | 0.081 | 0.033 |
| 128 | 0.081 | 0.070 | 0.068 |
| 256 | 0.082 | 0.062 | 0.079 |

Read of the two tables:

- **Expression has nothing to learn from the proteome's genotype subspace.** Expression's own
  rank-8 subspace already recovers its full-dimensional score (0.110 = 0.110), so 8 directions
  are all the task uses. The proteome's rank-8 subspace delivers 0.044, barely above the random
  floor of 0.030, and never exceeds the baseline at any rank. The transfer ceiling in this
  direction is at or below zero.
- **The proteome does gain from the expression task's subspace.** At r = 128 and 256 the
  expression-derived subspace (0.081, 0.082) beats both the proteome's own top-r subspace
  (0.070, 0.062) and the full 1024-d baseline (0.076), by +0.005 to +0.006.

Subspace overlap, fold 0: mean cosine of the principal angles 0.26 at r = 8, 0.33 at r = 32,
0.55 at r = 128, with `sum cos^2 / r` of 0.104 / 0.149 / 0.371 against a random expectation of
r/1024 = 0.008 / 0.031 / 0.125. So the two tasks share 3 to 13 times more genotype directions
than chance, and 63 to 90 percent of each subspace stays private.

Caveats, stated plainly. This bounds *linear* transfer through a ProtT5 representation of the
deleted gene; the CGT is nonlinear and reads a graph, so the bound is not a proof about the
CGT. The +0.005 to +0.006 is a point estimate from one fold draw, not an effect with a
confidence interval (a 5-seed paired repeat is running; see the note at the end of this
section).

The other channel a shared trunk has is extra rows. Learning curves, fixed penalty, one held-out
20 percent per label (`part6.json`):

| genotype -> expression | 155: 0.064 | 311: 0.093 | 622: 0.113 | 933: 0.129 | 1,244: 0.134 |
|---|---|---|---|---|---|
| genotype -> proteome | 223: 0.037 | 447: 0.054 | 895: 0.066 | 1,790: 0.083 | 3,581: 0.092 |

Both curves are still rising, so both heads are data-limited, but the rows joint training adds
are the wrong rows for the wrong head: **the proteome head gains 205 extra records (+4.6
percent, worth about +0.001 by this curve) and the expression head gains 3,127 records that
carry a different label**, whose transferable content is exactly what section 5's first table
measured as nil.

## 6. Where the current targets and losses hide or fake synergy

Four specific mechanisms, each traced to a line of `train_cgt_multitask.py` or a config key.

1. **The reveal schedule trains the expression head as a cross-modal imputer and scores it
   with the imputation switched off.** The whole v11 to v16 lineage inherits
   `observed_labels.enabled: true` and `mask_schedule: [0, 10, 100, 1000]` from
   `conf/cgt_expr_v9_mask.yaml`. In v16 `mask_head: per_gene` is the PROTEOME head, so on
   three of four training steps true proteome values are teacher-forced into the trunk
   (`_masked_step`, l.1260-1276: train picks ONE random k per step). The aux expression head's
   loss is inside the same `loss_k` at every k, so the expression head is trained, most of the
   time, with part of the answer's sister modality in the input, and section 4 says that
   channel is worth 0.226 against the genotype's 0.101. Validation reports k = 0 only
   (`_cache_epoch_metric` is called when `n_reveal == 0`). So the joint arm's expression head
   optimizes a capability that is off where it is scored. Whether this helps or hurts the k = 0
   number is not measured, and it is a confound in every v16 joint-versus-reference contrast.
2. **Per-feature z-scoring equalizes signal and noise columns.** The proteome's 28 percent
   zero-reliability columns (section 2) are scaled to unit variance like every other, so the
   pinball loss and the reported mean-over-features Pearson both give them full weight. A real
   +0.02 concentrated in the 50 percent of proteins with reliability above 0.5 shows up as
   +0.01 in the headline metric. This dilutes a true synergy by about a factor of two and adds
   variance to the statistic.
3. **Pinball at K = 19 with a point readout is scored on the median.** The metric comes from
   `DistHead.point()`, which for a quantile head is the tau = 0.5 knot. A head that has learned
   a sharper conditional *distribution* from the auxiliary label, without moving its median,
   scores identically. Cross-modal information is at least as likely to show up as
   calibration/sharpness as in the conditional median, and no current readout would see it.
   `save_loss_min: true` in v16 means the loss-minimum checkpoint exists and this is checkable
   offline.
4. **Sparse NaN scoring is correct, but the metric's denominator moves.**
   `_per_feature_pearson_sparse` drops any column with fewer than 3 finite pairs and any
   near-constant column, and `n_scored_genes` is logged per k. On 448 validation strains with
   2.6 percent missingness the drop count is small, so this is a caveat rather than a defect,
   but a joint-versus-reference difference should be recomputed on the intersection of scored
   columns rather than trusting two means taken over slightly different column sets.

One more data-side confound the addendum already names: `require_modalities []` in v16 makes
the expression-only arm iterate all 3,726 train genotypes, so a batch of 32 holds about 11
labeled rows. That is a *data-loading* property of the store, not of the model, and it is why
J_expr is not a usable reference. The joint arm's expression head sees the same 11 labeled rows
plus the proteome loss on the other ~31.

Example runs (v16, the joint round):

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/k7rzebp0

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hyasw3nx

## 7. What the data permits, in one paragraph

The honest summary for the PI. Cross-modal information between the knockout proteome and the
knockout transcriptome is large when one modality is OBSERVED (per-feature Pearson 0.22 both
ways, against 0.10 and 0.06 from the genotype) and is almost entirely orthogonal to what the
genotype carries. A shared trunk never observes the other modality at scoring time, so it
cannot collect that information; the channel it does have is a shared genotype representation,
and the linear measurement of that channel is **+0.005 to +0.006 on the proteome head and at or
below zero on the expression head**. At the measured paired-difference sd of 0.0167 (addendum
item 3), detecting +0.005 at 80 percent power needs 88 pairs two-sided or 69 one-sided; +0.006
needs 61 / 48. No design that fits in 14 days on three A40s plus a few cabbi cards reaches
that. Therefore **"joint >= single on BOTH heads" is not provable as a superiority claim within
two weeks**, and the two statements that ARE provable in that window are (a) non-inferiority at
a pre-registered 0.01 margin, which 12 to 18 clean pairs settle, and (b) superiority of an
explicitly CONDITIONED cross-modal model over a genotype-only model, where the effect is +0.11
to +0.16 and 3 pairs are overwhelming.

## Proposed experiments

Ranked by evidence value per GPU-day inside 14 days. "Pair" means one (arm, arm) contrast at
one split seed and one init seed; statistic throughout is the round template's
**mean of `val/<head>/pearson_per_feature` over a pre-registered fixed epoch window**, paired
within split seed and init seed, with the paired t on the differences. Assumed paired sd 0.0167
(addendum item 3). Throughput from the brief: v18 ran 1,200 epochs in 19 to 31 h at 3 runs per
card on cabbi; v16 joint ran 500 epochs in ~30 h at 3 per card on A40.

### E1. Conditioned cross-modal head, the provable claim (rank 1)

The only design whose effect size the data supports. Arms, all on `fig3_proteome` with
`require_modalities [protein_abundance, expression_log2_ratio]` so every row carries both
(1,103 train / 125 val / 121 test on seed 0):

- `C_geno`: expression head, genotype only, `mask_schedule [0]` (no reveal).
- `C_cond`: identical, but the trunk additionally receives the strain's OBSERVED proteome
  through the existing `observed_labels` channel with the proteome as the revealed modality,
  scored on expression at every step, and **evaluated with the proteome revealed** (the honest
  evaluation of an imputation model).
- Mirror pair `C_geno_P` / `C_cond_P` with the roles swapped (predict proteome, reveal
  expression).

Partitions x seeds: 3 split seeds x 2 init seeds = 6 pairs per direction, 24 runs. Epochs 600
(v14's proteome peak is at a median epoch of 150 and the expression rounds peak by ~400). 3
runs per card, ~20 h per task, 8 tasks: A40 `gpu` at `--array=0-7%3` finishes in about 3 days.
Pre-registered rule: one-sided paired t on `C_cond - C_geno`, reject at p < 0.05. Expected
effect from section 4: +0.11 on expression and +0.16 on the proteome; MDE at 6 pairs is 0.019,
so this is a 6-sigma design. Result the PI can state: "given a strain's proteome, the model
predicts its transcriptome at r = X against Y from genotype alone, p < 10^-4, on held-out
strains" -- a provable multimodal claim, honestly labeled as conditioning rather than shared-trunk
synergy.

### E2. Non-inferiority of the joint trunk, both heads (rank 2)

Replaces the unpowered superiority test with the claim the budget can actually support. Arms:
`J_ref` (proteome only, `require_modalities [protein_abundance]`), `J_joint` (both heads), and
a REPAIRED expression reference `J_expr2` that fixes the addendum's confound:
`require_modalities [expression_log2_ratio]` so it iterates 1,244 rows at 40 steps per epoch
like v17 L_ref, and the SAME unmasked `per_gene_aux`-style head the joint arm uses (not the
masked `per_gene` head). Add `mask_schedule [0]` to all three arms so the reveal confound of
section 6.1 is out of the contrast.

Partitions x seeds: 4 split seeds x 3 init seeds = 12 pairs per head, 36 runs. Epochs 1,200
(v18's budget). 3 per card, 19 to 31 h per task, 12 tasks: cabbi at 1 to 3 cards plus `gpu`
`%3`, about 5 to 6 days wall. Pre-registered rule, per head: two one-sided tests against a
margin of 0.01, i.e. declare non-inferiority when the one-sided 95 percent upper bound on
`single - joint` is below 0.01. At 12 pairs the one-sided half-width is 0.0087, so an observed
difference of +0.0013 or better establishes it. Result the PI can state: "on held-out strains,
joint training costs neither head more than 0.01 per-feature Pearson (95 percent one-sided),
while halving the number of trained models" -- a defensible claim, and the correct one given
that v16 measured -0.008 on the proteome with a two-sided MDE of 0.019.

### E3. Rescue the direction the data says can win: proteome head, expression subspace (rank 3)

Section 5 found the transfer is one-directional. Test it where it exists, with the aux weight
as the dose. Arms: `S_ref` (proteome only), `S_aux025`, `S_aux1`, `S_aux4` (expression aux head
at loss weight 0.25, 1, 4), all `mask_schedule [0]`, all on the union store. The dose-response
is the evidence, not a single contrast: a monotone trend across three weights at 4 pairs each
carries more information than one contrast at 12 pairs, because the linear ceiling (+0.005) is
below the single-contrast MDE but a trend test over 12 runs is not.

Partitions x seeds: 4 split seeds x 1 init seed x 4 arms = 16 runs, 4 pairs per weight.
Epochs 600. 4 per card, ~15 h per task, 4 tasks: cabbi, under 1 day per task, about 2 days
wall. Pre-registered rule: Page trend test (or a paired linear contrast with weights -1, 0, +1
across log aux weight) on the per-seed differences from `S_ref`, one-sided p < 0.05; plus the
non-inferiority bound of E2 as a secondary. Result the PI can state, if it lands: "the
auxiliary transcriptome head improves the proteome head monotonically in its weight
(trend p = X)". If it does not land, the recorded outcome is "measured null at the 0.015 MDE",
which is worth having.

### E4. Restrict the metric to reliable features (rank 4, zero extra GPU)

Section 2 and 6.2: 28 percent of the 1,850 proteins have zero clipped reliability, and
per-feature z-scoring gives them full weight in the metric. Recompute every existing v14, v16,
v17 and v18 readout on the subset of proteins with route-W reliability above 0.25 (63 percent
of columns) and above 0.5 (50 percent), from the checkpoints already on disk plus the logged
per-gene predictions where they exist. Cost: CPU only, hours, no GPU. Pre-registered rule: the
same paired statistic on the restricted column set; report the ratio of the restricted to the
full effect. This is a free doubling of effective effect size on every contrast already run,
and it must be declared BEFORE looking, so declare it now. Result the PI can state: whether
the -0.008 on the v16 proteome contrast is a null on signal-bearing proteins or an artifact of
averaging noise columns.

### E5. Distributional readout of the auxiliary gain (rank 5, zero extra GPU)

Section 6.3: the metric reads only the tau = 0.5 knot. v16 saved a loss-minimum checkpoint
(`save_loss_min: true`), so for the 6 joint and 6 reference runs compute, on the same held-out
strains, the mean pinball, the PIT calibration and the 80 percent interval coverage per head,
in addition to the point Pearson. Cost: inference only, a few GPU-hours total, fits in one
`gpu` slot overnight. Pre-registered rule: paired one-sided t on mean pinball (lower is
better), margin 0. Result the PI can state: whether joint training sharpens the conditional
distribution even when it does not move the median. A positive read here is a genuine synergy
claim that the current metric structurally cannot see.

### E6. Same-media anchor (rank 6, out of scope for 14 days, record it)

The proteome is SM and the expression is SC (`proteome-expression-eda.md`), so every cross-modal
number above is attenuated by a medium difference that no analysis in this store can remove.
No experiment on `fig3_proteome` can fix it. Recorded here so the two-week claim is written with
the caveat attached rather than discovered later: the cross-modal reads of section 4 are lower
bounds, and the shared-trunk ceiling of section 5 may be a lower bound too.
