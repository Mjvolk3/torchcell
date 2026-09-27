# Reviewer 3: everything measured about joint / multi-head training, and what it supports

Read time for every W&B number below: 2026-09-27 06:00 to 06:40 UTC. Sources: the W&B API
(`wandb.Api()`, entity `zhao-group`), full `scan_history` dumps of all 24 runs in
`torchcell_019_prot_v16` (saved under this directory as `hist/*.csv`), run configs, the
committed readouts in `experiments/019-simb-multimodal/results/`, and the notes-tex /
Dendron documents named inline. Takes the coordinator's `ADDENDUM_0207.md` as given.

## 1. Bottom line

Nothing in this campaign supports "joint training helps each label's own prediction."

- **Proteome head, genotype-only metric:** measured NULL to slightly negative. Pre-registered
  window statistic joint minus proteome-only = **-0.0076, sd 0.0133, t -1.41, 3 of 6 pairs
  positive**. On the biased roll_max the same contrast is **-0.0161, t -3.53, 0 of 6 positive**.
  On held-out test at the best-val checkpoint it flips sign to **+0.0048, t +0.62, 4 of 6**.
  The sign is not stable across statistics, which is what a null looks like.
- **Expression head:** the apparent **+0.060 (t +3.43)** win is an artifact of the control. Four
  of six expression-only runs never trained at all, and their head is a different objective from
  the joint arm's expression head (addendum item 2, confirmed independently below). Restricted to
  the two expression-only runs that did train, joint minus expression-only is **-0.0010 on val
  (roll_max, n=2)** and **-0.0674 on held-out test (n=2, both negative)**.
- **One real, unreported joint win:** with 10 proteins revealed, the joint arm's proteome head
  beats proteome-only by **+0.0902, sd 0.0109, t +20.3, 6 of 6 pairs**
  (`val/proteome/pearson_per_feature@k1`, same 299-499 window). With 1,000 revealed it **loses**
  by -0.0360, 0 of 6. Neither number appears in `v16_joint_readout.json`.
- **The joint arm fits the proteome WORSE on train**, by -0.0947 (sd 0.0150, t -15.4, 0 of 6) at
  epoch 499. So no "the second head regularizes the trunk" story is available: the joint arm is
  behind on train and on val simultaneously.
- **Every earlier multi-head experiment in the program is negative or uninterpretable.** The only
  auxiliary-head experiment run to completion anywhere (023 betaxanthin plus a 19-metabolite
  head) reads **-0.0279, sd 0.0377, n=5 cells, 2 of 5 positive**
  (`results/bx_aa_paired_summary.json`, `live_mean_delta`).

## 2. What the three v16 arms actually are (read from the run configs, not the comments)

`conf/cgt_expr_v16_joint.yaml` says the three arms "differ in nothing but the active heads."
They differ in four things. Verified per-arm from `run.config` and `run.summary`:

| | J_ref | J_joint | J_expr |
|---|---|---|---|
| `multitask.active_heads` | `[per_gene]` | `[per_gene, per_gene_aux]` | `[per_gene]` |
| `per_gene` label | `protein_abundance` | `protein_abundance` | `expression_log2_ratio` |
| `per_gene_aux` label | inactive | `expression_log2_ratio` | inactive |
| `mask_head` | `per_gene` | `per_gene` | `per_gene` |
| masked objective applies to | proteome | **proteome only** | **expression** |
| `cell_dataset.require_modalities` | `[protein_abundance]` | `[]` | `[]` |
| `n_train_supervised` | 3,581 | 3,722 | 1,244 |
| `n_val` / `n_test` | 448 / 447 | 478 / 481 | 155 / 155 |
| `trainer/global_step` at ep 500 | 56,000 (112/epoch) | 58,500 (117/epoch) | 58,500 (117/epoch) |
| checkpoint `monitor` | val/proteome/pf | val/proteome/pf | val/expression/pf |
| `metric_monitor` | val/mean/pf | **val/expression/pf** | val/mean/pf |

`mask_schedule` is `[0, 10, 100, 1000]` in all three and `val/mask/n_revealed@k` reads
0/10/100/1000 in all three, so the reveal machinery is identical. What differs is **which head
it is pointed at**. The metric keys prove it: `J_joint` logs
`val/proteome/pearson_per_feature@k0..k3` and `val/proteome/n_scored_genes@k*` and has **no**
`val/expression/...@k*` key at all; `J_expr` logs `val/expression/...@k0..k3`. So the joint arm's
expression head is supervised on all 6,127 genes at every step with nothing revealed, while the
expression-only arm's head is teacher-forced and scored on the hidden subset. Those are two
different training problems, and the campaign has compared them to each other.

Three consequences for the proteome contrast, which is the cleaner of the two:

1. The proteome partition IS matched to v14 (3,581 / 448 / 447 on every split seed, as
   `cgt_expr_v14_proteome.yaml` intends), and the masked objective on the proteome head is the
   same in J_ref and J_joint. **This contrast is honest.**
2. J_joint takes **4.5% more optimizer steps per epoch** than J_ref (117 vs 112) because its
   loader also iterates the 141 expression-only train rows. Small, but it is a data-quantity
   difference inside a contrast that is supposed to isolate the auxiliary head, and it is exactly
   the confound `scripts/optuna_joint_sweep.py`'s design note set out to avoid by pinning all
   conditions to the strains carrying both labels.
3. The joint arm's **expression** test number is read at the **proteome's** best-val checkpoint,
   because `monitor` is the proteome and the test pass loads the first ModelCheckpoint. Measured
   cost of that: at the proteome-monitored epoch the joint expression head's val Pearson is
   0.047 to 0.117 against its own best of 0.082 to 0.118, a shortfall of **-0.032 on average**
   (per-run: -0.055, -0.055, -0.001, -0.037, -0.035, -0.009). That is the same size as the effect
   being looked for.

## 3. Trajectories, per arm (5-epoch centered rolling mean of the val key; `traineval` = eval-mode train)

**J_ref, proteome-only.** Rises fast, peaks at epoch **92 to 368** (roll_max 0.0812 to 0.1155),
then gives back; `val/loss` bottoms at 150 to 200 and rises; `traineval/proteome/pf` reaches
**0.561 to 0.567** at 499 and is still climbing; `nmse_last` 1.08 to 1.09, i.e. worse than
predicting each protein's mean. Classic overfit with a 0.47 train-val gap. **Not still rising.**
The split-0 continuation confirms it: at epoch 547/549 val is 0.0823 / 0.0688, below the
epoch-150 values of 0.1157 / 0.1109.

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/rh9wgkoo

**J_joint, both heads.** Proteome peaks **later and lower** than J_ref in 6 of 6 pairs (raw-max
epochs 238-499 vs 92-368). Expression climbs monotonically and is **still rising at 499 and at
531**: from epoch 250 to 499 the expression head gains **+0.046 in 6 of 6** while the proteome
head is flat (+0.004, 3 of 6 negative). `traineval` at 499: proteome **0.452 to 0.486**,
expression **0.510 to 0.525**. Within-run across-epoch correlation of the two val curves is
+0.24 to +0.66 in five runs and -0.38 in one, so there is **no consistent within-run seesaw on
validation**; the trade-off shows up on the train fit instead (section 5).

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/k7rzebp0

**J_expr, expression-only.** Four of six never train: `traineval/expression/pf` stays in the
noise (-0.03 to +0.03) for all 500 epochs, `val/loss` is pinned flat from epoch 50,
`val/expression/pred_sd_ratio` is exactly **0.0000** (the head emits a constant per gene), and
`train/grad_norm` sits at 0.022 to 0.025 against 0.042 to 0.043 for the two that did train. Their
`calib/pit_ks` is the *best* of any arm (0.045 to 0.062) because a constant with the right
marginal spread is perfectly calibrated and useless. The two that trained reach `traineval`
0.218 and 0.297 at 499, which is still far under the joint arm's 0.51.

https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hyasw3nx
https://wandb.ai/zhao-group/torchcell_019_prot_v16/runs/hzca60i6

## 4. Every paired contrast, computed here

Paired within split seed x init seed; the two arms of each pair were co-resident on one card, so
card type and packing do not enter. Window statistic = mean of the val key over epochs 299 to
499, the pre-registered one named in both readout JSONs.

| contrast | statistic | n | mean | sd | t | n_pos |
|---|---|--:|--:|--:|--:|--:|
| joint - ref, proteome pf | window 299-499 | 6 | **-0.0076** | 0.0133 | -1.41 | 3/6 |
| joint - ref, proteome pf | roll_max | 6 | **-0.0161** | 0.0112 | -3.53 | 0/6 |
| joint - ref, proteome pf | TEST @ best val | 6 | **+0.0048** | 0.0190 | +0.62 | 4/6 |
| joint - ref, proteome pf@k1 (10 revealed) | window | 6 | **+0.0902** | 0.0109 | +20.34 | 6/6 |
| joint - ref, proteome pf@k3 (1000 revealed) | window | 6 | **-0.0360** | 0.0076 | -11.67 | 0/6 |
| joint - ref, traineval proteome pf @499 | point | 6 | **-0.0947** | 0.0150 | -15.43 | 0/6 |
| joint - expr, expression pf | window | 6 | **+0.0605** | 0.0433 | +3.43 | 5/6 |
| joint - expr, expression pf, trained controls only | roll_max | 2 | **-0.0010** | 0.0133 | -0.10 | 1/2 |
| joint - expr, expression pf | TEST @ best val | 6 | +0.0278 | 0.0859 | +0.79 | 3/6 |
| joint - expr, expression pf, trained controls only | TEST | 2 | **-0.0674** | 0.0253 | -- | 0/2 |

The first two rows and the `_excluding_plateau` block of `v16_joint_expr_readout.json` are the
committed versions of rows 1, 2 and 8. Rows 3, 4, 5, 6 and 10 I computed here; rows 3 and 10 are
missing from the committed readouts because `test_at_best_val` is taken from the **last** segment
and the split-0 continuation segments have not reached a test pass, so those cells read NaN.

Two readings worth keeping separate:

- **pf@k1 is the one place joint wins, and it wins hard.** Reveal 10 of the 1,850 proteins and
  the joint arm's proteome head is +0.090 better in every pair. Reveal 1,000 and it is -0.036
  worse in every pair. *Hypothesis (untested):* the auxiliary expression supervision buys a
  strain-level representation that helps when there is almost no proteome context to condition
  on, and at k=1000 the masked head can read the revealed proteins directly, where the joint
  arm's 0.095 deficit in proteome train fit costs it. This is a mechanism story, not a result.
- **Against the linear baseline the joint expression head is not competitive.** ProtT5 ridge on
  the same partitions reads val 0.135 / 0.076 / 0.117 (B2) and 0.130 / 0.113 / 0.111 (B3) on
  s0/s1/s2 (`baselines` block of `v16_joint_expr_readout.json`), against the joint head's window
  means of 0.087, 0.093, 0.090, 0.103, 0.058, 0.070. On the proteome the CGT is above its ridge
  (0.064-0.082 val): the proteome head clears its baseline, the expression head does not.

## 5. What the two heads do to each other

The decisive number is the train fit. At epoch 499, paired within split and seed:

```
split seed   ref traineval/proteome/pf   joint   diff
s0    0      0.5617                      0.4757  -0.0860
s0    1      0.5669                      0.4538  -0.1130
s1    0      0.5670                      0.4772  -0.0898
s1    1      0.5615                      0.4860  -0.0755
s2    0      0.5639                      0.4515  -0.1124
s2    1      0.5661                      0.4750  -0.0911
                       mean -0.0947  sd 0.0150  t -15.43  0/6 positive
```

The expression head costs the proteome head 0.095 of **training** Pearson, every time, with no
overlap between the two groups. Combined with a val delta of -0.008 to -0.016, the joint arm is
behind on both sides of the generalization gap. That excludes the only benign explanation for a
flat val contrast, namely that the second head trades train fit for generalization.

Direction of travel inside a joint run, epoch 250 to 499: expression +0.046 (6/6 up), proteome
+0.004 (3/6 down). So running longer moves the expression head and leaves the proteome head
where it is. That is consistent with the two heads not competing for a fixed val budget, and it
means a longer budget cannot rescue the proteome contrast.

## 6. The single-head references at longer budget

**Proteome (v14, 1,353 to 1,960 epochs, cabbi compute-3-3, `v14_proteome_readout.json`).**
P_ref roll_max per split seed: s0 0.1206 (ep 202) / 0.1087 (157); s1 0.0883 (418) / 0.0828 (246);
s2 0.1104 (124) / 0.1178 (97). The v16 J_ref roll_max on the same split and init seed:
0.1155, 0.1116, 0.0848, 0.0812, 0.1112, 0.1135. Per-pair differences (v16 minus v14) are
-0.0052, +0.0030, -0.0034, -0.0016, +0.0008, -0.0043: **J_ref reproduces v14's reference to
within 0.005 on all six**, and both peak in the same epoch band. Two conclusions:

- J_ref is a faithful restoration of the v14 reference, so the v16 proteome contrast is not
  compromised by the store change.
- **The proteome head does not need more than 500 epochs.** v14 ran 2.7 to 3.9 times longer and
  scored the same; its `nmse_last` is 1.11 to 1.15 and its `last` values are 0.055 to 0.100,
  below its own peaks. A 1,200-epoch or 2,000-epoch budget buys the proteome head nothing.

**Expression (v13, 5,180 to 5,731 epochs, `v13_split_readout.json` plus histories I pulled).**
V_ref roll_max by budget cut:

```
run       split seed  <=500ep  (at)  win299-499  <=1200ep  (at)   full   (at)   traineval@500  traineval@end
wq8y8nd5  s0 0        0.1146   219    0.0911     0.1997  1183   0.2097  3916   0.3714         0.7866
825on260  s0 1        0.0198   236   -0.0012     0.0198   236   0.0260  1344   0.0239         0.0024  (collapsed)
phkn895t  s0 2        0.1238   495    0.1040     0.1836  1178   0.1999  5680   0.3748         0.7887
8i75d8h1  s0 3        0.1235   498    0.0956     0.1814   834   0.2127  3341   0.3851         0.7888
58qsybms  s1 0        0.1007   396    0.0830     0.1499  1149   0.1706  4155   0.3771         0.7846
8ay2niuv  s1 1        0.1005   194    0.0714     0.1602  1191   0.1602  1191   0.3737         0.7853
lp6guytz  s2 0        0.0944   115    0.0570     0.1180  1097   0.1389  1755   0.3966         0.7916
hjx0y9f1  s2 1        0.0980   126    0.0592     0.1051  1144   0.1270  5260   0.3832         0.7932
bw7mxqcl  s3 0        0.0636   279    0.0372     0.1093   998   0.1485  4988   0.3707         0.7938
i4uw04mb  s3 1        0.0640   135    0.0472     0.1361   890   0.1435  1756   0.3848         0.7939
```

This is the single most important comparison for the expression side and it has not been made.
**A properly trained expression-only model, cut at epoch 500, scores roll_max 0.064 to 0.124 and
window mean 0.037 to 0.104.** The v16 joint arm's expression head at 500 scores roll_max 0.082 to
0.118 and window 0.058 to 0.103. The two distributions are **the same**. The joint expression head
is not ahead of a single-head expression model at matched epoch; it is on the same curve. (Not a
paired test: different store, different partition, and v13's partition is not reproducible on the
fig3_proteome store per `cgt_expr_v16_joint.yaml`. It is a comparison of distributions, and it is
the best available.) Expression at 500 epochs is about 55 to 60% of what it reaches by 5,700, and
`notes/experiments.019-simb-multimodal.phenotype-strand-retrospective.md` already records
"expression score is a function of the EPOCH BUDGET" as the campaign's finding.

## 7. Earlier multi-head rounds

**The expression-morphology optuna trio (the only earlier design with a correct control).**
`scripts/optuna_joint_sweep.py`'s note pins all three conditions to the 1,440 genotypes carrying
both modalities, precisely so the delta is the auxiliary-head effect and not a data-quantity
effect. Best final-epoch val Pearson, max over trials (an upward-biased order statistic over
different hyperparameter draws, and the budgets are not matched):

- expression alone, `torchcell_019_expr_v5`, 35 trials, <=151 epochs: **0.0532**
- morphology alone, `torchcell_019_morph_v5`, 23 trials, <=121 epochs: **0.0655**
- both heads, `torchcell_019_expr_morph_v5`, 32 trials, <=87 epochs: expression **0.0563**,
  morphology **0.0285**

So the second head cost morphology 0.037 and bought expression 0.003. Every score is in the
0.03-0.07 band, an order of magnitude below where either strand now sits, and the joint arm never
ran past 87 epochs; `9-campaign.tex` states this plainly ("It has never been run past 87 epochs")
and the retrospective's leaderboard carries the joint arm at 0.062 against morphology 0.082 and
expression 0.228. **Measured, but not a competent measurement of either task.** The v2/v3/v4
predecessors are shorter still (<=27 epochs, many crashed) and the 496-run
`torchcell_019-simb-multimodal_cgt_multitask` project mixes 323 single-head expression runs at
<=373 epochs with 32 two-head runs at <=116, which is not a contrast.

**The 023 betaxanthin metabolome head, the one auxiliary-head experiment run to completion.**
`results/bx_aa_paired_summary.json`: live pairs n=5, mean delta **-0.0279**, sd 0.0377, se 0.0169,
2 of 5 positive; fair-exposure pairs n=10, mean **-0.0181**, sd 0.0312, 4 of 10 positive. The
auxiliary head's own score is 0.007 to 0.147 against the 0.209 the amino-acid strand reaches
alone, so the run pays capacity for a head that is itself failing.
`1-summary.tex` and `8-directions.tex` both label this negative and naive rather than a verdict.

**Aggregate prior:** three independent attempts at a second head (expr+morph, betaxanthin+
metabolome, proteome+expression), three nulls or negatives on the first label.

## 8. What has been promised or planned

- `notes/experiments.019-simb-multimodal.experimental-plans.md` research question 1 is exactly
  this claim ("How much does jointly training on modality X ... improve prediction of modality
  Y"), and its proteome strand sequences EDA, then a linear map, then "joint train: few YKO
  expression + a little proteome -> does proteome help expression (and vice-versa)?"
- `7-common.tex` sec. "Cross-modality overlap is real and small" is the gating measurement and it
  is discouraging: median per-gene r across strains **0.08**, 1.7% of genes above 0.3, ridge
  held-out R^2 **0.035 / 0.025**. It adds the qualification that bounds any synergy claim: the
  two labels were measured in **different media** (synthetic minimal for the proteome, synthetic
  complete for the expression), so the overlap is certified non-redundant and **not** certified
  complementary, and "that distinction bounds how hard a 'the proteome helps expression' claim
  can be pushed until a same-media pair exists." The same section states "Joint proteome-and-
  expression training is not on the near roadmap."
- `9-campaign.tex` sec. "The joint question" says the campaign should be designed to **measure**
  the effect rather than confirm it, that "the control arm matters more than the joint arm," and
  that "a joint result is uninterpretable without a single-phenotype arm run to the same budget."
  v16 violated that last sentence on the expression side.
- `8-directions.tex` recommends the **fitness** pairing for morphology as the first full-scale
  joint test, on power grounds (4,220 strains rather than 1,440), and notes a null is publishable
  as the honest counterweight.

Nothing anywhere promises that joint proteome-plus-expression training helps. The documents are
consistent and correct; v16 is the first run at the question and it came back null.

## 9. Distance from the claim

**Measured.**

- Joint vs proteome-only on the proteome, genotype-only, 6 paired pairs, matched objective and
  matched proteome rows: null to negative, on val and on test, with signs that disagree.
- Joint vs proteome-only on masked conditioning: +0.090 at k=10 (6/6, t 20.3), -0.036 at k=1000
  (0/6, t -11.7).
- Joint costs the proteome head 0.095 of train Pearson (6/6, t -15.4).
- The proteome head is budget-saturated by epoch 500 (v14 at up to 1,960 epochs matches J_ref at
  500 to within 0.005).
- The expression head at 500 epochs is at 55 to 60% of its 5,700-epoch score, whether trained
  jointly or alone.
- Three earlier second-head attempts: all null or negative on the first label.

**Missing (not measured, as distinct from measured null).**

1. **An objective-matched, competent expression-only control.** No run anywhere trains the
   `per_gene_aux` configuration (unmasked, all genes supervised) as the only head. Without it
   there is no denominator for the joint expression head.
2. **A row-matched joint arm.** J_joint carries 141 extra train rows and 4.5% more steps than
   J_ref. `optuna_joint_sweep.py` already established that the instance set must be pinned; v16
   dropped that.
3. **A label-permutation control.** Nothing distinguishes "the expression labels carry
   information the proteome head can use" from "any second per-gene head with the right output
   statistics changes the optimization." This is the cheapest missing arm and it is the one that
   would make the pf@k1 result publishable.
4. **Per-head checkpoint selection.** One `ModelCheckpoint` per run means the joint arm's
   expression test number is read 0.032 below where its own monitor would have put it.
5. **An expression contrast at a budget where expression has converged.** Nothing exists past 558
   epochs on the fig3_proteome store.
6. **Gene-disjoint scoring.** `cgt_expr_v16_joint.yaml` states the expression metric "is to be
   read on all rows AND on the gene-disjoint rows" because 17 to 20 of the 155 held-out
   expression genotypes share a deleted gene with train. No readout splits it that way.
7. **Power.** At the pooled paired sd of 0.0167 (addendum item 3) six pairs detect only 0.027.
   The whole v16 round is underpowered for any effect the size of every other arm effect in this
   campaign (a few hundredths).

**What the running continuation (IGB 2410399) will and will not supply.** As of 06:36 UTC
2026-09-27 the W&B project holds exactly 6 continuation runs, all **split 0 only**, all resumed
from a first segment (`wandb.resumed_from` set), `trainer.max_epochs` 1,200, last logged epochs
**531 to 558**, all marked `finished`, last history row about 1.9 h stale. The addendum reports
tasks 0 and 1 at ~605 and tasks 2 to 5 pending behind the %3 throttle; either way fewer than 60
of the required 700 additional epochs have landed, on one partition of three.

- It **will** give the expression head 700 more epochs on split 0 and, if tasks 2 to 5 clear, on
  splits 1 and 2. That answers "where does the joint expression head plateau."
- It **will** supply a test pass at 1,200 epochs, restoring the `test_at_best_val` cells that are
  currently NaN for split 0.
- It **will not** fix the proteome contrast: v14 at up to 1,960 epochs equals J_ref at 500, and
  the joint proteome head gained +0.004 between epochs 250 and 499. There is no reason in the
  measured curves for epochs 500 to 1,200 to change a -0.008 delta.
- It **will not** fix the expression contrast, because it extends the same broken control. Four
  of the six J_expr runs are constant-output heads with `pred_sd_ratio` 0.0000; 700 more epochs
  of a head whose gradient norm is decaying does not un-collapse it, and even if it did, the
  control's objective still differs from the joint arm's.
- It **will not** add pairs. Six is six at 1,200 epochs too.
- It **will not** supply a row-matched arm, a permuted-label arm, or a per-head checkpoint.

Stated plainly: **after the continuation finishes, the PI still cannot say "joint training helps
each label's own prediction."** The strongest defensible statement the current data supports is
the conditional one: *when a small panel of proteins is already measured, a jointly trained model
predicts the rest of the proteome better than a proteome-only model* (+0.090 at 10 revealed, 6 of
6 pairs, pre-registered window) -- and that needs the permutation control and a test-set
confirmation before it is worth putting in front of anyone.

## Proposed experiments

Throughput measured on this store: 500 epochs in **24.2 to 30.0 h** at 3 runs/card (v16, 117
steps/epoch on 3,722 train rows). A store restricted to the ~1,349 both-label strains runs ~42
steps/epoch, so about 2.8x cheaper per epoch, matching the v17 rate (1,200 epochs in ~24 h at
3/card). Capacity: IGB `gpu` A40, at most 3 of our jobs (`--array=..%3`), unlimited wall; IGB
`cabbi`, 1 to 3 cards, every job <= 5 days. Assume 4 to 5 cards concurrently.

Pre-registered statistic for all of them, same as the round template: **mean of the val key over
the last 200 epochs of the budget**, paired within (split seed x init seed), one-sided paired t.
The claim is a conjunction, so run it as an intersection-union test: require **both** heads to
clear one-sided alpha 0.05, which is itself a valid level-0.05 test of the conjunction and needs
no multiplicity correction. Report the roll_max beside it, labeled as the biased order statistic.
Declare in advance that a run whose `val/<head>/pred_sd_ratio` is below 0.02 at the window start
is a failure-to-launch and its pair is dropped, with the count reported.

Ranked by evidence value per GPU-day inside 14 days.

**P0. Fix the instrument. Zero GPU-days.**
Two changes before any launch. (a) A second `ModelCheckpoint` per head so each head's test number
is read at its own best-val epoch; the joint arm's expression test number is currently 0.032 low
by construction. (b) Add the `@k` contrast and the gene-disjoint expression subset to
`v13_split_readout.py`, and make `test_at_best_val` fall back to the newest segment that ran a
test pass instead of returning NaN. Without (a) no test-set joint claim is interpretable, and
(b) surfaces a +0.090 result the committed readouts do not report.

**P1. The row-matched, permutation-controlled proteome experiment. ~9 to 11 GPU-days, 4 days wall.**
Store restricted to the strains carrying **both** labels (`require_modalities:
[protein_abundance, expression_log2_ratio]`, the existing intersection filter), so every arm sees
identical rows and identical steps per epoch. Arms, all with the proteome head masked exactly as
J_ref: `K_ref` (proteome only); `K_joint` (proteome + unmasked expression aux, weight 1);
`K_perm` (identical to `K_joint` but the expression label vector permuted across train strains
once, with a fixed seed, so the aux head has the right marginals and no strain information).
4 split seeds x 3 init seeds = **12 pairs per contrast**, 36 runs, 3 per card = 12 cards, **500
epochs** (v14 proves the proteome head is saturated there). At ~1,349 rows the epoch is 2.8x
cheaper, so a card finishes in ~10 to 12 h; 12 cards over 4 to 5 concurrent cards is 3 waves,
about **1.5 to 2 days** wall on `cabbi` (well inside 5 days) or `gpu`. MDE at pooled sd 0.0167 and
12 pairs is **0.015**.
Decision rule: `K_joint - K_ref` on the window statistic, one-sided. **Provable claim:** if
`K_joint - K_ref > 0` at p < 0.05 AND `K_joint - K_perm > 0` at p < 0.05, the PI can say "adding
the transcriptome as a second head improves proteome prediction on strains carrying both labels,
and the gain requires the real labels." If `K_joint - K_ref` is null the round retires the
proteome half honestly, at 12 pairs rather than 6. Also read `pf@k1` and `pf@k3` as secondary,
pre-declared: the k=10 result is the live hypothesis and this design is what would make it
defensible.

**P2. The competent expression control, same store. ~8 to 10 GPU-days, 4 days wall.**
Same row-matched store. Arms: `K_esolo` (the **unmasked** `per_gene_aux` configuration as the
only active head, so the objective matches the joint arm's expression head exactly);
`K_joint` (reuse P1's runs, same seeds and splits, so this costs only the control arm);
`K_emask` (the masked `per_gene` expression head alone, the J_expr configuration, included so the
objective difference is measured rather than argued). 4 splits x 3 seeds = 12 pairs, 24 new runs
(12 `K_esolo` + 12 `K_emask`), 8 cards. **2,400 epochs**: v13 says the expression head needs past
1,000 and its window at 1,200 is 0.105 to 0.200 against 0.127 to 0.213 at full budget, so 2,400 on
a 2.8x-cheaper epoch is ~48 to 60 h per card, 2 waves, about **4 to 5 days** wall. Fits `cabbi`'s
5-day cap per job; run the second wave as a separate job.
Decision rule: `K_joint - K_esolo` on the window statistic, one-sided.
**Provable claim:** P1 and P2 together, both significant, is exactly "joint training helps each
label's own prediction," with the instance set fixed, the objective matched per head, a
permutation control on one side, and 12 pairs. That is the experiment the PI can defend.
`K_esolo - K_emask` separately quantifies how much of v16's +0.060 was the objective, which is
worth stating in its own right.

**P3. Aux-weight ladder, only if P1 or P2 is positive-leaning. ~6 GPU-days, 2 days wall.**
`head_weights.per_gene_aux` in {0.25, 0.5, 1, 2} on the row-matched store, 2 splits x 2 seeds,
16 runs, 500 epochs (proteome readout) plus the aux head's own curve. `J_joint05` was declared in
`gh_expr_008_arm.sh` and **never launched** (`joint05_minus_ref` is `n: 0` in both readout JSONs),
so 1:1 is an untested choice and the optuna note flags head-weight balance as the first thing to
sweep if the joint arm underperforms both baselines, which is what happened. Do not spend the
GPU-days on this before P1 and P2 report.

**P4. The superset question, separately and honestly. ~9 GPU-days, 4 days wall.**
If the PI wants the v16 framing back (joint arm keeps the 205 expression-only strains), run it as
its own arm on top of P1: `K_joint_super` vs `K_joint` at the same splits and seeds isolates the
data-quantity term from the auxiliary-head term. 4 splits x 3 seeds, 12 runs, 500 epochs.
Without this, any positive v16-style result is confounded with 4.5% more rows and cannot be
reported as an auxiliary-head effect.

**Ranking rationale.** P0 is free and changes what the existing runs can say. P1 buys 12 clean
pairs on the head that is budget-saturated, at a third of the epoch cost of the current round, and
carries the permutation control that nothing in the campaign has. P2 is the only way to get a
denominator for the expression head and its control arm is the marginal cost because it reuses
P1's joint runs. P3 and P4 are refinements that are worth nothing until P1 and P2 report. Total
for P0 through P2: about **17 to 21 GPU-days over 6 to 7 days wall** at 4 to 5 concurrent cards,
which fits the two-week window with room for one requeue.

**What should not be spent on.** Extending v16 past 1,200 epochs on the current arms, and adding
more seeds to the current J_expr control. Both extend a comparison whose control is a different
training problem from the treatment, and no budget fixes that.
