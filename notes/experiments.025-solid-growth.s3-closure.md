---
id: isyv1unwr16v11iyxd7texj
title: S3 Closure
desc: ''
updated: 1789719760549
created: 1789719760549
---

## 2026.09.18 - The S3 closure pool against its own definition

Typeset document: `notes-tex/025-s3-closure/` (build with `make mermaid && make plots && make`).
Analysis script: [[experiments.025-solid-growth.scripts.s3_closure_recompute]]. Figure a:
[[experiments.025-solid-growth.s3-closure.mermaid.pipeline]].

The question: S3 carries every term of the trigenic identity for every triple, so does the
fitness stored in the build reproduce the interaction stored in the build (strength), and does
anything stored reproduce the p-value (confidence)? Both answered against the raw source tables.

Measured (summary at `experiments/025-solid-growth/results/s3_closure_recompute_summary.json`):

- Within one source screen the identity is exact: Costanzo 2016 r 0.999 (1,227,294 rows on
  closure pairs), Kuzmin 2018 r 1.000, Kuzmin 2020 r 0.996 once the missing query fitness is
  set to 1.0, which is what the source scoring did on 99.6 percent of its digenic rows (H5).
- In the build: digenic r 0.445 (n 728,856), trigenic r 0.230 (n 352,505). Doubles with no
  essentiality-tainted single r 0.598; with one r 0.452 and intercept 0.116; restoring the
  measured mean of the single (k/(k-1)) gives r 0.491, intercept 0.014, rmse halved.
- Confidence: a single-screen digenic p is ranked by a two-sided normal test on the double's own
  colony SD over 4 colonies at rho 0.984 (n 41,394); propagating the singles' stored SD lowers
  it to 0.813. A trigenic p is not reproducible from anything stored (rho 0.199 on the triple's
  SD, 0.097 with every term propagated). A merged record's p is the deduplicator's t-test over
  the k duplicate scores (df k-1): against the median source p rho 0.053; 1,754 closure doubles
  and 406 triples are significant in every source and not in the build; 94.4 percent of closure
  doubles (697,825) and 12,914 triples are merged.
- Hazards counted: 1,140 singles carry an essentiality entry (233 with no measurement, touching
  no closure record); 907 measured singles were averaged with a 0 (stored median 0.679 vs measured
  0.843), touching 141,099 doubles and 78,707 triples; 3 records carry two fitness values (masked
  by the trainer); 691 closure doubles carry a SynthLethDB 0 in their mean.

Consequence for the S3 arm: the identity cannot be the mechanism by which fitness supervision
helps (its ceiling on the stored values is r 0.23, below the additive ridge's 0.40); every
p-value in a merged record is unlabeled; three build changes are listed in Section 6 of the
document (keep the essentiality 0 out of the mean, combine source p-values by a rule that reads
them, keep the screen identity on merged records).

![](./assets/images/025-solid-growth/s3_closure_strength.svg)

![](./assets/images/025-solid-growth/s3_closure_confidence.svg)

![](./assets/images/025-solid-growth/s3_closure_hazards.svg)

## 2026.09.19 - The reproducibility ceiling of the trigenic score: 0.59, not 0.74-0.81

Two published numbers for how well the adjusted trigenic score reproduces itself, both from the
Kuzmin 2018 screens, and they differ because they are computed on different sets.

| source | set | replicate r, adjusted trigenic | digenic |
|---|---|---|---|
| Kuzmin 2018 SI, si1.md line 171 | significant scores only, p < 0.05 (Fig. S5B) | 0.74 to 0.81 | 0.90 to 0.91 (also raw triple) |
| Dango, zhangDANGOPredictingHigherorder2020 | all 91,050 triples of the diagnostic-array screen | 0.59 | 0.88 |

The Kuzmin p-value is a normal test of the score against a variance propagated from the query and
array fitness estimates (the array variance from screening a wild-type control query against the
diagnostic array, n = 91). Conditioning on p < 0.05 keeps scores large relative to their own
error, which raises a correlation computed inside the subset. **0.59 is the number to use for an
unselected pool like S3.** Dango calls it "an approximate upper bound of the performance that a
computational framework could possibly achieve."

Strictly it is the replicate-oracle value, not a ceiling: a model predicting the denoised value and
scored against a noisy one is capped by sqrt(reliability of the target), which is sqrt(0.59) = 0.77
against a single replicate and about 0.86 against a score combining two replicates of individual
reliability 0.59. Comparability caveat: Dango's set is one screen's 1,400-gene diagnostic array,
ours is the pooled S3 (376,732 triples) or 029 (299,146), and our val is the pinned 010 random
split.

Reference points on the same axis, all measured:

| quantity | value |
|---|---|
| two replicates of the same triple, all scores (Dango) | 0.59 |
| Dango's own model, random split (its Fig. 2b) | about 0.47 |
| our Dango reimplementation, val ([[experiments.005-kuzmin2018-tmi.results]]) | 0.41 to 0.42 |
| identity recomputed from 029 fitness, Kuzmin 2018 screen, Kuzmin-first policy | 0.517 |
| identity recomputed from 025 build fitness, all triples | 0.230 |
| S3 seed 1 (IGB 2409033), val triples, epoch 44, PARTIAL | 0.487 |
| S3 seed 1, train triples, epoch 44, PARTIAL | 0.727 |

Two readings. Validation at 0.487 sits in the same band as the identity recompute and the replicate
oracle, so the model is near the label's noise floor rather than near a modeling limit. And train
at 0.727 is above 0.59, so the model agrees with its training labels better than two independent
measurements of the same triple agree with each other; everything above 0.59 on train is fit to the
particular noise realization in the training copy. That is the overfit statement made quantitative,
and it is consistent with the curve shape (val flat at 0.48 to 0.49 since epoch 15 while train
triples climbed from 0.47 to 0.727). Recorded in section 8 of `notes-tex/025-s3-closure`.

## 2026.09.19 - The trigenic identity is exact inside a screen; one missing record explains the rest

`experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen.py`. The digenic
within-screen control fits on one raw row; the trigenic one does not, because the Kuzmin methods
put its terms in three measurements (si1.md line 191). On a trigenic row sit the double-mutant
query fitness f_ij, the array single fitness f_k and the triple fitness f_ijk. The two digenic
terms and the query singles come from the single-mutant control query strains, gene plus HO,
screened against the same array strain. Matching by array strain identifier and recomputing

    tau = f_ijk - f_ij f_k - eps_ik f_j - eps_jk f_i

| screen | terms | n | r | median abs residual |
|---|---|---|---|---|
| Kuzmin 2018 | every term from the same screen | 91,111 | 0.985 | 0.0016 |
| Kuzmin 2018 | f_ij replaced by f_i f_j | 91,111 | 0.538 | 0.046 |
| Kuzmin 2020 | every term from the same screen | 301,706 | 0.976 | 0.00003 |
| Kuzmin 2020 | f_ij replaced by f_i f_j | 301,706 | 0.419 | 0.041 |

**The formula is right and implemented right.** Nothing in the published correction is missing.
The whole loss is one term: the double-mutant query strain's fitness. Replacing only that term
collapses the reconstruction.

**That term is not a record.** Across the Kuzmin 2018 screen only 172 query pairs carry a released
double-mutant query fitness; 129 appear as a double in the 029 build, and those are a different
measurement of the same genotype (a Costanzo or Kuzmin array screen), agreeing with the query-strain
value at r = 0.777 with median absolute difference 0.045. Multiplied by f_k about 1 that difference
lands in tau, whose intermediate calling threshold is 0.08.

**Consistency check.** The 029 Kuzmin-first closure reaches 0.517 on the Kuzmin 2018 screen; the
within-screen reconstruction with only f_ij degraded reaches 0.538. A Kuzmin-first policy already
gets the other two digenic terms right, because Kuzmin's digenic records ARE those control query
screens. So the entire remaining gap is the one missing measurement.

**Action for the data model.** Ingest the query-strain fitness standard (Kuzmin Additional Data S4,
single and double mutant query fitness) as its own records carrying the query role, so a closure can
select f_ij. Without it no label policy over the current records can pass about 0.54 on trigenic.
Recorded in section 3 of `notes-tex/025-s3-closure` (Table 3, 13 pages, make check clean).

Record counts, for reference: 029 holds 299,146 triple gene sets carrying 309,122 stored trigenic
entries (a gene set reached by both Kuzmin screens keeps both); 289,062 have every term of the
identity present, 15,229 of the rest miss one of the three doubles and 8,545 one of the three
singles. The 025 comparable is 352,505 of 376,732.

## 2026.09.21 - S3 seed 1 read out to epoch 103: the plateau is flat for 88 epochs

IGB mmli 2409033, cancelled by another session at 2026-09-20 23:05 after 2 d 20 h 32 min, epoch 103
of 130. Rank-0 run `yb4gjh51`, synced. Validation interaction Pearson on the pinned 010 triples:

| window | value |
|---|---|
| ep 10 to 29 | 0.478 |
| ep 60 to 103 | 0.480 |
| ep 10 to 103 | 0.482 |
| max over epochs, upward-biased | 0.498 at ep 32 |
| S0 fitness 1.0, three seeds, ep 10 to 29 | 0.443 / 0.438 / 0.437 |

Validation is flat from epoch 15 to 103 while train interaction Pearson climbs from 0.47 to 0.768
and validation fitness sits at 0.950 throughout. Eighty-eight epochs after the plateau bought
nothing, so the 130-epoch budget was wrong and the replicates were correctly requeued at 50
(2409578, 2409579). Train at 0.768 is well above the 0.59 at which two independent measurements of
the same triple agree, so everything above that is fit to the training copy's particular noise.

The S3 closure arm therefore beats the S0 arms by about 0.04 and stops there. Where the ceiling
comes from is the subject of the recompute: the build's own fitness reproduces its own trigenic
scores at r 0.230, and the identity is exact within a source screen, so the limit is the join
rather than the model or the measurement.
