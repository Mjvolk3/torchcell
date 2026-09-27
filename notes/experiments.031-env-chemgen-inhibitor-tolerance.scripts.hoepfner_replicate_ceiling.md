---
id: 8ez1trxtir5nuc4k95tjqwl
title: Hoepfner_replicate_ceiling
desc: ''
updated: 1790533749436
created: 1790533749436
---

## 2026.09.27 - Can a ceiling be recovered for Hoepfner, which serves no uncertainty

Every other 031 dataset has a prediction ceiling because its source serves a per-record standard
error. Hoepfner serves none, and the loader records that as a typed provenance gap: the source
folds the replicate t-test into the adjusted score rather than releasing a per-cell error. Two
questions follow and they have different answers.

**Are the two technical replicates recoverable? NO.** A census of every column header in both
deposited score matrices:

| file | columns | adjusted-score | MADL | companion z-score | distinct experiments | experiment ids appearing twice |
|---|---|---|---|---|---|---|
| HIP_scores.txt | 5,912 | 2,905 | 51 | 2,956 | 2,905 | **0** |
| HOP_scores.txt | 5,846 | 2,904 | 19 | 2,923 | 2,904 | **0** |

One adjusted-score column per experiment, no experiment appearing twice. The two wells are
already combined inside that column. The only other column per experiment is the companion
gene-wise z-score, which is a second normalization of the same numbers, so correlating it with
its own score measures the normalization and not the measurement.

**This corrects an earlier claim.** The 031 document said the roughly 5,900 columns per file
were "2,956 experiments run in duplicate". The experiment count is right for HIP, the reason is
not: the doubling is the companion z-score column, not a replicate.

**Is there any repeated measurement? YES, and it is the more useful kind.** Fourteen compounds
were screened at the same concentration in more than one study, and a study is a separate
screen with its own control set. That gives 71 condition pairs. Two screens of one condition are
parallel measurements, so their correlation over genes estimates the reliability of one served
value directly, and the ceiling on correlation with the noise-free response is its square root.

| arm | condition pairs | compounds | median reliability | median ceiling |
|---|---|---|---|---|
| HIP | 35 | 13 | 0.636 | 0.797 |
| HOP | 36 | 14 | 0.574 | 0.758 |
| both | 71 | 14 | 0.611 | 0.781 |

So Hoepfner DOES have a ceiling, 0.781, comparable to Vanacloig 0.84, Hillenmeyer HET 0.79 and
Wildenhain 0.87. The best baseline of
[[experiments.031-env-chemgen-inhibitor-tolerance.scripts.per_dataset_baselines]] reaches 24% of
it.

**It is not the same quantity as a served-error ceiling, and the difference is conservative.** A
cross-screen repeat carries batch variation that two wells in one plate do not, so it measures
reliability against everything that changes between screens. That is closer to what a model
trained across screens faces, and closer to Vanacloig's three-batch replicate agreement than to
a within-plate standard error.

**Two cautions.** The fourteen repeated compounds are the ones a screening campaign repeats,
reference compounds rather than a random sample of the 148. And the pairs are spread unevenly:
methotrexate alone contributes 30 of the 71.

Result files: `results/hoepfner_column_audit.csv`,
`results/hoepfner_cross_screen_reliability.csv`. Rendered as table t14 of
`notes-tex/031-unified-representation`.
