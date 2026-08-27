---
id: 9pwhp71mblhy6tq4pswbfu3
title: Project_census
desc: ''
updated: 1787811160713
created: 1787811160713
---

## 2026.08.27 - Census of every 019-023 W&B project

Script: `experiments/019-simb-multimodal/scripts/project_census.py`
Results: `experiments/019-simb-multimodal/results/project_census.{json,csv}`

Summary-only (one paginated query per project), so unlike `pull_round_leaderboards.py` it
covers ALL projects in minutes rather than needing one history request per run.

**28 projects, 2,187 runs.**

| strand | projects | runs | n_train values seen |
|---|--:|--:|---|
| expression | 9 | 1,369 | 1,125 - 1,253 |
| expression + morphology joint | 5 | 214 | 1,161 |
| morphology | 3 | 183 | 1,161 |
| betaxanthin | 4 | 142 | 3,694 - 4,235 |
| beta-carotene | 3 | 136 | 3,846, 3,849 |
| amino acid | 3 | 97 | 4,234, 4,235 |
| betaxanthin + metabolome | 1 | 46 | 3,832 |

Two findings fall straight out of the n_train column:

- **Morphology is the only strand whose training-set size never varies**, and the value it
  never varies from is **1,161** out of a 4,718-strain screen. Every other strand's size
  moves across rounds, so the constant is a real restriction
  (`require_modalities: [expression_log2_ratio, calmorph]`) rather than an artifact.
- **The betaxanthin rounds are not comparable to each other**: 4,235 in the round that
  reached 0.4301, 3,698 in the round used for the Flux Cone Learning head-to-head, whose
  split pins that paper's 640 test genes out of training.

It also records WHICH METRIC NAME each project used. The honest per-feature metric was
`val/per_gene/pearson_per_gene` (expression) and `val/global/pearson_per_gene` (morphology)
before the rename to `val/<head>/pearson_per_feature`, so any reader of only the new names
sees nothing in the eleven oldest projects.

Context: [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
