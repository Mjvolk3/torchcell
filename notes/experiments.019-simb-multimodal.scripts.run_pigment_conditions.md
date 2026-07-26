---
id: x87j7yypjrzg50aq9907153
title: Run_pigment_conditions
desc: ''
updated: 1785040443967
created: 1785040443967
---

## 2026.07.25 - The four-condition metabolome-transfer contrast

Config: `experiments/019-simb-multimodal/conf/gh_pigment_base.yaml`
Results: `experiments/019-simb-multimodal/results/pigment_transfer_runs.json`

Runs A1 betaxanthin alone / A2 betaxanthin + mulleder19 / A3 beta-carotene alone /
A4 beta-carotene + mulleder19, composing ONE base config and overriding only
`multitask.active_heads` (and the seed). Four separate YAML files would let the arms drift;
an unmatched split invalidates a Delta, and a Delta is the whole result.

Metrics are Pearson AND Spearman per target, taken at their PEAK over validation epochs
(`{metric}_max`), matching the rest of 019 - these runs peak early and then collapse toward
the per-feature mean under MSE. **Spearman is primary for beta-carotene** (a subjective
ordinal), Pearson for betaxanthin.

Replicated over three seeds because the val split is only ~345 genotypes: a Pearson there
carries an SE around 0.05, so a single-seed Delta of a few hundredths is indistinguishable
from noise. Each seed reshuffles both the split and the init; within a seed all four
conditions share one split exactly, so each Delta is paired.
