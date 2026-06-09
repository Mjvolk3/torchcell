---
id: g1p7iez2kpywbmt4264d51o
title: Inference_dataset_2_setting_fitness_thresholds_simplest_assumptions
desc: ''
updated: 1781029020552
created: 1781029020552
---

## 2026.06.09 - A Worst-Case Sanity Floor for the Four-Level Fitness Thresholds

This script exists as the deliberately conservative companion to the replicate-aware threshold analysis: it strips away replicate averaging and dataset-specific noise to ask what fitness gaps would be defensible under the harshest assumptions. It exists so that the inference_2 thresholds have a known worst-case floor and so we can see how much rigor we sacrifice or gain by moving between correction schemes and tail choices.

### Motivation and Use

- Assumes worst-case SD = 0.07 and treats SD = SE (equivalent to n=1, no replicate gain), making the resulting gaps an upper bound on what we would ever need.
- Frames the four-level chain (WT, SMF, DMF, TMF) as three pairwise comparisons and applies Bonferroni correction, so the family-wise claim of monotonic improvement holds.
- Tabulates required thresholds across correction (none vs Bonferroni), alpha (0.05 vs 0.01), and tails (one vs two), surfacing the conservative/moderate/relaxed menu the experimenter chooses from.
- Console-only output; its product is the worst-case gap (roughly 0.1 per step under two-tailed Bonferroni) that bounds the more optimistic replicate-based thresholds.
