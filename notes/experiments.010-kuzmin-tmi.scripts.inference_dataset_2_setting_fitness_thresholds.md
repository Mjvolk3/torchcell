---
id: ak1dt5kzu6oet6ytqrtt3lu
title: Inference_dataset_2_setting_fitness_thresholds
desc: ''
updated: 1781029013629
created: 1781029013629
---

## 2026.06.09 - Defending the Iterative Fitness Improvement Claim with Replicate-Aware Thresholds

This script exists to answer a design question for inference_dataset_2: at what fitness values can we credibly claim that each successive gene deletion (WT to single to double to triple) produces a statistically significant improvement given real measurement noise? It exists to keep the candidate-selection thresholds honest, so that strains chosen for experimental validation are not artifacts of measurement scatter, and to expose the tradeoff between statistical rigor and how many genes/pairs survive filtering.

### Motivation and Use

- Grounds the SMF/DMF thresholds in the empirical noise of the source datasets (SmfCostanzo2016, DmfCostanzo2016, TmfKuzmin2018/2020) rather than round numbers, using one-sample and Welch two-sample t-tests against a fixed WT = 1.0.
- Demonstrates that the originally proposed gap (SMF > 1.10, DMF > 1.15) is too small to be significant at n=4, motivating either larger gaps or more replicates.
- Drives the experimental design recommendation toward 16-24 replicates (384-well plates), which shrinks the required gaps enough to keep thresholds feasible against data availability (it cross-checks how many pairs/triples survive each cutoff).
- Produces console-only output (no saved artifacts); its product is the recommended threshold scheme and replicate count that inference_2 filtering and downstream validation adopt.
