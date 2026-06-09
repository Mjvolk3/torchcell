---
id: pu6o5esl2hrdor0e7gp39yl
title: Inference_dataset_3_jonckheere_terpstra_thresholds
desc: ''
updated: 1781029034361
created: 1781029034361
---

## 2026.06.09 - Trading Pairwise Rigor for a Monotonic-Trend Test to Unlock a Larger Candidate Pool

This script exists because the pairwise-t-test thresholds of inference_2 were so strict that almost no genes qualified (only about seven singles above 1.12), starving the candidate pool. It reframes the central claim as an ordered-trend hypothesis (WT <= SMF <= DMF <= TMF) and uses the Jonckheere-Terpstra test, which needs no multiple-testing correction and detects small consistent gaps the chain of pairwise tests would miss, so inference_dataset_3 can relax its thresholds while still defending iterative fitness improvement.

### Motivation and Use

- Implements the JT statistic from scratch (sum of Mann-Whitney U counts, normal approximation) and uses Monte Carlo power simulation to find the minimum per-step gap that achieves 80 percent power at n = 4, 8, 16 replicates.
- Justifies the relaxed inference_3 scheme: only the best gene/pair must exceed baseline (max(smf) > 1.04, max(dmf) > 1.08) while all members merely stay viable (> 0.80), versus inference_2's requirement that all exceed 1.0.
- Quantifies the win directly against the Bonferroni pairwise approach and checks data availability at the relaxed cutoffs, confirming the candidate pool grows by orders of magnitude.
- Console-only output (a companion test file validates the JT statistic); its product is the inference_3 filtering thresholds and the experimental claim of a significant monotonic fitness trend.
