---
id: 6o9iimg0sik706vl5su569k
title: Reliability_reconciliation
desc: ''
updated: 1790478048901
created: 1790478048901
---

## 2026.09.26 - The two reliability estimates are reconciled, and only Vanacloig passes

The served-SE reliability index and the raw replicate Spearman differ in level by about a
factor of two, which left the prediction ceiling ambiguous. They are not estimates of the
same quantity: `replicate_rho` is the reliability of ONE replicate, and the served response
is the MEAN over that condition's replicates. Lifting the single-replicate value to the mean
of k by Spearman-Brown, `rel_mean = k * rel / (1 + (k - 1) * rel)`, makes them comparable.

| dataset | raw single | k | predicted index | measured index | residual |
|---|---|---|---|---|---|
| Vanacloig 2022 | 0.389 | 3 | 0.656 | 0.714 | +0.040 |
| Hillenmeyer HOM | 0.419 | 2 | 0.591 | 0.917 | +0.281 |
| Hillenmeyer HET | 0.251 | 2 | 0.423 | 0.841 | +0.360 |

Vanacloig's residual is 0.040, so its served standard error and its raw batch scatter
describe the same noise and its index is trustworthy in level, not only in rank. Hillenmeyer's
residuals are 0.281 and 0.360, so the served standard error there is blind to a large noise
component, confirming the earlier conclusion by a route that does not depend on the rank
correlation.

**The ceiling therefore depends on the target, and both numbers are real.**

- Predicting the served three-batch mean of a Vanacloig compound, the ceiling on r against
  the noise-free response is sqrt(0.714) = 0.845 at the median, 0.89 for isobutanol and 0.88
  for furfural.
- Predicting one fresh batch, the ceiling is sqrt(0.389) = 0.62 at the median, and two
  independent batches of the same compound agree at only 0.389.
- For Hillenmeyer the honest numbers are the raw ones: 0.419 single-array for HOM and 0.251
  for HET.

**Consequence for cross-dataset transfer (derivation, from the measured reliabilities).**
The attenuation limit on a correlation between two noisy measurements is
sqrt(rel_1 *rel_2). With Vanacloig at 0.714 and HOM at its honest 0.419, a
Vanacloig-by-HOM condition correlation could reach sqrt(0.714* 0.419) = 0.55. The measured
cross Spearman median is 0.010. Noise is therefore not what limits the transfer; the two
datasets genuinely do not share compound-level response structure, and the earlier
conclusion that only methyl methanesulfonate transfers is not a noise artifact.

Result file: `experiments/031-env-chemgen-inhibitor-tolerance/results/reliability_reconciliation.csv`
