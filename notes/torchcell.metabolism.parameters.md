---
id: 6xlz1zsfje9545oiiqza538
title: Parameters
desc: ''
updated: 1788314323947
created: 1788314323947
---

## 2026.09.01 - Kinetic and physical parameters, database-first

The organism-portability layer. Resolution is a policy with a per-value provenance tag:
experimental (BRENDA, Open Enzyme Database) before predicted (KcatNet, RealKcat, DEKP)
before an organism default, so a "published only" ablation is a filter rather than a re-run.

Measured coverage on yeast-GEM 9.0.2, which is the argument for the predictors:

| parameter | coverage |
| --- | ---: |
| molecular weight | 1,161 / 1,161 (100 %) |
| k_cat | **148 / 3,728 catalytic units (4.0 %)** |
| measured concentration | 191 / 2,806 metabolites (6.8 %) |

**A join bug found by hit rate.** The SwissProt `gene_id` column is an ALIAS LIST, not a
standard-plus-systematic pair: token counts run from 1 to 11. Taking the last token matched
430 of 1,161 genes, the first matched 38, and selecting the token that matches a systematic
ORF pattern matched all 1,161. Nothing errored in any case. The check that caught it was the
hit rate against a known key set.

Full write-up: [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]]
