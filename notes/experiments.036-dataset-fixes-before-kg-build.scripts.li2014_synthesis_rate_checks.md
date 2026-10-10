---
id: rdqep738ie4cameuw49pcqn
title: Li2014_synthesis_rate_checks
desc: ''
updated: 1791625462966
created: 1791625462966
---

## 2026.10.10 - Duplication by content and served-record round trip (#857)

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/li2014_synthesis_rate_checks.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/li2014_synthesis_rate_checks.json`. Read off the dev stores under `$DATA_ROOT/data/torchcell/`; findings are written up in [[torchcell.datasets.ecoli.li2014]].

1. Overlap of the three Li 2014 records with every record of the Schmidt 2016, Ishii 2007 and Brunk 2016 proteome stores and the Gupta 2024 turnover store: shared keys, keys whose two numbers are identical, Spearman. BW25113 tags are carried to MG1655 b-numbers through a one-to-one ECK synonym. Measured: 0 identical values in every store; Spearman 0.614 to 0.847 (Schmidt), 0.305 to 0.776 (Ishii), 0.447 to 0.855 (Brunk), -0.126 to 0.095 (Gupta).
2. Every record of the datasets `schema_impact_check.py` names as impacted through `ExperimentType`, validated through this branch's schema and dumped again: 0 of 622 records differ across 12 stores; `IsoprenolTiterDeSiqueira2025Dataset` is not valid under this branch because another branch's schema wrote it.
