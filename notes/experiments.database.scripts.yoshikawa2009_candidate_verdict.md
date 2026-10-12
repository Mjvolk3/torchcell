---
id: sz3mcga9ysp9f7nyzcc6wpt
title: Yoshikawa2009_candidate_verdict
desc: ''
updated: 1791802891025
created: 1791802891025
---

## 2026.10.12 - Re-gating an off-schedule key once its release is deposited

Yoshikawa 2009 has no row in the yeast candidate table (schedule row 85 is Yoshikawa 2011, a different paper), so `candidate-gate gate` cannot find it. This script re-runs the yeast Zotero audit's gate composition for the one entry, the [[experiments.database.scripts.messner2021_candidate_verdict]] pattern, with three changed inputs: G3 inventories the raw mirror's `manifest.json` (the publisher `.xls` and the YeastPhenome `.xlsx`), G4 reads `yoshikawa2009.py` as the key's own module (and takes the CLI's PMID inputs), and G5 is given the two stress compounds (ethanol, sodium chloride). With `--write` it replaces `database/candidates/yoshikawaComprehensivePhenotypicAnalysis2009.json`. Loader: [[torchcell.datasets.scerevisiae.yoshikawa2009]].
