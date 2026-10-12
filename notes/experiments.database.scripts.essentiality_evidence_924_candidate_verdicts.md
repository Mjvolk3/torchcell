---
id: qib2t4f1xtxkimxk3abcvai
title: Essentiality_evidence_924_candidate_verdicts
desc: ''
updated: 1791817458251
created: 1791817458251
---

## 2026.10.12 - G5 of Baba 2006 and Gerdes 2003 re-gated on the #924 evidence

`experiments/database/scripts/essentiality_evidence_924_candidate_verdicts.py` re-runs G5 of the two verdicts that recorded it as a gap on #924 (`database/candidates/babaConstructionEscherichiaColi2006.json`, `database/candidates/gerdesExperimentalDeterminationSystem2003.json`, both `admissible_with_gaps`). G1 to G4 read nothing the change touched, so they are kept as stored with their hand-written reasons (the inventory counts, the content comparison of the agreement script). G5 is `gates.evaluate_g5("GeneEssentialityPhenotype", ())` and its evidence is the loader's own sourced statements: the scoring rule, the definition of the call and the caveat, the same `SourcedValue` objects every record's `GeneEssentialityEvidence` carries.

Result: both `admissible` (G1 to G5 pass). Schema: [[torchcell.datamodels.schema]], 2026.10.12 section on #924.
