---
id: i3mkmilo9ielslpk9at3634
title: Messner2021_candidate_verdict
desc: ''
updated: 1791747830197
created: 1791747830197
---

## 2026.10.11 - Re-gate of the off-schedule Messner 2021 row

Off-schedule keys have no row in the yeast candidate table, so `candidate-gate gate` refuses them ("0 rows carry its DOI(s)"). This script loads the yeast audit script by path, takes its `AuditEntry` for `messnerUltrafastProteomicsScanning2021` (the `RowSpec`, the stop rule, the evidence), and gates it with what the loader added: the raw mirror (G3), the module's DOI (G4) and the loader's 19 compound labels (G5). Run after committing, so the verdict names a clean commit; `--write` replaces `database/candidates/messnerUltrafastProteomicsScanning2021.json`. Result on 2026-10-11: `admissible`, G1 to G5 pass. See [[torchcell.datasets.scerevisiae.messner2021]].
