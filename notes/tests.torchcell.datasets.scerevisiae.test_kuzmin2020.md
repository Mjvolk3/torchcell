---
id: vxe9okxo4d2q096fikdmshf
title: Test_kuzmin2020
desc: ''
updated: 1790769107170
created: 1790769107170
---

## 2026.09.30 - Phase 15: the S5-over-S1/S3 rule and the rest, no file-level skip

Four to thirteen tests, 0 to 95 percent alone (100 with the siblings); the import-time gate is gone. Smf allele single with a blank SD; Dmf allele query times unknown array stored as `SgaAllelePerturbation`, tm801 in S1 and S3 giving one record, Table S5's 0.5 / 0.006 winning over S1/S3's 0.55 labeled `bootstrap_se` n = 12 with the exact INFO and WARNING lines including "(max |diff| 0.0500)"; Tmf, Dmi and Tmi records with reference SD 0.025; `subset_n`; `download` for five; `transform_item` for five; `main` with exact stdout.

Findings: a blank SD becomes NaN in Smf (line 339) and Dmf (627); a duplicated Table S5 "Double mutant" row doubles that strain's record (merge at 173); `main` builds only Tmi at a root relative to the working directory (1384).
