---
id: hm6vudbkykk7ocjfdosj8sn
title: Test_nadal_ribelles2025
desc: ''
updated: 1790777393645
created: 1790777393645
---

## 2026.09.30 - Phase 17: per-batch storage, label lookup, the Heat table

New paired file, seven tests, 94.5 to 100 percent with the synthetic sibling (whose helpers are imported). A label with two `ptbs` rows (batch c1 60 cells, sd 1.5; c2 40, 0.9) stores 1.5 / 60 and WT stores 300 cells, not the pooled 500 (the per-batch storage behind the memory note that the assignment is impure across batches); NaCl scalars looked up by label, not row position (record 0.8 / 45, reference 1.05 / 250); a `Heat` table stored under `heat` with the control environment, no scalars and a third reference; every `resolve` branch; the build log; a stale `raw/ptb_summary.Rdata` kept while the mirror copy passes its sha256; `main` with the genome opened `overwrite=False`.

Findings: a repeated `ptbs` label keeps only the first row with no log and no batch field (line 400); any condition other than `nacl` gets the control environment (430-445); an existing raw file is never re-hashed (225).
