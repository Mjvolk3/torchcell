---
id: 6n5gbj4timsk6nlcnz58kqh
title: Test_kuzmin_released_zero_sd
desc: ''
updated: 1791806059028
created: 1791806059028
---

## 2026.10.12 - A released SD of 0 is kept as released

Pins the owner decision of 2026-10-12 for #894, #895 and #897: a combined-mutant SD the release gives as exactly 0 is stored as released (`fitness_uncertainty` = `fitness_std` = 0.0, `sample_sd`, `n_samples` 4, `colony`, `fitness_se` 0.0, no `ProvenanceGap`), never dropped. Hermetic: both Kuzmin uncertainty helpers on a 0. `--data`: Kuzmin 2018 Data S1 (4 digenic rows in the Dmf frame, 5 trigenic in the Tmf frame) and Kuzmin 2020 Tables S1 + S3 (18 + 15 digenic rows in the Dmf frame, by `screen_id`), each built into a record carrying the 0. See [[torchcell.datasets.scerevisiae.kuzmin2018]] and [[torchcell.datasets.scerevisiae.kuzmin2020]].
