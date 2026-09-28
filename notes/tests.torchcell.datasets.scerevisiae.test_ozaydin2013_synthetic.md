---
id: 3cvmxxayyzs8kz6yuisenu4
title: Test_ozaydin2013_synthetic
desc: ''
updated: 1790549946481
created: 1790549946481
---

## 2026.09.27 - The Ozaydin 2013 visual-screen loader on a synthetic SI workbook

The SI workbook is written with openpyxl into `<root>/raw/` so PyG never calls `download()`, and `process()` runs for real; no genome is involved because the loader validates ORF names with a regex. Six test functions (eleven cases): the records by `model_dump()` equality, the reference, the side files, the LMDB round trip, the download refusal on the real pinned digest. Loader coverage from this file 84%; the `urlopen` branch (lines 199 to 215) and `main()` stay unreachable. Findings: `gene_set.json` carries the cassette's `crtI`, `crtYB` and `YPL069C` beside the deletions (ozaydin2013.py 356 with experiment_dataset.py 497 to 501), and any SI strain other than BY4741 or BY4730 (W303 in the fixture) becomes BY4741 in the reference while `data.csv` keeps the source strain (lines 345 to 347). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
