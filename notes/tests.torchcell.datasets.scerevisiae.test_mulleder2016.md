---
id: app3wgcc3t5ki8s6dgcpriw
title: Test_mulleder2016
desc: ''
updated: 1790550001337
created: 1790550001337
---

## 2026.09.27 - The Mulleder 2016 amino-acid metabolome loader on a synthetic Table S3

The workbook is written with openpyxl under the loader's `.xls` filename (pandas sniffs the zip container and opens it with openpyxl) into `<root>/raw/`; no genome is involved. Six tests: the metabolite records with their amino-acid dicts, the reference, the side files, the download paths. Loader coverage from this file 83%; the `urlopen` branch (lines 137 to 149) and `main()` remain. Findings: the reference baseline is every row of the summary sheet, not only `AMINO_ACIDS`, so an extra `cysteine` row gives a 20-key reference against a 19-key experiment (mulleder2016.py lines 165 to 167); `download()` returns early without hashing an already-present raw file although its docstring says the sha256 is verified (line 135; Ozaydin, Cachera, da Silveira, Ohya, Ohnuki and O'Duibhir do verify). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
