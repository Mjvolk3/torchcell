---
id: app3wgcc3t5ki8s6dgcpriw
title: Test_mulleder2016
desc: ''
updated: 1790550001337
created: 1790550001337
---

## 2026.09.27 - The Mulleder 2016 amino-acid metabolome loader on a synthetic Table S3

The workbook is written with openpyxl under the loader's `.xls` filename (pandas sniffs the zip container and opens it with openpyxl) into `<root>/raw/`; no genome is involved. Six tests: the metabolite records with their amino-acid dicts, the reference, the side files, the download paths. Loader coverage from this file 83%; the `urlopen` branch (lines 137 to 149) and `main()` remain. Findings: the reference baseline is every row of the summary sheet, not only `AMINO_ACIDS`, so an extra `cysteine` row gives a 20-key reference against a 19-key experiment (mulleder2016.py lines 165 to 167); `download()` returns early without hashing an already-present raw file although its docstring says the sha256 is verified (line 135; Ozaydin, Cachera, da Silveira, Ohya, Ohnuki and O'Duibhir do verify). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 14: replicate counts, medium, the ledger, bad cells

Six to fifteen tests, 83 to 99 percent (only the `preprocess_raw` passthrough at line 221 is left). Findings: a `data_raw` sheet with three YAL001C rows still gives `n_replicates` 1 on all 19 keys because the loader never reads that sheet (line 249; issue #488); the reference is the MCD population mean but says n = 1 on all 19 keys (257; issue #489); both record and reference carry `SM_AGAR` (solid), not the liquid `SM` the amino acids are extracted from (issue #143); the ledger counts collided ORFs, not dropped rows, so four YBR001C rows log "1 ORF collisions deduped" although three rows were dropped (182-193); a blank concentration cell is served as NaN and the schema does not reject it, while a text cell (`n.d.`) stops the build with Python's `float()` error (186). Also pinned: `download()` and `main()`.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_download_trusts_a_present_file_without_hashing` is now `test_download_leaves_a_present_file_to_the_build_check`, plus the build-time refusal test. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
