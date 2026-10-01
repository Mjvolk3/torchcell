---
id: md779zftey2ciiy1av7jm13
title: Test_ohnuki2018
desc: ''
updated: 1790549985561
created: 1790549985561
---

## 2026.09.27 - The Ohnuki 2018 essential-gene heterozygote CalMorph loader

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` (YAL001C and YDR001C current, YBR002C renamed to YBR001C, YCR001W retired). Four tests: the heterozygous records, the renamed and retired handling, the side files, the download refusal. Loader coverage from this file 94% (the mirror-copy branch at line 166 and `main()` remain). Finding: a missing CalMorph cell is stored as `0.0` (ohnuki2018.py line 295), where Ohya 2005 (line 265) and Ohnuki 2022 (line 259) drop the row. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: the ledger, blank rows, the n.d. refusal

Four to eleven tests, 94 to 100 percent. The ledger; duplicate and blank-ORF rows; a wild-type column that is "n.d." everywhere refused with "CV measurement ACV103_A1B cannot be NaN"; `default_genome()`; both files copied.

Findings: blank or whitespace-only ORF rows are dropped with no log line while the count reads "0 dropped for naming" (lines 224-226); `create_experiment` returns None (254); duplicate spellings give two records.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Findings retired (issues #537, #546)

Retired the Phase 16 findings: blank ORF rows give a counted warning, two spellings of one strain are refused with no `data.csv` or store, `create_experiment` raises, and a refused WT reference leaves no `processed/lmdb`. The blank-cell-stored-as-0.0 finding stays pinned (not an item of #537).

## 2026.10.01 - Review follow-up on PR #591

Added `test_a_tcv_column_is_a_base_parameter_and_refused_by_the_schema` (fails with `TCV` restored); the refused-WT-reference test also asserts no `data.csv`.
