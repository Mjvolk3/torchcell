---
id: xgjv9u69v3sy5t0v1bxl5je
title: Test_oduibhir2014
desc: ''
updated: 1790549993352
created: 1790549993352
---

## 2026.09.27 - The O'Duibhir 2014 relative-growth-rate loader

The Dataset S2 text file is written into `<root>/raw/`; the loader's row-count self-checksum demands exactly 1312 data rows, so the five hand-made rows are padded with 1307 filler rows that resolve to nothing and are dropped (a fixture constraint, not a bug, oduibhir2014.py line 221). The genome stub carries `gene_attribute_table`. Six tests: the fitness records by `model_dump()` equality, the reference, the dropped-row accounting, the side files, the download refusal. Loader coverage from this file 91% (the mirror-copy branches at lines 179 and 186 and `main()` remain). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: the ledger line, duplicate and lowercase ORFs, the mirror copy

Six to fourteen tests, 91 to 100 percent; the schema-class identity test is now a `transform_item` round trip. The exact drop-ledger line (1309 dropped, raw tokens sorted); a duplicate and lowercase-ORF matrix with fitness 2^-0.5 and 2^-2; the mirror copy and the "Verified" log; `main` with `load_dotenv` stubbed.

Findings: a duplicate or lowercase ORF is stored once per row (lines 292-314); a blank `commonName` is stored as the string "nan" (301); a blank `log2relT` raises "Fitness cannot be NaN" after the store is opened (289), leaving an empty `processed/lmdb` so a retry serves 0 records; a mirror file that fails the sha256 check is left in `raw/` (179-180) and the next constructor builds from the unverified bytes.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: the unverified-raw Finding (#537) `test_a_mirror_file_off_the_pin_is_refused_but_left_in_raw_for_the_next_build` is now `test_a_mirror_file_off_the_pin_is_refused_and_nothing_lands_in_raw`. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Findings retired (issues #537, #546)

Retired the Phase 16 findings: a duplicated ORF (verbatim or lowercase), a blank `commonName` and a blank `log2relT` are now refused with exact `RuntimeError` messages, and each refusal leaves no `processed/lmdb` and no `gene_set.json`, with a second constructor on the same root refusing again.
