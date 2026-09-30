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
