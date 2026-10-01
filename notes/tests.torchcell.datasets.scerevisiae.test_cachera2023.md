---
id: v80etp562svfzvn9dyy1b03
title: Test_cachera2023
desc: ''
updated: 1790765131194
created: 1790765131194
---

## 2026.09.30 - Phase 14: a twelve-row CSV with the released 27-column header

Two to eighteen tests; the file-level data gate used to skip every test and now applies only to the two mirror tests, so the module goes from 0 percent (this file alone) to 100. Five kept records in order (ACN9 stored as SDH7, FLO8 storing its id in both fields); the whole AAC1 record with SE 1.2/4 = 0.3 and n 16 plus its reference and publication; whole RENAMED and NON_GENE_FEATURE records; a one-colony record with a NaN SE and n 1; a blank count discarding the released std (line 256); the single reference index [0..4]; the gene set carrying `CYP76AD1` and `DOD`; the exact `data.csv` and drop-summary line; refusal without a genome; `download` under 10,000 bytes, off the pin, and on the pin with an in-place re-check; `main` and the schema classes.

Findings: every record and reference store SC, synthetic, 30 C with a fluorescence `measurement_type`, against the paper's YPD + G418 plate, unstated temperature and color readout (issue #509; lines 339-342, 65); a raw file off the sha256 pin builds unverified because the check lives only in `download` (175-183).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_a_raw_file_off_the_pin_builds_unverified` (#528) is now `test_a_raw_file_off_the_pin_is_refused_at_build_time`. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Real genome only when its database is trusted

The data-gated genome construction now first calls `require_trusted_genome_database` (see [[tests.torchcell.conftest]]): when the real `data.db` would be built or migrated, the test fails by name instead of migrating the shared root.
