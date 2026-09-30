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
