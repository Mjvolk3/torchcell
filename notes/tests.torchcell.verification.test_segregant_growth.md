---
id: 8uvu656vnwo6g6krw7xoxrc
title: Test_segregant_growth
desc: ''
updated: 1790759643044
created: 1790759643044
---

## 2026.09.30 - Phase 12: every verdict on a synthetic release

Four to twenty-five tests, 75 to 99 percent (about 22 s alone, each test rebuilds the synthetic release). The passing panel pins the order of all 20 results and the exact message of the module's 14 own results (646 = 17 x 38 records, 80 = 16 x 5 marker checks); one table per failure mode with its verdict and message (L0 at record 5, an undocumented environment, NaN, a tampered value, a nonzero or missing reference, a dropped segregant at 608 records, a short block, a mixed measurement column, a row in two genotype files, a chrII reference base changed to a 0.6 match at a `>=` threshold, a gene outside SGD at 0.500, an empty gene set, a missing parent, a count mismatch); the L3 sourced-value audit against a mirror under `tmp_path` (53 audits, 50 failing on xls hash drift, one removed quote "NOT found", an edited `mapping.R`); `segregant_gene_set` and `__main__`. The frontmatter lines were added.

Findings: `mosaic_round_trip` can fail while its message still says the mosaics re-expand (lines 357-360); `__main__` exits 0 without running anything (645-646).
