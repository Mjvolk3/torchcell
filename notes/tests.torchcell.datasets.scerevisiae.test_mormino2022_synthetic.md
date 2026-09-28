---
id: 8207pn90fkzu9bf287be703
title: Test_mormino2022_synthetic
desc: ''
updated: 1790550051952
created: 1790550051952
---

## 2026.09.27 - The Mormino 2022 loader built end to end

`tests/torchcell/datasets/scerevisiae/test_mormino2022.py` covers the helpers and sourced constants; this file writes a stub PDF and a `paper.md` holding the tables under `<root>/raw/`, passes a duck-typed genome stub and runs `process()` for real. Nine tests: records by `model_dump()` equality, the side files asserted exactly, the mirror deposit and `download()` through a `tmp_path` `DATA_ROOT`. Loader coverage 96%; `main()` remains. Findings: `process()` asserts `self.genome is not None` before the first `_resolve`, so `genome=None` raises a bare `AssertionError` and the loader's "requires a genome" message is unreachable from a build (mormino2022.py line 620 against 552 to 554); an unresolved Table 1 gene raises after `_open_write_lmdb`, leaving an empty `processed/lmdb`, so a retry on the same root skips `process()` and serves zero records with no `dropped_records.json` or `gene_set.json` (lines 633 to 637; the audit failure at 619 happens before the store opens, so that retry rebuilds normally, pinned as the contrast); `gene_set.json` lists the two cassette slot names `BM3R1-HAA1-mTurquoise2` and `sfpHluorin` beside the 12 ORFs (lines 642 to 651 with experiment_dataset.py 469 to 473). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
