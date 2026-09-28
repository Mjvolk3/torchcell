---
id: lh2t343pw0bt3p5um4zgdih
title: Test_smith2016_synthetic
desc: ''
updated: 1790550035954
created: 1790550035954
---

## 2026.09.27 - The Smith 2016 CRISPRi chemogenomics loader built end to end

`tests/torchcell/datasets/scerevisiae/test_smith2016.py` covers the helpers and sourced constants; this file builds the dataset on a hand-made xlsx under `<root>/raw/` with a duck-typed genome stub (`gene_set`, `feature_index["standard_to_ids"]`, `resolve_gene_name`), so `process()`, the drop rules and `DropLog`, `_reference`, `_environment` and the LMDB round trip run for real. Eight tests: records by `model_dump()` equality, the side files (`gene_set.json`, `experiment_reference_index.json`, `dropped_records.json`, `build_manifest.json`, the `processed/interned` counts) asserted exactly, `deposit_raw_mirror` and `download()` exercised by pointing `DATA_ROOT` into `tmp_path` with the pinned sha256 constants monkeypatched to the fixture digests after the refusal on the real pins is asserted. Loader coverage 95% from this file, 96% with the in-memory file; `main()` and the unreachable drop-accounting `RuntimeError` (smith2016.py line 712) remain. Finding: the minus-ATc reference phenotype carries no `n_samples`, `sample_unit` or uncertainty while every record stores `n_samples=1`, `sample_unit=pooled` (lines 547 to 552 against 636 to 645). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
