---
id: x33uyzztnz5a4sm4dw468hl
title: Test_bloom2019_synthetic
desc: ''
updated: 1790550078396
created: 1790550078396
---

## 2026.09.27 - The Bloom 2019 segregant loader built end to end

`tests/torchcell/datasets/scerevisiae/test_bloom2019.py` covers the parsers; this file builds the dataset on hand-made cross tables under `tmp_path`. The loader's release constants (`CROSSES`, `EXPECTED_SEGREGANTS`, `N_SEGREGANTS`, `EXPECTED_RECORDS`) are repointed with `monkeypatch`, and because `read_cross_table` hardcodes `engine="xlrd"` (bloom2019.py line 737) and no BIFF writer is installed, `pandas.read_excel` is wrapped to substitute `openpyxl` for `xlrd`; every other line of the join runs. Seventeen tests: the segregant records by `model_dump()` equality, the marker and block expansion on the build path, `block_counts.json`, `data.csv` and the side files asserted exactly, the exact error messages. Loader coverage 96% (`main()` and four unreachable guards remain). Finding: the module docstring (lines 57 to 59) promises a `ProvenanceGap` on the 30 C representative temperature, but `_environment` (lines 861 to 873) stores `provenance_gaps == []` on every plate and `ConditionSpec.temperature_sourced` is never read by the build. A source change that would remove the engine wrapper: drop the `engine="xlrd"` kwarg, since pandas sniffs the format. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
