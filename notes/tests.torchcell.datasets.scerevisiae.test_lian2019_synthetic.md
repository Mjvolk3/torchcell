---
id: i1t1om2woninwspufm8ckr8
title: Test_lian2019_synthetic
desc: ''
updated: 1790550043940
created: 1790550043940
---

## 2026.09.27 - The Lian 2019 CRISPR MAGIC loader built end to end

`tests/torchcell/datasets/scerevisiae/test_lian2019.py` covers the helpers and sourced constants; this file writes the design workbook and the enrichment TSV under `<root>/raw/`, passes a duck-typed genome stub, and runs `process()`, the drop rules, `_reference`, `_environment` and the LMDB round trip for real. Seven tests: records by `model_dump()` equality per modality, the side files asserted exactly, the mirror deposit and `download()` through a `tmp_path` `DATA_ROOT` with the pinned digests repointed after the refusal on the real pins is asserted. Loader coverage 95% from this file, 96% with the in-memory file; `main()`, the unreachable drop-accounting `RuntimeError` (lian2019.py line 887) and the resolver cache hit (every fixture gene appears once) remain. Finding: the empty-value check precedes the own-background check, so a background gene's blank round cell is counted under `guide_round_has_no_enrichment_value`, not `guide_targets_its_own_round_background` (lines 777 against 780). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
