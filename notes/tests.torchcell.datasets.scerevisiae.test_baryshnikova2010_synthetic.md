---
id: ert48wiysw17st90gb8720q
title: Test_baryshnikova2010_synthetic
desc: ''
updated: 1790550070815
created: 1790550070815
---

## 2026.09.27 - The Baryshnikova 2010 SGA fitness loader built end to end

`tests/torchcell/datasets/scerevisiae/test_baryshnikova2010.py` covers the helpers; this file builds the dataset on hand-made raw tables under `tmp_path` with a duck-typed genome stub. The loader cannot build on anything but the exact release (`_EXPECTED_ROWS`, `_SV_COMPOSITION`, `_UNRESOLVABLE`, `EXPECTED_RECORDS`), so those module constants are repointed with `monkeypatch` (the YeastPhenome precedent) and one test pins the unpatched checksums' exact rejections. Ten tests: records by `model_dump()` equality for the deletion, DAmP and temperature-sensitive allele kinds, the references and the side files asserted exactly. Loader coverage 98% (`main()` remains). Findings: for an alias row the stored `perturbed_gene_name` is the resolved current ORF, not the released token, and only `strain_id` keeps the raw id verbatim (baryshnikova2010.py lines 671 to 680 and 763; Hoepfner keeps the source ORF); `process` builds one reference per allele kind but the deletion and DAmP references are identical objects, so the content-hashed index has two entries, not three (line 741). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
