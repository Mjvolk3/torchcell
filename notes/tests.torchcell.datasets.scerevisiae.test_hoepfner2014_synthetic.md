---
id: cxg4zr39x3yutzhjypup4b7
title: Test_hoepfner2014_synthetic
desc: ''
updated: 1790550085894
created: 1790550085894
---

## 2026.09.27 - The Hoepfner 2014 HIP-HOP loader built end to end

`tests/torchcell/datasets/scerevisiae/test_hoepfner2014.py` covers the helpers; this file builds the dataset on hand-made raw tables with `DATA_ROOT` pointed into `tmp_path` for the raw-data mirror and the genomes-tier manifest, a duck-typed genome stub, and `_fetch_from_dryad` exercised through a stubbed `_dryad_get`. Sixteen tests: records by `model_dump()` equality, the compound identities (InChIKeys, CIDs, ChEBI ids, boromycin's `UNRESOLVED_PUBLIC` reason), `table_s5_affected_strains.json`, `sourced_values.json`, the drop ledger and the side files asserted exactly, exact error messages. Loader coverage 92%; the `_dryad_get` network loop (hoepfner2014.py lines 738 to 779), `_resolver()` opening a real `SCerevisiaeGenome` (924 to 932), the 500k-record batch commit and `main()` remain. Findings: a repeated `Systematic Name` row is not detected, the cached genotype is reused and a second (strain, condition) record is written with nothing in the ledger (lines 1266 to 1269); a current-status ORF absent from the R64 FASTA universe is dropped with `status: "current"` (line 1248). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
