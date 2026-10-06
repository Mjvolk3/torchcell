---
id: jdmx7m2hcyr6gg82tzvgtjf
title: Test_sameith2015_synthetic
desc: ''
updated: 1790562898040
created: 1790562898040
---

## 2026.09.27 - The Sameith 2015 loader on in-memory GEOparse objects

Ten tests on real in-memory GEOparse objects pickled to the loader's paths. Loader coverage 71%. Findings: in the single-mutant dataset a double-mutant array is stored as a single deletion of its first gene (sameith2015.py lines 483 and 242); `"wt" in title.lower()` classes `swt1-del-a` (SWT1, YOR166C) as wildtype, so that deletion is never written (line 239); a `MATa` comment maps to BY4742 and `matA`/`MATalpha` to BY4741, the reverse of standard nomenclature (BY4741 is MATa), pinned as written and not yet checked against the supplementary tables (lines 954 to 957); a single 0 signal drops that gene's log2 ratio but keeps its expression value, and phenotype validation then aborts the whole build (lines 671 to 674 against 782 to 788). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.28 - The strain mapping is now asserted the right way round

The Finding test of 2026.09.27 pinned `MATa -> BY4742`. With the loader fix on branch `fix/kemmeren-sameith-channel-strain` ([[torchcell.datasets.scerevisiae.sameith2015]]) the test asserts `MATa strain -> BY4741`, `matA -> BY4741`, `MATα -> BY4742`, blank or unrelated -> BY4742, with the paper quote and the four GEO `strain: BY4741` samples in its docstring. Because both fixture pairs are now BY4741 with the same all-ones refpool they share one reference, so `experiment_reference_index.json` has one entry with `member_indices` `[0, 1]`. 10 tests, all passing under the sentinel `DATA_ROOT`.

## 2026.10.06 - Phase 21: download, the batch path and the static helpers

Fixture: the existing in-memory GSE fixtures; `GEOparse.get_GEO` and `urllib.request.urlretrieve` are replaced by recorders that write hand-made bytes under `tmp_path`. No network.

- `download` (both classes): `get_GEO(geo="GSE42536", destdir=<raw_dir>, silent=False)`, `urlretrieve(<Springer ESM URL>, <raw_dir>/12915_2015_222_MOESM1_ESM.xlsx)`, the pickle round-trips the returned object, `raw/` holds exactly the pickle and the workbook. The single-mutant class re-fetches an existing workbook, the double-mutant class does not. Refusal messages are exact (`GEO download failed` / `Failed to download GSE42536 from GEO`, `Failed to download supplementary data`).
- `process` re-fetches the GEO object through `get_GEO` when the pickle is missing and builds the same records.
- `DmMicroarraySameith2015Dataset._process_batch` and `_process_sequential` on six hand-made groups write the same three records in order (D1 + D2, D3, and the off-SI pair YFL001W/YEL001C at the default strain BY4742); not-double, one-gene and no-data groups are skipped.
- `_calculate_replicate_statistics_static`: A = [1, 3] gives mean 2, SD sqrt(2), SE 1, variance 2.0000000000000004; B = [5] gives NaN SE and variance; equals the instance method.
- `_extract_expression_from_gsm_static` equals both instance extractors on six branches (no table, no `ID_REF`, no Cy3, no probe map, unmapped / non-numeric / zero-signal rows, a `RefPool` source swapping channels: log2(8/2) = 2).
- `_process_sequential` (single) skips not-single, gene-less and no-data groups.

Findings:

- `download` (sameith2015.py lines 155-185 and 928-960) records no sha256 and no retrieval record for either file (under the stub `raw/` holds just the pickle and the workbook; a real `get_GEO` also leaves `GSE42536_family.soft.gz`, likewise unrecorded).
- `_extract_probe_to_gene_mapping` (lines 605-613 and 1631-1639): the three "validate" arms all store `gene_name.upper()`, so a control probe `Empty` becomes gene `EMPTY` and a `None` cell becomes gene `NONE`; only a float NaN is dropped. Those two never occur in the real platform, but the same path stores `SNR10`, a non-systematic name, in every served record (82 single-mutant, 72 double-mutant; audit 1).
