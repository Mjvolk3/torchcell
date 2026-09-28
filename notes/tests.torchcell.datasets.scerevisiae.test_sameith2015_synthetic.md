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
