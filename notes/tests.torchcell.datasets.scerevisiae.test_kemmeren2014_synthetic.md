---
id: rmt8lv8803ph5cdqp8tvohu
title: Test_kemmeren2014_synthetic
desc: ''
updated: 1790562890516
created: 1790562890516
---

## 2026.09.27 - The Kemmeren 2014 loader on in-memory GEOparse objects

Eight tests on real `GEOparse.GEOTypes` GSE, GSM and GPL objects built in memory and pickled to the paths the loader checks, so the network fallback never runs; `process_workers=1` goes through `_process_parallel` (its forked body is invisible to coverage). Loader coverage 70%. Findings: the `ValueError` for a missing "orf name" column is swallowed by the `except Exception` at line 1700, so a malformed Table S1 silently yields two empty maps (line 1591); the channel rule checks `"-a"` before `-b`, so a dye-swapped array for a gene with `-A` in its name reads the refpool channel as the deletion (line 902). Noticed, not pinned: a hardcoded 2633 divided by the deletion count raises `ZeroDivisionError` with none (line 459). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.28 - Rewritten for the GEO-metadata channels and within-array ratios

The 2026.09.27 fixture encoded the loader's belief (deletion in Cy5 on a "-a" title). With the loader fix on branch `fix/kemmeren-sameith-channel-strain` ([[torchcell.datasets.scerevisiae.kemmeren2014]]) every fixture array carries GEO-style metadata: `label_ch1` Cy5, `label_ch2` Cy3, and `source_name_ch1` / `source_name_ch2` naming the reference pool in one channel and the culture in the other. GSM1 `[HS1991] cup9-del-a` has the refpool in Cy5 (4, 4, 1) and the deletion in Cy3 (2, 8, 0), GSM2 `cup9-del-b` the swap, so CUP9's within-array ratios are -1 and +1 for YAL001C (mean 0, SE 1, variance 2.0000000000000004), the mirror for YBR001C, and +2 alone for Q0010; the linear `expression` is the mean deletion signal over the kept arrays (5, 5, 4) and the reference `expression` the mean refpool signal (4, 4, 1). GSM3 names its gene only in `characteristics_ch1`, exercising the classification that now reads both channels. The MATalpha wildtype array names its reference `ref1`, as 193 GSE42217 arrays do.

New tests: `_channel_columns` from metadata regardless of the title (a "-b" array of `ycr087c-a-del` reads the deletion from Cy5; `ref1` is a reference; `ref2-del` is not; two references, none, or labels other than Cy5 + Cy3 raise), `_extract_channels_from_gsm_static` pairing the two signals row by row and raising on a missing column, `_collect_replicate_pairs_static` giving one pair per array in array order, `_validate_channel_assignment` returning 0.5 for one depleted and one raised array and NaN with nothing to check, `_extract_refpool_from_wt_gsm` keeping positive refpool values only, and `create_expression_experiment` on pairs (the skip cases and an exact phenotype dump). 12 tests, all passing under the sentinel `DATA_ROOT`; the parallel path builds the same records.
