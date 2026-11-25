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

## 2026.10.06 - Phase 21: download, batch path, resolution passes, Table S1 branches

Fixture: the existing in-memory GSE fixture; `requests.get` and `GEOparse.get_GEO` are replaced by recorders, Table S1 variants are written with openpyxl under `tmp_path`.

- `download`: `requests.get(<Box URL>, timeout=60)`, then `get_GEO` for GSE42527, GSE42526, GSE42241, GSE42240, GSE42217, GSE42215 in that order; an existing workbook is not re-fetched; a body of 5 or 999 bytes or an HTTP error raises (full-match message), a body of exactly 1000 bytes is written verbatim and all six series follow; with URL, cause and manual-save path; a GEO failure names the failing series and stops there.
- `process` re-fetches a missing deletion pickle once.
- `_process_batch` equals `_process_sequential` record by record on five genes (CUP9 and HSN1 written; no strain, no arrays, only non-positive pairs skipped).
- `resolve_gene_name_comprehensive`: every pass with exact counters (gene table 3, alias 5, reconciler 1, unresolved 2), including the case-insensitive alias pass and the sorted choice among several candidates.
- `_load_mating_type_map`: `MATα`, `mat alpha` -> BY4742, `mata` -> BY4741, `MATa/alpha` skipped, repeated orf keeps both common names and the last strain.
- A build where genes are named only in `characteristics` (with an `[HS1991]` prefix) and four identical arrays give log2 1, SE 0, variance 0, n 4.

Findings:

- `download` fetches Table S1 from a `uofi.box.com` share link while the missing-file message names Cell's `mmc1.xlsx` (line 968); no sha256 or retrieval record for any file (the stubbed run pins the seven files the loader writes; a real `get_GEO` also leaves the SOFT files, 13 files in the real raw dir).
- `_extract_probe_to_gene_mapping` (lines 923-937): same three identical arms as sameith2015 (`EMPTY`, `NONE` become genes in the fixture). Real reach: `SNR10`, a non-systematic name, is an expression key in all 1484 served Kemmeren records (audit 1).
- A Table S1 without a mating-type column returns two empty maps with only a log line (lines 1108-1114), and a missing `orf name` column raises at line 1007 only to be swallowed by `except Exception` at line 1116. The build then does not end quietly: the LMDB has 0 entries and `post_process` raises `ValueError("Cannot set an empty or None value for gene_set")` (experiment_dataset.py line 809), which does not name Table S1. (Corrected 2026.10.06 after audit 1; the first wording said zero records and no exception.)
- `already_assigned` is accepted and ignored by `resolve_gene_name_comprehensive` (lines 1139-1140) although `process` maintains it (lines 344, 379, 409, 439).
- `convert_gene_name` (lines 1266-1295) has no caller.
- `_log_processing_summary` counts dict keys (line 1320), so its duplicate branch (lines 1324-1326) is unreachable.
- `_process_parallel` logs the written count as "Total gene deletions attempted" and can never log skipped genes, because `_process_batch` drops skips instead of returning `None` (5 genes in, log says 2 attempted; the sequential path says 5 and 3 skipped). Reach: the served build takes the sequential path (`build_dataset_lmdb` passes no `process_workers`); only `experiments/012-sameith-kemmeren/scripts/kemmeren_volcano.py` (`process_workers=10`) reaches the log line, and no record differs.

## 2026.11.24 - One record per array (#1178)

Rewritten for the per-array loader. The fixture adds the second spot of YAL001C (probe 5), the GSE42215 pickle and a MATalpha WT array in GSE42215, and drops the `wt control` array from the deletion series (a non-deletion array there now raises). `VALUE` is 99 on every fixture row, because the loader no longer reads it.

- Nine arrays give nine records in release order: four deletion arrays (CUP9's two dye-swapped arrays stay two records), then the WT series GSE42215, GSE42217, GSE42240, GSE42241 with empty genotypes. Each record: `n_replicates` 1, no SE or variance, two `not_applicable_single_measurement` gaps, `replicate_id` the GEO title, reference the refpool on the same array.
- `_array_signals`: a gene's spots averaged (YAL001C on GSM2: log2 1 and 2 -> 1.5), `VALUE` never read, a 0 or missing signal raises, a non-numeric cell raises, an unmapped probe is ignored.
- Refusals: a deletion-series array naming no deletion, a WT-series array naming one, a WT culture whose `matA` naming disagrees with the series strain, and a deletion ORF without a Table S1 mating type all raise.
- Reference index: records share a reference only when two refpool readings are identical (fixture: [[0], [1], [2], [3, 4, 5], [6, 8], [7]]).
- `@pytest.mark.slow` real-series test: 3,061 records, per-series counts, 1,149 mutants with two arrays and 335 with one, unique titles, 6,169 genes per record.
- Removed with the code they pinned: the replicate-pair, per-mutant mean/SE, reference `n_replicates` (#484), `_process_batch`, `_log_processing_summary` and skipped-gene log tests.
