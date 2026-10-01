---
id: 4a56wjvhtoopyf50y02hdka
title: Test_ohnuki2022
desc: ''
updated: 1790549977946
created: 1790549977946
---

## 2026.09.27 - The Ohnuki 2022 quadruple-deletion CalMorph loader

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` (YAL001C, YBR001C and the background gene YGL013C current; YDR012W renamed to the background gene YDR011W; YCR001W retired). Three tests: the records with the 3Delta background genotype, the side files, the download refusal. Loader coverage from this file 86%; the mirror-copy branch (lines 207 to 214) and `main()` remain. Finding: `gene_set.json` includes the three 3Delta background genes (ohnuki2022.py line 336); a missing CalMorph cell drops the row (line 259). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 15: the ledger, blank cells, downloads

Three to twelve tests, 86 to 100 percent. `transform_item`; inert `preprocess_raw` and `create_experiment`; the drop ledger as logged (`['YBR001C']`, then `['YGL013C', 'YDR011W']`, then the retained count); a clean base-only matrix (CV None, nothing logged); a non-numeric reference cell raising `ValueError`; `default_genome` called once when no genome is passed; `download` with the pins patched to the synthetic files, a skip of verified raw files once the mirror is gone, the "sha256 mismatch after copy" refusal; `main`.

Findings: a duplicated ORF gives two records; the reference mean skips a blank cell (line 313) while a blank mutant cell drops the whole strain (239).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Issue #533 Findings Retired

- Retired: a repeated ORF giving two records and the reference mean skipping a blank cell.
- Now asserted: `test_repeated_target_orf_refuses` and `test_blank_reference_cell_refuses` with exact messages and no LMDB written; `test_clean_base_only_matrix` keeps the base-only path.
