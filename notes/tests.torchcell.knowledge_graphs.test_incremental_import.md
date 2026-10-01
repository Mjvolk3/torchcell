---
id: nptpyz09vwicvlx9tzsafsz
title: Test_incremental_import
desc: ''
updated: 1790773286517
created: 1790773286517
---

## 2026.09.30 - Phase 16: every refusal, the call script byte for byte, the edge filter

Eleven to twenty-three tests, 91 to 100 percent. The module has no admission verdict (that is `kg_manifest`); pinned are every refusal message, the whole call script, the saved analysis JSON, sample caps, batched existing-edge queries through a scripted session, and `main` through a fake driver. Finding: re-running `filter_existing_edges` re-reads `unfiltered/` and writes back a row the first run found already in the store (lines 481, 493-508).

## 2026.09.30 - Finding retired (issue #538)

- Retired: a rerun of `filter_existing_edges` writing back a row the first run dropped.
- Now asserted: run 2 asks only about the current pairs and leaves the part file empty, a third run with an empty lookup changes nothing, and `unfiltered/` still holds run 1's original.

## 2026.09.30 - Crash-atomic rewrite asserted (issue #570)

- Added `test_filter_existing_edges_crash_before_rename_leaves_the_part_intact`: with `os.replace` raising once, the part file holds both original rows, `unfiltered/ExperimentMemberOf-part000.csv.filtering` holds the one kept row, and `discover_csv_groups` sees only the part file; the rerun completes the rewrite, removes the staged file and counts one drop.
