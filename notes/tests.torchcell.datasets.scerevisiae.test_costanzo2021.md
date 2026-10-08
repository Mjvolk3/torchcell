---
id: og3pkma0w9vii5roqolvkdj
title: Test_costanzo2021
desc: ''
updated: 1790762183461
created: 1790762183461
---

## 2026.09.30 - Phase 13: a real xlsx in the released layout

Nine to thirty tests, 75 to 95 percent (the rest is `main()`). The fixture is an xlsx with a decoy first sheet, the `Diff. Mutant fitness_Conditions` sheet and a `" benomyl "` header; nine rows (a second allele of the same ORF, a renamed ORF, a non-gene feature, a retired ORF, a padded copy of row 1, a blank Systematic Name, an AMBIGUOUS gene, blank and 0.0 cells) give eight records in row-major then condition order; `model_dump()` equality for a deletion row and a ts-allele-on-galactose row plus the reference and publication; the drop log as whole JSON (5 of 9 strains kept, 6 cells dropped); one reference over members 0 to 7; all 13 doses with units, 26 C as a derivation, `n_samples` 3 screens; refusals with exact messages (no genome, a missing condition column, a missing sheet, no mirror file, a sha256 mismatch naming both digests, `download` hashing an existing raw file); `deposit_raw_mirror` with a frozen time.

Findings: the sha256 pin is checked only inside `download` (lines 595-622) and a failed check leaves the copied bytes in `raw/`, so the next construction builds all 8 records from the unverified file; a refused deposit has already created `<mirror>/data/` (516, 518); a blank Systematic Name is resolved and logged as `"nan"` (770); a repeated row is stored twice with no ledger entry (769-821); an AMBIGUOUS drop keeps no candidate list.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_a_failed_sha256_check_leaves_bytes_the_next_build_uses_unverified` and `test_deposit_refuses_a_source_off_the_pin_after_creating_the_mirror_dir` (#524) are now `test_a_failed_sha256_check_leaves_nothing_in_raw_and_every_retry_refuses` and `test_deposit_refuses_a_source_off_the_pin_before_creating_the_mirror_dir`; `test_download_called_on_an_existing_raw_file_hashes_it_in_place` became the build-time refusal test. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.

## 2026.10.01 - Fix PR for the pinned findings

Retired the blank-name, repeated-row and AMBIGUOUS-candidate findings (issue #524). The fixture sheet is now seven rows (6 records); the blank and repeated rows are added only in the refusal tests, which assert `BlankSystematicNameError` and `RepeatedStrainRowError` with exact messages and no store or drop log. The drop log asserts `candidates` on every entry.

## 2026.10.08 - No processed/ after a sha256 refusal; raw/ links the mirror

Issue #524 (Costanzo part). A build-time refusal used to leave an empty `processed/`, because PyG's `_process` creates it before calling `process()`. The loader now verifies Data File S1 in an `_process` override ahead of that, so `test_a_raw_file_off_the_pin_is_refused_at_build_time` asserts no `processed/`, no `preprocess/` and no drop log. New `test_a_corrupted_raw_file_refuses_before_processed_exists_and_on_every_retry` links the fixture file from the mirror under its pin, appends one byte to it, and asserts the refusal names the `raw/` path, the pin and the observed digest, leaves only `raw/` under the root, and repeats identically on a second construction with the link untouched. `test_download_links_the_mirror_file_and_verifies_the_pin` (was `..._copies_...`) asserts `raw/` holds a symlink to the mirror file. The two refusal tests fail on the previous loader (checked by running them against the `origin/main` module: 3 failed, 32 passed).
