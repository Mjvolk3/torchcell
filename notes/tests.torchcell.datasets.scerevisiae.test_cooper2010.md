---
id: l7b462wyouuihb9w46wfqz9
title: Test_cooper2010
desc: ''
updated: 1790756800423
created: 1790756800423
---

## 2026.09.30 - Phase 11: the full build on an eight-row synthetic Table 4

Seventeen to thirty-four tests, 68 to 96 percent. The build runs against a fake essentiality LMDB with `DATA_ROOT` under `tmp_path`: record order and keys, full `model_dump()` equality for a full and a ragged row (every header-to-key mapping, `glutamine+valine`, the linear-ratio `measurement_type`, reference 1.0 per present key, `n_replicates` 1, `temperature` None with its typed gap), the exact `dropped_records.json` (the first duplicate kept, the lowercase duplicate to the ledger, the rename, the normalizations, the pseudogene drop, the BY4742 exclusion, the essentiality flag), `data.csv`, `gene_set.json`, the reference index `[[0,2,3,4],[1]]`, download refusals (missing mirror with the recipe, wrong sha256 with both digests), `load_sgd_essential_genes` with interned `$ref` records, `read_table4` refusing a short row and a reordered header, `deposit_raw_mirror`.

Finding: the Table 4 sha256 pin is checked only inside `download()` (lines 1102-1107); a file placed directly in `raw/` is consumed unchecked.

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Finding retired: `test_a_raw_file_placed_in_raw_dir_is_never_sha256_checked` (#518) is now `test_a_raw_file_placed_in_raw_dir_is_refused_at_build_time`. Every sha256 refusal now asserts `RawSha256MismatchError` with its exact message (`sha256 mismatch for <file or URL>: expected <pin>, observed <digest>`) and the on-disk state after it: a download refusal leaves nothing in `raw/` (no `.partial`), a deposit refusal leaves no mirror directory, and a build-time refusal leaves `processed/` empty and the raw file as found. The build tests run under the `raw_pin_calls` recorder from [[tests.torchcell.conftest]]; the refusal tests restore the real check.
