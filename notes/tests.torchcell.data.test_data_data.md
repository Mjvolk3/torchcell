---
id: w15wpuz1ix3ar0hebwpgj9w
title: Test_data_data
desc: ''
updated: 1790777386101
created: 1790777386101
---

## 2026.09.30 - Phase 17: the small helpers of torchcell/data/data.py

New file, ten tests (the basename `test_data.py` is owned by `tests/torchcell/sequence/`, so the pairs table in `pyproject.toml` maps this file); 88 to 100 percent with the existing reference-index tests. `from_stored` keeps the concrete reference class in both formats and its two refusals are exact; `repr` truncates the member list after five indices; `mask(n)` raises `IndexError` for a member at or beyond `n`; `ReferenceIndex` indexing and iteration return the entries and the gap refusal is exact; the SHA-256 vectors hold over UTF-8 bytes.

Finding: `ReferenceIndex` accepts two entries with equal references because the partition check reads indices only (lines 114-119).

## 2026.10.01 - Equal references refused (issue #541)

The split-reference Finding is retired: `test_reference_index_refuses_one_reference_split_over_two_entries` asserts the exact pydantic message naming entries 0 and 2 and that the wrapped error is `DuplicateReferenceError`.
