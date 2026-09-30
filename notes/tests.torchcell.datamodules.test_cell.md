---
id: 55s1xg92u1qnmqtz1bf6ppd
title: Test_cell
desc: ''
updated: 1790756785218
created: 1790756785218
---

## 2026.09.30 - Phase 11: split arithmetic, reassignment, caches, loaders

Twenty-one to forty-four tests, 69 to 99 percent. The seed-42 split of ten records is the stdlib shuffle `[7,3,2,8,5,6,9,4,0,1]` sliced 8/1/1 (train `[2..9]`, val `[0]`, test `[1]`); the reassignment of records filed under two keys with ratio targets (17, 2, 1) gives val `[0, 6]`, test `[3]`, train 17; a key emptied by `index_subset` is skipped but appears in the details with count 0; the cache names (`_pin3-e0d22c21_sub3-c0be322c`) and contents, the corrupt-file message, `setup()` building once, the DataLoader options at 0 and 2 workers, the exact `df_summary` rows.

Findings: records under no split-index key land in no split (lines 501-528); `split_indices=None` makes all three splits empty (451-528); a key of 10 or fewer records holding a disputed record raises `ZeroDivisionError` because `int(10 * 0.09999999999999995) = 0` (line 523); an empty details object raises `KeyError('split')` in `__str__` so the `"(empty)"` branch never runs (114, 128-129); a custom `collate_fn` is silently discarded by PyG's `DataLoader` (line 679); `overlap_dataset_index_split` raises a validation error whenever one split has no overlap (210-212).
