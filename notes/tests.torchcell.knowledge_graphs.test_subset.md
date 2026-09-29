---
id: lvj2iuc0f3vavaq4g4mri1e
title: Test_subset
desc: ''
updated: 1790649299037
created: 1790649299037
---

## 2026.09.28 - Record filters and seeded subsets over a tiny LMDB (Phase 9)

10 tests: an all-unset filter is rejected with its full message; `matches` applies every set criterion as an AND; the perturbation summary; gene-set files skip comments and blank lines and a relative path resolves against the torchcell package (the committed `sameith_double_pairs.txt`, 72 pairs, first `YBL054W YER088C`); inline sets are folded before the file; `select_indices` returns matching keys in numeric order on a twelve-record store, where LMDB's bytewise order (`0, 1, 10, 11, 2, ...`) would otherwise leak through; `subset_dataset` returns the dataset itself when nothing narrows, takes the whole pool when the size covers it, and samples with the seed and sorts (`random.Random(42).sample(range(10), 3)` is `[0, 1, 4]`). Coverage 0% to 100%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
