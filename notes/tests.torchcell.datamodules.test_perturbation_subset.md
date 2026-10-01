---
id: l6qmuu6ir670q6t3ipcbcvk
title: Test_perturbation_subset
desc: ''
updated: 1790549264373
created: 1790549264373
---

## 2026.09.27 - PerturbationSubsetDataModule on an in-memory dataset stub

Thirteen tests on a dataset object that exposes only what the datamodule reads (`perturbation_count_index`, the label and dataset-name indices, `len` and `__getitem__`): `_set_size`, `_create_subset` on every branch with the exact stratified indices under a fixed seed and the exact counts from the ratio arithmetic, the size clamp on a shortfall, `_create_index_details`, `_create_dataset_split`, the `_save_index`, `_cached_files_exist` and `_load_or_compute_index` round trip through `tmp_path` as exact JSON, `setup`, and the four loader factories (batch composition, `DensePaddingDataLoader` against the plain loader, `PrefetchLoader`). Coverage of the module from this file 99% (three partial branches). Findings: `dense=True` with `num_workers=0` is unconstructible because `prefetch_factor` is passed unguarded and torch raises `ValueError` (perturbation_subset.py line 411; the plain branch guards it); the `collate_fn` argument is stored and forwarded but PyG's `DataLoader.__init__` pops it and installs its own `Collater`, so a caller's collate is never called (lines 427 to 428; `CellDataModule._get_dataloader` has the same shape); on a shortfall `self.size` is rewritten and the cache files are named after the shrunken size inside a directory named after the requested size, so the next construction with the original size misses the cache and recomputes (lines 302 and 351). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Collate finding retired (issue #553)

Retired the finding `test_collate_fn_is_stored_but_discarded_by_the_pyg_loader`. Replaced by `test_custom_collate_fn_is_honored_through_a_torch_loader`, which asserts the loader is exactly `torch.utils.data.DataLoader`, its `collate_fn` is the caller's function, the batch size and worker options, and the exact batches `[("batch", [18.0]), ("batch", [19.0])]` on the parent's two-record test split at batch size 1 (and `[("batch", [18.0, 19.0])]` at batch size 2), and `test_without_a_collate_fn_the_pyg_loader_applies_follow_batch`, which asserts the no-collate path is exactly PyG's `DataLoader` with a plain `Collater` carrying `follow_batch == ["x"]` and `x_batch == [0, 1]`. Fourteen tests, up from thirteen. The dense `prefetch_factor` finding is unchanged.

Review follow-up: the collate test also builds (without iterating) a loader at `num_workers=1, pin_memory=True` and asserts `(num_workers, persistent_workers, pin_memory, start method) == (1, True, True, "spawn")`, so the worker options are pinned at non-default values. The dense finding's docstring now cites line 421 and states that the plain branch omits `prefetch_factor` rather than guarding it.

## 2026.10.01 - Dense finding retired, loader options asserted (issue #580)

Retired the finding `test_dense_selects_the_dense_padding_loader_but_needs_workers` (dense at `num_workers=0` raised on the unguarded `prefetch_factor`). Now asserted: `test_dense_builds_at_zero_workers_and_pads_the_parent_test_pair` (dense at zero workers has prefetch `None`, timeout 0, and pads the parent's test pair to `x == [[[18.0]], [[19.0]]]` with an all-True mask); `test_dense_with_workers_carries_the_shared_worker_options` (prefetch 3, timeout 10800, spawn, persistent, follow_batch); `test_dense_with_a_collate_fn_is_refused_by_name` (exact `ValueError` message, nothing written under the cache dir); `test_non_dense_loaders_pass_prefetch_factor_and_the_worker_timeout` (PyG and torch-collate loaders report `(1, 7, 10800)` with a worker and `(0, None, 0)` without).
