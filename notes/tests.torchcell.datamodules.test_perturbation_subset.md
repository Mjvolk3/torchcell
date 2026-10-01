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
