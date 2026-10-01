---
id: 77v1j9ovclxhxdgyv9bi2r9
title: Perturbation_subset
desc: ''
updated: 1728315260499
created: 1728315260499
---

## 2026.09.30 - A custom collate_fn now collates the batches

Previously `_get_dataloader` put `collate_fn` into the kwargs of PyG's `DataLoader`, whose `__init__` pops it and installs its own `Collater`, so a caller's collate was never called. With `is_perturbation_subset=True` the 006 lazy scripts (`experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py` and `..._lazy_preprocessed.py`) lost their `LazyCollater`. Now, on the non-dense path, a given `collate_fn` builds the plain `torch.utils.data.DataLoader` with the same worker options and no `follow_batch` (that collate owns the batching); without one the PyG loader with `follow_batch` is unchanged. This is the contract `CellDataModule._get_dataloader` adopted in PR #549. The two datamodules share no base class, so the branch is repeated rather than factored. `collate_fn` is now typed `Callable[[list[Any]], Any] | None`. The dense branch still ignores `collate_fn` and still passes `prefetch_factor` unguarded; neither is part of issue #553. Evidence: issue #553, `test_custom_collate_fn_is_honored_through_a_torch_loader` and `test_without_a_collate_fn_the_pyg_loader_applies_follow_batch` in [[tests.torchcell.datamodules.test_perturbation_subset]].

## 2026.10.01 - Shared loader options and a refused dense collate

Previously the non-dense `_get_dataloader` never passed the constructor's `prefetch_factor` (a loader built with `num_workers=1, prefetch_factor=7` reported torch's default 2) and set no `timeout`, while `CellDataModule` passed both. The dense branch passed `prefetch_factor` unguarded, so `dense=True, num_workers=0` raised torch's "prefetch_factor option could only be specified in multiprocessing" `ValueError`, and a `collate_fn` given with `dense=True` was silently dropped, because `DensePaddingDataLoader` pops it and always batches with its own `DensePaddingCollater` (issue #580).

Now every branch takes its options from `torchcell.datamodules.cell.worker_dataloader_kwargs`, the helper `CellDataModule` uses: prefetch factor and a 10800 s worker timeout with workers, `None` and 0 at zero workers. `dense=True` with a `collate_fn` raises `ValueError(DENSE_COLLATE_REFUSAL)` at construction, before the cache directory is created. Refusal was chosen over honoring because the dense loader's padding collater is its whole purpose; no call site in torchcell/ or experiments/ passes both.

Call-site impact: scripts that pass `prefetch_factor` from config into this module (experiments 006 `cell_graph_transformer.py`, `equivariant_cell_graph_transformer.py`, `hetero_cell_bipartite_dango_*`, `hetero_cell_nsa_retry.py`; 009, 010 and 011 `equivariant_cell_graph_transformer*.py`; 025 `equivariant_cell_graph_transformer.py`) now get their configured factor (1 to 4) instead of 2 when `num_workers > 0`, and every call site with workers gets the 3 h worker timeout. Neither changes what a batch contains.

Evidence: `test_dense_builds_at_zero_workers_and_pads_the_parent_test_pair`, `test_dense_with_workers_carries_the_shared_worker_options`, `test_dense_with_a_collate_fn_is_refused_by_name`, `test_non_dense_loaders_pass_prefetch_factor_and_the_worker_timeout`.
