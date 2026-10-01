---
id: 77v1j9ovclxhxdgyv9bi2r9
title: Perturbation_subset
desc: ''
updated: 1728315260499
created: 1728315260499
---

## 2026.09.30 - A custom collate_fn now collates the batches

Previously `_get_dataloader` put `collate_fn` into the kwargs of PyG's `DataLoader`, whose `__init__` pops it and installs its own `Collater`, so a caller's collate was never called. With `is_perturbation_subset=True` the 006 lazy scripts (`experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_lazy.py` and `..._lazy_preprocessed.py`) lost their `LazyCollater`. Now, on the non-dense path, a given `collate_fn` builds the plain `torch.utils.data.DataLoader` with the same worker options and no `follow_batch` (that collate owns the batching); without one the PyG loader with `follow_batch` is unchanged. This is the contract `CellDataModule._get_dataloader` adopted in PR #549. The two datamodules share no base class, so the branch is repeated rather than factored. `collate_fn` is now typed `Callable[[list[Any]], Any] | None`. The dense branch still ignores `collate_fn` and still passes `prefetch_factor` unguarded; neither is part of issue #553. Evidence: issue #553, `test_custom_collate_fn_is_honored_through_a_torch_loader` and `test_without_a_collate_fn_the_pyg_loader_applies_follow_batch` in [[tests.torchcell.datamodules.test_perturbation_subset]].
