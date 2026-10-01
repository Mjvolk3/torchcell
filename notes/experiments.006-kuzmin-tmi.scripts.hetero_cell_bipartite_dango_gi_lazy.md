---
id: ijrrasuekgodms6nnmvsuyv
title: Hetero_cell_bipartite_dango_gi_lazy
desc: ''
updated: 1762192269746
created: 1762192269746
---

## 2026.10.01 - follow_batch goes to the LazyCollater (issue #572)

The script passed `follow_batch_list = ["x", "x_pert", "perturbation_indices"]` to the datamodules, which since PRs #549/#571 do not apply it when a custom `collate_fn` is given. It now passes the same list to `LazyCollater`, which (after the [[torchcell.datamodules.lazy_collate]] fix) builds `perturbation_indices_ptr` / `_batch` and `gene.ptr` exactly as PyG's `Collater` did for the runs before those PRs.
