---
id: ijrrasuekgodms6nnmvsuyv
title: Hetero_cell_bipartite_dango_gi_lazy
desc: ''
updated: 1762192269746
created: 1762192269746
---

## 2026.10.01 - follow_batch goes to the LazyCollater (issue #572)

The script passed `follow_batch_list = ["x", "x_pert", "perturbation_indices"]` to the datamodules, which since PRs #549/#571 do not apply it when a custom `collate_fn` is given. It now passes the same list to `LazyCollater`, which (after the [[torchcell.datamodules.lazy_collate]] fix) builds `perturbation_indices_ptr` / `_batch` and `gene.ptr` as PyG's `Collater` does.

Which list PyG applied before PRs #549/#571, by commit date of the script and the slurm files (as committed; the working tree at run time is not recoverable from git): slurm 062-065 (4760f653c, 2025-10-29) and 069 (8b68e55d5, 2025-10-31) ran the script before `follow_batch_list` existed, so the datamodule default `["x", "x_pert"]` applied and the local predictor term was all zeros for multi-genotype batches; slurm 073, 076 and 078-084 (53c257c22, 2025-11-13) committed with this `["x", "x_pert", "perturbation_indices"]` list (083 names no script).
