---
id: 2bdnrd72g6ir18w0szvfobl
title: Test_int_hetero_cell
desc: ''
updated: 1790872904260
created: 1790872904261
---

## 2026.10.01 - Batch sizing in the hetero trainers (issue #567)

New file. For both `RegressionTask` and `DiffusionRegressionTask`: a batch with vector [0, 0, 1] and `num_graphs = 3` is size 3; the dataloader-profiling step logs exactly `{"val/dataloader_profile_loss": (0.0, 3), "val/dataloader_profile_batch_size": (3.0, 3)}`; and the ladder below `num_graphs` gives [5, 3, 2, 1] for `x` rows, perturbed genes, values, empty.
