---
id: s8zyxc6dd8zcgng1ap95agn
title: Int_hetero_cell
desc: ''
updated: 1765217693705
created: 1765217693705
---

## 2026.10.01 - Batch size reads num_graphs (issue #567)

`RegressionTask._get_batch_size` and `DiffusionRegressionTask._get_batch_size` sized a perturbation batch (no `gene.x`) as `max(perturbation_indices_batch) + 1`, which drops a trailing genotype with no perturbed gene. Both now return `int(batch.num_graphs)` in that branch; the `x` rows / perturbed-gene count / value count / 1 ladder is unchanged. With the lazy collate now writing `num_graphs` (issue #572), every 006 lazy batch carries it. Tests: [[tests.torchcell.trainers.test_int_hetero_cell]].
