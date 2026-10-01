---
id: apb6u33q54kfuxktj0aavuu
title: fix-dcell-trainers-loss-order
desc: ''
updated: 1790812886119
created: 1790812886119
---

## 2026.09.30

- [x] PR: FIX(trainers) issue #516, DCell trainers call `DCellLoss(predictions, outputs, target)`, root-only metrics, `_root` validation names, target validation, no sanity leak, per-epoch checkpoint artifacts. [[torchcell.trainers.dcell_regression]], [[torchcell.trainers.dcell_regression_slim]], [[torchcell.trainers.simple_linear_regression]], [[tests.torchcell.trainers.test_dcell_regression]], [[tests.torchcell.trainers.test_dcell_regression_slim]], [[tests.torchcell.trainers.test_simple_linear_regression]]
