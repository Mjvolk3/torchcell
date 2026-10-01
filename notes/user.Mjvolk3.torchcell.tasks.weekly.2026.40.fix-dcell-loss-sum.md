---
id: 3quzd6nvhrksauzkkytgaz8
title: fix-dcell-loss-sum
desc: ''
updated: 1790818910948
created: 1790818910948
---

## 2026.09.30

- [x] PR #574: FIX(losses) issue #554, `DCellLoss` takes a required `aux_reduction` (`"sum"` is Ma et al. 2018), skips the root by its declared key, refuses broadcast shapes; `int_dcell` passes `[B]` shapes; historical runs in issue #578. Notes: [[torchcell.losses.dcell]], [[torchcell.models.dcell]], [[torchcell.models.dcell_opt]], [[torchcell.trainers.int_dcell]], [[torchcell.trainers.dcell_regression]], [[torchcell.trainers.dcell_regression_slim]]; tests in [[tests.torchcell.losses.test_losses_dcell]], [[tests.torchcell.models.test_dcell]], [[tests.torchcell.models.test_dcell_opt]], [[tests.torchcell.trainers.test_int_dcell]], [[tests.torchcell.trainers.test_dcell_regression]], [[tests.torchcell.trainers.test_dcell_regression_slim]].
