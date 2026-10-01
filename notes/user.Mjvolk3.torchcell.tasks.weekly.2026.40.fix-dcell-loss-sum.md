---
id: 3quzd6nvhrksauzkkytgaz8
title: fix-dcell-loss-sum
desc: ''
updated: 1790818910948
created: 1790818910948
---

## 2026.09.30

- [x] PR: FIX(losses) issue #554, `DCellLoss` sums the auxiliary subsystem losses as in Ma et al. 2018 (`aux_reduction="mean"` reproduces the 005/006 runs) and skips the root by key only, in [[torchcell.losses.dcell]], [[torchcell.models.dcell]], [[torchcell.models.dcell_opt]]; tests in [[tests.torchcell.losses.test_losses_dcell]], [[tests.torchcell.trainers.test_dcell_regression]], [[tests.torchcell.trainers.test_dcell_regression_slim]], [[tests.torchcell.models.test_dcell]].
