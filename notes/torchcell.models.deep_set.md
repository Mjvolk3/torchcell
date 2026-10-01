---
id: xdjcskpe3hgtat18hb7xfeg
title: Deep_set
desc: ''
updated: 1695866541137
created: 1695866035821
---
## Dropout on Last Layer Only

When I say last layer and mean the last layer for the global embedding vector @liUnderstandingDisharmonyDropout2018

## 2026.09.30 - Aggregation check and scoped anomaly detection

Issue #525. Previously the `sum`/`mean` check was an `assert` in `__init__` only, so an `aggregation` reassigned later reached the set layers with `x_aggregated` unbound (`UnboundLocalError`); `main()` called `torch.autograd.set_detect_anomaly(True)` and never turned it off. Now `_check_aggregation` raises `ValueError("Unknown aggregation 'max'; expected one of ('sum', 'mean')")` at construction and again at the start of `forward`; `main()` runs its forward/backward inside `torch.autograd.detect_anomaly()`, so detection is off after it returns. Evidence: [[tests.torchcell.models.test_deep_set]].
