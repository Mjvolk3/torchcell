---
id: 5tyc681jaqeq6lmbtyq8uu2
title: Test_int_dcell
desc: ''
updated: 1790820782092
created: 1790820782092
---

## 2026.09.30 - Paired shapes into DCellLoss (issues #554, #578)

New. `RegressionTask._shared_step` with a weight-free stand-in (root [0, -1, 1], GO:1 [0, 1, 2], GO:2 [0, 2, 1], y = [1, 0, 0.5]) returns 1.7 under `"sum"` and 1.225 under `"mean"`, reachable only with paired `[B]` shapes; the old broadcast path gave 1.075 on this fixture. The returned predictions and targets stay `[B, 1]` for the metrics. Issues #554, #578.
