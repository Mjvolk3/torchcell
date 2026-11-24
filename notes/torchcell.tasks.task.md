---
id: hpxizxknzhaex8k7wmu8ahp
title: Task
desc: ''
updated: 1795561765215
created: 1795561765215
---

## 2026.11.24 - Task object and the test lock

`BenchmarkTask` holds samples, genes, labels, the excluded records and the genome. `Split.test` raises `TestSplitLocked`; `Split.unlock_test(reason)` returns the test indices and emits `TestSplitWarning`. `regression_metrics` gives Pearson, Spearman, R2, MSE and MAE per target; a constant prediction has an undefined correlation (nan).
