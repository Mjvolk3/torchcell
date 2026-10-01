---
id: 9erbiezyjaw5zsbcghyya41
title: fix-cell-datamodule-splits
desc: ''
updated: 1790812815924
created: 1790812815924
---

## 2026.09.30

- [x] PR: `FIX(datamodules)` split integrity in [[torchcell.datamodules.cell]] (issue #517): remainder balancing targets with a named zero-target refusal, orphan and missing-`split_indices` refusals, custom `collate_fn` honored through a torch loader, empty details `str`, empty overlap splits; tests in [[tests.torchcell.datamodules.test_cell]]
