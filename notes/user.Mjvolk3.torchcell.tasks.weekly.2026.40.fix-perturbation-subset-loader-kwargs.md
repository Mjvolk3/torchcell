---
id: nliz542vucatscrdd8j9a0x
title: fix-perturbation-subset-loader-kwargs
desc: ''
updated: 1790872260969
created: 1790872260969
---

## 2026.10.01

- [x] PR: FIX(datamodules) issue #580, one shared loader-kwargs helper for [[torchcell.datamodules.cell]] and [[torchcell.datamodules.perturbation_subset]], dense collate refused, dense zero-worker loader constructible; `s288c_gb.py` retired, see [[test-campaign.2026.09.25]]; tests in [[tests.torchcell.datamodules.test_cell]] and [[tests.torchcell.datamodules.test_perturbation_subset]]
