---
id: 76a0ij07ewkxm8ae7a4jmgo
title: fix-dango-616
desc: ''
updated: 1790979131686
created: 1790979131686
---

## 2026.10.02

- [x] PR: FIX(dango) issue #616, one model forward per step (the second call existed only because `forward` dropped `reconstructions`), output sized by genotype with empty and duplicate genotypes refused, NaN targets masked before the loss, empty epochs refused instead of logged as NaN, one gene table for the oversmoothing log, unread accumulation schedule and scheduler type refused, lambda table refuses unknown networks. Notes: [[torchcell.models.dango]], [[torchcell.trainers.int_dango]]; tests in [[tests.torchcell.models.test_dango]], [[tests.torchcell.trainers.test_int_dango]].
