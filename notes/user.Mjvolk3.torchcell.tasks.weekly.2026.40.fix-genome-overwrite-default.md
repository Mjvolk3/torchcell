---
id: ti5smis5z6helfpugsp9mnv
title: fix-genome-overwrite-default
desc: ''
updated: 1790877176365
created: 1790877176365
---

## 2026.10.01

- [x] PR: `FIX(genome)` default `overwrite=False` in [[torchcell.sequence.genome.scerevisiae.s288c]], immutable shared `data.db` (atomic recorded builds, counts verified at open, one-time migration of untrusted files, drops on a private per-instance copy), pickling never rebuilds (issue #606); explicit `overwrite=True` removed from [[torchcell.datasets.esm2]], [[torchcell.datasets.protT5]], [[torchcell.graph.graph_analysis]], [[torchcell.scratch.load_batch_005]] and the 006 inference script; tests in [[tests.torchcell.sequence.genome.scerevisiae.test_s288c_synthetic]] and [[tests.torchcell.sequence.genome.scerevisiae.test_s288c]]
