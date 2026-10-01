---
id: mahb5vbpvfzm3l43thtf2mt
title: Graph_analysis
desc: ''
updated: 1790877168690
created: 1790877168690
---

## 2026.10.01 - genome built with overwrite=False

The three `SCerevisiaeGenome` constructions here (`go_gaf_investigation`, `old_main`, `main`) passed `overwrite=True`, rebuilding the shared `data.db` under every other reader. They now pass `overwrite=False`, which opens the existing database after checking its source record (see [[torchcell.sequence.genome.scerevisiae.s288c]]). `go_gaf_investigation` and `main` still pass `data_root=`, which the class does not accept, so they raise `TypeError` before any build; that is unchanged.
