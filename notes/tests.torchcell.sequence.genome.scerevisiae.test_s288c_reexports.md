---
id: khxk7wdb468okxxl5b2ya7c
title: Test_s288c_reexports
desc: ''
updated: 1791372112655
created: 1791372112655
---

## 2026.10.07 - The s288c compatibility contract after the base extraction

Five tests: every top-level name `base` defines (66, all but its logger) is reachable from `s288c` as the identical object; `monkeypatch.setattr` and `monkeypatch.delattr` of a shared name on `s288c` land in `base` and `undo` restores both; a name `s288c` owns (`download_url`, `SCerevisiaeGenome`) is not mirrored; and the pre-extraction unpickling target `_restore_genome(cls, genome_root, go_root, private_db_path)` hands the base restore the init fields with `overwrite=False`. Source: [[torchcell.sequence.genome.scerevisiae.s288c]], [[torchcell.sequence.genome.base]].
