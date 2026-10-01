---
id: wuk19p1e91cv9k9gbzy2y3q
title: Test_s288c
desc: ''
updated: 1790813185745
created: 1790813185745
---

## 2026.09.30 - YAL037W 5' window (issue #543)

The data-gated YAL037W `window_five_prime(9, include_start_codon=False)` now expects `[74010, 74019)` = `AACACTGCT`, ending before the gene's first base (0-based 74019). The bases were read from `S288C_reference_sequence_R64-4-1_20230830.fsa` (chrI `[74010:74019]`); the old expectation `ACACTGCTA` ended with the `A` of the start codon.

## 2026.10.01 - genome fixture passes overwrite=False

The data-gated `genome` fixture omitted `overwrite`, so under `--data` it rebuilt the shared `$DATA_ROOT/data/sgd/genome/data.db` with the old `True` default. It now passes `overwrite=False` explicitly and first calls `require_trusted_genome_database`, so under `--data` a record-less or modified real database fails the module by name instead of being migrated by a test.
