---
id: 6ze0wna8t8r7b8tj29b3lp2
title: Test_ohya2005_adapter
desc: ''
updated: 1790780611899
created: 1790780611899
---

## 2026.09.30 - Phase 18: the exact node and edge lists

New file, seven tests, 27 to 100 percent. The real `ScmdOhya2005Dataset` built in `tmp_path` from two-row synthetic SCMD tables with a genome stub, run through the real adapter and its own conf: ids as the sha256 of the written-out JSON; all 22 nodes by value in table order (the CalMorph base and CV fields as literal JSON strings, the `phenotype_<id>` versus `calmorph phenotype` preferred ids, the per-record duplicate environment, media, temperature and publication nodes) and the 15 enabled node methods; all 22 edges and the 13 enabled edge methods; an interned store (the 808-byte reference becomes a `$ref` pointer, the 315-byte environment and 181-byte publication stay inline under 512) emitting the same graph; chunk and batch sizes refused or logged exactly; a missing conf refused before wandb; `main` with 7 CPUs faked splitting ceil(1.4) = 2 io and 5 process workers.

Finding: `ohya2005.py` `process` writes records inline and never calls `_intern_record` (lines 214-221), while thirteen other loaders intern, so an Ohya store has no `interned` env.

## 2026.10.01 - Findings retired (issues #537, #546)

Retired the #546 finding: the built Ohya store now carries the reference as a `$ref` pointer (environment and publication inline), and the adapter's node and edge lists are unchanged.
