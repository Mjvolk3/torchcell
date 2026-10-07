---
id: oytlov2nr751c7fs3y0e0w1
title: Test_kt2440
desc: ''
updated: 1791374741196
created: 1791374741196
---


## 2026.10.07 - KT2440 genome tests

Covers [[torchcell.sequence.genome.pputida.kt2440]]. Synthetic tests (everywhere): members
read, gene set with named RNA tags, a minus-strand gene without a symbol, GOA coverage,
and resolution round trips. Tier tests (`@pytest.mark.data`, skipped without the set):
5,786 / 5,729 / 57 loci, GO coverage 3,912 tags and 2,619 terms, round trips, and the CDS
translation check (only PP_0489 differs, a selenoprotein).

Run: `DATA_ROOT=/scratch/projects/torchcell-scratch pytest tests/torchcell/sequence/genome/pputida --data`.
