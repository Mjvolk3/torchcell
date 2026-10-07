---
id: 29fozxhl618i1xhktf2em4b
title: Test_k12
desc: ''
updated: 1791374733358
created: 1791374733358
---


## 2026.10.07 - K-12 genome tests

Covers [[torchcell.sequence.genome.ecoli.k12]]. Synthetic tests (everywhere): the real
strain classes read synthetic MG1655 and BW25113 sets through a stubbed `resolve` with
the network refusing; they pin the members read and their order, the `data.db` record,
the gene set, gene fields, resolution round trips (including the `Pro2`/`pro2` case rule),
the GO route summaries, the D9 default and derived view, the ECK crosswalk on a toy join,
`drop_empty_go`, `remove_deprecated_go_terms`, pickling, a GFF/GenBank disagreement and
`for_strain`. Tier tests (`@pytest.mark.data`, skipped without the sets): the counts and
the crosswalk in [[torchcell.sequence.genome.ecoli.k12]], built into pytest temporary
roots.

Run: `DATA_ROOT=/scratch/projects/torchcell-scratch pytest tests/torchcell/sequence/genome/ecoli --data`.
