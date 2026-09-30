---
id: odcftqk9fdc9w48h62mwl52
title: Test_sgd_gene_graph
desc: ''
updated: 1790756761366
created: 1790756761366
---

## 2026.09.30 - Phase 11: a three-gene graph by hand

Thirteen tests, 13.5 to 78 percent (the rest is `main()`, which needs the genome). The loader takes a networkx graph, so the fixture is three genes with the attributes the code consumes: exact raw and normalized feature rows including the lower-median fill (`median([100, 300]) = 100`), categorical indices `[0, 0, 1]`, pathway indices and slices, the files under `processed/`, reload from disk on a second construction, and the exact errors (invalid model name, `KeyError(None)`, a missing attribute, the empty `min()`).

Findings: a feature value of 0 is treated as missing and replaced by the median (line 107, pattern on 103-114); categorical indices are positions in a set that is still growing, so the same chromosome or pathway gets different indices and changes with `PYTHONHASHSEED` (lines 120-153); a constant feature normalizes to NaN with no error (lines 141-143); `data/embedding.py` lines 77-90 raise `AttributeError` (`dna_windows`) on a by-name lookup even when the gene is present.
