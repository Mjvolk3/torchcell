---
id: 2avr83351zak0f7m0jbuy0w
title: Compute_isolate_embeddings
desc: ''
updated: 1790818737356
created: 1790818737356
---

## 2026.09.30 - Isolate stores keep a [1, D] row per gene

Previously `compute_isolate_esm2` and `compute_isolate_fudt` squeezed each embedding to `[D]`, so the collated store was FLAT `[n_genes * D]`, and they set no `dna_windows`. `BaseEmbeddingDataset.__getitem__` by gene id reads `_data.dna_windows` and slices `embeddings[index : index + 1]`, so a gene-id lookup on these stores raised `AttributeError: 'GlobalStorage' object has no attribute 'dna_windows'`, and on a flat tensor the slice would have returned one scalar rather than the gene's row. Since PR #552 `Esm2Dataset.process` stores `[1, hidden]` per gene, and `FungalUpDownTransformerDataset` always did, so the isolate stores no longer matched the datasets they claim to mirror.

Both functions now keep `embed(...).cpu().reshape(1, -1)` and set `dna_windows` (the protein for ESM2, the window sequence for FUDT), so the collate is `[n_genes, D]` and gene-id lookup returns the gene's row (issue #555).

Evidence: the ESM2 path run with a faked backbone (two genes, `D = 4`) collated to `(2, 4)`; lookup of `YAL001C` and `YAL002W` by id returned `[[3.0, 77.0, 1.0, 2.0]]` and `[[5.0, 65.0, 1.0, 2.0]]` with their proteins, and integer index 1 returned the second row. The same probe on the `origin/main` version raised the `dna_windows` AttributeError. The FUDT path (needs BLAST+ and the 1011 assemblies) and the real `--dry-run` (needs the Peter tarball under `DATA_ROOT`) were not run.

Isolate stores already on disk stay flat and still load by integer index; gene-id lookup works only on stores written after this change.
