---
id: knxlp1h37b58gc5olo1e89f
title: Fast_csv
desc: ''
updated: 1792072607496
created: 1792072607496
---

## 2026.10.15 - The sink refuses a node id that arrives under a second label

`FastCsvSink.seen_nodes` and the chunk renderer's per-chunk set were one set of ids across every label, matching BioCypher's global deduplicator and the single id space of `neo4j-admin import`. That is still the dedup rule, but it dropped the second row silently, which is how a whole-genotype `interned constant` (whose id was the bare hash of the same bytes the genotype node hashes) took the place of 218,258 `Genotype` and `PerturbedSegregantGenotype` nodes on the job 4901 partial and 14,573 on the served KG 3.0 store ([[torchcell.datamodels.interned_constant]]).

The table is now `id -> PascalCase label`. A repeat under the same label is the legitimate dedup of a shared constant and is counted in `node_dups` as before; an id under a different label raises `ValueError` naming both labels, from `RenderedChunk.from_rows` (before rendering, as before) and from `_add_nodes`. The namespaced interned id means this never fires on the adapters as written; it is the mechanical guard for the next derivation that hashes the same bytes under two labels. Cost: a dict entry instead of a set entry per node id, about 8 bytes more per id on top of the id string itself (about 1 GB on a 100 M-id full build); the label strings are shared objects. Test: `test_an_id_under_a_second_label_is_refused_not_dropped`.
