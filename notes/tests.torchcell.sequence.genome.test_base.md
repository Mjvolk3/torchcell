---
id: bau64raig145avshp80qp93
title: Test_base
desc: ''
updated: 1791372104943
created: 1791372104943
---

## 2026.10.07 - The base through a non-yeast subclass

Thirteen tests over `ToyGenome`, a host with one linear replicon (`contig1`, chromosome 1), its own locus types (`gene`, `pseudogene`), GO in a `go_terms` attribute, its own annotation labels, and no SGD convention (no roman numerals, no `chrmt`, no `go_root`, no download). `resolve` is stubbed at `torchcell.sequence.genome.base.resolve`. They pin the four tier resolutions and the recorded `data.db` source, the replicon key, the gene set, both strands, GO from the hook (an `Ontology_term` row is ignored), all seven resolver outcomes with the toy note text, the subclass refusal text, the generic pickle restore, `remove_deprecated_go_terms` rewriting only the hook attribute while the shared file stays byte-identical, `drop_empty_go`, `get_seq`, the abstract hook surface, and that `base` imports no organism subpackage. Source: [[torchcell.sequence.genome.base]].
