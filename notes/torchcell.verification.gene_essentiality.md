---
id: 91rxc45ismg7jj9lvrgxynv
title: Gene_essentiality
desc: ''
updated: 1791686815992
created: 1791686815992
---

## 2026.10.10 - Gene-essentiality verifier (#889)

Single-pass battery for `gene_essentiality_sgd`: `structural`, `reference_structural`, `count`, `single_gene`, `one_record_per_gene_and_publication` (a record identical down to its publication is a duplicate; two papers for one gene are two records), `essential_label`, `released_annotations` (stored (gene, PubMed) multiset against the SGD per-gene JSON release, pinned by `released.sgd_json_digest`), `reference_viable`, caller rows, shared rules. `essential_without_viable_deletion_result` is the cross-dataset L4 row the runner adds against `smf_costanzo2016`.
