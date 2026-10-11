---
id: 91rxc45ismg7jj9lvrgxynv
title: Gene_essentiality
desc: ''
updated: 1791686815992
created: 1791686815992
---

## 2026.10.10 - Gene-essentiality verifier (#889)

Single-pass battery for `gene_essentiality_sgd`: `structural`, `reference_structural`, `count`, `single_gene`, `one_record_per_gene_and_publication` (a record identical down to its publication is a duplicate; two papers for one gene are two records), `essential_label`, `released_annotations` (stored (gene, PubMed) multiset against the SGD per-gene JSON release, pinned by `released.sgd_json_digest`), `reference_viable`, caller rows, shared rules. `essential_without_viable_deletion_result` is the cross-dataset L4 row the runner adds against `smf_costanzo2016`.

## 2026.10.11 - Module removed: SGD essentiality is a provenance record

`Environment.media` stays required (owner, [[torchcell.datamodels.schema]]) and no SGD annotation states a medium the schema can carry, so `GeneEssentialitySgdDataset` left the served graph and its source is a provenance record ([[torchcell.datasets.scerevisiae.sgd]]). There is no gene-essentiality store left to verify: `torchcell/verification/gene_essentiality.py` and `runners.run_gene_essentiality` are removed, with `GENE_ESSENTIALITY_DATASETS`, `viable_deletion_genes` and their tests. `run_all` no longer calls the family, and the registry-root pin in `test_runners.py` reads 56 rather than 59 (this family's one root and the two SynLethDB roots left with it).

The checks the verifier made are not lost, they moved to where the evidence now lives: the allele rule, the condition inventory and the SGD-versus-Costanzo disagreement split are measured functions and constants of the provenance record module, pinned by its own tests against the pinned dump. The 7 unconditional disagreements stay unexplained and are recorded as `UNCONDITIONAL_DISAGREEMENTS`.
