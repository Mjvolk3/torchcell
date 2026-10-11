---
id: 91rxc45ismg7jj9lvrgxynv
title: Gene_essentiality
desc: ''
updated: 1791686815992
created: 1791686815992
---

## 2026.10.10 - Gene-essentiality verifier (#889)

Single-pass battery for `gene_essentiality_sgd`: `structural`, `reference_structural`, `count`, `single_gene`, `one_record_per_gene_and_publication` (a record identical down to its publication is a duplicate; two papers for one gene are two records), `essential_label`, `released_annotations` (stored (gene, PubMed) multiset against the SGD per-gene JSON release, pinned by `released.sgd_json_digest`), `reference_viable`, caller rows, shared rules. `essential_without_viable_deletion_result` is the cross-dataset L4 row the runner adds against `smf_costanzo2016`.

## 2026.10.11 - The drop ledger and SGD's conditions (#901)

The SGD loader now stores only annotations whose allele is the whole-gene deletion and ledgers the 42 others, and carries SGD's condition qualifiers ([[torchcell.datasets.scerevisiae.sgd]]). Three verifier changes follow, none loosening a check: `released_annotations_result` takes the loader's ledger (`ledgered`, the (gene, PubMed id) multiset of `preprocess/dropped_records.json`), fails when a ledgered annotation is not released, and compares the store with the release less the ledger; `states_no_condition` and `essential_gene_set(..., unconditional_only=True)` keep only records whose environment states no medium, temperature or edit, which the runner's L4 uses (an annotation inviable only under nitrogen starvation is not contradicted by a deletion grown on the SGA medium); `essential_without_viable_deletion_result(..., scope=...)` names the compared set in its message. The registry oracle is 1,287 (1,329 less 42). On the rebuilt store the one failing rule is L4: 7 of 1,119 unconditionally essential genes are viable deletions in Costanzo 2016.
