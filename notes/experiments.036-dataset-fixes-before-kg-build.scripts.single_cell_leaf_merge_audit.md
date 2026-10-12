---
id: cwog4ory426agyap2w7gw77
title: Single_cell_leaf_merge_audit
desc: ''
updated: 1791774301167
created: 1791774301167
---

## 2026.10.11 - The #1016 merge measurement

Decides and then checks the merge of `PseudobulkUmiCountPhenotype` (Jackson 2020) into `SingleCellExpressionPhenotype` (Boocock 2025). Three modes, each reading LMDB stores read-only and appending to `experiments/036-dataset-fixes-before-kg-build/results/single_cell_leaf_merge_audit.json`:

- `round-trip <processed>...`: run on the stores built before the merge. Every stored phenotype goes through `CandidateLeaf` (both leaves' fields, the source-specific ones nullable with a typed gap, `total_umis` bounded below by the stored counts) and every stored field must come back with an identical value and JSON type.
- `old-vs-new <old processed> <new root>`: after the rebuild, pairs the retired store with its successor by LMDB key and counts every old field that was dropped or changed (additions listed separately).
- `cells <root>`: per reference of a rebuilt store, cells versus distinct bare barcodes versus distinct (library, barcode) pairs, and the reference phenotype's JSON size.

Results and the decision are in [[torchcell.datamodels.eqtl-data-model]] (2026.10.11, #1016).
