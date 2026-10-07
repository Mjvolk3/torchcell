---
id: svic3j8yzrpfeiagbzmfead
title: Neo4j_cell_demo
desc: ''
updated: 1791330588328
created: 1791330588328
---

## 2026.10.06 - Origin

The five demo functions of `torchcell/data/neo4j_cell.py` built a `Neo4jCellDataset` from the 003-fit-int small-build query and inspected it: `main` pulled one training batch through `CellDataModule`, `main_incidence` did the same with incidence (hypergraph) graphs, `main_transform_standardization` checked label standardization with the metabolic network and plotted before and after distributions, and `main_transform_categorical` and `main_transform_categorical_dense` checked the label binning transforms (the latter with dense conversion on a perturbation subset). They moved verbatim, in order, with the `if __name__ == "__main__":` block and its commented-out alternatives (it runs `main_transform_standardization`), to `torchcell/scratch/neo4j_cell_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/neo4j_cell_demo.py` (it needs `DATA_ROOT` in `.env` with the genome, GO, STRING, TFLink and embedding data, and the small-build dataset or a reachable Neo4j). The helpers `_label_values` and `_print_label_stats` stay in the module, since the tests import them; the demo imports `Neo4jCellDataset` and `_print_label_stats` from it. Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original ranges at main 3e88685ff: `main` L1326 to 1453 (the AST ends the function at L1421; L1423 to 1453 are its trailing indented commented-out code, which moved with it), `main_incidence` L1456 to 1587, `main_transform_standardization` L1610 to 1838, `main_transform_categorical` L1841 to 2047, `main_transform_categorical_dense` L2050 to 2279, and the `if __name__ == "__main__":` block L2282 to 2286.
