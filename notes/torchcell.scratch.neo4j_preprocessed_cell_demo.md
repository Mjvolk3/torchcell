---
id: bz1btlv12igvn4py8udpqst
title: Neo4j_preprocessed_cell_demo
desc: ''
updated: 1791330596009
created: 1791330596009
---

## 2026.10.06 - Origin

The `main_preprocess` of `torchcell/data/neo4j_preprocessed_cell.py` built the 006-kuzmin-tmi small-build `Neo4jCellDataset` and wrote its preprocessed copy as a `Neo4jPreprocessedCellDataset` under `001-small-build-preprocessed-lazy`. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/neo4j_preprocessed_cell_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/neo4j_preprocessed_cell_demo.py` (it needs `DATA_ROOT` and `EXPERIMENT_ROOT` in `.env` and the small-build dataset or a reachable Neo4j). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/data/neo4j_preprocessed_cell.py` L450 to 543 and the `if __name__ == "__main__":` block L546 to 547.
