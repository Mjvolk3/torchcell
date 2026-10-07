---
id: 7hys4rozr24zpn3iymphbcy
title: Hetero_cell_bipartite_dango_demo
desc: ''
updated: 1791330473180
created: 1791330473180
---

## 2026.10.06 - Origin

The hydra `main` instantiated `HeteroCellBipartite` from `experiments/003-fit-int/conf/hetero_cell_bipartite_dango.yaml`, ran a demonstration training loop on the 003-fit-int sample batch (loaded through `torchcell.scratch.load_batch`), and wrote embedding and correlation plots to `ASSET_IMAGES_DIR`. It now lives in `torchcell/scratch/hetero_cell_bipartite_dango_demo.py` ([[torchcell.scratch.hetero_cell_bipartite_dango_demo]]), moved verbatim, and runs from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_bipartite_dango_demo.py`. It needs `DATA_ROOT` and `ASSET_IMAGES_DIR` in `.env`, the 003-fit-int 001-small-build dataset under `DATA_ROOT`, and a GPU when the config asks for one. Executing `torchcell/models/hetero_cell_bipartite_dango.py` directly now exits with status 1 and a pointer to the demo. Reason: demo code is not library code, so it no longer counts in the coverage denominator (test campaign Phase 23).

Original location at main 3e88685ff: `torchcell/models/hetero_cell_bipartite_dango.py`, L686 to 908 (the `@hydra.main` decorator through the end of `main`) and the `if __name__ == "__main__":` block L911 to 912.
