---
id: lq5d2abhbn533m76nwtdiqe
title: Equivariant_cell_graph_transformer_demo
desc: ''
updated: 1791330449691
created: 1791330449691
---

## 2026.10.06 - Origin

The hydra `main` trained the Equivariant Cell Graph Transformer on the 006-kuzmin-tmi perturbation sample batch (loaded through `torchcell.scratch.load_batch_006_perturbation`, config `experiments/006-kuzmin-tmi/conf/equivariant_cell_graph_transformer.yaml`) and wrote training plots to `ASSET_IMAGES_DIR`. It now lives in `torchcell/scratch/equivariant_cell_graph_transformer_demo.py` ([[torchcell.scratch.equivariant_cell_graph_transformer_demo]]), moved verbatim, and runs from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/equivariant_cell_graph_transformer_demo.py`. It needs `DATA_ROOT`, `EXPERIMENT_ROOT` and `ASSET_IMAGES_DIR` in `.env`, the 006-kuzmin-tmi 001-small-build dataset under `DATA_ROOT`, and a GPU when the config asks for one. Executing `torchcell/models/equivariant_cell_graph_transformer.py` directly now exits with status 1 and a pointer to the demo. Reason: demo code is not library code, so it no longer counts in the coverage denominator (test campaign Phase 23).

Original location at main 3e88685ff: `torchcell/models/equivariant_cell_graph_transformer.py`, L3059 to 3719 (the `@hydra.main` decorator through the end of `main`) and the `if __name__ == "__main__":` block L3722 to 3723.
