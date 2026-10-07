---
id: 0rin3umtso95perrekkn2b9
title: Cell_graph_transformer_metabolism_demo
desc: ''
updated: 1791330488829
created: 1791330488829
---

## 2026.10.06 - Origin

The `main` ran one real batch of the 019-simb-multimodal `fig6_pigment_transfer` build through `CellGraphTransformerMetabolism` and printed the head-input variance table (`h_CLS`, the genome-wide gene pool, and the perturbed-gene pool) and the head output shapes. The worktree import bootstrap block near the top of the module, which only mattered for direct execution, moved with it to the top of the demo. It now lives in `torchcell/scratch/cell_graph_transformer_metabolism_demo.py` ([[torchcell.scratch.cell_graph_transformer_metabolism_demo]]), moved verbatim, and runs from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/cell_graph_transformer_metabolism_demo.py`. It needs `DATA_ROOT` and `EXPERIMENT_ROOT` in `.env` and the `fig6_pigment_transfer` dataset under `DATA_ROOT`; it runs on the CPU. Executing `torchcell/models/cell_graph_transformer_metabolism.py` directly now exits with status 1 and a pointer to the demo. Reason: demo code is not library code, so it no longer counts in the coverage denominator (test campaign Phase 23).

Original location at main 3e88685ff: `torchcell/models/cell_graph_transformer_metabolism.py`, L592 to 738 (`main`), the `if __name__ == "__main__":` block L741 to 742, and the worktree import bootstrap block L60 to 78.
