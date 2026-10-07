---
id: 73tze947uaqbso4s9n9lz6d
title: Hetero_cell_bipartite_dango_diff_gi_demo
desc: ''
updated: 1791330479985
created: 1791330479985
---

## 2026.10.06 - Origin

The hydra `main` of `torchcell/models/hetero_cell_bipartite_dango_diff_gi.py` overfit `GeneInteractionDiff` on a single batch from the `experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_diff_gi.yaml` config: it built the 006-kuzmin-tmi `001-small-build` dataset under `DATA_ROOT`, trained with the diffusion loss (on the GPU when available) and wrote training, uncertainty and summary plots under `ASSET_IMAGES_DIR`. It moved verbatim, with the TODO comment above it and its `if __name__ == "__main__":` block (spawn start method included), to `torchcell/scratch/hetero_cell_bipartite_dango_diff_gi_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_bipartite_dango_diff_gi_demo.py` (hydra still reads `experiments/006-kuzmin-tmi/conf`, since `config_path` resolves against `os.getcwd()`; it needs `DATA_ROOT`, `EXPERIMENT_ROOT` and `ASSET_IMAGES_DIR` in `.env`). The TODO comment carried the hard-coded machine path allowlisted in `tests/torchcell/test_import_all.py`, so that allowlist entry was removed. Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/hetero_cell_bipartite_dango_diff_gi.py` L360 to 1376 (the TODO comment at L360, the `@hydra.main` decorator at L361, `def main` at L366) and the `if __name__ == "__main__":` block L1379 to 1383.
