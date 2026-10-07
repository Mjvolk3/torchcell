---
id: nmthlk5mdo0tj4oiffogk3n
title: Hetero_cell_bipartite_dango_gi_lazy_demo
desc: ''
updated: 1791330472178
created: 1791330472178
---

## 2026.10.06 - Origin

The hydra `main` of `torchcell/models/hetero_cell_bipartite_dango_gi_lazy.py` ran a manual smoke test of the lazy `GeneInteractionDango` from the `experiments/006-kuzmin-tmi/conf/hetero_cell_bipartite_dango_gi.yaml` config: it loaded the lazy 006 sample batch, trained on it (on the GPU when `cfg.trainer.accelerator` is gpu) and wrote training and embedding plots under `ASSET_IMAGES_DIR`. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/hetero_cell_bipartite_dango_gi_lazy_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_bipartite_dango_gi_lazy_demo.py` (hydra still reads `experiments/006-kuzmin-tmi/conf`, since `config_path` resolves against `os.getcwd()`; it needs `DATA_ROOT` and `ASSET_IMAGES_DIR` in `.env`). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/hetero_cell_bipartite_dango_gi_lazy.py` L1508 to 3302 (the `@hydra.main` decorator at L1508, `def main` at L1513) and the `if __name__ == "__main__":` block L3305 to 3306.
