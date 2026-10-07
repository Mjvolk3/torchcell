---
id: jfpqt4x2jizw2bhreg556bk
title: Hetero_cell_nsa_retry_demo
desc: ''
updated: 1791330487751
created: 1791330487751
---

## 2026.10.06 - Origin

The hydra `main` of `torchcell/models/hetero_cell_nsa_retry.py` ran a manual smoke test of `HeteroCellNSA` from the `experiments/006-kuzmin-tmi/conf/hetero_cell_nsa_retry.yaml` config: it loaded the 005 sample batch, trained with `ICLoss` (on the GPU when `cfg.trainer.accelerator` is gpu) and wrote a training loss plot under `ASSET_IMAGES_DIR`. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/hetero_cell_nsa_retry_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/hetero_cell_nsa_retry_demo.py` (hydra still reads `experiments/006-kuzmin-tmi/conf`, since `config_path` resolves against `os.getcwd()`; it needs `DATA_ROOT` and `ASSET_IMAGES_DIR` in `.env`). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/hetero_cell_nsa_retry.py` L423 to 651 (the `@hydra.main` decorator at L423, `def main` at L428) and the `if __name__ == "__main__":` block L654 to 655.
