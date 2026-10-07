---
id: 65ktq3hlkg4n93hp2w2yvql
title: Dango_demo
desc: ''
updated: 1791330603657
created: 1791330603657
---

## 2026.10.06 - Origin

The hydra `main` of `torchcell/models/dango.py` tested the DANGO model by overfitting on a single sample batch from `torchcell.scratch.load_batch_005`, plotting losses and final results. It moved verbatim, decorator included, with its `if __name__ == "__main__":` block, to `torchcell/scratch/dango_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/dango_demo.py` (the hydra `config_path` resolves against `os.getcwd()`, so it reads the same `experiments/005-kuzmin2018-tmi/conf` as before; it needs `DATA_ROOT` in `.env`, the sample batch, and a GPU where available). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/dango.py` L607 to 1032 (decorator at L607, `def` at L612) and the `if __name__ == "__main__":` block L1035 to 1036.
