---
id: uml5dwaofdkavgqk52whsie
title: Dcell_opt_demo
desc: ''
updated: 1791330611317
created: 1791330611317
---

## 2026.10.06 - Origin

The hydra `main` of `torchcell/models/dcell_opt.py` trained and evaluated the optimized DCell model on a sample batch from `torchcell.scratch.load_batch_005`, plotting training progress. It moved verbatim, decorator included, with its `if __name__ == "__main__":` block (which sets the `spawn` start method), to `torchcell/scratch/dcell_opt_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/dcell_opt_demo.py` (the hydra `config_path` resolves against `os.getcwd()`, so it reads the same `experiments/006-kuzmin-tmi/conf` as before; it needs `DATA_ROOT` in `.env`, the sample batch, and a GPU where available). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/dcell_opt.py` L793 to 1148 (decorator at L793, `def` at L798) and the `if __name__ == "__main__":` block L1151 to 1155.
