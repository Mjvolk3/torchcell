---
id: 1p821dht4nrq3pma4lgtsik
title: Sameith2015_demo
desc: ''
updated: 1791330618961
created: 1791330618961
---

## 2026.10.06 - Origin

The `main` of `torchcell/datasets/scerevisiae/sameith2015.py` built or loaded both Sameith2015 datasets (double mutants and the BY4742 single mutants) and printed a summary. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/sameith2015_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/sameith2015_demo.py` (it needs `DATA_ROOT` in `.env` with the SGD genome and GO data). The module's `from dotenv import load_dotenv`, used only by `main`, was dropped. Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/datasets/scerevisiae/sameith2015.py` L1983 to 2083 and the `if __name__ == "__main__":` block L2086 to 2087.
