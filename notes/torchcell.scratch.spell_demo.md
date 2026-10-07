---
id: q7nierzpsei7f3i6xkw2oi5
title: Spell_demo
desc: ''
updated: 1791330626661
created: 1791330626661
---

## 2026.10.06 - Origin

The `main` of `torchcell/datasets/scerevisiae/spell.py` loaded SPELL expression data from all studies, exported and quality-checked the condition metadata, and plotted per-gene expression histograms. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/spell_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/spell_demo.py` (it needs the SPELL archive extracted under `DATA_ROOT/data/sgd/spell`, where `DATA_ROOT` and `ASSET_IMAGES_DIR` are the module constants, which the demo imports along with the four functions it calls). The module's `timestamp` import, used only by `main`, was dropped. Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/datasets/scerevisiae/spell.py` L1082 to 1169 and the `if __name__ == "__main__":` block L1172 to 1173.
