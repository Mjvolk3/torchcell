---
id: h5vizfnf3mooky621t0pbv5
title: Stoichiometric_hypergraph_conv_demo
desc: ''
updated: 1791330481033
created: 1791330481033
---

## 2026.10.06 - Origin

The `main` ran a small `StoichHypergraphConv` example on random toy tensors, with and without stoichiometric gating, and printed the output shapes and features. It now lives in `torchcell/scratch/stoichiometric_hypergraph_conv_demo.py` ([[torchcell.scratch.stoichiometric_hypergraph_conv_demo]]), moved verbatim, and runs from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/stoichiometric_hypergraph_conv_demo.py`. It needs no `.env`, no data and no GPU. The older commented-out `main` sketch just above it stays in the module, since the move takes only live `main` functions. Executing `torchcell/nn/stoichiometric_hypergraph_conv.py` directly now exits with status 1 and a pointer to the demo. Reason: demo code is not library code, so it no longer counts in the coverage denominator (test campaign Phase 23).

Original location at main 3e88685ff: `torchcell/nn/stoichiometric_hypergraph_conv.py`, L231 to 337 (`main`) and the `if __name__ == "__main__":` block L340 to 341.
