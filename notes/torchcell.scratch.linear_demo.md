---
id: 3n0c199e64wa8zoburhjekq
title: Linear_demo
desc: ''
updated: 1791330495559
created: 1791330495559
---

## 2026.10.06 - Origin

The `main` of `torchcell/models/linear.py` ran a small forward and backward pass of `SimpleLinearModel` on random dummy data on the CPU and printed the output shape, the loss and a success line. It moved verbatim, with its `if __name__ == "__main__":` block, to `torchcell/scratch/linear_demo.py`; run it from the repo root with `PYTHONPATH=$PWD python torchcell/scratch/linear_demo.py` (it needs no environment variables or data). Executing the module directly now exits with a pointer to the demo. Reason: demo code is not library code (test campaign Phase 23).

Original range at main 3e88685ff: `torchcell/models/linear.py` L45 to 75 and the `if __name__ == "__main__":` block L78 to 79.
