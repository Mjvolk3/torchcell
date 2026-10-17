---
id: 6l61xx47yyo1btfa8t4oels
title: hpc-smoke-tests
desc: ''
updated: 1792225880002
created: 1792225880002
---

## 2026.10.17

- [x] `hpc-smoke-tests` worktree: torchcell installed on ALCF Polaris, ALCF Aurora and NERSC Perlmutter from GilaHyper (own Miniforge, python 3.13, torch 2.11.0 cu128 or xpu, torch_geometric 2.8.1), a `.env` and data root per machine, and a GPU smoke job on each that trains and logs to W&B online: Polaris 7757261, Perlmutter 59896274, Aurora 8938709 (one Max 1550 tile). Scripts, queue limits, the RAPIDS and CFS-lock findings and the W&B links in [[experiments.042-hpc-smoke-tests]]; the test itself in [[experiments.042-hpc-smoke-tests.scripts.smoke_train]].
- [ ] Aurora real training blocked on two items: nine files import `torch_scatter` directly (no XPU build) and Lightning 2.6 has no XPU accelerator.
- [ ] Carry a dataset store and run a real torchcell training job on Polaris `capacity` (first dispatch target).
