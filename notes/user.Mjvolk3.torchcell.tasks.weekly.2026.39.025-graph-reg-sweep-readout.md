---
id: z3zqfflk2imehrysynzh43m
title: 025-graph-reg-sweep-readout
desc: ''
updated: 1790475578843
created: 1790475578843
---

## 2026.09.26

- [x] Graph-regularization sweep read out of W&B at three seeds per arm: `graph_reg_sweep_readout.py` (runs, history, summary, LaTeX tables, `--regroup` of W&B groups by arm) and `graph_reg_sweep_plots.py` (ladder, curves, random-graph control) [[experiments.025-solid-growth.scripts.graph_reg_sweep_readout]] [[experiments.025-solid-growth.scripts.graph_reg_sweep_plots]]
- [x] notes-tex `025-graph-reg-sweep` (five sections, `make check` clean): mask = no penalty (0.421 vs 0.425 at epoch 29); KL rises monotonically to 0.447 at lambda 1 with no interior peak; best-epoch reading compresses all arms to 0.444 to 0.454; contraction saturates at lambda 1e-2; random-graph control at 1e-3 underpowered [[experiments.025-solid-growth.graph-reg-sweep]]
- [ ] Submit the random-graph arm at lambda 1 (three seeds) and lambda 10 and 100 on Delta; read the Delta log of job 22055161 (lambda 1e-5 seed 2, no W&B run); `make refresh` once random seed 3 (22056269) finishes
- [ ] Panels d and e of the mock-up from the Delta checkpoints (discovery, overruling)
