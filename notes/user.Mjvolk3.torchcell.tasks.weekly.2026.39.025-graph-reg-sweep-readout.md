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

## 2026.09.27

- [x] Figure reworked on review: one 3 x 3 figure, one palette color per arm, readings by marker with paired-t stars per reading, no text under 5 pt (mathtext superscripts were 4.2 pt; ladder axis is now log10 lambda), draw.io mock-up as Figure 1 of the document, third reading at minimum validation loss (none separates there; lambda 0.1 closest at p 0.08)
- [x] Round 2 planned, not submitted: 16 arms, 46 runs, about 2,970 GPU-h; wireframe of the three figures; three small model changes (symmetric KL target, directed mask, two-hop mask) [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]
- [x] Approved ("do them all"): `khop_reach` and four model flags with 7 tests; composite configs 040 / 041; k-hop density table (undirected 2 hops of physical, coexpression, experimental cover 60 to 70 percent of pairs, so reach masks are directed); `delta_submit_round2.sh` generated from the plan (80 runs, 6 chains, round 1b first) [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]] [[experiments.025-solid-growth.scripts.graph_reg_khop_density]]
- [x] Delta: all 80 runs submitted 03:47 through the open tmux ssh pane (jobs 22461957 to 22462041, six lanes, round 1b heads 22461957 to 22461962) from detached worktree `025-graph-reg-round2` at ef4135f2; preflight hung, paths checked by hand [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]
- [ ] Real-size CPU smoke on GilaHyper (job 2894, pending on Priority): if it fails, `scancel` the `025-r2-*` and `025-r3-*` jobs before their lanes reach them
- [ ] After round 1b lands: `make refresh` in `notes-tex/025-graph-reg-sweep`; pull best checkpoints for the discovery and overruling panels
- [x] mmli 2409708 (025 S3 composite + fitness, seed 1) synced from the IGB login node: epoch 33, val GI Pearson 0.4815 max at epoch 28, partial
