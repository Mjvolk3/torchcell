---
id: znurbved334nd6nr3skxt0o
title: '38'
desc: ''
updated: 1789365911262
created: 1789365911262
---

## 2026.09.14

- [x] Delta seed-1 fitness arm 22030924 (`fit_014`) COMPLETED at 00:08 after 16 h 01 m for 30 epochs on the staged NVMe path (W&B `b3n4ax4a`, best validation Pearson 0.4523 at epoch 18, an upward-biased max); its checkpoints are on `/scratch`. Chain continues with 22034665 (ctrl s1, Priority) then 22034666 to 22034673 ([[experiments.025-solid-growth.scripts.delta_cgt]])
- [x] Graph-regularization sweep rebased onto the 025 build: every arm a one-key override of `ctrl_013`, hard mask as `cgt_s0_r_mask_028`, `delta_submit_sweep.sh` rewritten as a 21-job staggered chain on 24 h clocks; submitted 01:50 as 22055147 to 22055169 (lambda 0, mask, 1e-2, 1e-1, 1e-4, 1e-5, 1 per seed; 1e-3 = the three ctrl_013 seeds in the fitness chain). Still open for the figure: random-graph arm, gradient probe, checkpoint readouts ([[experiments.025-solid-growth.scripts.delta_cgt]])
- [x] Whole-build arm for cabbi answered: the requested set (triples + all doubles + singles + synthetic lethality) is the full 025 build, 13,525,071 records; SGD essentiality contributed no record (not in the served graph). `CellDataModule(unpinned_to_train=True)` plus `subset.indices: null` make it runnable as `cgt_s5_r_kl_fit_029`, evaluated on the pinned trigenic splits; `cgt_s3_r_kl_fit_030` is the 1.12M-record closure version. Cost by scaling, unmeasured: 10 to 14 h/epoch for S5, about 72 min/epoch for S3 on cabbi
- [ ] Decide the cabbi arm: S3 closure (about 36 h) now, or S5 with a per-epoch cap on the doubles (sampler not written)
- [x] Random-graph arm built: `torchcell.graph.rewire` (seeded double-edge swaps, in/out degree kept; on the nine real graphs 13 to 46 percent of edges survive, under a minute), `model.random_graph` in the 025 trainer, config `cgt_s0_r_kl_rand_031` at lambda 1e-3, four tests; CPU smoke through two epochs ([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]])
- [x] Gradient probe for panel c: `RegressionTask(gradient_probe_epochs)` logs `probe/grad_norm/{point,dist,graph_reg,fitness,total}` on the first batch of epochs 0, 1, 2, 5, 10, 20 (set in `cgt_s0_r_kl_000`, so every pending Delta job logs it at start). Smoke, epoch 0, 64 records: point 12.25, dist 0.66, graph penalty 1.03, total 12.66
- [ ] Submit the three random-graph seeds on Delta behind the sweep (`delta_submit_sweep.sh random`, AFTER=22055169)
- [ ] Panel readouts from checkpoints: divergence at lambda 0 and at the mask (a), off-graph edge recovery (d), attention on Costanzo digenic edges (e)
