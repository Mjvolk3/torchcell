---
id: znurbved334nd6nr3skxt0o
title: '38'
desc: ''
updated: 1789365911262
created: 1789365911262
---

## 2026.09.14

- [x] Delta seed-1 fitness arm 22030924 (`fit_014`) COMPLETED at 00:08 after 16 h 01 m for 30 epochs on the staged NVMe path (W&B `b3n4ax4a`, best validation Pearson 0.4523 at epoch 18, an upward-biased max); its checkpoints are on `/scratch`. Chain continues with 22034665 (ctrl s1, Priority) then 22034666 to 22034673 ([[experiments.025-solid-growth.scripts.delta_cgt]])
- [x] Graph-regularization sweep rebased onto the 025 build: every arm a one-key override of `ctrl_013`, hard mask as `cgt_s0_r_mask_028`, `delta_submit_sweep.sh` rewritten as a 21-job staggered chain on 24 h clocks; both configs composed and checked. Set to submit at 09:01 CDT after the scheduler maintenance ([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]])
- [x] Whole-build arm for cabbi answered: the requested set (triples + all doubles + singles + synthetic lethality) is the full 025 build, 13,525,071 records; SGD essentiality contributed no record (not in the served graph). `CellDataModule(unpinned_to_train=True)` plus `subset.indices: null` make it runnable as `cgt_s5_r_kl_fit_029`, evaluated on the pinned trigenic splits; `cgt_s3_r_kl_fit_030` is the 1.12M-record closure version. Cost by scaling, unmeasured: 10 to 14 h/epoch for S5, about 72 min/epoch for S3 on cabbi
- [ ] Decide the cabbi arm: S3 closure (about 36 h) now, or S5 with a per-epoch cap on the doubles (sampler not written)
- [ ] Random-graph arm: the degree-matched rewiring option, then three seeds on Delta
- [ ] Panel readouts from checkpoints: divergence at lambda 0 and at the mask (a), off-graph edge recovery (d), attention on Costanzo digenic edges (e)
