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
- [x] Random-graph seeds submitted on Delta behind the sweep: 22056267 to 22056269 (`delta_submit_sweep.sh random`, after 22055169). Everything the figure trains on is now queued, 32 jobs ([[experiments.025-solid-growth.scripts.delta_cgt]])
- [ ] Panel readouts from checkpoints: divergence at lambda 0 and at the mask (a), off-graph edge recovery (d), attention on Costanzo digenic edges (e)
- [x] Both chain heads (22034665 ctrl s1, 22055147 lambda 0 s1) OOM'd at their first batch: the KL arms ran at 95 to 96 percent of the A40 (W&B system metrics of b3n4ax4a) because every layer materialized the full attention matrix while only layer 1 is regularized. Model fixed to take the manual path only in regularized layers (13.5 GiB peak at 64 records per GPU, measured on GilaHyper, against 41 GiB before); peak memory now logged per epoch. Delta worktree at 14c02243; resubmitted 22080055 (ctrl s1), 22080056 (fit_014 s1 rerun, one code path for the fitness experiment), 22080057 (lambda 0 s1). 33 jobs queued ([[experiments.025-solid-growth.scripts.delta_cgt]])

## 2026.09.15

- [x] cabbi: `fit_014` seed 2 (2400109) started, then cancelled in favor of the closure composite: 2400200 = `cgt_s3_r_kl_fit_030` seed 1, joint fitness + interaction on S3 (1,121,645 records: all singles, the 739,222 doubles inside some triple, the triples; 694 synthetic-lethal doubles; no essentiality in the build), evaluated on the pinned trigenic splits. About 36 h expected ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]])
- [ ] Sync the flanks run (2395008, 100 epochs) and rerun the disjoint readout
- [x] Per-order metrics (`per_order_metrics`) landed and the closure arm restarted to carry them: 2401111, pending on cabbi behind a 019 wave-5 array that took the freed GPUs; 2400200 had reached batch 14 at 25 s/batch (cold reads) ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]])
- [x] Delta: second OOM at the epoch-10 diagnostic validation (22034666, 22055149, both after ten clean epochs at 20 to 25 min); diagnostics now return attention for one validation batch and validation runs at 32 per rank (measured peak 16.4 GiB, was 42); worktree at 14f92459; re-queued as 22107790 (fit_015 s1) and 22107791 (mask s1). Chain heads have waited 1 to 2.4 days; Slurm estimates starts tonight ([[experiments.025-solid-growth.scripts.delta_cgt]])
- [x] IGB disjoint runs synced (20 offline runs) and read at two to three seeds: control window means 0.140 / 0.144 / 0.125, composite 0.215 / 0.235, random-vector matched control 0.151, flanks 100 ep 0.157; report republished ([[experiments.025-solid-growth.scripts.equivariant_cell_graph_transformer]])
- [x] 2026.09.16 synced mmli 2397848 / 2397876; composite Q at three seeds (window mean 0.218 vs control 0.136); composite+fitness 0.209 at one seed [[experiments.025-solid-growth.scripts.disjoint_embedding_readout]]
- [x] 2026.09.16 Delta: fit_014 seeds 2 and 3 and lambda-0 seed 2 completed; joint arm 0.4523 / 0.4476 / 0.4522 max; 30 jobs still pending, heads estimated Wed 09-17

## 2026.09.18

- [x] S3 closure seed 1 (mmli 2408888, `cgt_s3_r_kl_fit_031`) synced at epoch 4 as W&B group `s3_fit_031` (rank 0 `kj03xx8y`): cold epoch 0 4 h 21 min, epoch 1 on 36 min 24 s, projected end Sunday morning inside the 4-day clock. Partial, no result yet ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]])
- [x] Order-1 fitness was never logged: the per-order counter only advanced for the interaction collection and gated the fitness collection on it, so the 5,694 singles' fitness metrics were dropped each epoch. Fixed in 2e96f24f (own count per collection, key `n_records/<label>/order<k>`, test added); replicates resubmitted from IGB worktree `-h` as 2409031 (seed 2) and 2409032 (seed 3)
- [ ] Read seed 1's order-1 fitness from its checkpoints once it finishes, since the running job carries the old logging
- [x] User's call: S3 seed 1 (2408888) given up at epoch 4 for single-gene logging; restarted as 2409033 from IGB worktree `-h`, started at once (mmli idle). Triplicate 2409033 / 2409031 / 2409032 all on the fixed code ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]])
- [x] Disjoint single-region arms queued to three seeds behind the S3 chain: CaLM, ProtT5, flanks at seeds 1 and 2 (2409034 to 2409039); flanks replicates on the new 30-epoch `cgt_s0_q_kl_fudt_032` ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]])
- [x] S3 closure document: `notes-tex/025-s3-closure/` (9 pages, gate clean) with the pipeline diagram as figure a and three multipanel figures; recompute of every interaction and p-value from the pool against the raw sources. Identity exact within a screen (r 0.999 / 1.000 / 0.996), r 0.445 digenic and 0.230 trigenic in the build; essentiality 0 in the mean costs half the digenic error; merged records' p is a t-test that discards the source p (rho 0.05); Kuzmin 2020 scored 99.6 percent of digenic rows with query fitness 1.0 ([[experiments.025-solid-growth.s3-closure]])
- [x] Build stages optimized and tested (commit 2824b6335): conversion copies unconvertible records byte for byte, raw commits every 4,096 records and accumulates the gene set in-stream, aggregation keys off bytes with a parse guard; 12 tests
- [x] 029 deletion-only build prepared: query validated on the served graph (deletion predicate, SynthLethDB self-pairs dropped by node count), no dedup stage, root on /db; launched under slurm ([[experiments.029-solid-growth-ko]])
- [ ] Label policy at read time (precedence Kuzmin > Costanzo 30 C > 26 C, converted 0 only without a measurement, Stouffer over source p) before any 029 training
- [ ] 025 build intermediates archived to /bulk (tar.zst, in progress); delete from /db once verified
