---
id: z7ilz4svvtdie333nw9w3a3
title: 030-per-entry-dataset-token
desc: ''
updated: 1790417343189
created: 1790417343189
---

## 2026.09.26

- [x] Per-entry training path with the source-dataset token at the readout, essentiality holdout loader, committed normalization constants [[experiments.030-solid-growth-multi.scripts.equivariant_cell_graph_transformer]]
- [x] 030 split, holdout and eval artifacts; smoke launcher; IGB sync and mmli launcher (PR #444)
- [x] Smoke on GilaHyper: PASS on job 2867 after the twin-per-source redesign (2861 inverse-compose bug, 2863 and 2866 single-token design) [[experiments.030-solid-growth-multi.scripts.smoke_report_030]]
- [x] `gh_sync_igb_030.slurm` (job 2862) mirrors the build and the split cache to IGB
- [x] Detached IGB worktree at 5d3371e2; seed 0 of `cgt_030_s3_r_tok_embfit_001` queued on mmli as job 2413344 (behind 2409708 and the chained 025 jobs)
- [x] 2026.09.28: 2413344 CANCELLED before it started. The holdout arm removed 698 singles from the pool and added a second validation loader, so it was not the 025 S3 training set; the first 030 read must be apples to apples with `cgt_s3_r_kl_fit_031`. `subset.exclude` is now optional (commit a47b5afd3); new arms `cgt_030_s3_r_tok_fit_002` (table, whole S3 pool, 60 epochs, constants `normalization_stats_030_s3_r.json` over 1,046,316 training records) and `cgt_030_s3_r_tok_embfit_003` (same, composite). Split cache warmed for seeds 0-2 under the new pool tag (`utt_sub1121662-9fb708df`) and rsynced to IGB; IGB worktree advanced to a47b5afd. Queued: **2413832** (`030-r1w1-tab-s0`, fit_002 seed 0) then **2413833** (`030-r1w1-emb-s0`, embfit_003 seed 0, afterany). 025 seed 3 (2409710) HELD so 030 follows the running seed 2 (2409709); `scontrol release 2409710` to put it back.
- [ ] Seeds 1 and 2 after epoch-1 wall time and MaxRSS of 2413832; `wandb sync` the offline run from the login node
- [ ] Convergence: 025 composite seed 1 (anx8rrdn) over epochs 30-49 mean 0.4784, slope +0.010 per 10 epochs (still rising), while the 025 table seeds 2 and 3 (8fuwsk48, srl1xzdf) sit at 0.4907 / 0.4927 with slope 0.000 / +0.002 (flat). The 60-epoch budget of fit_002 adds a 50-59 tail to read against that
