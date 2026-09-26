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
- [ ] Seeds 1 and 2 after epoch-1 wall time and MaxRSS of 2413344; `wandb sync` the offline run from the login node
