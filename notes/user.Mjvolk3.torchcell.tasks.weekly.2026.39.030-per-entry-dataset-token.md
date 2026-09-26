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
- [ ] Smoke job 2861 on GilaHyper: read the PASS/FAIL report [[experiments.030-solid-growth-multi.scripts.smoke_report_030]]
- [ ] After PASS: `gh_sync_igb_030.slurm`, detached worktree on the IGB login node, seed 0 of `cgt_030_s3_r_tok_embfit_001` on mmli
