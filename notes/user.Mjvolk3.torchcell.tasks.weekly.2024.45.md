---
id: gze5rnzin8yb1h9qlo36f7q
title: '45'
desc: ''
updated: 1731867413987
created: 1730794204188
---
## 2024.11.05

- [x] Summary of sagpool sweep runs don't shed much light. No prediction better than mean here. [wandb sweep](https://wandb.ai/zhao-group/torchcell_003-fit-int_cell_sagpool_5e4/sweeps/ow1gqtkc?nw=nwusermjvolk3)
- [x] Long sagpool `5e4` shows no sign of learning anything other than mean. [wandb run](https://wandb.ai/zhao-group/torchcell_003-fit-int_cell_sagpool_5e4/runs/noo5u8hz?nw=nwusermjvolk3)
- [x] `sagpool` run `5e5` still only predicts the mean. [wandb run](https://wandb.ai/zhao-group/torchcell_003-fit-int_cell_sagpool_5e5/runs/9ae9bdk6/overview) → This could be due to over regularization... It could be due to model. Really not sure.
- [x] Implement `sagpool_inception`
- [x] Run DDP 2gpu `sagpool_inception` `5e4` → I have alleviated some of the regularization and the `mlp` should provide stronger supervision on the intermediate clusters.
- [x] Implement `gin_diffpool`
- [x] Change `{hostname_slurm_job_id}_{group}` to just `{group}` since it already contains hostname.
- [x] Submit `gh` performance ticket
- [x] Run DDP `gin_diffpool
