---
id: apndmuju1m93c4akuu79mbm
title: '44'
desc: ''
updated: 1730715111984
created: 1730424081057
---

## 2024.10.31

- [ ] Implement early diffpool join to try and get some speed up. I suspect this should at least `1.5x` our speeds
- [ ] In `diffpool` make sure model saving is happening according to the `data root`
- [ ] Check that the metrics are being recorded properly... we have some nans specifically in gene interactions

## 2024.11.03

- [x] Implement GIN cell diff pool
- [x] Diffpool early metrics → delayed
- [x] See how many isolated nodes we have in graphs. [[Cell_diffpool_dense|dendron://torchcell/experiments.003-fit-int.scripts.cell_diffpool_dense]] → I thought this might be causing the issues with little to no learning.
- [ ] Change `{hostname_slurm_job_id}_{group}` to just `{group}` since it already contains hostname.
- [ ]

***

- [ ] in `IndexSplit` changes `indices` to `index`...
