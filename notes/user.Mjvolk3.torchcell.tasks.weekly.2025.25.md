---
id: zfnut7p12bhr1rt6imnlkg2
title: '25'
desc: ''
updated: 1750222128146
created: 1750220864493
---

- [ ] Add Gene Expression datasets
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.
- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work
- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.
- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...

***

## 2025.06.16

[[2025.06.16 - With Use Transform No Other Change|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250616---with-use-transform-no-other-change]]

```bash
(torchcell) mjvolk3@biologin-2 ~/projects/torchcell $ squeue -p cabbi
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           1826088     cabbi     HCPD  mjvolk3  R 1-07:24:54      1 compute-3-3
           1826116     cabbi     HCPD  mjvolk3  R   15:44:35      1 compute-3-3
```

## 2025.06.17

- [x] [[2025.06.17 - I Cannot Find Pattern Of Jumping Correlation|dendron://torchcell/experiments.005-kuzmin2018-tmi.results#20250617---i-cannot-find-pattern-of-jumping-correlation]] → Our model looks state of the art now according to our own benchmarking. Still have disagreement with the original DCell paper.
- [x] Transfer data via globus

- [ ] Run model on `006` locally

- [ ] Run Dango on `006` query
