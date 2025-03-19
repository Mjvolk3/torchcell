---
id: 50c2p7axdt22bp6alnafuqi
title: '12'
desc: ''
updated: 1742406430847
created: 1742343814779
---
## 2025.03.16

- [ ] Prepare report. Topics: ISAB failure, long runs, Node-Self Attention Integration, Hetero Node Set Attention math, Hetero Node Set Attention math integration.
- @Andrew-Dudzik - "GNN's are state of the art in algorithmic alignment. LLM's are not." [The Problems in Mathematics of Deep Learning](https://www.youtube.com/watch?v=btF19HOrWC4)

## 2025.03.18

- [x] Load model from file config. → works on hetero_cell.

- [ ] Try to fix no overfitting `batch_size=2`. → overfit on `["S"]` → overfit on `["M"]` →

- [ ] Test `DDP` on IGB

- [ ] Multi-node `DDP` on Delta  ?


- [ ] Sweep `hetero_cell_nsa` DDP from checkpoint. 16 gpus.
- [ ] Batch size 4?, `DDP` 16 gpu, > `1e6` parameters. Before we do this need to check that batch size 8 fits or not.


- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]


- [ ] Is it possible to scale the database over `num_nodes` on slurm cluster. If we can pool memory across `cpu` nodes and use more `cpu` we will be able to not only build db faster but completer build on `Delta`. This allows for horizontal scaling of db builds.


## 2025.03.19

- [ ] implement bipartite fully `GatV2` model


***

- [ ] Should rename normalization to scaling
