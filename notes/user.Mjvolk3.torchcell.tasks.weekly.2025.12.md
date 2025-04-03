---
id: 50c2p7axdt22bp6alnafuqi
title: '12'
desc: ''
updated: 1743634579068
created: 1742343814779
---

## 2025.03.18

- [x] Load model from file config. → works on hetero_cell.
- [x] Try to fix no overfitting `batch_size=2`. → overfit on `["S"]` → overfit on `["M"]` → Can overfit but we still just have issue related to density.
- [x] Test `DDP` on IGB
- 🔲 Sweep `hetero_cell_nsa` DDP from checkpoint. 16 gpus.
- [x] Batch size 4?, `DDP` 16 gpu, > `1e6` parameters. Before we do this need to check that batch size 8 fits or not.

- 🔲 Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [x] Is it possible to scale the database over `num_nodes` on slurm cluster. If we can pool memory across `cpu` nodes and use more `cpu` we will be able to not only build db faster but complete build on `Delta`. This allows for horizontal scaling of db builds. → 2025.04.02 shelving as we can build on `GH`

## 2025.03.19

- [x] implement bipartite fully `GatV2` model

## 2025.03.22

- [x] [[2025.03.22 - IGB Singularity Container|dendron://torchcell/containerize.igb#20250322---igb-singularity-container]]
- [x] [[2025.03.22 - Delta Apptainer Container|dendron://torchcell/containerize.delta#20250322---delta-apptainer-container]]
