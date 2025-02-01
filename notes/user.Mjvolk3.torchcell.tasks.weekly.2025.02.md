---
id: ngv3l4ot5sgfmxnk5mlvqid
title: '02'
desc: ''
updated: 1738205262285
created: 1736127191615
---

⏰ One thing: Metabolism Label Split Regression Run

## 2025.01.05

- [x] [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.stoichiometric_hypergraph_conv]]

## 2025.01.06

- [x] Launch `1e6` DDP on on multiple nodes entropy regularization → Some inter node communication issue.
- [x] Launch `1e6` DDP on on node entropy regularization → Fails due to some memory build up. ![](./assets/images/user.Mjvolk3.torchcell.tasks.weekly.2025.02.md.memory-build-up-during-train-ddp-1e6.png) [one of 4 ddp runs](https://wandb.ai/zhao-group/torchcell_003-fit-int_hetero_gnn_pool_1e6/runs/2ridstrq?nw=nwusermjvolk3)

Run crashes after 15 hours. From a quick look there aren't any obvious memory hogs 🐗 in the training loop.

- [x] Launch some DDP `5e5` on one node for entropy regularization

## 2025.01.07

- [x] Issue on multi-node DDP → [NCSA Jira Issue](https://jira.ncsa.illinois.edu/servicedesk/customer/user/requests?status=open)
- [x] Memory build up clarification with image and links previous day
- [x] Submitted for more `Delta` allocation. Running low.
- [x] Pass incidence metabolism hypergraph through dataset
- [x] Adjust `to_cell_data` to handle incidence graphs → have a prototype but needs fixing

## 2025.01.08

- [x] Prepare slides → ![ordinal classificaiton](/assets/drawio/ordinal_classification.drawio.png)

## 2025.01.10

- [x] Fix current failure in trying to overcome memory hog. → Running experiment on `Delta`
- [x] Adjust `to_cell_data` to handle incidence graphs
- [x] Write new phenotype processor to handle message passing on multigraph → `Identity()`
