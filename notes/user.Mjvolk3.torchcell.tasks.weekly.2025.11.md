---
id: 25cmv32yhofqeg6pm6a9n9l
title: '11'
desc: ''
updated: 1741647005682
created: 1741634702492
---

## 2025.03.10

- [ ] image dump

![](./assets/images/stochastic_block_adjacency_matrix.png)
![](./assets/images/ppi_reg_adjacency_matrices_with_rcm.png)
![](./assets/images/hetero_cell_isab_gene_interaction_split_training_loss_2025-03-04-20-17-32.png)

- when you try to back prop on set operator over genes in wildtype graph that are removed from integrated graph.

![](./assets/images/hetero_cell_isab_training_loss_2025-03-05-06-01-46.png)
![](./assets/images/hetero_cell_pma_training_loss_2025-03-06-15-37-43.png)

- Not sure why the `PMA` seems to converge much slower. I believed these implementations to be equivalent. They differ in that I started directly using the `PMA` implemented in `pyg` instead of using an `ISAB` down to size 1 to get pooling.

![](./assets/images/standardization_with_metabolism_comparison.png)

- Standardizing labels since it is likely fitness was dominating.

![](./assets/images/original_ppi_matrix.png)
![](./assets/images/reordered_ppi_matrix.png)

- PPI - note the differences when we add self loops. Many PPI have self loops.

[](./assets/images/original_reg_matrix.png)
![](./assets/images/reordered_reg_matrix.png)

- REG - note the differences when we add self loops and we make it undirected. Since `pyg` has `edge_index` object making undirected makes this vector bigger as we actually add edges both ways.

- [ ] Fix reaction map with no rxn if $\emptyset$. Fix the compound map so that we have all reactions. They can be named according to `Yeast9` reaction scheme. We just need to have associated genes as properties. Direction of reaction can be captured with directionality of the graph.
- [ ] Make sure `SubgraphRepresentation` works

- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]

- [ ] See if we can improve the convergence on `pma`. → On training we have the issue that it likes to predict just a few values, in other words the distribution is spiky.

- [ ] Is it possible to scale the database over `num_nodes` on slurm cluster. If we can pool memory across `cpu` nodes and use more `cpu` we will be able to not only build db faster but completer build on `Delta`. This allows for horizontal scaling of db builds.

- [ ] Last experiments should be relatively simple.

***

- [ ] Should rename normalization to scaling
