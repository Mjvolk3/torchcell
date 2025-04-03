---
id: 0d6g26fwevdbs52is7g5o3h
title: Plot_ppi_reg_rcm
desc: ''
updated: 1741658175511
created: 1741658172193
---
![](./assets/images/original_ppi_matrix.png)
![](./assets/images/reordered_ppi_matrix.png)

- PPI - note the differences when we add self loops. Many PPI have self loops.

[](./assets/images/original_reg_matrix.png)
![](./assets/images/reordered_reg_matrix.png)

- REG - note the differences when we add self loops and we make it undirected. Since `pyg` has `edge_index` object making undirected makes this vector bigger as we actually add edges both ways.
