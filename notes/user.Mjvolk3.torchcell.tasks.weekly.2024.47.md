---
id: p1k8gaf6bgm9i3zklcmoyij
title: '47'
desc: ''
updated: 1731870494634
created: 1731867464345
---

## 2024.11.17

- #ramble Would be difficult to implement transforms at the level of the lightning module since they are by default applied at the level of the dataset. If user wants to have the binning reflected on different dataset sizes they would have to run multiple queries. Maybe this isn't ideal, but changing it would require rethinking the perturbation subsetting. It would have to be done at the level of the dataset, meaning we would generate a new dataset from the subsetting instead of just indexing into the larger dataset. This would be the proper way of getting transforms to work since this is the default `pyg` behavior.
- [ ]

***

- [ ] Spearman needs to be fixed unbounded. I saw values around 3.
