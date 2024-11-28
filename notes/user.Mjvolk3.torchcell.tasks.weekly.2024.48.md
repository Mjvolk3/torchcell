---
id: y593q189gi18kzc4ppdhtqd
title: '48'
desc: ''
updated: 1732633257185
created: 1732548647061
---
## 2024.11.25

- [x] Added weighted phenotype loss according to subset. Added to config.
- [x] Added `ce` loss_type to config.
- [ ] Use inverse transform to convert back
- [ ] #ramble Losses on binary hetero don't seem to be progressively dropping with epoch. Also the val metrics are constant which seems very strange. Performance on fitness is near perfect but not so for gene interactions. We could weight by total number of interactions on perturbation.

## 2024.11.26

- [ ] Parameterize head prediction.

***

- [ ] Spearman needs to be fixed unbounded. I saw values around 3.
