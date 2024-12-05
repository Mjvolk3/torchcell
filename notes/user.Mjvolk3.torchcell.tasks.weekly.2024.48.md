---
id: y593q189gi18kzc4ppdhtqd
title: '48'
desc: ''
updated: 1733260269028
created: 1732548647061
---
## 2024.11.25

- [x] Added weighted phenotype loss according to subset. Added to config.
- [x] Added `ce` loss_type to config.
- [x] Use inverse transform to convert back
- [x] #ramble Losses on binary hetero don't seem to be progressively dropping with epoch. Also the val metrics are constant which seems very strange. Performance on fitness is near perfect but not so for gene interactions. We could weight by total number of interactions on perturbation.

## 2024.11.26

- [x] Parameterize head prediction.

