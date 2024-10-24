---
id: x7735rux2u7r93pv1nejg20
title: '42'
desc: ''
updated: 1729524644320
created: 1728854629343
---

## 2024.10.13

- [x] Add self loops
- [x] fix issue with visualization.. → added nan % for predictions. Don't really like this aesthetically but will help monitor any model instabilities.
- [x] Fix indexing [[torchcell.data.neo4j_cell]] → had to make sure keys align

## 2024.10.14

- [x] Keys are still off. → Recompute indices
- [x] Recompute small set of traditional machine learning datasets.
- [x] Check against traditional machine learning datasets.
- [x] [[2024.10.14 - Idea for Supporting Different Base Graphs|dendron://torchcell/torchcell.data.neo4j_cell#20241014---idea-for-supporting-different-base-graphs]]

## 2024.10.15

- [x] Make `get_init_graphs` just process a `gene_set` this would remove dependence on `raw_db`. We could also pass `raw_db` as arg for dependency in injection. It could optionally be `None` if the `db` has already been computed. → now passing db due to relationship between query.
- [x] Check the perturbation count for 1e04 because it seems off according to the traditional ml models. → fixed now when comparing to traditional ml models. The issue was `lmdb` indices not matching. The key used on lmdb is in bytes.

## 2024.10.17

- [x] `1e5` models on `GH` are failing with nans in loss. We need more regularization. We are not yet doing a phenotype difference. I am not sure how much sense it makes now considering that we have both `fitness` and `gene_interaction`. Fitness on `wt` is 1, but the `gene_interaction` is 0. → I have some suspicion that batch normalization doesn't work well in pooling because the features will start to behave very differently in pooling layers. This might account for some of the instability. → It is unclear what is causing the nans now. They occur after one step of back propagation on M1.

