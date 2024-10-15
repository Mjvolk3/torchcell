---
id: x7735rux2u7r93pv1nejg20
title: '42'
desc: ''
updated: 1728957658815
created: 1728854629343
---

## 2024.10.13

- [x] Add self loops
- [x] fix issue with visualization.. → added nan % for predictions. Don't really like this aesthetically but will help monitor any model instabilities.
- [x] Fix indexing [[torchcell.data.neo4j_cell]] → had to make sure keys align

## 2024.10.14

- [x] Keys are still off. → Recompute indices
- [x] Recompute small set of traditional machine learning datasets.
- [ ] Check against traditional machine learning datasets.

- [ ] Make `get_init_graphs` just process a `gene_set` this would remove dependence on `raw_db`. We could also pass `raw_db` as arg for dependency in injection. It could optionally be `None` if the `db` has already been computed.

- [ ] in `IndexSplit` changes `indices` to `index`...
- [ ] Check the perturbation count for 1e04 because it seems off according to the traditional ml models.
