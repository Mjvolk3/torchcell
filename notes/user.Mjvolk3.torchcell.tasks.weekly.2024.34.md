---
id: 65w5kdtce7q047b7ollalhq
title: '34'
desc: ''
updated: 1724674560835
created: 1724101120841
---
## 2024.08.19

- [x] Solve speed issue with loader by using a cpu loader. Current issue is that the iteration speeds are super low because we are not using multiprocessing on loading datasets for building `train`, `val`, `test` → switched to trying gpu but still slow... → using `torch_scatter` helps some. → Best solution is just to use m1... very annoying.
- 🔲 Bring in kuzmin2020 dataset
- 🔲 Plots on data distributions.

## 2024.08.22

- [x] Check if `Neo4jCell` can be pickled without deleting vars. → can be.
