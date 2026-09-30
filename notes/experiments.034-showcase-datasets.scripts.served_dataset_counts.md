---
id: 5lvf53srypj096tlbklgre5
title: Served_dataset_counts
desc: ''
updated: 1790751414381
created: 1790751414381
---

## 2026.09.30 - Size of the collection on the datasets index

Reads the newest committed release snapshot (`database/releases/<release>.json`, the file [[torchcell.knowledge_graphs.release_snapshot]] writes at each KG build) and writes `docs/source/datasets/_generated/served_counts.md`: the release, the number of datasets and total experiments it serves, the node count, and one row per dataset with its experiments served and the dataset page that shows it (from `docs_page` of each supported query and the `dataset.id` values its Cypher selects). Included on `docs/source/datasets/index.md` under "Size of the collection". For `2026.09.21-ab6d8c5d`: 51 datasets, 52,743,047 experiments, 99,724,909 nodes. Rerun after every release snapshot lands and whenever a dataset page is registered.
