---
id: 5lvf53srypj096tlbklgre5
title: Served_dataset_counts
desc: ''
updated: 1790751414381
created: 1790751414381
---

## 2026.09.30 - Size of the collection on the datasets index

Reads the newest committed release snapshot (`database/releases/<release>.json`, the file [[torchcell.knowledge_graphs.release_snapshot]] writes at each KG build) and writes `docs/source/datasets/_generated/served_counts.md`: the release, the number of datasets and total experiments it serves, the node count, and one row per dataset with its experiments served and the dataset page that shows it (from `docs_page` of each supported query and the `dataset.id` values its Cypher selects). Included on `docs/source/datasets/index.md` under "Size of the collection". For `2026.09.21-ab6d8c5d`: 51 datasets, 52,743,047 experiments, 99,724,909 nodes. Rerun after every release snapshot lands and whenever a dataset page is registered.

## 2026.10.12 - Grouped by organism

A release may now serve a subset of organisms ([[torchcell.knowledge_graphs.dataset_adapter_map]]), so the fragment states which. The header line gives the totals by organism and says where they came from, and the table is one table per organism (in `Organism` order) with its own total row.

- A snapshot written since the switch records `organisms` and each dataset's `organism` ([[torchcell.knowledge_graphs.release_snapshot]]); those are used, and a dataset outside the recorded set is an assertion failure.
- A snapshot from before the switch records neither. The fragment says so in its header line and reads each dataset's organism off its loader class (`resolve_dataset_class` then `dataset_organism`), the attribute the build selects by.

Rerun on 2026-10-12. The newest committed snapshot is now `2026.10.06-4b293d34` (it was `2026.09.21-ab6d8c5d` when the fragment was last written): 51 datasets, 52,668,931 experiments, 100,127,515 graph nodes, every dataset `scerevisiae`, and the release records no organism set.
