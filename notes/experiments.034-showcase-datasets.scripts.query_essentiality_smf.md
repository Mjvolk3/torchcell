---
id: uwzhlfe3q7dh4twhs5mochw
title: Query_essentiality_smf
desc: ''
updated: 1790725116913
created: 1790725116913
---

## 2026.09.29 - Supported query run, job 3029

Runs `torchcell/knowledge_graphs/queries/essentiality_smf.cql` through `Neo4jCellDataset` under slurm (`gh_query_essentiality_smf.slurm`, submitted from the worktree with `TORCHCELL_SRC`). The build is the no-merge shape `torchcell.data.label_policy` is written for: `CompositeFitnessConverter`, no deduplicator, `GenotypeAggregator`.

The first run (job 3021) had no aggregator and failed in `Neo4jCellDataset.compute_phenotype_info`, which expects every processed value to be a list of entries; a converter-only build stores one dict per key. Adding `GenotypeAggregator` fixed it; the stale cache was moved to the session scratchpad, not deleted.

Job 3029 (completed in 4 min 58 s, MaxRSS 2.07 GB) against release `2026.09.21-ab6d8c5d` (version 1.2):

- Query: 1,140 essentiality entries and 20,358 SMF entries. The release serves 1,140 and 20,484 experiments; the 126 SMF experiments not returned belong to 53 genes outside `SCerevisiaeGenome.gene_set` (checked against the dev store). The store's 1,329 essentiality records become 1,140 graph nodes because the node id is the sha256 of the experiment dump, which excludes the publication, and all 347 records of the 158 multi-annotation genes share identical experiment content.
- 5,673 processed records; 1,140 carry a `converted_zero` entry, 233 of them with nothing else and 907 beside a measured Costanzo fitness.
- `label_df` keeps the last non-missing entry, which is the converted 0 for 126 of those 907. The default `LabelPolicy` refuses the converted 0 in all 907.

Numbers from `experiments/034-showcase-datasets/results/essentiality_smf_query.json`.
