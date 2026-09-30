---
id: j6diexml871malq6mpxusvh
title: Query_amino_acid_betaxanthin
desc: ''
updated: 1790742054980
created: 1790742054980
---

## 2026.09.29 - Supported query run, job 3056

Runs `torchcell/knowledge_graphs/queries/amino_acid_betaxanthin.cql` through `Neo4jCellDataset` under slurm (`gh_query_amino_acid_betaxanthin.slurm`, submitted from the worktree with `TORCHCELL_SRC`): no converter, no deduplicator, `GenotypeAggregator`. Job 3055 completed but its fragment counted metabolite keys by summing values (a bug in the script); the cache was moved to the session scratchpad and job 3056 rebuilt it from the query (36 s, MaxRSS 1.13 GB).

Job 3056 against release `2026.09.21-ab6d8c5d` (version 1.2):

- Query: 4,678 Mulleder, 4,313 Cooper and 4,706 Cachera entries (the release serves 4,719 Cachera experiments; 13 delete genes outside `gene_set`).
- 9,600 processed records: 4,052 hold one Mulleder and one Cooper entry, 22 one Mulleder and two Cooper entries (renamed genes), 604 Mulleder alone, 217 Cooper alone, 4,704 one Cachera entry, 1 two Cachera entries (the *ARO4* and *ARO7* deletions). No record mixes Cachera with an amino-acid entry.

Numbers from `experiments/034-showcase-datasets/results/amino_acid_betaxanthin_query.json`.
