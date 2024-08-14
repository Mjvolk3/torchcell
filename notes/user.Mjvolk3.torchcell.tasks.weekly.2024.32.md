---
id: epexvgtp427jnjyj40bupb8
title: '32'
desc: ''
updated: 1723481102337
created: 1722806044510
---

Changing as of this week, days are going down chronologically. Before it made sense with extremely long notes to have them reversed. [[user.Mjvolk3.torchcell.tasks.deprecated]]. We do it this way in all other notes.

## 2024.08.04

- [x] Check on db build. → failed due to time limit... Great...
- [x] We can optimize this build per edge node type batch sizes. Custom batch sizing in adapters. → we have added some memory reduction mechanisms via yaml

## 2024.08.08

- [x] biocypher-out combine so we can combine partial knowledge graph builds → [[2024.08.08 - How it works|dendron://torchcell/torchcell.database.biocypher_out_combine#20240808---how-it-works]]
- [x] Test that the combine gives the same as all at once. → [[2024.08.08 - Checking that Combine Produces the Same Import Summary|dendron://torchcell/torchcell.knowledge_graphs.dmf_tmi_combine_kg#20240808---checking-that-combine-produces-the-same-import-summary]] → Not working correctly.

## 2024.08.10

- [x] [[2024.08.10 - Troubleshooting Combine to Match Simultaneous Graph Builds|dendron://torchcell/torchcell.knowledge_graphs.dmf_tmi_combine_kg#20240810---troubleshooting-combine-to-match-simultaneous-graph-builds]]
- [x] Get apoc working for run. We need this to run queries with random number permutations. → [apoc release](https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/tag/4.4.0.30) → [[Apoc|dendron://torchcell/neo4j.apoc]] → [[apoc.coll.randomItems|dendron://torchcell/neo4j.apoc.coll.randomItems]] → put on GilaHyper and update slurm script.
- [x] Note to track builds [[GilaHyper Builds|dendron://torchcell/database.docker.builds#gilahyper-builds]] → [[GilaHyper 2024-08-08_18-24-33|dendron://torchcell/database.docker.builds#gilahyper-2024-08-08_18-24-33]]
- [x] Start script for `traditional_ml`[[Traditional_ml_dataset|dendron://torchcell/experiments.002-dmi-tmi.scripts.traditional_ml_dataset]]
- [x] Construct `dmi` and `tmi` `kuzmin` kgs

