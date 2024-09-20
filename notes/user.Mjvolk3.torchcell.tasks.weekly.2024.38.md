---
id: jculhuhy90fej0ip6fe2am5
title: '38'
desc: ''
updated: 1726777197524
created: 1726425036900
---

## 2024.09.15

- [x] [[2024.09.15 - Rename to Match other Datasets|dendron://torchcell/torchcell.datasets.scerevisiae.sgd#20240915---rename-to-match-other-datasets]]
- [x] #ramble If want to be able to create kgs in container we would need all data dirs. Especially relevant for when we have a path dependency like `data/sgd/genome` [[torchcell.datasets.scerevisiae.sgd]]
- [x] Write adapters for remaining datasets → trying to generalize.
- [x] Add a kg for each dataset class

## 2024.09.16

- [x] Build kgs → failed due to config error some nans
- [x] Fix [[torchcell.datasets.scerevisiae.kuzmin2020]] to filter out nans → had to fix `smf` and `dmf`

## 2024.09.17

- [x] Build kgs → failed due to config error floats when they should be bool.
- [x] `GH` ssh security  → now we require pass phrase, password and 2FA... 😅
- [x] Launch builds for local small build [[scerevisiae_small_global_kg|dendron://torchcell/torchcell.knowledge_graphs.conf.scerevisiae_small_global_kg.yaml]] and large `GH` build [[torchcell.knowledge_graphs.conf.scerevisiae_global_kg.yaml]]

## 2024.09.18

- [x] Check on local build → Looks like everything worked now we can create `1e03`, `1e04`, and `1e05` datasets. → this build takes `8 hr`
- [x] Check on `GH` build → we had a process in pool terminate abruptly... investigating. → looks like we ran out of memory... The issue seems to be event 15 which is publication node. I don't remember if we changed the representation of publication after `tmiKuzmin2018` worked where we didn't have any memory reduction on publication nodes. Regardless we will try this and relaunch. → [[2024.09.18 - OOM on Publication Node|dendron://torchcell/torchcell.adapters.conf.dmf_costanzo2016_adapter.yaml#20240918---oom-on-publication-node]] → relaunched after change.
- [x] Fix torchcell availability on local db → import from shell
- [x] There is an issue that we don't have all of the datasets in the db.. Specifically we don't have the `SyntheticRescue` or `GeneEssentiality` Datasets. → fixed in steps below.
- [x] Why so few phenotypes in `SynthLethal` → [[2024.09.18 - Few Phenotype Nodes|dendron://torchcell/torchcell.adapters.conf.synth_lethality_yeast_synth_leth_db.yaml#20240918---few-phenotype-nodes]] → this is not an issue
- [x] SynthRescue has an error. → [[2024.09.18 - Troubleshooting Biocypher Build|dejndron://torchcell/torchcell.adapters.synth_leth_db#20240918---troubleshooting-biocypher-build]]
- [x] Check gene essentiality. → issue was were trying to add `label_statistic_name` to property in [[torchcell.adapters.cell_adapter]] when this doesn't exist
- [x] Run `Synth` adapters without build.
- [x] Run `GeneEssentiality` adapter without build.
- [x] Run `Synth` and `GeneEssentiality` adapter with build.
- [x] Inspect graph in neo4j browser → All datasets exists and look correct.
- [x] Run rebuild of small kg in it's entirety
- [x] Run rebuild of whole kg on `GH`

## 2024.09.19

- [x] Check small build
- [x] Check large build → Running, looks a little faster than before not sure why.
- [x] Get `raw_query` to work → had to just add publication key
- [x] Fix circular import → moved `BaseEmbedding` to `data`

- [ ] 
- [ ]
- [ ] Add

- [ ] Change "interaction" to "gene_interaction"
- [ ] Rebuild local database
- [ ] Rebuild large database


- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.

***
