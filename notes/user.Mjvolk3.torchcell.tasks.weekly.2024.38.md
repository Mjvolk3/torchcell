---
id: jculhuhy90fej0ip6fe2am5
title: '38'
desc: ''
updated: 1726687055870
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
- [ ] There is an issue that we don't have all of the datasets in the db.. Specifically we don't have the `SyntheticRescue` or `GeneEssentiality` Datasets.

- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.

***
