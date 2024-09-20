---
id: 2udnurpk5ce6e78o55plv93
title: '30'
desc: ''
updated: 1722369681071
created: 1721590753227
---

## 2024.07.25

- [x] pr to biocypher with proper testing.

## 2024.07.24

- [x] Add edge between `phenotype to experiment reference`. Add to existing configs.

## 2024.07.23

- [x] Add pubmed id to dataset in schema.
- [x] After merge clean up commented code in [[torchcell.adapters.cell_adapter]] and [[torchcell.dataset.experiment_dataset]]
- [x] Add gene essentiality dataset. → [[2024.07.23 - Gene Essentiality Duplicates|dendron://torchcell/torchcell.datasets.scerevisiae.sgd_gene_essentiality#20240723---gene-essentiality-duplicates]]
- [x] Add synthetic lethality. Do same as for essentiality. → [[2024.07.23 - We Will Not Be Able to Manually Populate Meta Data for all Synthetic Lethality and Synthetic Rescue Experiments|dendron://torchcell/torchcell.datasets.scerevisiae.syn_leth_db_yeast#20240723---we-will-not-be-able-to-manually-populate-meta-data-for-all-synthetic-lethality-and-synthetic-rescue-experiments]]
- [x] #ramble we might have an issue with adding pub information to experiment reference... This will create many nodes that look like duplicates which really aren't. → This is obviously going to be an issue... need to handle immediately. Shame I overlooked it. → This is mainly an issue for reference. Since we will end up with many `experiment reference` that are essentially the same but belong to different publications... We want forced deduplication based on essential data.
- [x] Split out publication information into a separate node.

## 2024.07.21

- [x] After merge delete feature branch. Drop all stashes
- [x] Add genes essentiality dataset. → requires fix on genome looks like SGD changed path names. → fixed url for [[torchcell.sequence.genome.scerevisiae.S288C]] but this doesn't contain gene data needed for essentiality. → in the works.
