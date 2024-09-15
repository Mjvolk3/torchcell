---
id: z824gvi4f507r9ky9poh0zj
title: '37'
desc: ''
updated: 1726425224184
created: 1725815628586
---


## 2024.09.08

- [x] Either add dataset to experiment and reference or to lmdb structure. lmdb probably makes most sense. Check that splitting from this would work. It should since we iterate over data like this. `label = data["experiment"]`... Check that this aligns with adapters. → we added dataset to experiment and reference as I think this will be more useful when inspecting data in `repl`.
- [x] Change Schema to generic structure as described here [[torchcell.datamodels.schema]]. → [[2024.08.12 - Making Sublcasses More Generic For Downstream Querying|dendron://torchcell/torchcell.datamodels.schema#20240812---making-sublcasses-more-generic-for-downstream-querying]] and [[2024.08.26 - Generic Subclasses Need to Consider Phenotype Label Index|dendron://torchcell/torchcell.datamodels.schema#20240826---generic-subclasses-need-to-consider-phenotype-label-index]]
- [x] `torch_scatter` directly from `https://data.pyg.org/whl/torch-2.4.0+cpu.html` fixed issue with `database/build/build_linux-arm.sh`. Not sure how to fix this in general for installation.

## 2024.09.09

- [x] [[Versus Delta Speed|dendron://Kbase/gilahyper.versus-delta-speed]]
- [x] Planning [[2024.09.09 - Thinking About Pooling GNNs|dendron://torchcell/experiments.003-fit-int-leth#20240909---thinking-about-pooling-gnns]]
- [x] #ramble GO graph looks like the levels don't necessarily feed into levels directly above... We might be able to git fewer levels by looking at direct connections to levels above which could give a smart number pooling layers for the model.

- [x] Fix all of [[torchcell.datasets.scerevisiae.costanzo2016]]
- [x] Fix all of [[torchcell.datasets.scerevisiae.kuzmin2018]]
- [x] Fix [[torchcell.datasets.scerevisiae.sgd_gene_essentiality]]

## 2024.09.10

- [x] `kuzmin2018` alternate data source. [[2024.09.10 - Kuzmin2018 Alternative Download Source|dendron://torchcell/torchcell.datasets.scerevisiae.kuzmin2018#20240910---kuzmin2018-alternative-download-source]]
- [x] [SGA automated image analysis](http://sgatools.ccbr.utoronto.ca/) called SGATools.
- [x] [[2024.09.10 - There are Negative Fitness Values|dendron://torchcell/torchcell.datasets.scerevisiae.kuzmin2020#20240910---there-are-negative-fitness-values]] validate negative fitness to `0.0`.
- [x] Write `DmfKuzmin2020Dataset`.
- [x] Build all datasets

## 2024.09.11

- [x] Fix adapters for new data → did partially

## 2024.09.12

- [x] Meet with Cheng and Ge to discuss project
- [x] `GH` security research → lots of work to do here.

## 2024.09.13

- [x] Fix adapter yaml names to match
- [x] Add multiple edge for phenotypes → just had to add input label for all of the phenotypes.
- [x] Update edge creation in [[torchcell.adapters.cell_adapter]] → not needed since we fixed by adding `input_label`
