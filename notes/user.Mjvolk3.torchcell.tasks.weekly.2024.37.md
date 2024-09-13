---
id: z824gvi4f507r9ky9poh0zj
title: '37'
desc: ''
updated: 1726245631590
created: 1725815628586
---


## 2024.09.08

- [x] Either add dataset to experiment and reference or to lmdb structure. lmdb probably makes most sense. Check that splitting from this would work. It should since we iterate over data like this. `label = data["experiment"]`... Check that this aligns with adapters. → we added dataset to experiment and reference as I think this will be more useful when inspecting data in `repl`.
- [x] Change Schema to generic structure as described here [[torchcell.datamodels.schema]]. → [[2024.08.12 - Making Sublcasses More Generic For Downstream Querying|dendron://torchcell/torchcell.datamodels.schema#20240812---making-sublcasses-more-generic-for-downstream-querying]] and [[2024.08.26 - Generic Subclasses Need to Consider Phenotype Label Index|dendron://torchcell/torchcell.datamodels.schema#20240826---generic-subclasses-need-to-consider-phenotype-label-index]]
- [x] `torch_scatter` directly from `https://data.pyg.org/whl/torch-2.4.0+cpu.html` fixed issue with `database/build/build_linux-arm.sh`. Not sure how to fix this in general for installation.

## 2024.09.09

## First Experiment

```python
repeats = 5
matrix_size = 1000
```

GilaHyper no Slurm

```bash
(torchcell) michaelvolk@gilahyper torchcell % /home/michaelvolk/miniconda3/envs/torchcell/bin/python /home/michaelvolk/Documents/projects/torchcell/torc
hcell/scratch/cpu_performance_benchmark.py
Raw Data: [20.143832683563232, 19.094918727874756, 20.667733669281006, 19.605305671691895, 19.61408519744873]
Mean Time: 19.83 seconds
Standard Deviation: 0.54 seconds
```

GilaHyper Slurm. There is a slight overhead.

```bash
Raw Data: [20.507216453552246, 20.640620231628418, 20.882274389266968, 21.411606311798096, 20.875935316085815]
Mean Time: 20.86 seconds
Standard Deviation: 0.31 seconds
```

Delta Slurm

```bash
Raw Data: [1.1937663555145264, 1.5070643424987793, 1.3696041107177734, 1.2952206134796143, 1.2181971073150635]
Mean Time: 1.32 seconds
Standard Deviation: 0.11 seconds
```

## Second Experiment

```python
repeats = 5
matrix_size = 4000
```

GilaHyper Slurm.

```bash
Raw Data: [2717.3176045417786, 2077.3530197143555, 2640.1949455738068, 2294.7787528038025, 2614.683986902237]
Mean Time: 2468.87 seconds
Standard Deviation: 243.28 seconds
```

Delta Slurm

```bash
Raw Data: [15.837791681289673, 9.31566071510315, 12.116915702819824, 9.68659520149231, 9.873022317886353]
Mean Time: 11.37 seconds
Standard Deviation: 2.44 seconds
```

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
- [ ] Add multiple edge for phenotypes
- [ ] Update edge creation in [[torchcell.adapters.cell_adapter]]
- [ ] Write adapters for remaining datasets
- [ ] Add a kg for each dataset class

- [ ] `GH` ssh security

- [ ] Build kgs

- [ ] Run KG build for kuzmin2020 interactions

- [ ] Update combine to add a `README.md` which can serve as a trace to combined data.
- [ ] Combined datasets and update readonly db.

- [ ] Create `1e03`, `1e04`, and `1e05` datasets with positive `tmi`. → This will be difficult because it'll be hard to balance mutant types. We could just use triple mutants with the plan to down select by enriched double mutants.

***

- [ ] Zendron on `zotero_out`
- [ ] Add in transformation to essentiality to growth type phenotype. This should probably be enforced after querying during data selection and deduplication. The rule is something like if we can find some reasonable fixed function for transforming labels we add them. Don't know of a great way of doing this but. Possible we can even add these relations to the Biolink ontology. In theory this could go on indefinitely but I think one layer of abstraction will serve a lot of good at little cost.
- [ ] Add expression dataset for mechanistic aware single fitness
- [ ] Add expression from double fitness
- [ ] Add fitness from singles
- [ ] Add fitness from doubles
- [ ] We need a new project documents reproducible procedure on `gh` for restarting slurm, docker, etc.
- [ ] Run container locally with [[torchcell.knowledge_graphs.minimal_kg]] → Had to restart to make sure previous torchcell db was deleted. → struggling with `database/build/build_linux-arm.sh` retrying from build image. → Cannot install CaLM... →
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.
- [ ] Expand [[paper-outline-02|dendron://torchcell/paper.outline.02]]
- [ ] `ExperimentReferenceOf` looks broken.
- [ ] Make sure ports are getting forwarded correctly and that we can connect to the database over the network. We need to verify that we can connect with the neo4j browser.
- [ ] Try to link docker and slurm with `cgroup`
- [ ] Run build bash script for testing.
- [ ] `gh` Test build under resource constraints.
- [ ] Change logo on docs → to do this we need a `torchcell_sphinx_theme`. → cloned, changed all `pyg_spinx_theme` to `torchcell_sphinx_theme`, pushed, trying rebuild.
- [ ] Remove software update on image entry point
- [ ] dataset registry not working again because circular import
