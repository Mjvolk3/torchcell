---
id: 7o9vl88x5ktyaj2401bpsvm
title: '04'
desc: ''
updated: 1737674889190
created: 1737507166367
---
## 2025.01.21

- [x] Recording note [[143744|dendron://torchcell/scratch.2025.01.16.143744]] → rough table outlining progress.
- [x] [[2025.01.21 - Choice of Data Preprocessing|dendron://torchcell/torchcell.models.isomorphic_cell#20250121---choice-of-data-preprocessing]]
- [x] [[torchcell.models.isomorphic_cell]] → started

## 2025.01.22

- [x] [[Cell-vs-SupCR|dendron://torchcell/torchcell.losses.Cell-vs-SupCR]]
- [x] [[2025.01.22 - Update Mermaid|dendron://torchcell/torchcell.models.isomorphic_cell#20250122---update-mermaid]]
- [x] [[2025.01.21 - Algorithm|dendron://torchcell/torchcell.models.isomorphic_cell#20250121---algorithm]]
- [x] Mermaid alone [[233025|dendron://torchcell/scratch.2025.01.21.233025]]
- [x] Algorithm alone [[212028|dendron://torchcell/scratch.2025.01.22.212028]]

## 2025.01.23

- [ ] Implement isomorphic cell.
- [ ] Implement `SupCR` loss.


***

- [ ] Update `cell_latent_perturbation`
- [ ] `cell_latent_perturbation` remove stoichiometry for reaction aggregation
- [ ] `cell_latent_perturbation` unify embeddings across graphs
- [ ] `cell_latent_perturbation` use `ISAB` set transformer for `intact_whole`
- [ ] unified model

- [ ] Memory issue with regression to classification scripts. We still have issue of processing memory accumulation. Unsure where it is coming from. Will only need to be solved if we use these losses.
- [ ] Implement intact and pert phenotype processor.
- [ ] Synthesize Results in report. Discussion on consideration of use alternative methods like mse plus a divergence measure.
- [ ] Run metabolism label split regression run
- [ ] Information Diff., WL Kernel

## Notes on Metabolism

- Can get Gibbs Free Energy of reaction from [MetaCyc](https://biocyc.org/reaction?orgid=META&id=D-LACTATE-DEHYDROGENASE-CYTOCHROME-RXN)
- To preserve sign information in [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.met_hypergraph_conv]] we should use activations that can handle negative input like leaky relu, elu, or tanh.

## Notes Related to Dango

Breakout into specific notes on Dango.

- [ ] Verify

> Pearson correlation between the trigenic interaction scores of two individual replicates is around 0.59, which is much lower than the Pearson correlation between the digenic interaction score of two replicates from the same data source (0.88). ([Zhang et al., 2020, p. 3](zotero://select/library/items/PJFDVT8Y)) ([pdf](zotero://open-pdf/library/items/AFBC5E89?page=3&annotation=D8D949VF))

- [ ] Plot P-Values of current dataset to compare to predicted interactions. Can do for both digenic and trigenic interactions. Do this over queried datasets.

- [ ] What is purpose of the pretraining portion? Why not just take embeddings and put into this hypergraph embedding portion?
