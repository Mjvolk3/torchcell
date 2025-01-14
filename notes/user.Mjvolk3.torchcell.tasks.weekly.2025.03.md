---
id: wqwnnd3noe8djjjsc7m440v
title: '03'
desc: ''
updated: 1736891838087
created: 1736717919222
---
⏰ One thing: Metabolism Label Split Regression Run

## 2025.01.12

- [x] Working prototype [[Cell_latent_perturbation|dendron://torchcell/torchcell.models.cell_latent_perturbation]] →

## 2025.01.13

- [x] Working prototype [[Cell_latent_perturbation|dendron://torchcell/torchcell.models.cell_latent_perturbation]] → Added in stoichiometry for [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.met_hypergraph_conv]]
- [x] Overfit on one batch size of 16
- [x] Optimized some for loop to matrix ops

## 2025.01.14

- [x] Overfitting batch and plotting training curves and correlation plot.
- [x] [[Hetero_data|dendron://torchcell/torchcell.data.hetero_data]] → monkey patch `__repr__` to get better printing. [[2025.01.14 - Sample of Dataset|dendron://torchcell/torchcell.models.cell_latent_perturbation#20250114---sample-of-dataset]]
- [x] Begin RMA process for GH
- [ ] Add skip connections to `hypergraph_conv` [[torchcell/models/cell_latent_perturbation.py#^ebhvzdc73ycd]]
- [ ] Model walk through check each method [[Cell_latent_perturbation|dendron://torchcell/torchcell.models.cell_latent_perturbation]]


***

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
