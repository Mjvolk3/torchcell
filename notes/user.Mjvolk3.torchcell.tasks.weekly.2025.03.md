---
id: wqwnnd3noe8djjjsc7m440v
title: '03'
desc: ''
updated: 1738205262286
created: 1736717919222
---
⏰ One thing: Metabolism Label Split Regression Run

## 2025.01.12

- [x] Working prototype `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard) →

## 2025.01.13

- [x] Working prototype `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard) → Added in stoichiometry for [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.stoichiometric_hypergraph_conv]]
- [x] Overfit on one batch size of 16
- [x] Optimized some for loop to matrix ops

## 2025.01.14

- [x] Overfitting batch and plotting training curves and correlation plot.
- [x] [[Hetero_data|dendron://torchcell/torchcell.data.hetero_data]] → monkey patch `__repr__` to get better printing. `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard)
- [x] Begin RMA process for GH
- [x] Exploring alternative data structure. → `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard) → Added `Perturbation(GraphProcessor)`
- [x] Add skip connections to `hypergraph_conv`
- [x] Remove excess data `attrs` for improved virtual mem → `reaction_to_genes_indices`
- [x] Check model works on `cpu` and `gpu`
- [x] Add `log(cosh)`
- 🔲 Memory issue with regression to classification scripts. We still have issue of processing memory accumulation. Unsure where it is coming from. Will only need to be solved if we use these losses. → still not solved.
- [x] Experiment scripts and lightning module for `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard)
- 🔲 Model walk through check each method `torchcell/models/cell_latent_perturbation.py` (retired 2026-10-06 to the graveyard)

## 2025.01.16

- [x] Investigate results → Things don't look like they are working well. Maybe it is still too early to tell in the training process. We aren't seeing distribution matching on quantile loss. Other losses `dist`, `mse`, `logcosh` don't look any good.
- [x] See if all genes are in encoded vectors from genome. If this is true, we could distribute embeddings at the beginning of training. → Confirmed all genes in GEM are in genome. `YeastGEM().gene_set -  SCerevisiaeGenome().gene_set -> GeneSet(size=0, items=[])`
