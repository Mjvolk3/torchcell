---
id: r1orjyberqc5mathlsswg03
title: '07'
desc: ''
updated: 1739417142633
created: 1739216011880
---

## 2025.02.11

- [x] Add images [[Analyze_feature_distributions|dendron://torchcell/experiments.003-fit-int.scripts.analyze_feature_distributions]]
- [x] [[Problem Formulation|dendron://torchcell/003-fit-int.problem-formulation]]

## 2025.02.12

🏆 of the day - 4 hour sweeps - learning rate

- [x] Collected key papers for sparse graph transformer
- [ ] Reviewed data of IGB run. Model didn't fit... Bit troubling. We plan do changed the model a bit to get matching parameters in both sides of model. Maybe use `GAT` instead. Then run sweep over fitness and gene interactions. First sweep should be short fast, attempt to get quick smooth learning. I also want to use that as an opportunity to add plot logging and adjust the logged metrics.


- [ ] Add learnable embedding.
- [ ] 



- [ ] Fix pin memory not working due to some data being sent to device prior to the pinning.
- [ ] Fix slow subgraph representation.

- [ ] Add after we get run going on GPU [[torchcell.viz.visual_regression]]

- [ ] Edit to get most up to date formula of the problem. [[Isomorphic_cell|dendron://torchcell/torchcell.models.isomorphic_cell]]

## 2025.02.12

🏆 of the day - 10 hour sweeps.

***
**Node Embeddings for Whole Genome**

- [ ] Delay to feature transfer
- [ ] Find and replace str for moving node embeddings.
- [ ] Move all node embeddings `M1`.
- [ ] Delete Node embeddings on `Delta`.
- [ ] Transfer Node embeddings to `Delta`.
- [ ] Remove node embeddings on `GH` so when we get machine on return things will break until we transfer node embeddings back.

- [ ] Wait on this... plots for enrichment of all genes. Contains any of all graphs.
- [ ] Make sure y distributions look like they y from other datasets.
- [ ] Histogram of genes usage in subset... we have these plots somewhere.
- [ ] Evaluate
500,000 (5e5) - cannot do with current db we have
500,000 (5e5)

- [ ] Want small datasets with most possible perturbations. Check network overlap?

***

### Writing

- [ ] [[Outline 02|dendron://torchcell/paper.outline.02]] - Move on to outline 3 too much has changed. [[Outline 03|dendron://torchcell/paper.outline.03]]

- [ ] Start Draft pipeline. Bring in thesis latex template.

***

- [ ] Test edge attention on hypergraph conv
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
- To preserve sign information in [[Met_hypergraph_conv|dendron://torchcell/torchcell.nn.stoichiometric_hypergraph_conv]] we should use activations that can handle negative input like leaky relu, elu, or tanh.

## Notes Related to Dango

Breakout into specific notes on Dango.

- [ ] Verify

> Pearson correlation between the trigenic interaction scores of two individual replicates is around 0.59, which is much lower than the Pearson correlation between the digenic interaction score of two replicates from the same data source (0.88). ([Zhang et al., 2020, p. 3](zotero://select/library/items/PJFDVT8Y)) ([pdf](zotero://open-pdf/library/items/AFBC5E89?page=3&annotation=D8D949VF))

- [ ] Plot P-Values of current dataset to compare to predicted interactions. Can do for both digenic and trigenic interactions. Do this over queried datasets.
- [ ] What is purpose of the pretraining portion? Why not just take embeddings and put into this hypergraph embedding portion?

***

[[04|dendron://torchcell/user.Mjvolk3.torchcell.tasks.weekly.2025.04]]
