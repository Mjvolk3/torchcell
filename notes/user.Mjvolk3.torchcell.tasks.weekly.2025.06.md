---
id: 9iva3m2lluhgot3xu1cq9nl
title: '06'
desc: ''
updated: 1738549091480
created: 1738521316295
---

## 2025.02.02

- [x] Confirm that we have embeddings for max genome size since these genes are needed for metabolism. [[experiments.003-fit-int.scripts.check_node_embedding_sizes_whole_genome]] ran on `GH` but failed do to model naming, fixed simple type on M1 but now it is computing for `6607`. → #TODO COMPLETION STATUS?
- [ ] Verify [[experiments.003-fit-int.scripts.check_node_embedding_sizes_whole_genome]] works on `GH`
- [ ] #ramble Should we be saving the embeddings in a different folder according to a particular genomic setup? `torchcell/data/scerevisiae/nucleotide_transformer_embedding/processed/nt_window_5979_max.pt` → Yes we should migrate these to something lke `torchcell/data/scerevisiae/<sgd_genome_max_6607_genes>/nucleotide_transformer_embedding/processed/nt_window_5979_max.pt` → Will have to move all of them and do sweeping search and replace over dir name, since it is just str shouldn't be bad. After completion of moving on M1 can redistribute to `Delta`.

### 2025.02.02 - Node Embeddings for Whole Genome

- [ ] Find and replace str for moving node embeddings.
- [ ] Move all node embeddings `M1`.
- [ ] Delete Node embeddings on `Delta`.
- [ ] Transfer Node embeddings to `Delta`.
- [ ] Remove node embeddings on `GH` so when we get machine on return things will break until we transfer node embeddings back.

### 2025.02.02 - Iso Attentional Batch Overfit

- [ ]
- [ ]
- [ ]

### 2025.02.02 - QM9 Data Grid Search

- [x] 8 ish pm check on progress. Decide on more jobs. → 6pm all finished. → Dist plots bad due to num bins. Down sampling for plotting → 10 min/epoch with `5e4` data, little less than 1/3 of all data... waiting.
- [ ] Run 2 epoch test run then predict time for run.
- [ ] 

### 2025.02.02 - Iso Attentional 2.5e4 Data Sweep

- [ ] Write training scripts.
- [ ] Decide on sweep parameters prioritizing lambdas. Fix most layer dims in model. Need to really be careful in not over doing it!
- [ ] Test speed of sweep on `GH`
- [ ] Move to `Delta` and start sweep.

***

- [ ] Wait on this... plots for enrichment of all genes. Contains any of all graphs.
- [ ] Make sure y distributions look like they y from other datasets.
- [ ] Histogram of genes usage in subset... we have these plots somewhere.
- [ ] Evaluate
500,000 (5e5) - cannot do with current db we have
500,000 (5e5)

- [ ] Want small datasets with most possible perturbations. Check network overlap?

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
