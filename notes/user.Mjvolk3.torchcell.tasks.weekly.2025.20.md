---
id: t56ofd5vftd2cgtz6a8uxwi
title: '20'
desc: ''
updated: 1747352212796
created: 1747059074017
---

## 2025.05.12

- [x] Get `String11` working.
- [x] Launch `String11` jobs. 3 in total spaced out so that configs load. → Instead just wrote 2 more configs and passed via `hydra`
- [x] Run DCell to see if we can overfit. → Added layer norm instead of `batchnorm` since we saw differences across batch this might help in the overfitting setting. Added some plots. → Layer norm didn't seem to help
- [x] Train DCell overfit with increased alpha. → Doesn't working. We have tried a lot of small rapid, poorly tracked experiments, trying to find solution. None have worked. Documenting tomorrow.
- 🔲 Run [[Dcell_batch_005_verify_mutant_state|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.dcell_batch_005_verify_mutant_state]] with larger bach size ... if we try overfit with larger batch.
- [x] [[2025.05.12 - Dango Reproducibility Run|dendron://torchcell/wandb-experiments.wandb-sync-agent-dirs.igb-sync-log#20250512---dango-reproducibility-run]]

## 2025.05.13

- #ramble We have tried to basically change every possible original config from the `DCell` model and cannot get above a 0.45 pearson correlation. We tried to add some additional options to see if it helps at all including `layer_norm`, and `gelu` activation. We also added having additional layers... Not sure if these are just linear... which would be pointless need to double check.

- [x] Try learnable embedding over genes instead of binary. We can apply perturbation by multiplying by 0. We could do this generically by either initializing with `nn.embedding` of `embedding_size` or by initializing with ones. Then no matter what we just apply perturbation by taking the vectors of genes and multiple perturbed genes by 0. Then distribute their representations to the different subsystems. → We are trying this... We made a commit right before trying to be able to revert to a working version.
  
This approach could help because:

1. Gene embeddings can capture more nuanced relationships than binary indicators
2. The model gains capacity to learn gene-specific contributions to subsystems
3. It maintains the interpretable subsystem structure while enhancing representational power

- [x] Find points that cannot be learned. Plot them as red dots on the gene ontology DAG. This should help identify why certain relationships cannot be learned. I assume that it is just weaker connectivity that limits expressiveness. → These are plotted for now. Can try to explain DCell limits later.
- [x] Adding `nn.embedding` to `dcell`

## 2025.05.14

- [x] Continuation of GO diff in submodule count from paper.
- [x] Rest of experiments on `DCell` overfit.
- [x] We have 37,854 go annotations whereas there are 120,735 reported by [geneontology.org](https://current.geneontology.org/products/pages/downloads.html) - This is a pretty big discrepancy that probably accounts for the differences we are seeing. → This is misleading because there are genes outside of the reference genome etc. Looks like the number is actually much smaller and the non-overlapping region doesn't make huge difference to sgd json go data.

## 2025.05.15

- [x] Check total GO Terms to make sure that Union increased number. → abandoned for now due to complexity
- [x] From adding GFF it created a lot of complexity in filter because the data has different formats in SGD json versus GO GAF from source. From preliminary looks there are discrepancies between these two sources of GO but the json GO seemed to dominate so it is probably fine to stick with it for now. Also `geneontology.org` says that they are using the gene ontology that comes from SGD... Another frustrating data discrepancy. `geneontology.org` doesn't appear to have versions readily available so it is difficult to know if we are even using up to date GO. Or if it is more up to date than the json. The best think to do will be to compare the `graph.G_raw` with the GAF then state why we use json. We just want to provide sufficient evidence for justifying this decison.
- [x] [[2025.05.15 - Approximating Original DCell Given Discrepancies|dendron://torchcell/torchcell.models.dcell#20250515---approximating-original-dcell-given-discrepancies]]
- [x] Summary Note [[2025.05.15 - Experiments Over Last Few Days|dendron://torchcell/torchcell.models.dcell#20250515---experiments-over-last-few-days]]
- [x] Sort out notes for commits - Split out `DCell` notes for commit. Add to model file.
- [x] Fixed filters to use `deepcopy`. [[Go_term_filter_analysis|dendron://torchcell/experiments.005-kuzmin2018-tmi.scripts.go_term_filter_analysis]]

***

- [ ] DCell - Working

- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.

- [ ] Morphology Random Forest Baseline

- [ ] Morphology animation ? for fun...

- [ ] HeteroCell?

- [ ] Contrastive DCell head on HeteroCell.

- [ ] sync scripts for delta

***

- [ ] Add Gene Expression datasets
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.

- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work

- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.

- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
-
