---
id: 26cw7x2rsicu2lscj23kgjy
title: '03'
desc: ''
updated: 1768415191929
created: 1768264787823
---
## 2026.01.13

- [x] Wandb sync of recent #IGB
- [x] #GH Get a inference comparison on deletion versus larger dataset [[Performance Diff 010 009|experiments.010-kuzmin-tmi.performance-diff-010-009]]

## 2026.01.14

- [x] Collected 668 uncharacterized and 684 dubious genes [[experiments.013-uncharacterized-genes.scripts.count_dubious_and_uncharacterized_genes]]
- [x] Analyzed 44,619 triple interactions involving uncharacterized genes [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
- [x] Found 3 essential ∩ uncharacterized genes, identified YCR016W annotation discrepancy [[experiments.013-uncharacterized-genes.scripts.uncharacterized_essential_overlap]]

- [x] Refactored inference/eval configs from r0/E006_M## to m00/m01/m02 naming convention
- [x] ADH/ALD triple knockout case study for 2,3-butanediol production (Ng et al. 2012) [[Trigenic_interaction_adh1_adh3_adh5|experiments.010-kuzmin-tmi.scripts.trigenic_interaction_adh1_adh3_adh5]]
- [x] Costanzo2016 SMF/DMF/ε lookup for ADH1/ADH3/ADH5/ALD6 genes [[Adh_ald_costanzo2016_lookup|experiments.010-kuzmin-tmi.scripts.adh_ald_costanzo2016_lookup]]
- [x] Improved graph recovery metric documentation with intuition/interpretation format [[Graph_recovery|torchcell.viz.graph_recovery]]

- [ ] Clean up and commit 014
- [ ] Clean up and commit 012
- [ ] Add n to fitness interaction data
- [ ] Put Spell in right place
- [ ] Add examples or clarify the schema design to avoid ambiguity in property definitions.

***

Review

## 2025.12.17

- [ ] Add a check to the final gene list to see if we have any genes overlapped with the double gene expression #GH

- [ ] Kemmeren, Sameith dataset verify metadata #M1

- [ ] Follow up on the dataset outlier comparison by reporting the spearman at snapshot for very best model across the different scenarios. → From quick comparison it looks like spearman for datasets with more data are still higher. Test datasets are obviously not exactly the same. →

- [ ] Expression datasets write adapters #M1

- [ ] Start DB build over night #GH

## 2025.12.20

- [x] Is expression data from SGD web browser available in the gene json? → It is not it comes from SPELL
- [ ] Are there images associated with [yeast-gfp data](https://yeastgfp.yeastgenome.org/). Yes if needed.. Maybe they do have some more information then just categorical classification? Maybe not.

***

 Move to ideas notes

- Studying epistasis through knockouts ... maybe not feasible... too many parts.
- Do locations of proteins change depending on populations changes of different kinds of proteins. We know that proteins go to different places. Maybe you can track this probabilistically, but can you shift the distribution of other proteins?
- The GFP-tagged library is distributed by < Invitrogen >.
- The RFP-tagged strains used for the colocalization studies can be
- If we could obtain these strains

***

- [ ] Email CW
