---
id: 26cw7x2rsicu2lscj23kgjy
title: '03'
desc: ''
updated: 1768423699917
created: 1768264787823
---
## 2026.01.13

- [x] Wandb sync of recent #IGB

## 2026.01.14

- [x] Collected 668 uncharacterized and 684 dubious genes [[experiments.013-uncharacterized-genes.scripts.count_dubious_and_uncharacterized_genes]]
- [x] Analyzed 44,619 triple interactions involving uncharacterized genes [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
- [x] Found 3 essential ∩ uncharacterized genes, identified YCR016W annotation discrepancy [[experiments.013-uncharacterized-genes.scripts.uncharacterized_essential_overlap]]
- [x] Complete 3-step pipeline: gene collection, triple interaction analysis, essential overlap [[experiments.013-uncharacterized-genes.scripts.013-uncharacterized-genes]]
- [x] Identified 2,852 extreme TMI interactions involving 178 uncharacterized genes [[experiments.014-genes-enriched-at-extreme-tmi.scripts.analyze_extreme_interactions]]
- [x] Created 4 multi-panel figures showing extreme interaction patterns [[experiments.014-genes-enriched-at-extreme-tmi.scripts.visualize_extreme_interactions]]
- [x] Refactored analysis pipeline into modular data processing and visualization steps [[experiments.014-genes-enriched-at-extreme-tmi.scripts.014-genes-enriched-at-extreme-tmi]]
- [x] Enhanced frontmatter script with shebang preservation, clean note naming, and smart test file logic [[notes.assets.scripts.add_frontmatter]]
- [x] Fixed 7 experiment note files to follow H2 header convention with summaries: [[experiments.006-kuzmin-tmi.2025.11.04.storage-calculations]], [[experiments.006-kuzmin-tmi.2025.11.06.dango-vs-lazy-profile-comparison]], [[experiments.006-kuzmin-tmi.2025.11.06.ddp-device-fix]], [[experiments.006-kuzmin-tmi.2025.11.06.gpu-mask-vectorization]], [[experiments.006-kuzmin-tmi.2025.11.06.preprocessing-workflow]], [[experiments.006-kuzmin-tmi.2025.11.06.uint8-preprocessing-solution]], [[experiments.006-kuzmin-tmi.2025.11.06.vectorization-final-fix]]
- [x] Cancelling out graph reg [[Key Question to Answer - Does Graph Reg Help|torchcell.losses.point_dist_graph_reg#key-question-to-answer---does-graph-reg-help]]
- [x] Updated microarray description and txt string - no code update. [[schema|torchcell.datamodels.schema]]

- [ ] Clean up and commit 012

- [ ] Reorganize Spell we don't want any analysis module. Build out into experiment.

- [ ] Add n to fitness interaction data
- [ ] Put Spell in right place
- [ ] Add examples or clarify the schema design to avoid ambiguity in property definitions.

- [ ] Do we have robust way of handling labels? Did we do by some other method?  [[Generate_calmorph_labels|scripts.generate_calmorph_labels]]

- [ ] Build db

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
