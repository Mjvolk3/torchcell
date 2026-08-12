---
id: sl8aydabdeg8jbrjljuwng0
title: '30'
desc: ''
updated: 1784566139426
created: 1784566139426
---

## 2026.07.20

- [x] Quantified how much of the model's triple space survives the plate's gene swap: 66 of 122 constructible triples (31 of the top-52) are still buildable from the 10 genes actually constructed [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]]
- [x] Recomputed the doubles set-cover against the strains on hand, cutting the construction plan to 8 doubles that still reach every buildable high-ranking triple [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]]
- [x] Added an SMF baseline figure set (between-source forest + Gaussian ridgeline) for the 10 constructed genes so the assay is read against a like-for-like published reference [[experiments.010-kuzmin-tmi.scripts.constructed_10_smf_figures]]
- [x] Pulled published double-mutant fitness ± s.d. for all 45 pairs among the 10 genes so each constructed double can be judged against the literature at the bench [[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]]
- [x] Selected a 13-double construction+validation set (8 for triple reconstruction, 5 adding DMF/interaction dynamic range) so the echo-plating assay has real fitness range and interaction signal to validate against, and confirmed only 3 significant interactions exist in the whole panel [[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]]
- [x] Typed the SMF and DMF uncertainty per source with derived SE columns (Costanzo SMF is a bootstrap SE, DMF a 4-colony sample SD) so the assay is compared against the right statistic rather than mislabelled noise [[experiments.010-kuzmin-tmi.scripts.validation_panel_smf_reference]] [[experiments.010-kuzmin-tmi.scripts.constructed_10_dmf_reference]]
- [x] Migrated the two Ohnuki CalMorph morphology loaders (2018 diploid essential-gene heterozygotes; 2022 drug-hypersensitive quadruple deletions) onto the shared layered gene-name resolver `SCerevisiaeGenome.resolve_gene_name` — retain-all + collision-safe, same pattern as ohya/cachera. Counts unchanged (1112 / 1979); 2022 recovered 7 stale ORF names, retained 3 non-gene loci + 1 retired. Both rebuilt + L0-L4 PASS #high [[torchcell.datasets.scerevisiae.ohnuki2018]] [[torchcell.datasets.scerevisiae.ohnuki2022]]
- [x] Content-addressed interning of constant sub-objects (Environment/reference/publication) into a separate sibling LMDB env (`processed/interned/`) so the ~7.9 KB component-Media isn't denormalized per-record (ballooned dmi_costanzo2016 45→159 GB); centralize read-resolve in ExperimentDataset.get_single_item, wire the 4 SGA loaders, verify on smf_baryshnikova2010, then re-rebuild the SGA set on the lean encoding #high [[plan.experiment-dataset-interning.2026.07.15]]

## 2026.07.21

- [x] Overhauled the SGA colony segmenter to a grid-constrained gitter-style method (6-sided gel-boundary gate + one shared accept predicate) that removes the phantom-boundary artefacts [[torchcell.sga.image]]
- [x] Re-analysed the run-2 volume/timepoint sweep on full-resolution 72 h captures and added Spearman ordering to the reference scatter [[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]]
- [x] Green-lit Cellpose for colony segmentation after a clean zero-shot web-demo result, and drafted the workstation/rsync + integration plan [[experiments.W019-echo-crispr-array.cellpose-segmentation-plan]]
- [x] Moved segmentation to Cellpose on gila: rsynced the run-2 images, wired a `seg_method='cellpose'` branch, and validated against the classical numbers #high [[experiments.W019-echo-crispr-array.cellpose-segmentation-plan]]
