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
- [ ] Pre-adapter cleanup + canonical rebuild for the β-carotene (Ozaydin 2013) and betaxanthin (Cachera 2023) datasets before BioCypher adapters: fix stale metabolite-verifier test, add sha256 verification to both loaders, document the intentional Cachera gene-drop (AAD6/CRS5/FLO8), add build-smoke tests, rebuild both stale LMDBs, promote the adopted cassette design in the notes #high [[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]
- [ ] Migrated the two Ohnuki CalMorph morphology loaders (2018 diploid essential-gene heterozygotes; 2022 drug-hypersensitive quadruple deletions) onto the shared layered gene-name resolver `SCerevisiaeGenome.resolve_gene_name` — retain-all + collision-safe, same pattern as ohya/cachera. Counts unchanged (1112 / 1979); 2022 recovered 7 stale ORF names, retained 3 non-gene loci + 1 retired. Both rebuilt + L0-L4 PASS #high [[torchcell.datasets.scerevisiae.ohnuki2018]] [[torchcell.datasets.scerevisiae.ohnuki2022]]
- [ ] Content-addressed interning of constant sub-objects (Environment/reference/publication) into a separate sibling LMDB env (`processed/interned/`) so the ~7.9 KB component-Media isn't denormalized per-record (ballooned dmi_costanzo2016 45→159 GB); centralize read-resolve in ExperimentDataset.get_single_item, wire the 4 SGA loaders, verify on smf_baryshnikova2010, then re-rebuild the SGA set on the lean encoding #high [[plan.experiment-dataset-interning.2026.07.15]]
- [ ] Commit the gilahyper phenotype/gzip script so the phenotype-dataset and integrated-graph tables come from a script instead of a hand transcription; they are the last numbers in the paper without a generating script, and one of them is sourced from a scratch note #high [[paper.information-accounting]]
- [ ] Re-examine the SVR interaction fits at random (d=1000), where a CV s.d. of 0.383 against a mean of 0.458 sits in the same cell that produced a diverged MSE #medium [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [ ] Reconcile the paper-facing classical-ML plot script with the new figure standard, since its PNG output conflicts with the palette SVG route that now feeds the classical-ML figure #medium [[experiments.smf-dmf-tmf-001.traditional_ml-plot_paper]]
