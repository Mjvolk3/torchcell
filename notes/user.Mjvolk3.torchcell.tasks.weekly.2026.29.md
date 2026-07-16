---
id: 34o942tb7way1avtfhxt10m
title: '29'
desc: ''
updated: 1783986627743
created: 1783986627743
---

## 2026.07.13

- [x] Fixed the SI classical-ML MSE table running off the page; the overflow traced to two diverged SVR fits rather than styling, and all three benchmark tables now fit the real journal column width [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [x] Added a Pearson SI table so the correlations quoted in Supplementary Note 6 can be checked against the data [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [x] Measured the persistent-entity corpora (UniProt, NCBI nt/RefSeq, PubChem, wwPDB) straight from the public archives so Fig 1c's bit counts are reproducible rather than asserted, and pinned them to a dated snapshot that can be cited if a referee challenges them [[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]]
- [x] Wrote the information-accounting Supplementary Note behind Fig 1c, which says exactly what the bit counts do and do not mean and keeps compressed storage separate from usable supervision, so the new numbers do not quietly break the identifiability argument [[paper.information-accounting]]
- [x] Corrected the supplementary-citation style across the whole manuscript after finding that "Fig. S1" is a form Nature explicitly rejects; SI floats now number from 1 and are cited only through cross-reference macros, so inserting a Note can no longer silently invalidate the references to it [[paper.proof-writing-standard]]
- [x] Documented the true-size SVG export that stops draw.io silently shrinking every matplotlib panel to 72 percent of its authored width, which had been quietly breaking the panel-width contract every figure depends on [[torchcell.utils.utils]]
- [x] Built the Fig 7 classical-ML dataset panels showing each benchmark's perturbation-order makeup and its 80/10/10 stratified split, so the performance panels beside them are read on equal footing [[experiments.002-dmi-tmi.scripts.dataset_composition_palette]]
- [x] Established repo-wide figure standards - strict panel-width templates, loose height, boxed plots, and one ordered palette - as `torchcell.utils` constants so every plot pulls from a single source [[torchcell.utils]]
- [x] Consolidated the figure colors into one ordered green-free draw.io line-and-fill reference matching Fig 1 and repainted the classical-ML bar and progression panels to line-color faces with solid black hatching and tenth gridlines [[notes.assets.scripts.generate_color_palette]]
- [ ] Re-examine the SVR interaction fits at random (d=1000), where a CV s.d. of 0.383 against a mean of 0.458 sits in the same cell that produced a diverged MSE #medium [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [ ] Commit the gilahyper phenotype/gzip script so the phenotype-dataset and integrated-graph tables come from a script instead of a hand transcription; they are the last numbers in the paper without a generating script, and one of them is sourced from a scratch note #high [[paper.information-accounting]]
- [ ] Reconcile the paper-facing classical-ML plot script with the new figure standard, since its PNG output conflicts with the palette SVG route that now feeds the classical-ML figure #medium [[experiments.smf-dmf-tmf-001.traditional_ml-plot_paper]]

## 2026.07.15

- [ ] Pre-adapter cleanup + canonical rebuild for the β-carotene (Ozaydin 2013) and betaxanthin (Cachera 2023) datasets before BioCypher adapters: fix stale metabolite-verifier test, add sha256 verification to both loaders, document the intentional Cachera gene-drop (AAD6/CRS5/FLO8), add build-smoke tests, rebuild both stale LMDBs, promote the adopted cassette design in the notes #high [[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]
- [ ] Migrated the two Ohnuki CalMorph morphology loaders (2018 diploid essential-gene heterozygotes; 2022 drug-hypersensitive quadruple deletions) onto the shared layered gene-name resolver `SCerevisiaeGenome.resolve_gene_name` — retain-all + collision-safe, same pattern as ohya/cachera. Counts unchanged (1112 / 1979); 2022 recovered 7 stale ORF names, retained 3 non-gene loci + 1 retired. Both rebuilt + L0-L4 PASS #high [[torchcell.datasets.scerevisiae.ohnuki2018]] [[torchcell.datasets.scerevisiae.ohnuki2022]]
- [ ] Content-addressed interning of constant sub-objects (Environment/reference/publication) into a separate sibling LMDB env (`processed/interned/`) so the ~7.9 KB component-Media isn't denormalized per-record (ballooned dmi_costanzo2016 45→159 GB); centralize read-resolve in ExperimentDataset.get_single_item, wire the 4 SGA loaders, verify on smf_baryshnikova2010, then re-rebuild the SGA set on the lean encoding #high [[plan.experiment-dataset-interning.2026.07.15]]
