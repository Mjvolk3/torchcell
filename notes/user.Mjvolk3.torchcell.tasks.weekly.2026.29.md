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

## 2026.07.14

- [x] Locked in the 18-color ordered plotting palette with named tiers and wrote the repo-wide rules so every plot draws from one green-free, warm-primaries-first source [[notes.assets.scripts.generate_color_palette]]
- [x] Introduced a component-based Media schema and corrected the SGA double/triple-mutant selection medium and temperature so growth phenotypes are sourced from the real assay conditions rather than a generic YEPD assumption [[torchcell.datamodels.media-components]]
- [x] Remade the 017 background-mutation figures in the ordered palette and exported them as true-size SVGs so they match the rest of the manuscript [[experiments.017-hoepfner-background-mutations.analysis]]

## 2026.07.15

- [x] Built one shared R64 gene-name reconciler that retains renamed/retired features instead of silently dropping them and migrated the Ohnuki 2018/2022 CalMorph loaders onto it, alongside deprecate-move tooling for safe rebuilds [[torchcell.sequence.genome.scerevisiae.s288c]]
- [x] Provisioned the Ohya 2005 CalMorph morphology data with a sha256-pinned mirror and sourced corrections so the dataset rebuilds deterministically from scratch [[torchcell.datasets.scerevisiae.ohya2005]]
- [x] Rebuilt Hoepfner 2014 restricted to the 150 encodable SMILES compounds and quantified the proprietary/non-encodable remainder, so the chemogenomic dataset carries only structures the model can actually use [[torchcell.datasets.scerevisiae.hoepfner2014]]
- [x] Cleaned up the Ozaydin and Cachera product-proxy screens ahead of adapter ingestion so their records map cleanly onto the schema [[plan.ozaydin-cachera-preadapter-cleanup.2026.07.15]]
- [x] Fixed the Sameith 2015 loader to recompute log2 ratios from the per-array Signal columns with the correct per-array sign instead of the released VALUE column, correcting a sign error in the stored expression changes [[torchcell.datasets.scerevisiae.sameith2015]]
- [x] Reproduced the Kemmeren-Sameith overlap correlation on the sign-fixed data and rebuilt the per-KO expression distributions and correlation histogram as Nature-sized palette panels embedded as true-size SVGs [[experiments.012-sameith-kemmeren.scripts.kemmeren_sameith_overlap_analysis]]
- [x] Made Caudal 2024 emit gene-absence edits for every reference ORF so the natural-isolate genotypes are complete edits to the reference rather than partial ones [[torchcell.datasets.scerevisiae.caudal2024]]
- [x] Corrected the schema so synthetic lethality/rescue and digenic interactions are represented as edges rather than hyperedges, fixing the synth-leth-db genome mapping in the process [[torchcell.datamodels.schema]]
- [x] Stood up a keyed read-only literature endpoint, containerized it with Docker/compose, and backfilled all 41 manifests so the OCR mirror is served with integrity checking [[plan.literature-keyed-endpoint.2026.07.14]]
- [x] Added schema-dependency tracking to flag datasets for rebuild when their schema changes, and registered 15 newly built datasets in the supported-datasets table [[torchcell.provenance.schema-dependency-tracking]]
- [x] Drafted the 018 expression-modeling setup plan for inferring expression on the natural-isolate corpus [[experiments.018-natural-isolate-genomics.expression-modeling-setup]]

## 2026.07.16

- [x] Content-addressed constant sub-objects by interning them into a sibling LMDB env and made the L0-L4 readers interning-aware, shrinking the built datasets while a graveyard retention janitor keeps rebuilds clean [[plan.experiment-dataset-interning.2026.07.15]]
- [x] Added BioCypher cell adapters for the abstract expression/morphology/metabolite datasets, including Zelezniak metabolome+proteome and Mülleder amino-acids, plus a protein-abundance phenotype node [[torchcell.adapters.cell_adapter]]
- [x] Resolved Kemmeren 2014's alias-only KO names through the shared R64 reconciler so its perturbations map to current gene identifiers [[torchcell.datasets.scerevisiae.kemmeren2014]]
- [x] Built the 018 Fig 4 dataset-comparison panels contrasting KO versus natural-isolate genotype and transcriptome, recolored by dataset size with explanatory sub-panels [[experiments.018-natural-isolate-genomics.dataset-comparison]]
- [x] Set up a nightly sync that captures newly added Zotero database-collection papers into the OCR mirror [[torchcell.literature.capture]]
- [x] Published a preview PDF of the supported-datasets table so collaborators can see the current dataset coverage [[paper.supported-datasets-and-databases]]

## 2026.07.17

- [x] Computed the minimum doubles set-cover for the panel-12 triples so only 11 doubles (not all 66) need constructing to build every one of the 52 constructible triples, an 83% build reduction [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover]]
- [x] Built an SMF reference table (Costanzo/Kuzmin fitness ± s.d.) for the droplet validation panel, recovering the SPH1/YLR313C and LCL1/YPL056C fitness values the inference panel never queried [[experiments.010-kuzmin-tmi.scripts.validation_panel_smf_reference]]
- [x] Added a shared-axis ridgeline of the 12 Costanzo SMF Gaussians so each gene's fitness spread is visually comparable across the panel [[experiments.010-kuzmin-tmi.scripts.panel12_smf_gaussian_ridgeline]]
- [x] Built the general KG subset mechanism where a small build is the full build over every dataset with per-dataset caps, parameterized the build config, and wired wandb node/edge monitoring into slurm [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] Containerized the Neo4j/BioCypher build on a py3.13 conda-forge env with in-container env sourcing and deps pulled from torchcell@main, so the KG build runs reproducibly [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] Made the dataset base class skip the PyG download step when a processed LMDB already exists, avoiding needless re-downloads on rebuild [[torchcell.data.experiment_dataset]]
- [x] Added CABBI metabolism adapters for Yoshida 2012, Lopez 2024, and Xue 2025 so those datasets ingest into the knowledge graph [[torchcell.adapters.cell_adapter]]
- [x] Made SourcedValue non-generic so provenance pickles across the KG-build worker queue, and added defensive strain_id and loader teardown to fix the adapter pickle-hang [[torchcell.literature.provenance]]
- [x] Extended the nightly literature sync to also capture the Zotero paper collection [[torchcell.literature.capture]]
- [x] Wrote the Results-6 experimental plan for the regulatory-network double-KO chassis test, including an acetyl-CoA to TAL cross-host production strategy [[paper.results-and-discussion.6.experimental-plan]]
- [x] Polished the 018 main figure with a Sameith split, a size-aligned d2 panel, and clarified overlap/decoupling panels [[experiments.018-natural-isolate-genomics.dataset-comparison]]

## 2026.07.18

- [x] Brought the subset KG build to a queryable database of 77,488 nodes and 179,446 edges by fixing the O(P^2) genotype hashing and compacting Caudal genotypes, and flagged three datasets needing rebuild [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] Added a Neo4j graph-contents table describing what the graph represents versus its source datasets [[paper.supported-datasets-and-databases]]
- [x] Declared protein_abundance_se on the protein-abundance phenotype node so its uncertainty is carried in the schema [[torchcell.datamodels.schema]]
- [x] Added a monitor-tcdb-build skill to introspect, diagnose, and restart tcdb KG builds [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] De-duplicated the two 018 notes and reframed the decoupling framing [[experiments.018-natural-isolate-genomics.dataset-comparison]]

## 2026.07.19

- [x] Retired the Mjvolk3/biocypher fork for stock biocypher 0.15.2 and upgraded Neo4j from 4.4.30 to the 5.26.28 LTS with a pinned build container [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] Added a KG build-time projection utility calibrated on a real build job so build duration can be estimated ahead of time [[plan.kg-database-build-environment-fix.2026.07.18]]
- [x] Added a curated YeastPhenome growth-phenome loader with a ProvenanceGap affordance for missing-provenance fields [[plan.yeastphenome-ingestion-harness.2026.07.15]]
- [x] Fixed the DmfCostanzo2016Dataset default root that pointed at the smf tree instead of the dmf tree [[torchcell.datasets.scerevisiae.costanzo2016]]
