---
id: 8cuoylxkfbn6ccfscelebbw
title: Fitness_browser_putida
desc: ''
updated: 1791803950088
created: 1791803950088
---

## 2026.10.12 - The Fitness Browser's KT2440 experiments, keyed by publication

`FitnessBrowserPutidaDataset` (`torchcell/datasets/pputida/fitness_browser_putida.py`), the P. putida half of the decomposition (#931, issues #932 to #940). One loader for the organism, reading the Fitness Browser deposit through [[torchcell.datasets.fitness_browser]]; the inventory and the decisions are in [[torchcell.candidates.aggregation-decomposition]] (2026.10.12). `CITATION_KEY` is the deposit's mirror key, `priceMutantPhenotypesThousands2018`; each record's `Publication` is its sample's study.

### Studies (`SOURCE_STUDIES`), all `source_kind = primary`

| `pubId` | study | records' identifier | issue | stored samples |
|---|---|---|---|---|
| Schmidt22 | Schmidt 2022 nitrogen metabolism | doi:10.1128/aem.02430-21 (Borchert 2024's `SourceStudy`) | #932 | 128 |
| Thompson20 | Thompson 2020 fatty acid and alcohol metabolism | doi:10.1128/AEM.01665-20 (Borchert 2024's; the browser links the preprint) | #933 | 46 |
| Incha20 | Incha 2020 bisdemethoxycurcumin | doi:10.1016/j.mec.2019.e00119 | #934 | 18 |
| Rand17 | Rand 2017 levulinic acid | doi:10.1038/s41564-017-0028-z (Borchert 2024's) | #935 | 16 |
| Thompson18 | Thompson 2019 lysine metabolism | doi:10.1128/mBio.02577-18 (raw mirror manifest; the browser links the preprint) | #936 | 7 |
| Garber20 | Garber 2020 sigma factor domain swap | doi:10.1101/2020.09.30.321588 | #937 | 6 |
| Price18b | Price 2019 deoxyribose catabolism | doi:10.1128/msystems.00297-18 | #938 | 6 |
| Price21b | Price 2021 GapMind for carbon sources | doi:10.1101/2021.11.02.466981 | #939 | 3 |
| Thompson19 | Thompson 2019 valerolactam | doi:10.1016/j.mec.2019.e00098 (raw mirror manifest; the browser links PubMed 31720214) | #940 | 2 |
| Pearson24 | Pearson 2024 AraC/XylS regulators | PubMed 38113125 (no DOI in the bytes) | none | 2 |
| VillalobosEscobedo23 | Villalobos-Escobedo 2023 Trichoderma exometabolites | doi:10.1371/journal.pgen.1010909 | none | 0 (R2A) |

Not stored: 34 experiments with no `pubId` (sets 8 and 9, the bioreactor series Borchert 2024 also drops, and 12 carbon-source samples it lacks), an open owner decision of #931; `VillalobosEscobedo23`'s 46, every one on `R2A`, which has no MEDIA_LIBRARY entry (the recipe and the quote route are in the decomposition note).

### Records

One `BacterialEnvironmentResponseExperiment` per (locus, stored sample): 234 samples x 4,778 loci = 1,118,052 records. Against Borchert 2024's copy of the same samples this adds 46 loci per sample (4,778 against 4,732), full precision (the compendium rounds to three decimals), the browser's study attribution for 53 samples the compendium leaves to itself, the released moderated t (`environment_response_t_statistic`), and `nGenerations` as `duration_generations` for the 60 stored samples that state it. The environment is built from the same columns exactly as `borchert2024.build_environment` builds it (same media objects, same perturbations; a test pins the equality). Genotype: `TransposonInsertionPerturbation` of the locus tag in the sample's own library (`Putida_ML5_JBEI` 209 samples, `Putida_ML5` 25), transposon `borchert2024.TRANSPOSON`; loci reconciled against the KT2440 GenBank annotation (`reconcile_locus_tags`, at least 0.99 resolved). `screen_id` is `Putida:<expName>`.

`RbTnseqBorchert2024Dataset` stays served unchanged as the L4 sanity check, `compendium_cross_check_borchert2024`: per-sample and pooled Pearson r over shared (locus, sample) values against a bar of 0.9999 (the compendium copies the browser to the half unit, 0.0005).

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from the Fitness Browser deposit (`feba.db`), Putida. Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: each stored study's values are the browser's recomputation under its current statistics version, and the medium, dose and condition are the browser's `Experiment` columns; a study's paper can state replicate structure or conditions the browser does not carry. The studies' own SI tables, where they exist, are the source to collect.
