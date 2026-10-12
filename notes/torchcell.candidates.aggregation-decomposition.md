---
id: xj73lqw46z3cujnq49a6ibg
title: Aggregation Decomposition
desc: ''
updated: 1791694246053
created: 1791694246053
---

## 2026.10.11 - The decomposition policy and the four served compendia

Policy (owner, 2026-10-10; plan decision 16, revising decision 15 of [[plan.dataset-admission-pipeline.2026.10.10]]): a served compendium is a processing reference and a sanity check, not the source of record. Each original study inside a served compendium is collected and loaded from its own per-record release when it has one, attributed to that study, and cross-checked against the compendium's copy of the same samples. The compendium stays served only for samples whose only artifact is the compendium itself. The aim is more detail than the compendium kept: conditions, replicate structure, eliminated loci, per-study methods.

Code: `AggregationRecord.sources` in `torchcell/candidates/verdict.py` ([[torchcell.candidates.verdict]]), one `SourceDecomposition` per (source study, served class), folded into a verdict by `python -m torchcell.candidates gate --decomposition <audit.json>` ([[torchcell.candidates.cli]]). Measurement: `experiments/036-dataset-fixes-before-kg-build/scripts/aggregation_decomposition_audit.py` ([[experiments.036-dataset-fixes-before-kg-build.scripts.aggregation_decomposition_audit]]), results `experiments/036-dataset-fixes-before-kg-build/results/aggregation_decomposition_audit.json`, probes `aggregation_decomposition_probes.json`, tables `aggregation_decomposition_tables.md`. Verdicts: `database/candidates/{priceMutantPhenotypesThousands2018,borchertMachineLearningAnalysis2024,limMachinelearningPseudomonasPutida2022,lamoureuxMultiscaleExpressionRegulation2023}.json`, all four `admissible`; the ledger's 2026.10.11 section renders them ([[torchcell.candidates.ledger]]).

### Vocabulary

| term | meaning |
|---|---|
| compendium-only | no artifact other than the compendium holds the samples (`own_release_kind = none`) |
| compendium paper | the compendium's own paper, for samples it first described; never a separate study to load |
| loadable | an own release exists, it is not the compendium's, and no registered dataset serves it yet |
| per-record values | the study's own release carries the per-gene values of these samples |
| raw reads only | the own release is sequencing reads; values must be recomputed with the compendium's processing as the reference |
| unattributed release | samples another artifact releases with no study named for them (a Fitness Browser experiment with an empty `pubId`, a BioProject with no publication) |

### How it was measured

Read-only: the raw and literature mirrors, the dev stores' `preprocess/` ledgers (no LMDB was opened; slurm array 3733 was rebuilding `rbtnseq_borchert2024`, so Borchert 2024 was read off its raw release with the loader's own `attribute_sample` and `drop_reason`), and a snapshot of the Fitness Browser database.

The Fitness Browser's HTML pages answer scripts with HTTP 403 (`org.cgi?orgId=Putida`, probed 2026-10-11), which is what the Schmidt 2022, Thompson 2020, Thompson 2019 lysine and Rand 2017 subsumption records measured. Its static data files are not walled: `https://fit.genomics.lbl.gov/cgi_data/feba.db` (the whole browser as SQLite) and `cgi_data/db.StrainFitness.Putida` answer 200. The snapshot used here:

| field | value |
|---|---|
| URL | `https://fit.genomics.lbl.gov/cgi_data/feba.db` |
| bytes | 10,205,732,864 |
| sha256 | `250d4a03ffc41ea918027ed7adcb2fa9ed3ddc127ad3de29b40602fc6264b4ef` |
| Last-Modified / ETag | Fri, 09 Oct 2026 18:08:11 GMT / `"2604f2000-65d6c3b3c80ac"` |
| retrieved | 2026-10-11T04:29:05Z, `curl -s -D headers.txt -A "torchcell-audit/1.0" -o feba.db <URL>` |
| mirrored | no (10.2 GB; owner decision below) |

The browser records a publication per experiment (`Experiment.pubId`, `Publication` table), an attribution independent of any compendium. GEO supplementary listings, ENA read-run reports and PubMed DOIs were probed (250 probes; 16 ENA reports answered 500 and were re-asked with `--reprobe-failed`, all then 200). A GEO series counts as releasing per-record values when its supplementary listing (or its `RAW.tar` file list) names a file that is not a browser track, image or sequence; the classification is by file name, the files were not opened.

### Counts per compendium

| compendium (served class) | sources | original studies | unattributed releases | compendium paper | loadable (scriptable / walled) | per-record values / reads only | compendium-only | loaded as own dataset | samples (served) | samples loadable | samples compendium-only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Price 2018 Keio (`RbTnseqPrice2018EcoliDataset`) | 2 | 1 | 0 | 1 | 1 (1 / 0) | 1 / 0 | 1 | 0 | 162 (147) | 92 | 70 |
| Borchert 2024 (`RbTnseqBorchert2024Dataset`) | 12 | 10 | 1 | 1 | 10 (10 / 0) | 10 / 0 | 1 | 1 | 332 (290) | 254 | 36 |
| Lim 2022 (`PutidaPrecise321Lim2022Dataset`) | 22 | 12 | 9 | 1 | 21 (21 / 0) | 9 / 12 | 1 | 0 | 321 (180) | 225 | 96 |
| PRECISE-1K (`RnaseqLamoureux2023Dataset`) | 26 | 25 | 0 | 1 | 24 (24 / 0) | 21 / 3 | 2 | 0 | 1,035 (241) | 551 | 484 |
| Public K-12 (`RnaseqPublicK12Lamoureux2023Dataset`) | 90 | 38 | 52 | 0 | 90 (90 / 0) | 29 / 61 | 0 | 0 | 1,675 (240) | 1,675 | 0 |

Every own release found is scriptable; none is walled. Loaded as its own dataset: one study, Borchert 2023 (`RbTnseqBorchert2023Dataset`). Wetmore 2015 is being loaded from its own release on branch `feat/ecoli-wetmore2015-own-release` (not on main when measured), so it counts as loadable, not loaded. `GeneEssentialityPrice2018EcoliDataset` (Price 2018's own call) and `GrowthRateLamoureux2023Dataset` (PRECISE-1K metadata cells, whose source papers' own growth rates were not read) are not decomposed.

### Per-compendium tables

Generated by `aggregation_decomposition_audit.py` (copied from `results/aggregation_decomposition_tables.md`; regenerate with `--tables-only`, never edit by hand). Samples are compendium samples, served in parentheses. "mirrored" names the mirror key.

#### priceMutantPhenotypesThousands2018

Counts: studies 2, loadable_scriptable 1, loadable_walled 0, loadable_per_record_values 1, loadable_raw_reads_only 0, compendium_only 1, loaded_as_own_dataset 0, mirrored 2, samples 162, samples_served 147, samples_loadable_from_own_release 92, samples_compendium_only 70.

`RbTnseqPrice2018EcoliDataset` (2 sources)

| source | role | samples (served) | own release | access | mirrored | loaded | L4 cross-check | compendium's attribution |
|---|---|---|---|---|---|---|---|---|
| doi:10.1128/mbio.00306-15 | original_study | 92 (84) | per_record_values | scriptable | wetmoreRapidQuantificationMutant2015 | no | min_pearson_r 0.989 (pin; 91 samples) | wetmoreRapidQuantificationMutant2015 92 (browser: Wetmore15 87, not-in-browser 4, Price18 1) |
| doi:10.1038/s41586-018-0124-0 | compendium_paper | 70 (63) | none | - | priceMutantPhenotypesThousands2018 | no | not measured | priceMutantPhenotypesThousands2018 70 (browser: Price18 70) |

Dropped or recomputed relative to the originals:

- statistics recomputed: the compendium re-ran the RB-TnSeq code (1.0.3) over Wetmore 2015's samples scored at 1.0.0; per-sample Pearson r min 0.9890, median 0.9971, largest single-gene difference 2.573 over 91 shared samples
- genes: Wetmore 2015 values 3646 genes, the compendium 3789; every Wetmore gene is in the compendium
- samples Wetmore scored successful and the compendium dropped: ['set1IT029']; failed there, carried here: ['set2IT045']
- withdrawn by the authors' 2021 note and still in the compendium release: ['set1IT007', 'set1IT008', 'set1IT043', 'set1IT044']; the browser no longer carries them (4 of 4 absent)
- the browser's values equal the compendium release: max abs diff 8.88e-16 over 158 samples x 3,789 genes
- the browser holds 25 more Keio experiments than Table S5 ({'Price21b': 6, 'no-pubId': 19}); they are not in the served compendium
- loader drops by source: kept {'wetmoreRapidQuantificationMutant2015': 84, 'priceMutantPhenotypesThousands2018': 63} of {'wetmoreRapidQuantificationMutant2015': 92, 'priceMutantPhenotypesThousands2018': 70}

Not decomposed:

- GeneEssentialityPrice2018EcoliDataset: Price 2018's own call over the insertion maps, not an aggregation of other studies' values

#### borchertMachineLearningAnalysis2024

Counts: studies 12, loadable_scriptable 10, loadable_walled 0, loadable_per_record_values 10, loadable_raw_reads_only 0, compendium_only 1, loaded_as_own_dataset 1, mirrored 7, samples 332, samples_served 290, samples_loadable_from_own_release 254, samples_compendium_only 36.

`RbTnseqBorchert2024Dataset` (12 sources)

| source | role | samples (served) | own release | access | mirrored | loaded | L4 cross-check | compendium's attribution |
|---|---|---|---|---|---|---|---|---|
| doi:10.1128/aem.02430-21 | original_study | 128 (128) | per_record_values | scriptable | schmidtNitrogenMetabolismPseudomonas2022 | no | max_abs_diff 0.0005 vs 0.0005 (pass; 128 samples) | borchert2024 5; schmidt2022 123 (browser: Schmidt22 128) |
| doi:10.1128/aem.01665-20 | original_study | 46 (46) | per_record_values | scriptable | thompsonFattyAcidAlcohol2020 | no | max_abs_diff 0.0005 vs 0.0005 (pass; 46 samples) | thompson2020 46 (browser: Thompson20 46) |
| doi:10.1016/j.ymben.2023.04.007 | original_study | 42 (42) | per_record_values | scriptable | borchertRBTnSeqIdentifiesGenetic2023 | RbTnseqBorchert2023Dataset | max_abs_diff 0.0005 vs 0.0005 (pass; 42 samples) | borchert2023 42 (browser: not-in-browser 42) |
| doi:10.1128/msystems.00942-23 | compendium_paper | 36 (36) | none | - | borchertMachineLearningAnalysis2024 | no | not measured | borchert2024 36 (browser: not-in-browser 36) |
| fitness-browser:no-pubId | unattributed_release | 22 (0) | per_record_values | scriptable | no | no | max_abs_diff 0.0005 vs 0.0005 (pass; 22 samples) | borchert2024 22 (browser: no-pubId 22) |
| doi:10.1016/j.mec.2019.e00119 | original_study | 18 (18) | per_record_values | scriptable | no | no | max_abs_diff 0.0005 vs 0.0005 (pass; 18 samples) | borchert2024 18 (browser: Incha20 18) |
| doi:10.1038/s41564-017-0028-z | original_study | 16 (0) | per_record_values | scriptable | randMetabolicPathwayCatabolizing2017 | no | max_abs_diff 0.0005 vs 0.0005 (pass; 16 samples) | rand2017 10; borchert2024 6 (browser: Rand17 16) |
| doi:10.1128/mbio.02577-18 | original_study | 7 (7) | per_record_values | scriptable | thompsonMassivelyParallelFitness2019 | no | max_abs_diff 0.0005 vs 0.0005 (pass; 7 samples) | borchert2024 7 (browser: Thompson18 7) |
| doi:10.1101/2020.09.30.321588 | original_study | 6 (6) | per_record_values | scriptable | no | no | max_abs_diff 0.0005 vs 0.0005 (pass; 6 samples) | borchert2024 6 (browser: Garber20 6) |
| doi:10.1128/msystems.00297-18 | original_study | 6 (2) | per_record_values | scriptable | no | no | max_abs_diff 0.0005 vs 0.0005 (pass; 6 samples) | borchert2024 6 (browser: Price18b 6) |
| doi:10.1101/2021.11.02.466981 | original_study | 3 (3) | per_record_values | scriptable | no | no | max_abs_diff 0.0005 vs 0.0005 (pass; 3 samples) | borchert2024 3 (browser: Price21b 3) |
| doi:10.1016/j.mec.2019.e00098 | original_study | 2 (2) | per_record_values | scriptable | thompsonOmicsdrivenIdentificationElimination2019 | no | max_abs_diff 0.0005 vs 0.0005 (pass; 2 samples) | borchert2024 2 (browser: Thompson19 2) |

Dropped or recomputed relative to the originals:

- loci eliminated: the compendium keeps 4732 genes; the browser values [4778] genes for each of the 254 browser-sourced samples, so 46 loci per sample are recoverable from the browser alone (Borchert 2023's SI recovers 271 for its 42 samples)
- values rounded: every browser-sourced sample equals the browser to the three-decimal half unit (0.0005)
- attribution: the compendium names 5 studies; the browser's pubId names 10 for its samples, including Incha 2020, Thompson 2019 (lysine and valerolactam), Garber 2020, Price 2018b and Price 2021b, which the compendium leaves compendium-attributed
- browser metadata the compendium release does not carry (samples with a value): {'vessel': 254, 'shaking': 239, 'pH': 35, 'nGenerations': 60, 'dateStarted': 254, 'timeZeroSet': 254, 'growthPlate': 0, 'gMed': 254, 'cor12': 254, 'mad12': 254, 'adjcor': 254, 'gccor': 254}
- loader drops (served store): {'medium_not_in_media_library': 20, 'kept': 290, 'reactor_process_not_representable': 22}
- the browser holds 60 Putida experiments not in the compendium ({'Pearson24': 2, 'no-pubId': 12, 'VillalobosEscobedo23': 46})
- the browser lists the RCH2_defined_noCarbon recipe the loader lacks (28 MediaComponents rows; issue #860)
- per-strain fitness (barcode, insertion position) is served at https://fit.genomics.lbl.gov/cgi_data/db.StrainFitness.Putida (HTTP 200, 262636894 bytes), while the browser's HTML pages answer 403

#### limMachinelearningPseudomonasPutida2022

Counts: studies 22, loadable_scriptable 21, loadable_walled 0, loadable_per_record_values 9, loadable_raw_reads_only 12, compendium_only 1, loaded_as_own_dataset 0, mirrored 1, samples 321, samples_served 180, samples_loadable_from_own_release 225, samples_compendium_only 96.

`PutidaPrecise321Lim2022Dataset` (22 sources)

| source | role | samples (served) | own release | access | mirrored | loaded | L4 cross-check | compendium's attribution |
|---|---|---|---|---|---|---|---|---|
| doi:10.1016/j.ymben.2022.04.004 | compendium_paper | 96 (31) | none | - | limMachinelearningPseudomonasPutida2022 | no | not measured | lim2022 96 |
| doi:10.1099/mic.0.000875 | original_study | 36 (36) | per_record_values | scriptable | no | no | not measured | doi:10.1099/mic.0.000875 36 |
| doi:10.1128/AEM.03236-16 | original_study | 21 (21) | per_record_values | scriptable | no | no | not measured | doi:10.1128/AEM.03236-16 21 |
| doi:10.1038/s41598-018-23858-6 | original_study | 18 (0) | raw_reads_only | scriptable | no | no | not measured | doi:10.1038/s41598-018-23858-6 18 |
| doi:10.3389/fmicb.2020.01099 | original_study | 18 (2) | raw_reads_only | scriptable | no | no | not measured | doi:10.3389/fmicb.2020.01099 18 |
| doi:10.1039/D0GC01663B | original_study | 16 (10) | per_record_values | scriptable | no | no | not measured | doi:10.1039/D0GC01663B 16 |
| doi:10.1021/acssuschemeng.1c03765 | original_study | 14 (0) | per_record_values | scriptable | no | no | not measured | doi:10.1021/acssuschemeng.1c03765 14 |
| pmid:30298597 | original_study | 11 (11) | per_record_values | scriptable | no | no | not measured | pmid:30298597 11 |
| doi:10.1016/j.ymben.2020.01.001 | original_study | 9 (3) | per_record_values | scriptable | no | no | not measured | doi:10.1016/j.ymben.2020.01.001 9 |
| project:FleQ (PRJNA535401) | unattributed_release | 9 (6) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (FleQ) 9 |
| project:RelA (PRJNA455888,PRJNA455889...) | unattributed_release | 9 (9) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (RelA) 9 |
| project:Phase (PRJNA338850,PRJNA338876) | unattributed_release | 8 (8) | per_record_values | scriptable | no | no | not measured | no publication in the sheet (Phase) 8 |
| project:ZnO (PRJEB9788) | unattributed_release | 8 (8) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (ZnO) 8 |
| doi:10.1128/AEM.01210-18 | original_study | 6 (3) | raw_reads_only | scriptable | no | no | not measured | doi:10.1128/AEM.01210-18 6 |
| doi:10.3389/fmicb.2020.573857 | original_study | 6 (6) | raw_reads_only | scriptable | no | no | not measured | doi:10.3389/fmicb.2020.573857 6 |
| pmid:25711694 | original_study | 6 (2) | per_record_values | scriptable | no | no | not measured | pmid:25711694 6 |
| project:Carbon2 (PRJNA457015,PRJNA457016...) | unattributed_release | 6 (6) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (Carbon2) 6 |
| project:Fuel (PRJNA455030,PRJNA455031...) | unattributed_release | 6 (6) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (Fuel) 6 |
| project:Mobile_gene2 (PRJDB5812) | unattributed_release | 6 (0) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (Mobile_gene2) 6 |
| project:Myristic_acid (PRJNA480068) | unattributed_release | 6 (6) | per_record_values | scriptable | no | no | not measured | no publication in the sheet (Myristic_acid) 6 |
| doi:10.1039/C8EE00460A | original_study | 4 (4) | raw_reads_only | scriptable | no | no | not measured | doi:10.1039/C8EE00460A 4 |
| project:Bioreactor1 (PRJNA520374,PRJNA520375) | unattributed_release | 2 (2) | raw_reads_only | scriptable | no | no | not measured | no publication in the sheet (Bioreactor1) 2 |

Dropped or recomputed relative to the originals:

- values recomputed: the compendium re-aligned every library to AE015451.2 and re-quantified log2(TPM + 1) (lim2022.REFERENCE_GENOME, EXPRESSION_UNIT); no source's own processed values are carried
- conditions dropped: the sample sheet carries no medium or temperature column for any sample, so every environment but the aromatic project's is a composition-deferred placeholder (lim2022 ENVIRONMENT); the sources' own GEO sample records are where those conditions are stated
- replicate structure: Lim 2022 discarded libraries with replicate R^2 < 0.95 (lim2022.REPLICATE_FILTER) and kept 321 of 541 collected
- loader drops (served store): {'engineered_strain_code': 71, 'non_reference_background': 24, 'kept': 180, 'engineered_evolved_sugar_strain': 14, 'sequence_variant_allele': 2, 'evolved_isolate': 4, 'deleted_gene_unresolved': 11, 'plasmid_content': 15}
- 16 ionic-liquid ALE samples pair with in-house SBRG count columns, not their own SRX columns, which do not reproduce the published values (lim2022 pair_count_columns)

#### lamoureuxMultiscaleExpressionRegulation2023

Counts: studies 116, loadable_scriptable 114, loadable_walled 0, loadable_per_record_values 50, loadable_raw_reads_only 64, compendium_only 2, loaded_as_own_dataset 0, mirrored 1, samples 2710, samples_served 481, samples_loadable_from_own_release 2226, samples_compendium_only 484.

`RnaseqLamoureux2023Dataset` (26 sources)

| source | role | samples (served) | own release | access | mirrored | loaded | L4 cross-check | compendium's attribution |
|---|---|---|---|---|---|---|---|---|
| doi:10.1093/nar/gkad750 | compendium_paper | 470 (74) | none | - | lamoureuxMultiscaleExpressionRegulation2023 | no | not measured | empty-DOI 470 |
| doi:10.1101/749887 | original_study | 101 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1101/749887 101 |
| doi:10.1016/j.ymben.2020.07.002 | original_study | 94 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1016/j.ymben.2020.07.002 94 |
| doi:10.1038/s41467-019-13483-w | original_study | 43 (33) | per_record_values | scriptable | no | no | not measured | doi.org/10.1038/s41467-019-13483-w 43 |
| doi:10.1128/msystems.00980-20 | original_study | 32 (20) | per_record_values | scriptable | no | no | not measured | doi.org/10.1128/mSystems.00980-20 32 |
| doi:10.1128/msphere.00443-21 | original_study | 30 (10) | per_record_values | scriptable | no | no | not measured | doi.org/10.1128/mSphere.00443-21 30 |
| doi:10.1093/nar/gky752 | original_study | 28 (28) | per_record_values | scriptable | no | no | not measured | doi.org/10.1093/nar/gky752 28 |
| doi:10.1093/molbev/msz251 | original_study | 24 (4) | per_record_values | scriptable | no | no | not measured | doi.org/10.1093/molbev/msz251 24 |
| doi:10.1101/310946 | original_study | 22 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1101/310946 22 |
| doi:10.1371/journal.pone.0197272 | original_study | 20 (12) | raw_reads_only | scriptable | no | no | not measured | doi.org/10.1371/journal.pone.0197272 20 |
| doi:10.1128/aem.00410-17 | original_study | 18 (4) | per_record_values | scriptable | no | no | not measured | doi.org/10.1128/AEM.00410-17 18 |
| doi:10.1038/s41564-018-0340-2 | original_study | 16 (4) | per_record_values | scriptable | no | no | not measured | doi.org/10.1038/s41564-018-0340-2 16 |
| doi:10.1186/s12918-018-0653-z | original_study | 14 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1186/s12918-018-0653-z 14 |
| url:escholarship.org/uc/item/0k4709x6 | original_study | 14 (0) | none | - | no | no | not measured | escholarship.org/uc/item/0k4709x6 14 |
| doi:10.1016/j.cels.2016.04.003 | original_study | 13 (2) | per_record_values | scriptable | no | no | not measured | doi.org/10.1016/j.cels.2016.04.003 13 |
| doi:10.1038/ncomms13091 | original_study | 12 (4) | per_record_values | scriptable | no | no | not measured | doi.org/10.1038/ncomms13091 12 |
| doi:10.1093/nar/gky069 | original_study | 12 (12) | raw_reads_only | scriptable | no | no | not measured | doi.org/10.1093/nar/gky069 12 |
| doi:10.1073/pnas.1909987116 | original_study | 10 (2) | per_record_values | scriptable | no | no | not measured | doi.org/10.1073/pnas.1909987116 10 |
| doi:10.1093/molbev/msu209 | original_study | 10 (1) | per_record_values | scriptable | no | no | not measured | doi.org/10.1093/molbev/msu209 10 |
| doi:10.15252/msb.20145243 | original_study | 10 (6) | per_record_values | scriptable | no | no | not measured | doi.org/10.15252/msb.20145243 10 |
| doi:10.1016/j.celrep.2015.07.043 | original_study | 8 (8) | per_record_values | scriptable | no | no | not measured | doi.org/10.1016/j.celrep.2015.07.043 8 |
| doi:10.1016/j.celrep.2021.108961 | original_study | 8 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1016/j.celrep.2021.108961 8 |
| doi:10.1038/ncomms5910 | original_study | 8 (8) | per_record_values | scriptable | no | no | not measured | doi.org/10.1038/ncomms5910 8 |
| doi:10.1038/ncomms8970 | original_study | 7 (7) | raw_reads_only | scriptable | no | no | not measured | doi.org/10.1038/ncomms8970 7 |
| doi:10.1128/aem.02246-14 | original_study | 7 (0) | per_record_values | scriptable | no | no | not measured | doi.org/10.1128/AEM.02246-14 7 |
| doi:10.1038/s41598-017-02110-7 | original_study | 4 (2) | per_record_values | scriptable | no | no | not measured | doi.org/10.1038/s41598-017-02110-7 4 |

`RnaseqPublicK12Lamoureux2023Dataset` (90 sources)

| source | role | samples (served) | own release | access | mirrored | loaded | L4 cross-check | compendium's attribution |
|---|---|---|---|---|---|---|---|---|
| pmid:32385082 | original_study | 45 (45) | per_record_values | scriptable | no | no | not measured | pmid:32385082 45 |
| pmid:32518228 | original_study | 35 (23) | per_record_values | scriptable | no | no | not measured | pmid:32518228 35 |
| pmid:23716638 | original_study | 30 (12) | per_record_values | scriptable | no | no | not measured | pmid:23716638 30 |
| pmid:29686109 | original_study | 30 (30) | per_record_values | scriptable | no | no | not measured | pmid:29686109 30 |
| bioproject:PRJEB36111 | unattributed_release | 28 (5) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJEB36111 28 |
| pmid:29046437 | original_study | 18 (2) | per_record_values | scriptable | no | no | not measured | pmid:29046437 18 |
| pmid:29150605 | original_study | 18 (18) | per_record_values | scriptable | no | no | not measured | pmid:29150605 18 |
| pmid:28448512 | original_study | 16 (8) | raw_reads_only | scriptable | no | no | not measured | pmid:28448512 16 |
| bioproject:PRJEB46447 | unattributed_release | 15 (12) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJEB46447 15 |
| bioproject:PRJNA360103 | unattributed_release | 14 (14) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJNA360103 14 |
| bioproject:PRJNA716815 | unattributed_release | 14 (14) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJNA716815 14 |
| bioproject:PRJNA422780 | unattributed_release | 13 (13) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJNA422780 13 |
| bioproject:PRJNA559809 | unattributed_release | 10 (6) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJNA559809 10 |
| pmid:27713404 | original_study | 9 (9) | per_record_values | scriptable | no | no | not measured | pmid:27713404 9 |
| bioproject:PRJEB23326 | unattributed_release | 8 (4) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJEB23326 8 |
| pmid:25275371 | original_study | 8 (8) | raw_reads_only | scriptable | no | no | not measured | pmid:25275371 8 |
| pmid:29902774 | original_study | 8 (3) | per_record_values | scriptable | no | no | not measured | pmid:29902774 8 |
| bioproject:PRJEB14391 | unattributed_release | 6 (6) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJEB14391 6 |
| bioproject:PRJNA448156 | unattributed_release | 6 (6) | raw_reads_only | scriptable | no | no | not measured | bioproject:PRJNA448156 6 |
| pmid:31270363 | original_study | 2 (2) | per_record_values | scriptable | no | no | not measured | pmid:31270363 2 |
| 70 more, none served | original_study per_record_values: 15, original_study raw_reads_only: 12, unattributed_release per_record_values: 5, unattributed_release raw_reads_only: 38 | 1342 (0) | | | | | | |

Dropped or recomputed relative to the originals:

- attribution not stored: both loaders write Lamoureux 2023 as every record's Publication (lamoureux2023.py and lamoureux2023_public_k12.py build Publication(doi=PAPER_DOI)); the per-sample DOI, PMID, GEO and BioProject cells exist only in the release metadata
- PRECISE-1K: 565 of 1035 samples name an earlier paper in the DOI column; the rest are first released in PRECISE-1K
- accessions missing: of the 470 first-released samples, 430 carry no GEO or SRX accession and are named only by an s3://precise-db path in R1, although the Data Availability says the accession numbers are listed in the metadata file (lamoureux2023.DATA_AVAILABILITY's paragraph)
- values recomputed: every library re-aligned to NC_000913.3 and re-quantified (lamoureux2023.ALIGNMENT_REFERENCE, UNIT); the Public K-12 log2[TPM] is not released at all, so its served TPM is computed from released counts (lamoureux2023_public_k12 docstring)
- replicate structure: PRECISE-1K and Public K-12 keep only replicates with Pearson r >= 0.95 (lamoureux2023.REPLICATE_QC); conditions with one library were dropped from Public K-12 (lamoureux2023_public_k12.REPLICATE_RULE)
- served: PRECISE-1K keeps 241 of 1035 (dropped_records 241 kept); Public K-12 keeps 240 of 1675 (dropped_records 240 kept)

Not decomposed:

- GrowthRateLamoureux2023Dataset: the 89 served rates are PRECISE-1K metadata cells; whether each source paper reports its own growth rate needs the papers read, so its own-release status is not measured here

### What the measurement changes in earlier records

- **The Fitness Browser is a scriptable per-record release.** The Schmidt 2022 and Thompson 2020 subsumption records (`$DATA_ROOT/torchcell-raw/<key>/subsumption_record.json`) concluded "no bytes here to load" because the browser's HTML answered 403. `cgi_data/feba.db` answers 200, and its values equal the Borchert 2024 copy to the three-decimal half unit for all 254 browser-sourced samples (max abs diff 0.0005 over 4,732 loci each). Those rows are loadable from their own release.
- **The browser attributes 53 of the 111 samples Borchert 2024 leaves compendium-attributed.** By `pubId`: Incha 2020 18, Thompson 2019 lysine (`Thompson18`) 7, Rand 2017 6 more (set1 glucose 20 mM and acetate 5 mM, set5 levulinic acid 20 mM), Garber 2020 6, Price 2019 deoxyribose (`Price18b`) 6, Schmidt 2022 5 more (set10 Arg, His, glucose with the 20-amino-acid mix; set12 lysine twice), Price 2021 GapMind (`Price21b`) 3, Thompson 2019 valerolactam 2. Of the rest, 22 (sets 8 and 9, the bioreactor series) carry no `pubId` and 36 (Beckham sets 100 and 101 outside Borchert 2023) are not in the browser. The compendium's `HYPOTHESES` for set6, set7 and set12 are measured by this column, not by the compendium.
- **Thompson 2019 lysine is 7 samples, not 4.** The value match of #908 identified four; the browser names seven (set6IT057, set7IT042, set7IT044, set7IT055, set7IT056, set7IT057, set7IT062). #908's owner decision is unchanged in kind and larger in scope.
- **Rand 2017 is 16 samples, not 10.** All 16 are on `RCH2_defined_noCarbon` and served 0 until #860. The browser's `MediaComponents` table lists that medium (28 rows, both Wolfe's mixes expanded), a second source beside Price 2018 Table S18.
- **Price 2018's Keio release is the browser's.** The compendium's `fit_logratios_good.tab` equals the browser to 8.9e-16 over 158 samples x 3,789 genes. The four samples the authors withdrew in 2021 are still in the compendium release and absent from the browser. Wetmore 2015's own 1.0.0 statistics differ: per-sample Pearson r min 0.9890, median 0.9971, largest single-gene difference 2.573, over 91 shared samples and 3,646 genes.
- **The Lamoureux 2023 stores name no source study.** Both loaders write Lamoureux 2023 as every record's `Publication`; the release's DOI, PMID, GEO and BioProject cells are not carried. Of the 470 PRECISE-1K samples first released there, 430 carry no GEO or SRX accession and are named only by an `s3://precise-db` path, although the Data Availability says the accessions are listed in the metadata file.
- **Per-strain fitness is released.** `cgi_data/db.StrainFitness.Putida` answers 200 (262,636,894 bytes). Hypothesis (untested, the file was not opened): it carries barcode and insertion position per strain, which would fill the `TransposonInsertionPerturbation` slots the Rand 2017 record says no release fills.

### Loading order recommended

1. **The nine browser-attributed studies of Borchert 2024,** largest first: Schmidt 2022 (128 samples), Thompson 2020 (46), Incha 2020 (18), Rand 2017 (16, after #860), Thompson 2019 lysine (7, after #908), Garber 2020 (6), Price 2019 (6), Price 2021 (3), Thompson 2019 valerolactam (2). One scriptable retrieval covers all of them, the L4 cross-check already passes for every sample, and each record gains what the compendium dropped: 46 loci per sample (4,778 in the browser against 4,732), the browser's per-experiment metadata (vessel 254 samples, shaking 239, pH 35, generations 60, start date and time-zero set 254, quality metrics 254), and a study attribution for 53 samples the compendium leaves to itself.
2. **Wetmore 2015,** in flight on its own branch: 1.0.0 statistics, the fumarate success (set1IT029) the compendium drops, Data Set S1's generations and end OD.
3. **Lim 2022's sources,** the nine with processed GEO tables first (7 original studies, 2 BioProject-only projects), then the twelve with reads only. The compendium's sample sheet carries no medium or temperature for any sample; the sources' GEO sample records are where those are stated, so environment is the main gain. 141 of 321 samples are dropped by the served loader today (engineered strain codes 71, non-reference background 24, plasmid content 15, evolved sugar strains 14, unresolved deletions 11, evolved isolates 4, an allele 2); whether an own release resolves any of those reasons is untested.
4. **PRECISE-1K's 24 earlier papers.** Same lab and pipeline, so the values gain least; the gain is attribution (no served record names its study today) and, for evolved isolates, the hypothesis (untested) that a paper's own SI lists the mutations the PRECISE-1K release omits, which would let some of the 794 dropped samples be written.
5. **Public K-12,** 90 sources of which 20 contribute a served sample: an owner decision on scope (below).

### Cross-check design (L4, per sample)

The cross-check compares the original's values with the compendium's copy of the SAME sample, matched by a shared identifier (browser `expName` = compendium sample name; GEO GSM or SRX = compendium SRX) or, where none is shared, by content with a nearest-wrong-sample margin (Borchert 2023, Thompson 2019 lysine, Rand 2017 precedents). The statistic and tolerance follow each compendium's stated reprocessing:

| compendium | reprocessing it states | statistic | tolerance | measured |
|---|---|---|---|---|
| Borchert 2024 | copies browser and Borchert 2023 values, three decimals | max abs diff per sample over shared loci | 0.0005 (half unit) | pass, 254 browser samples and 42 Borchert 2023 arms |
| Price 2018 | re-ran the statistics (1.0.3, 1.1.0 or 1.1.1) over Wetmore 2015's 1.0.0 | per-sample Pearson r, identity by sample name and metadata | none stated: the measured minimum 0.9890 is a pin, a bound is an owner decision | Wetmore 91 samples |
| Lim 2022 | re-aligned to AE015451.2, featureCounts, log2(TPM + 1); refused replicates with R^2 < 0.95 | per-library R^2 of log2(x + 1), the source's own normalized value against the compendium's | R^2 >= 0.95, the compendium's own replicate floor: a re-processing of one library must agree at least as well as two replicates it kept | not measured |
| PRECISE-1K and Public K-12 | re-aligned to NC_000913.3; refused replicates with Pearson r < 0.95 | per-library Pearson r of log2(x + 1) | r >= 0.95 (`lamoureux2023.REPLICATE_QC`) | not measured |
| reads-only sources | the compendium's pipeline is the processing reference | counts re-derived with that pipeline against the compendium's released counts | equality where the tool versions match, else the replicate floor above | not measured |

A failed cross-check blocks the own-release loader for that sample until the cause is named (a label swap, a different library, a processing version); the compendium's copy stays served meanwhile.

### Coupling with the compendium loaders

Serving a study from its own release and from the compendium would store each sample twice. Proposal (an owner decision, not taken): each compendium loader reads a committed list of samples served by an own-release dataset and drops them with a counted reason, so the compendium serves exactly its compendium-only samples (Price 70, Borchert 2024 36, Lim 2022 96, PRECISE-1K 484). Dropping served records is a full rebuild, which KG 4.0 is.

### Open decisions (owner)

- Where the browser snapshot lives: deposit `feba.db` (10.2 GB) once in the raw mirror, or have each loader deposit only its organism's slice. The audit's copy is a session scratch file, so a re-run needs the bytes re-fetched and the pin re-checked.
- Public K-12: file one issue per source (90) or treat it as one loader keyed by source.
- Whether the 22 unattributed browser samples (bioreactor series) and the 9 Lim 2022 BioProject-only projects count as studies to load.
- The Price 2018 cross-check bound.
- Seven registered classes sit under the four admissible verdicts' keys and could leave `GRANDFATHERED`. Five declare a module-level `CITATION_KEY`; `lamoureux2023_public_k12.py` and `lamoureux2023_growth.py` do not, so those two need one first. The enforcement test pins `len(GRANDFATHERED) == 134`, so any departure moves that pin.

### Issues filed (2026-10-11)

Umbrella #931 (`dataset`, `before-next-kg-build`), with the counts, the owner decisions and the Public K-12 list. One issue per original study loadable from its own release and not yet loaded, in the recommended order:

- Borchert 2024 (9): #932 Schmidt 2022; #933 Thompson 2020; #934 Incha 2020; #935 Rand 2017; #936 Thompson 2019 lysine (browser pubId Thompson18); #937 Garber 2020; #938 Price 2019 (browser pubId Price18b); #939 Price 2021 (browser pubId Price21b); #940 Thompson 2019 valerolactam (browser pubId Thompson19).
- Lim 2022 (12): #941 Pobre 2020; #942 Bojanovič 2017; #943 Miyazaki 2018; #944 Nakamura 2020; #945 Lim 2020; #946 Lim 2021; #947 D'Arrigo 2019; #948 Bentley 2020; #949 Xiao 2018; #950 Lei 2020; #951 La Rosa 2015; #952 Jayakody 2018.
- PRECISE-1K (24): #953 Tan 2019; #954 Tan 2020; #955 Sastry 2019; #956 Choudhary 2020; #957 Sastry 2021; #958 Gao 2018; #959 Anand 2019 (OxyR Is a Convergent Target for...); #960 Guzmán 2018 (Enzyme promiscuity shapes evolutionary innovation and...); #961 Latif 2018; #962 Sandberg 2017; #963 Anand 2019 (Pseudogene repair driven by selection pressure...); #964 Guzmán 2018 (Reframing gene essentiality in terms of...); #965 Utrilla 2016; #966 Ebrahim 2016; #967 Kim 2018; #968 Anand 2019 (Adaptive evolution reveals a tradeoff between...); #969 Sandberg 2014; #970 Bordbar 2014; #971 Seo 2015 (Genome-wide Reconstruction of OxyR and SoxRS...); #972 Anand 2021; #973 Seo 2014; #974 Seo 2015 (Decoding genome-wide GadEWX-transcriptional regulatory networks reveals...); #975 LaCroix 2015; #976 Seo 2017.

Not filed per study: Wetmore 2015 (in flight on `feat/ecoli-wetmore2015-own-release`), the unattributed releases, and Public K-12 (owner decisions in #931). Comments with the browser measurement went to #908 (7 Thompson 2019 lysine samples, not 4), #860 (the browser lists `RCH2_defined_noCarbon`; 16 Rand 2017 samples on it) and #861 (per-strain fitness file answers 200).

### Preparing for SPELL and YEASTRACT

Nothing yeast is loaded here. Measured by `aggregation_decomposition_audit.py --yeast-only` (`results/aggregation_decomposition_yeast_sources.json`) and, for SPELL, recorded in [[experiments.015-spell.publications]].

**SPELL** is SGD's archive of published expression datasets as SGD-processed PCL tables. Per [[experiments.015-spell.publications]] (2026.10.04): 603 study folders, one publication each, named by first author, year and PMID; 515 READMEs name a GEO series and 88 (1997 to 2006) do not; 778 PCL files and 16,160 conditions; 107 files carry `knn` in their name (11.4% of cells), where imputed values cannot be told from measured ones. Attribution is per study folder, so the decomposition is one `SourceDecomposition` per folder with `source_label` `pmid:<id>`. The own release of the 515 is their GEO series (scriptable, per-sample values in the series matrix or supplementary files, by file name to be checked as above); the 88 need the data location read from each paper. Measured 2026-10-11: `$DATA_ROOT/data/sgd/spell/` holds 0 files and `all_spell_datasets.tar.gz` (sha256 `d021ba29...` in the 015 note) is not there, so the archive must be restored before any SPELL work.

**YEASTRACT+** is a curated table of regulator-target associations. The mirrored 2022 flat file (`torchcell-raw/teixeiraYEASTRACTPortalExploitation2023`, shared by the maintainers by email, `manual_browser`, sha256 `2f2e9f8b...`) has 356,181 S. cerevisiae rows, each naming one PubMed id: 1,671 PMIDs, 226 regulators, 6,943 targets. Evidence is `Indirect` (an expression change in a regulator mutant or overexpression) for 264,772 rows, `Direct` (binding) for 91,155 and `N/A` for 254, over 125 methods led by "Microarray analysis - wt vs TF mutant" (215,867). Two PMIDs hold 141,049 rows: Salin 2008 (74,129) and Reimand 2010 (66,920), the latter itself a re-analysis of transcription-factor knockout microarrays, so one source is an aggregation inside the aggregation. A row carries a sign and an evidence class, never a value: each source's own release (expression series, binding peaks) carries the measurement the association was called from.

What the schema would need, as proposals:

- The decomposition applies as built: `AggregationRecord.sources` with `pmid:` labels; YEASTRACT's `value_origin` is `derived_by_aggregator` (curated calls), SPELL's `re_measured` (SGD re-normalized).
- For SPELL, a per-value imputation flag (the knn files), the two-channel reference of each study, and the platform; `MicroarrayExpressionPhenotype` exists and the Kemmeren 2014 loader is the precedent.
- For YEASTRACT, the association is not a phenotype of a genotype in an environment. An `Indirect` row is a derived statement over an expression dataset that would be loaded from its own release (a regulator deletion's transcriptome); a `Direct` row needs a binding-measurement record class that does not exist. Hypothesis (untested): most `Indirect` rows reduce to expression datasets already planned through SPELL or GEO, and the association becomes an edge derived from them rather than a stored record.
- A re-analysis source (Reimand 2010) needs a `source_label` for the re-analysis and a pointer to the original study it re-processed, so the decomposition can recurse.

## 2026.10.11 - Wetmore 2015 is loaded as its own dataset; its Price 2018 cross-check gets a bound

`RbTnseqWetmore2015Dataset` ([[torchcell.datasets.ecoli.wetmore2015]], branch `feat/ecoli-wetmore2015-own-release`) loads Wetmore 2015 from its own 1.0.0 release. The audit's Wetmore entry is now built by `wetmore_source` in `aggregation_decomposition_audit.py`, which reads `loaded_as_own_dataset` and `own_dataset_class` off the registry instead of hard-coding them, and the entry was recomputed with that function (the full audit needs the 10 GB browser snapshot and re-measures all four compendia; only this entry's inputs moved). The Price 2018 verdict was re-gated with `--decomposition` and the ledger section re-rendered.

| field | before | now |
|---|---|---|
| `loaded_as_own_dataset` / `own_dataset_class` | no / None | yes / `RbTnseqWetmore2015Dataset` |
| cross-check artifact | `fit_logratios_good.tab` (91 samples, 331,786 values) | `fit_logratios.tab`, every sample (92 samples, 335,432 values; `set2IT045`, failed by Wetmore's own cor12 rule, included) |
| statistic, value | min Pearson r 0.98895 | min Pearson r 0.98895 (unchanged: `set2IT045` is 0.9903) |
| tolerance | none (a pin) | **0.87**, Price 2018's "median correlation of 0.87 for gene fitness values from defined media experiments" between exact replicates |
| Price counts | loadable 1, loaded 0, 92 samples loadable | loadable 0, loaded 1, 0 samples loadable |

The bound is an owner decision taken by recommendation (the table above marked it so): Price states it analysed fitness "as previously described9" (ref 9 is Wetmore) at statistics 1.0.3, so a shared sample's two values are two analyses of the same reads and should agree at least as well as two independent exact replicates do. The same bar is the Wetmore loader's L4 row `compendium_cross_check_price2018`, which compares the two dev stores (83 samples in both, 301,124 values, min r 0.9889).

## 2026.10.12 - The Fitness Browser inventory and one browser loader per organism

Loader shape decided 2026-10-12 (#931): ONE Fitness Browser loader per organism, keyed by the browser's own `pubId` per sample, reading the deposit `$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018/data/fitness_browser/feba.db` (10,205,732,864 bytes, sha256 `250d4a03ffc41ea918027ed7adcb2fa9ed3ddc127ad3de29b40602fc6264b4ef`, `DEPOSIT.md` beside it), each source study a `SourceStudy` with `source_kind` and cross-checked against the served compendium as L4. Code: [[torchcell.datasets.fitness_browser]] (the pinned reader), [[torchcell.datasets.ecoli.fitness_browser_ecoli]] (`FitnessBrowserEcoliDataset`), [[torchcell.datasets.pputida.fitness_browser_putida]] (`FitnessBrowserPutidaDataset`).

### Inventory from the bytes (G3)

Generated by `experiments/database/scripts/fitness_browser_inventory.py` ([[experiments.database.scripts.fitness_browser_inventory]]) off the deposit with its sha256 re-verified, written to `experiments/database/results/fitness_browser_inventory.json` (every experiment's publication, set, condition columns, medium, temperature, library, generations, loci with a fit and a t value, Price 2018 Table S5 membership by `(orgId, name)`, and the served store that keeps it by that loader's own sample rule) and `fitness_browser_inventory_tables.md`, copied here. "Served by a torchcell store" counts `RbTnseqPrice2018EcoliDataset` and `RbTnseqWetmore2015Dataset` for Keio and `RbTnseqBorchert2024Dataset` for Putida, computed from the pinned release bytes through the loaders' own classification, never from a dev store.

Deposit: `data/fitness_browser/feba.db`, sha256 `250d4a03ffc41ea918027ed7adcb2fa9ed3ddc127ad3de29b40602fc6264b4ef` (verified this run: True), 72 organisms, 10,053 experiments, 28 publications cited by at least one experiment (28 rows in `Publication`).

| orgId | organism | experiments | publications | no pubId | loci with values | in Price 2018 Table S5 | served by a torchcell store |
|---|---|---|---|---|---|---|---|
| ANA3 | Shewanella sp029835735 ANA-3 | 107 | 2 | 0 | 3668 | 101 | 0 |
| BFirm | Paraburkholderia phytofirmans PsJN | 129 | 3 | 16 | 5428 | 87 | 0 |
| Bifido | Bifidobacterium breve UCC2003 | 698 | 1 | 0 | 1713 | 0 | 0 |
| Bosea_OAE506 | Allobosea sp040546595 OAE506 | 6 | 0 | 6 | 3838 | 0 | 0 |
| Brev2 | Brevundimonas vesicularis GW460-12-10-14-LB2 | 4 | 1 | 0 | 2394 | 0 | 0 |
| Btheta | Bacteroides thetaiotaomicron VPI-5482 | 542 | 1 | 23 | 4055 | 0 | 0 |
| Burk376 | Paraburkholderia caledonica 376MFSha3.1 | 54 | 2 | 24 | 5153 | 0 | 0 |
| Burkholderia_OAS925 | Paraburkholderia graminis OAS925 | 244 | 1 | 28 | 5667 | 0 | 0 |
| Bvulgatus_CL09T03C04 | Phocaeicola vulgatus CL09T03C04 | 22 | 0 | 22 | 3346 | 0 | 0 |
| CL21 | Ralstonia flatus UNC404CL21Col | 34 | 0 | 34 | 4014 | 0 | 0 |
| Caulo | Caulobacter vibrioides NA1000 | 198 | 1 | 6 | 3312 | 192 | 0 |
| Cola | Echinicola vietnamensis KMM 6221, DSM 17526 | 202 | 2 | 0 | 3954 | 190 | 0 |
| Cup4G11 | Cupriavidus basilensis_F FW507-4G11 | 132 | 2 | 26 | 6384 | 103 | 0 |
| Dda3937 | Dickeya dadantii 3937 | 58 | 1 | 23 | 3705 | 0 | 0 |
| Ddia6719 | Dickeya dianthicola 67-19 | 33 | 1 | 0 | 3528 | 0 | 0 |
| DdiaME23 | Dickeya dianthicola ME23 | 69 | 1 | 33 | 3761 | 0 | 0 |
| Dino | Dinoroseobacter shibae DFL-12 | 186 | 2 | 0 | 3187 | 182 | 0 |
| DvH | Nitratidesulfovibrio vulgaris Hildenborough JW710 | 757 | 1 | 0 | 2741 | 0 | 0 |
| Dyella79 | Dyella_D marensis_A UNC79MFTsu3.2 | 69 | 1 | 0 | 3700 | 68 | 0 |
| HerbieS | Herbaspirillum seropedicae SmR1 | 96 | 4 | 0 | 3879 | 62 | 0 |
| Kang | Kangiella aquimarina DSM 16071 | 108 | 1 | 0 | 2003 | 108 | 0 |
| Keio | G047199095 sp047199095 BW25113 | 183 | 3 | 19 | 3789 | 158 | 147 |
| Korea | Sphingomonas koreensis DSMZ 15582 | 162 | 2 | 0 | 3393 | 144 | 0 |
| Koxy | Klebsiella grimontii M5al | 208 | 3 | 0 | 4608 | 171 | 0 |
| Lysobacter_OAE881 | Lysobacter_J soli OAE881 | 34 | 1 | 9 | 3084 | 0 | 0 |
| MR1 | Shewanella oneidensis MR-1 | 176 | 2 | 0 | 3782 | 176 | 0 |
| MT058 | Pantoea eucalypti MT58 | 18 | 0 | 18 | 3820 | 0 | 0 |
| Magneto | Phaeospirillum magneticum AMB-1 | 36 | 1 | 0 | 3221 | 0 | 0 |
| Marino | Marinobacter adhaerens HP15 | 255 | 2 | 0 | 3650 | 249 | 0 |
| Methanococcus_JJ | Methanococcus maripaludis JJ | 129 | 1 | 0 | 1383 | 0 | 0 |
| Methanococcus_S2 | Methanococcus maripaludis S2 | 386 | 1 | 15 | 1244 | 0 | 0 |
| Miya | Cupidesulfovibrio sp019931275 Miyazaki F | 178 | 2 | 0 | 2531 | 170 | 0 |
| Mucilaginibacter_YX36 | Mucilaginibacter yixingensis YX-36 DSM 26809 | 75 | 0 | 75 | 3566 | 0 | 0 |
| MycoTube | Mycobacterium tuberculosis H37Rv | 212 | 0 | 212 | 2881 | 0 | 0 |
| PS | Azospira oryzae PS | 79 | 2 | 0 | 2561 | 79 | 0 |
| PV4 | Shewanella loihica PV-4 | 160 | 1 | 0 | 3009 | 160 | 0 |
| Par6119 | Pectobacterium aroidearum 61-19 | 33 | 0 | 33 | 3836 | 0 | 0 |
| PatSCRI1043 | Pectobacterium atrosepticum SCRI1043 | 33 | 0 | 33 | 3882 | 0 | 0 |
| Pbr123 | Pectobacterium brasiliense 123-1 | 40 | 0 | 40 | 3595 | 0 | 0 |
| Pbr1692 | Pectobacterium brasiliense 1692 | 35 | 0 | 35 | 3557 | 0 | 0 |
| PccWPP14 | Pectobacterium carotovorum WPP14 | 61 | 0 | 61 | 3732 | 0 | 0 |
| Pedo557 | Pedobacter sp002151545 GW460-11-11-14-LB5 | 177 | 2 | 0 | 4423 | 163 | 0 |
| Phaeo | Phaeobacter inhibens DSM 17395 | 274 | 4 | 0 | 3099 | 256 | 0 |
| Phaga5 | Hydrogenophaga  GW460-11-11-14-LB1 | 15 | 0 | 15 | 4601 | 0 | 0 |
| Ponti | Pontibacter actiniarum KMM 6156, DSM 19842 | 104 | 1 | 0 | 3685 | 104 | 0 |
| PpaNY1722A | Pectobacterium parmentieri NY1722A | 31 | 0 | 31 | 3856 | 0 | 0 |
| Putida | Aquipseudomonas alloputida KT2440 | 314 | 11 | 34 | 4778 | 0 | 232 |
| PveNY1715C | Pectobacterium versatile NY1715C | 38 | 0 | 38 | 3944 | 0 | 0 |
| RPal_CGA009 | Rhodopseudomonas palustris_F CGA009 | 208 | 0 | 208 | 4322 | 0 | 0 |
| RalstoniaBSBF1503 | Ralstonia solanacearum IBSBF1503 | 65 | 0 | 65 | 3779 | 0 | 0 |
| RalstoniaGMI1000 | Ralstonia nicotianae GMI1000 | 59 | 0 | 59 | 3980 | 0 | 0 |
| RalstoniaPSI07 | Ralstonia syzygii PSI07 | 71 | 0 | 71 | 4029 | 0 | 0 |
| RalstoniaUW163 | Ralstonia solanacearum UW163 | 6 | 0 | 6 | 3795 | 0 | 0 |
| SB2B | Shewanella amazonensis SB2B | 190 | 2 | 0 | 3121 | 190 | 0 |
| Smeli | Sinorhizobium meliloti 1021 | 90 | 2 | 0 | 5133 | 84 | 0 |
| SynE | Synechococcus elongatus PCC 7942 | 129 | 1 | 0 | 1899 | 129 | 0 |
| SyringaeB728a | Aquipseudomonas fragariae pv. syringae B728a | 46 | 2 | 12 | 4314 | 0 | 0 |
| SyringaeB728a_mexBdelta | Aquipseudomonas fragariae pv. syringae B728a &Delta;mexB | 23 | 1 | 0 | 4282 | 0 | 0 |
| Variovorax_OAS795 | Variovorax sp040546685 OAS795 | 92 | 1 | 0 | 4532 | 0 | 0 |
| WCS417 | Aquipseudomonas simiae WCS417 | 201 | 4 | 0 | 4419 | 144 | 0 |
| Xantho | Xanthomonas campestris pv. campestris strain 8004 | 19 | 1 | 0 | 3700 | 0 | 0 |
| acidovorax_3H11 | Acidovorax sp001633105 GW101-3H11 | 140 | 1 | 0 | 3935 | 140 | 0 |
| azobra | Azospirillum baldaniorum Sp245 | 91 | 1 | 0 | 4833 | 91 | 0 |
| psRCH2 | Stutzerimonas stutzeri_AE RCH2 | 350 | 4 | 0 | 3349 | 301 | 0 |
| pseudo13_GW456_L13 | Aquipseudomonas jessenii GW456-L13 | 106 | 1 | 0 | 4350 | 106 | 0 |
| pseudo1_N1B4 | Aquipseudomonas sp000282495 FW300-N1B4 | 147 | 2 | 0 | 4336 | 138 | 0 |
| pseudo3_N2E3 | Aquipseudomonas helvetica FW300-N2E3 | 211 | 2 | 0 | 5028 | 201 | 0 |
| pseudo5_N2C3_1 | Aquipseudomonas fluorescens_AA FW300-N2C3 | 184 | 2 | 0 | 5193 | 172 | 0 |
| pseudo6_N2E2 | Aquipseudomonas fluorescens_Q FW300-N2E2 | 203 | 3 | 15 | 5133 | 178 | 0 |
| rhodanobacter_10B01 | Rhodanobacter denitrificans FW104-10B01 | 211 | 1 | 181 | 2933 | 0 | 0 |
| rhodanobacter_R12 | Rhodanobacter sp000427505 FW510-R12 | 6 | 0 | 6 | 2946 | 0 | 0 |
| rhodanobacter_T8 | Rhodanobacter sp001617365 FW510-T8 | 11 | 0 | 11 | 2657 | 0 | 0 |
| **total** | | 10053 | | 1543 | | 4797 | 379 |

Price 2018 Table S5 rows absent from the browser, per organism: ANA3 2; BFirm 2; Caulo 4; Cola 6; Cup4G11 1; Dino 2; Dyella79 1; HerbieS 1; Keio 4; Korea 4; Koxy 2; Pedo557 3; Phaeo 6; SB2B 4; Smeli 2; WCS417 4; acidovorax_3H11 5; azobra 2; psRCH2 2; pseudo13_GW456_L13 4; pseudo1_N1B4 2; pseudo3_N2E3 4; pseudo5_N2C3_1 4; pseudo6_N2E2 2.

**Keio (E. coli BW25113)**, by browser publication:

| pubId | experiments | in Table S5 | kept by RbTnseqPrice2018EcoliDataset | kept by RbTnseqWetmore2015Dataset | media |
|---|---|---|---|---|---|
| Wetmore15 | 87 | 87 | 83 | 83 | M9 minimal media_noCarbon 56; M9 minimal media_noNitrogen 26; MOPS Rich Defined media_noCarbon 4; LB 1 |
| Price18 | 71 | 71 | 64 | 0 | LB 63; M9 minimal media_noNitrogen 6; MOPS minimal media_noCarbon 2 |
| no-pubId | 19 | 0 | 0 | 0 | SDM_noCarbon 15; M63_no_sulfur 4 |
| Price21b | 6 | 0 | 0 | 0 | RCH2_defined_noCarbon 6 |

**Putida (P. putida KT2440)**, by browser publication:

| pubId | experiments | in Table S5 | in the Borchert 2024 release | kept by RbTnseqBorchert2024Dataset | media |
|---|---|---|---|---|---|
| Schmidt22 | 128 | 0 | 128 | 128 | MOPS minimal media_Glucose_noNitrogen 104; MOPS minimal media_noCarbon 24 |
| Thompson20 | 46 | 0 | 46 | 46 | MOPS minimal media_noCarbon 46 |
| VillalobosEscobedo23 | 46 | 0 | 0 | 0 | R2A 46 |
| no-pubId | 34 | 0 | 22 | 0 | M9_1percentGlucose 22; MOPS minimal media_noCarbon 12 |
| Incha20 | 18 | 0 | 18 | 18 | MOPS minimal media_noCarbon 18 |
| Rand17 | 16 | 0 | 16 | 16 | RCH2_defined_noCarbon 16 |
| Thompson18 | 7 | 0 | 7 | 7 | MOPS minimal media_noCarbon 7 |
| Garber20 | 6 | 0 | 6 | 6 | MOPS minimal media_noCarbon 6 |
| Price18b | 6 | 0 | 6 | 6 | RCH2_defined_noCarbon 4; MOPS minimal media_noCarbon 2 |
| Price21b | 3 | 0 | 3 | 3 | MOPS minimal media_noCarbon 3 |
| Pearson24 | 2 | 0 | 0 | 0 | MOPS minimal media_noCarbon 2 |
| Thompson19 | 2 | 0 | 2 | 2 | MOPS minimal media_noCarbon 2 |

What the inventory establishes:

- Every Keio experiment carries values for the same 3,789 loci and every Putida experiment for the same 4,778 (the browser's matrix is complete within an organism).
- The browser holds 4,797 of Price 2018 Table S5's rows across 32 organisms. Of the 4 Keio rows it lacks, all 4 are the sucrose and D-mannitol samples the authors withdrew in 2021.
- Putida is not a Price 2018 organism (0 rows in Table S5); its compendium is Borchert 2024, which holds 254 of the browser's 314 Putida experiments.
- The six Keio `Price21b` samples (set52, D-mannitol at 10, 5 and 2.5 mM on `RCH2_defined_noCarbon`, library `Keio_ML9a`) were started on 3-Aug-21 (`Experiment.dateStarted`), before the 2021-09-10 withdrawal note. The note names "this publication" (Price 2018), and the browser, which no longer carries the four withdrawn samples, still serves these six under the 2021 GapMind paper. They are stored, not refused; whether the stock-solution problem reaches set52 is not stated in any deposited byte and is surfaced for the owner.

### What each loader stores, and why (G4 duplication by content)

| organism | experiments | stored (`primary`) | own release already loaded | unattributed (no `pubId`) | medium not in MEDIA_LIBRARY |
|---|---|---|---|---|---|
| Keio | 183 | 6 (`Price21b`) | 158 (`Wetmore15` 87 by `RbTnseqWetmore2015Dataset`; `Price18` 71 by `RbTnseqPrice2018EcoliDataset`) | 19 | 0 |
| Putida | 314 | 234 (Schmidt22 128, Thompson20 46, Incha20 18, Rand17 16, Thompson18 7, Garber20 6, Price18b 6, Price21b 3, Pearson24 2, Thompson19 2) | 0 | 34 | 46 (`VillalobosEscobedo23`, all on `R2A`) |

- A sample whose publication is already loaded from its own release is not stored again. `Price18` is the compendium's own paper (`source_kind = compendium_only`): its own release is the compendium release `RbTnseqPrice2018EcoliDataset` serves, which equals the browser to 8.9e-16 (the 2026.10.11 audit), so a second copy would be the same values under a second dataset. The browser loader's L4 rows re-measure that equality against the served stores.
- Every Putida study in Borchert 2024 is served there only through the compendium, so the browser loader stores it as `primary`, at full precision (the compendium rounds to three decimals) and over 4,778 loci (the compendium keeps 4,732). `RbTnseqBorchert2024Dataset` stays served unchanged as the sanity check: no served loader changes in this PR.
- The unattributed releases (Keio 19, Putida 34) are not stored: the browser names no study for them and whether they count as studies to load is an open owner decision of #931.
- `VillalobosEscobedo23` (46): `R2A` has no MEDIA_LIBRARY entry. The deposit's `MediaComponents` table states its recipe (8 rows, 0.5 g/L each of Bacto Peptone, casamino acids, yeast extract, D-glucose and starch, 0.3 g/L potassium phosphate dibasic and sodium pyruvate, 0.05 g/L magnesium sulfate heptahydrate), but the deposit is SQLite, so a verbatim quote has no text file to be audited against (`audit_sourced_value` reads the source as text). The route is a deterministic text extract of the deposit's tables with its own pin; not built here. Its supernatant groups (Trichoderma atroviride culture supernatants) also have no compound identity.

### The Borchert 2024 verdict

`aggregation_decomposition_audit.py --mark-fitness-browser-loaded` re-derives only the browser sources' loaded fields off `fitness_browser_putida.STORED_PUB_IDS` (a full run applies the same rule), and `database/candidates/borchertMachineLearningAnalysis2024.json` was re-gated with `--decomposition` (G1 to G5 unchanged, admissible): loaded as own dataset 1 -> 10 studies, loadable from an own release 10 -> 1 (the unattributed bioreactor release), samples loadable 254 -> 22. The Price 2018 decomposition is unchanged: its two sources are Wetmore 2015 (loaded) and the compendium paper.

### What remains

- The other 70 browser organisms (5,225 experiments in the 32 Price 2018 organisms besides Keio, and the rest): no loader; the inventory JSON lists every experiment.
- Keio and Putida unattributed releases (owner decision, #931) and `VillalobosEscobedo23` (the text-extract route above).
- Lim 2022 (#941 to #952) and PRECISE-1K (#953 to #976) sources: not Fitness Browser studies; untouched here.

## 2026.10.12 - Revision: the browser loaders store genes as the deposit names them, and #908

Owner rule, 2026-10-12: "the data is the data ... the genome is separate.. it doesn't gate anything". Both Fitness Browser loaders now store every locus under the deposit's own `Gene.sysName` with no assembly consulted: Keio as MG1655 b-numbers (the 21 loci the ECK crosswalk could not place onto BW25113 are now stored, 22,608 -> 22,734 records), KT2440 as its `PP_` tags (1,118,052 records, unchanged in count; the common names now come from the deposit's `Gene.gene`, not the genome). The genome is not read to build either store, and the verifiers take the deposit's own gene table as their gene universe.

#908 is folded in: Borchert 2024's four samples Thompson 2019 lysine measured (set7IT062, set7IT055, set7IT044, set6IT057; 18,928 records) carry `thompson2019_lysine` as their `SourceStudy` ([[torchcell.datasets.pputida.borchert2024]], 2026.10.12). A vessel is not stored for any bacterial record (no `Environment` field), so set6IT057's vessel is pinned as a quote and the storage question is a schema decision.

Still not stored by the browser loaders, each for a reason that is not the genome:

| samples | organism | why | what would store them |
|---|---|---|---|
| 19 (no `pubId`) | Keio | no `SourceType` member for a data release with no paper; `SDM_noCarbon` and `M63_no_sulfur` have no MEDIA_LIBRARY row and their recipes exist only in the SQLite deposit; a sulfur source has no `PhysicalFactor` member | a `SourceType` member and a `sulfur_source` factor (schema records), two media rows quoted from a pinned text extract of the deposit's `MediaComponents` |
| 34 (no `pubId`) | Putida | the same `SourceType` gap; 22 are bioreactor samples whose dissolved-oxygen set point, feed regime and solvent overlay have no `Environment` field | the `SourceType` member; for the 22, process fields (schema) |
| 46 (`VillalobosEscobedo23`) | Putida | `R2A` has no MEDIA_LIBRARY row; 6 samples add a Trichoderma culture supernatant, which no perturbation type represents, and 8 add Vogel's fungal medium (a mixture) | an R2A row quoted from Price 2018 Table S18 (it prints the recipe as "Reasoner's 2A media"); a typed carrier for a conditioned medium |
