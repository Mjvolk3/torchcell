---
id: 72ij0qdv3yusenq5yn1p34x
title: Fitness_browser_ecoli
desc: ''
updated: 1791803941338
created: 1791803941338
---

## 2026.10.12 - The Fitness Browser's Keio experiments, keyed by publication

`FitnessBrowserEcoliDataset` (`torchcell/datasets/ecoli/fitness_browser_ecoli.py`), the E. coli half of the Price 2018 decomposition (#931; owner policy 2026-10-10, plan decision 16). One loader for the organism, reading the Fitness Browser deposit through [[torchcell.datasets.fitness_browser]]; the inventory and the per-organism decisions are in [[torchcell.candidates.aggregation-decomposition]] (2026.10.12). `CITATION_KEY` is the deposit's mirror key, `priceMutantPhenotypesThousands2018`, whose candidate verdict is admissible; each record's `Publication` is its sample's study.

### Studies (`SOURCE_STUDIES`)

| `pubId` | study | `source_kind` | own release already loaded by | browser samples | fate |
|---|---|---|---|---|---|
| Wetmore15 | Wetmore 2015, doi:10.1128/mBio.00306-15 | primary | `RbTnseqWetmore2015Dataset` | 87 | not stored (G4) |
| Price18 | Price 2018, doi:10.1038/s41586-018-0124-0 | compendium_only | `RbTnseqPrice2018EcoliDataset` | 71 | not stored (G4) |
| Price21b | GapMind for carbon sources, doi:10.1101/2021.11.02.466981 (the browser's URL; a preprint) | primary | none | 6 | stored |
| (empty) | unattributed release | | | 19 | not stored (#931 owner decision) |

`check_studies` compares each study's browser title and URL with the deposit's `Publication` row on every build and refuses a difference; a `pubId` missing from the table is refused, never defaulted.

### Decisions taken by recommendation (owner to confirm)

- **Price18 is not stored again.** It is the compendium's own paper: its release is the bigfit release `RbTnseqPrice2018EcoliDataset` already serves, and the browser equals it to 8.9e-16 (2026.10.11 audit). A second copy is the same values under a second dataset, which is G4 duplication by content. The L4 row `compendium_cross_check_price2018` re-measures the equality against the served store at an identity bar (Pearson r 0.999999).
- **Wetmore15 is not stored again** for the same reason, against its own 1.0.0 release; the L4 row `own_release_cross_check_wetmore2015` compares the browser (1.0.3 analysis) with that store at Price 2018's median exact-replicate r, 0.87 (`wetmore2015.COMPENDIUM_MIN_PEARSON_R`).
- **Price21b's D-mannitol samples are stored.** The 2021 withdrawal (`wetmore2015.AUTHOR_WITHDRAWAL`) names "this publication", Price 2018; its scope here is `WITHDRAWAL_SCOPE = {Wetmore15, Price18}`. The browser removed the four withdrawn samples and still serves these six under the GapMind paper. Their set was started 3-Aug-21, before the note of 2021-09-10, and no deposited byte says which stock set52 used: surfaced to the owner, not resolved by assumption.
- **The transposon is None** for `Keio_ML9a`: Price 2018 Table S20 states Tn5 for `KEIO_ML9`; nothing deposited states how `Keio_ML9a` relates to it.

### Records

One `BacterialEnvironmentResponseExperiment` per (placed gene, stored sample): 6 samples x 3,768 genes = 22,608 records (the browser's 3,789 loci, of which 3,768 place on BW25113 through a one-to-one ECK pair, the route `price2018.map_genes` takes; every one of the 3,789 names a b-number in `Gene.sysName`, and 21 have no one-to-one ECK pair). The value is `FitByExp_Keio.fit`, the browser's moderated t is `environment_response_t_statistic`, and `environment_response_se` is `sqrt((f / t)^2 - 0.1^2)`. Environment: `RCH2_DEFINED_NOCARBON_PRICE2018`, D-mannitol as carbon source (`EnvironmentPhysicalPerturbation`, mM), 30 C, aerobic; `nGenerations` is empty, so `duration_hours` and `duration_generations` are typed gaps. D-mannitol carries the compound table's `inchikey` gap (#726). `screen_id` is `Keio:<expName>`, the key Price 2018 and Wetmore 2015 also use.

Reports in `preprocess/`: `samples.json` (every browser experiment with its fate and its typed row), `dropped_records.json` (fates, by publication, the study table), `identifier_mapping.json`, `compendium_cross_check.json`, `verification_report.json`.

### Build and L0-L4 (dev store, slurm job 4196, 2026-10-12)

Built with `build_dataset_lmdb --dataset FitnessBrowserEcoliDataset --retire-existing --verify` (array script, worktree source): 22,608 records in 40 s, gene set 3,768, 6 references; `--list-stale` reads the store `fresh`. Of the browser's 3,789 Keio loci, 21 do not place on BW25113 (`b_number_not_in_mg1655_annotation` 8, `eck_absent_from_bw25113` 7, `eck_not_one_to_one` 6), the same 21 the Price 2018 loader drops. Report: `$DATA_ROOT/data/torchcell/rbtnseq_fitness_browser_ecoli/preprocess/verification_report.json`.

| level | rule | result |
|---|---|---|
| L0 | structural | pass: 22,608 records validated |
| L1 | count | pass: 22,608 observed, 22,608 expected |
| L1 | pair_uniqueness | pass: 22,608 unique (study, strain, condition) records |
| L1 | provenance_gaps | pass: 135,648 documented gaps; deferred field `inchikey` (D-mannitol, #726) |
| L1 | canonical_gene_names | pass: 3,768 systematic names, each current in the BW25113 genome |
| L1 | records_carry_their_browser_publication | pass: 6 stored samples, each record names Price21b's DOI |
| L2 | value_fidelity, se_nonnegative | pass: 22,608 values |
| L2 | interval_orientation, uncertainty_sanity | pass |
| L3 | measurement_type_consistent | pass: `log2_ratio` only |
| L3 | reference_zero | pass: reference response 0 for all records |
| L3 | environment_perturbed | pass: every record carries its carbon source |
| L3 | compound_identity, media_compound_identity, media_membership | pass: 1 medium, `RCH2_DEFINED_NOCARBON_PRICE2018` |
| L4 | gene_containment_sgd, current_genome_genes | pass: 1.000 of 3,768 genes are BW25113 genes |
| L4 | compendium_cross_check_price2018 | pass: 151 shared samples, 568,968 (gene, sample) values, pooled Pearson r 1.000000, smallest per-sample r 1.000000 (bar 0.999999), largest difference 8.88e-16 |
| L4 | own_release_cross_check_wetmore2015 | pass: 83 shared samples, 301,124 values, pooled Pearson r 0.997034, smallest per-sample r 0.988897 (bar 0.87), largest single-gene difference 2.57 |

The L4 rows measure the browser's copy of the 151 Keio samples the served Price 2018 store keeps (the store was rebuilt at 05:56 by array 4011 and now keeps 151, not the 147 of the 2026.10.11 audit) against that store, and of the 83 samples Wetmore 2015's store keeps against its own 1.0.0 release: the Price copy is identical to the float, so storing it again would duplicate it (G4), and the Wetmore agreement reproduces the audit's minimum 0.98890.

## 2026.10.12 - Revision: every gene under the deposit's own locus tag

Owner rule, 2026-10-12: "the data is the data ... the genome is separate.. it doesn't gate anything". The build above placed each browser locus on a BW25113 tag through the ECK crosswalk and left out the 21 that do not place; that is a genome lookup deciding what is stored, so it is gone.

- Every gene is stored under the locus tag the deposit gives it, `Gene.sysName`, an MG1655 b-number (namespace `ecoli_k12_mg1655_bnumber`, no `identifier_mapping`), with `Gene.gene` as its common name (the tag when the deposit has none: 105 of 3,789). `fitness_browser.released_genes` refuses only a change in the deposit's shape (a locus with values and no `Gene` row or no `sysName`, or a repeated `sysName`). No genome is injected or read to build; the reference genome still states the library strain, BW25113 (`Organism.strain`).
- Records: 6 samples x 3,789 loci = **22,734** (was 22,608; the 21 loci the crosswalk could not place are now stored).
- The verifier's gene rules use the deposit's own Keio `Gene.sysName` set as the universe, with no resolver.
- The L4 rows still align the deposit to the served Price 2018 and Wetmore 2015 stores through `price2018.map_genes`, because those stores key a gene by the BW25113 tag their own loaders placed it on. It is a measurement of samples this loader does not store (G4) and decides nothing stored.

### Build and L0-L4 (slurm job 4332, `--retire-existing --verify`)

22,734 records in 23 s, gene set 3,789, 6 references; verification PASSED. Report: `$DATA_ROOT/data/torchcell/rbtnseq_fitness_browser_ecoli/preprocess/verification_report.json`.

| level | rule | result |
|---|---|---|
| L0 | structural | pass: 22,734 records validated |
| L1 | count | pass: 22,734 observed, 22,734 expected |
| L1 | pair_uniqueness | pass: 22,734 unique records |
| L1 | provenance_gaps | pass: 136,404 documented gaps; deferred field `inchikey` (D-mannitol) |
| L1 | canonical_gene_names | pass: 3,789 systematic names, one spelling each (no resolver supplied) |
| L1 | records_carry_their_browser_publication | pass: 6 of 6 samples name Price21b's DOI |
| L2 | value_fidelity, se_nonnegative, interval_orientation, uncertainty_sanity | pass |
| L3 | measurement_type_consistent, reference_zero, environment_perturbed | pass |
| L3 | compound_identity, media_compound_identity, media_membership | pass: 1 medium, `RCH2_DEFINED_NOCARBON_PRICE2018` |
| L4 | gene_containment_sgd, current_genome_genes | pass: 1.000 of 3,789 genes are in the deposit's Keio `Gene.sysName` set |
| L4 | compendium_cross_check_price2018 | pass: 151 samples, 568,968 values, pooled r 1.000000, smallest per-sample r 1.000000 (bar 0.999999), largest difference 8.88e-16 |
| L4 | own_release_cross_check_wetmore2015 | pass: 83 samples, 301,124 values, pooled r 0.997034, smallest per-sample r 0.988897 (bar 0.87), largest difference 2.57 |

## 2026.10.12 - #1045 part 2: the rest of the released rows, under their released identifiers

Every row #1057 still dropped that is not an author withdrawal is stored, and no identifier is translated: the stored key is the one the release prints, and a crosswalk to an annotation is a modeling-time join outside the record. Rebuilt from the part-2 branch with `--retire-existing --verify` (slurm array 4583; the stores with no per-dataset verifier entry point verified by jobs 4645 and 4644).

### `rbtnseq_fitness_browser_ecoli`: 22,734 -> 22,734 records

Every gene under its released b-number; the loader reads no genome.

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 22734 records validated |
| L1 | `count` | pass | observed 22734, expected 22734 |
| L1 | `pair_uniqueness` | pass | 22734 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 22734 values checked |
| L2 | `se_nonnegative` | pass | 22734 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 22734 records |
| L3 | `environment_perturbed` | pass | all 22734 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media="RCH2 defined minimal medium, no carb... |
| L1 | `provenance_gaps` | pass | 136404 documented provenance gaps over 22734/22734 records; 1 deferred field(s): ['inchikey']; 682020 undeclared None values over 16 carrier fields (top: Compound.inchi x159138,... |
| L1 | `canonical_gene_names` | pass | 3789 systematic names, one canonical spelling each (no resolver supplied: spelling checked, annotation not) |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 22734 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 90936 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 22734 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 3789 measured genes are the deposit's Keio Gene.sysName genes (>= 1.0) |
| L4 | `current_genome_genes` | pass | every one of the 3789 measured systematic names is a gene of the current genome |
| L1 | `records_carry_their_browser_publication` | pass | SUPPLEMENTARY: 6 stored samples, 6 expected; 0 carry a publication other than their pubId's |
| L4 | `compendium_cross_check_price2018` | pass | SUPPLEMENTARY: 158 shared samples against data/torchcell/rbtnseq_price2018_ecoli, 598662 (gene, sample) values; pooled Pearson r 1.000000, smallest per-sample r 1.000000 against... |
| L4 | `own_release_cross_check_wetmore2015` | pass | SUPPLEMENTARY: 88 shared samples against data/torchcell/rbtnseq_wetmore2015_ecoli, 320848 (gene, sample) values; pooled Pearson r 0.996969, smallest per-sample r 0.988954 agains... |

## 2026.10.13 - Compendium-derived: hidden assumptions and missed representation

This store is derived from the Fitness Browser deposit (`feba.db`), Keio. Data collected from a compendium, a database aggregate or a follow-up study may embed that source's preprocessing (normalization, filtering, re-keying, merged conditions) and may lack ontological representation the primary release carries (per-sample metadata, replicate structure, the released statistic). Its records are a derived copy, and the primary release is still to be collected from source ([[torchcell.knowledge_graphs.dataset-admission-loop]], 2026.10.13).

What it may hide here: the stored `Price21b` samples are the browser's recomputation under its current statistics version, and the medium and condition are the browser's `Experiment` columns rather than the preprint's Methods. The preprint's own tables, if released, are the source to collect.
