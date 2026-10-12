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

One `BacterialEnvironmentResponseExperiment` per (placed gene, stored sample): 6 samples x 3,768 genes = 22,608 records (the browser's 3,789 loci, of which 3,768 place on BW25113 through a one-to-one ECK pair, the route `price2018.map_genes` takes; one locus has no `sysName`). The value is `FitByExp_Keio.fit`, the browser's moderated t is `environment_response_t_statistic`, and `environment_response_se` is `sqrt((f / t)^2 - 0.1^2)`. Environment: `RCH2_DEFINED_NOCARBON_PRICE2018`, D-mannitol as carbon source (`EnvironmentPhysicalPerturbation`, mM), 30 C, aerobic; `nGenerations` is empty, so `duration_hours` and `duration_generations` are typed gaps. D-mannitol carries the compound table's `inchikey` gap (#726). `screen_id` is `Keio:<expName>`, the key Price 2018 and Wetmore 2015 also use.

Reports in `preprocess/`: `samples.json` (every browser experiment with its fate and its typed row), `dropped_records.json` (fates, by publication, the study table), `identifier_mapping.json`, `compendium_cross_check.json`, `verification_report.json`.
