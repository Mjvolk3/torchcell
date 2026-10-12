---
id: zksmckh6uefum8qnbsint29
title: '11'
desc: ''
updated: 1791770797617
created: 1791770797617
---

## 2026.10.11 - Yeast expansion wave 1: what landed, what the owner decides

Rows 1 to 10 of [[database-expansion-100|notes-tex/database/database-expansion-100]] plus three off-schedule items from [[database.yeast-zotero-collection-audit.2026.10.11]] were loaded on 2026-10-11 and 2026-10-12 by one agent per dataset, landed through the merge queue in a chained rebase, and verified L0 to L4 on their dev stores. Counts are from the built stores, never the abstracts. The verdicts are in the 2026.10.12 section of [[torchcell.candidates.ledger]].

### Landed

| Row | Dataset | PR | Records (dev store) | Schema additions |
|---|---|---|---|---|
| 1 | Muenzner 2024 | #1012 | 794 proteome + 55 turnover | `YeastProteinTurnoverExperiment` pair |
| 2 | Peter 2018 | #993 | 33,985 (971 isolates x 35 conditions) | `MeasurementType.condition_growth_ratio`; `natural_isolate_response` verifier |
| 3 | De Chiara 2022 | #1009 | 4,039 life cycle; 36,295 doubling time; 36,286 cell yield | `LifeCyclePhenotype`, `MeasurementType.cell_yield`, `ConcentrationUnit.mg_nitrogen_per_l` |
| 4 | Boocock 2025 | #1011 | 79,516 (393 pooled segregants + 3 de novo crosses) | single-cell expression leaf, segregant expression pair |
| 5 | Hale 2024 | #1023 | 8,692,635 (170 backgrounds x 7,963 guides x 3 assays) | `PerturbedSegregantGenotype`, `MeasurementType.relative_fitness`, tier set `hale2024_322134S_variants` |
| 6 | Jackson 2020 | #1020 | 784 pseudobulk (38,225 cells) | `PseudobulkUmiCountPhenotype` |
| 7 | Puddu 2019 | #1019 | 8,826 colonies over 4,453 strains | `GenomeArchitecturePhenotype` family |
| 8 | N'Guessan 2025 | #1021 | 0; loader present, class unregistered | `cell_cycle_stage_counts` required-nullable on the single-cell leaf |
| 9 | Hackett 2020 | #1030 | 1,752 arrays over 224 courses | inducible-promoter leaf, induction time-course family |
| 10 | Hu 2007 | #1029 | 263 (1,559,117 score and p-value pairs) | `MicroarrayExpressionScorePhenotype` |
| audit 2 | Parsons 2006 | #1034 | 291,673 | none; 33 compound rows |
| audit 4 | Messner 2021 | #1033 | 70 runs (138,662 values) | none; 10 compound rows |
| audit 1 | Piotrowski 2017 | #1028 | 3,435,820 | none; open until slurm 3925 verifies |

Audit rank 3, Yoshikawa 2009, is held: the publisher copy is walled and the only scriptable copy is the YeastPhenome repository.

### Published-data errors found by measurement

- De Chiara 2022 Supplementary Table 3: the normalized growth block is shifted by one isolate name on 519 of 1,011 rows; only the absolute values are loaded (#1006).
- Jackson 2020: 8 strain-condition pairs have no cells in the release, against the paper's "every strain in every condition".
- Jackson 2020 Table S2 names the wrong STP gene for 7 strains; the label, the parent row and the data agree, so the label governs.
- Hu 2007 GEO GSE4654 releases error-model X scores and p-values, not log ratios.
- Boocock 2025: the 27,744 cells are a de novo cross, not the 393 segregants; three crosses; no galactose condition. The candidate row was corrected.
- Hale 2024 crosses BY4716 with 322134S, not RM11-1a; the last author is Ehrenreich. The candidate row was corrected; the notes-tex tables were not regenerated.
- Parsons 2006 stores control over treated, so the YeastPhenome copy is its exact negative (r = -0.999993 over 291,530 shared cells).

### Owner decisions (open issues)

Schema, before KG 4.0:

- #1001, #1018: ploidy and whole-chromosome aneuploidy have no genotype slot (168 of 794 Muenzner isolates, 45 of 55 turnover isolates, all of Puddu).
- #987, #988, #1002: natural loss-of-function alleles need a type; the Peter-panel genotype construction is reused through Caudal's private helpers and should be one shared module; 122 HO-deleted isolates are not on the genotype.
- #1016: one single-cell UMI-count leaf for Boocock and Jackson, or two.
- #1022: the microarray score phenotype as a sibling class, now landed, to confirm.
- #1031: `log2_ratio` fixes treatment over control, but Hillenmeyer HET and Parsons store control over treatment; the sign lives only in `units`.
- #1032: prototrophic BY4741 backgrounds have no typed background on protein and metabolite references.
- #1007: engineered edits on an isolate background (De Chiara validation strains).
- #1013: the 3S parent is pinned to a variant table, not an assembly.
- #1017: `EndpointRule` needs a member for a culture read at its own saturation.
- #989, #990, #991, #992: Gasch 2000 ratio-only arrays, Zhang 2020 promoter drivers, Vaishnav 2022 variable promoter sequences, Khaiwal 2025 cytometry and lifespan traits.

By hand (walled or unmirrored sources):

- #1014 N'Guessan: the Nguyen Ba 2022 Dryad genotypes (4 files, about 5.09 GB); every record needs them. #1015: Nguyen Ba 2022 is not mirrored.
- #1004 Boocock Dryad raw counts; #1026 Hackett's McIsaac 2011/2012/2013 and Saldanha 2004; #994 to #999 Lee 2014, Trikka 2015, DeLoache 2015, Lian 2017, Guzman 2024, Zackrisson 2016.

Curation:

- Suthers 2020 (Issatchenkia orientalis) sits in `database/Saccharomyces-cerevisiae` while `database/Pichia-kudriavzevii` is empty.
- Yoshikawa 2009: load from the YeastPhenome copy, or capture the publisher original by hand first (#1000 covers the duplication with YeastPhenome).
- Khaiwal 2025's antifungal data carries a no-redistribution notice.
- The Peter 2018 DOI has two mirror keys (group and personal library); loaders use `peterGenomeEvolution10112018`.

Housekeeping:

- About 77 GB of superseded store copies under `/scratch/projects/torchcell-deprecated` and the Piotrowski `*.superseded.*` directories wait for a purge.
- The Girgis 2009 dev store must be rebuilt before KG 4.0: Parsons's compound rows changed what "doxycycline hyclate" resolves to, and the staleness check does not see it. It is in the 2026-10-12 rebuild array.

### Next build

Every landing widened the experiment-type unions, so 151 dev stores read stale under main 32bb760c12. The rebuild array is slurm 3953 (`--array=1-151%3`, classes file `/scratch/projects/torchcell/database/slurm/stale_classes_2026-10-12.txt`). When it drains, `python -m torchcell.database.build_dataset_lmdb --list-stale --include-private` must read 0 before the live-rebuild preflight; the KG 4.0 rebuild itself is the owner's.
