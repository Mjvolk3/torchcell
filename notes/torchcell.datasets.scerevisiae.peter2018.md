---
id: uhw3n1nmrh2jjqoavgocpwi
title: Peter2018
desc: ''
updated: 1791716753689
created: 1791716753689
---

## 2026.10.11 - Peter 2018 phenome: 971 isolates x 35 conditions, loaded

Loader: `torchcell/datasets/scerevisiae/peter2018.py` (`IsolateGrowthPeter2018Dataset`, dev store `data/torchcell/isolate_growth_peter2018`). Source: Peter J, De Chiara M, Friedrich A, et al., Schacherer J. Nature 2018;556:339-344, doi:10.1038/s41586-018-0030-5, group-library citation key `peterGenomeEvolution10112018` (Zotero item `YBDGXVH6`). The personal-library key `peter1011GenomeEvolution2018` is the same DOI; issue #271 already decided to leave per-library duplicates alone. Yeast expansion schedule row 2 (`notes-tex/database/database-expansion-100`). Verifier: [[torchcell.verification.natural_isolate_response]]. Adapter: [[torchcell.adapters.peter2018_adapter]]. Every count below that is not a verifier row comes from `experiments/036-dataset-fixes-before-kg-build/scripts/peter2018_release_inventory.py` (`results/peter2018_release_inventory.json`).

### What the release is

`phenoMatrix_35ConditionsNormalizedByYPD.tab.gz` from the 1002 Yeast Genome host: 971 isolates x 35 conditions, no missing cell, every value in [0, 1.903133028], 197 exact zeros. The Data availability line: "-phenoMatrix_35ConditionsNormalizedByYPD.tab.gz: growth ratio between 35 stress conditions and standard YPD medium at $3 0 ^ { \circ } \mathrm { C } ,$ for 971 isolates." (paper.md line 292, sha256 `21c49934...`). The paper's "In total, we analysed 34,956 phenotypic measurements that covered 36 traits" is 971 x 36: the YPD 30 C control is the denominator of every ratio and is not released, so 971 of the 34,956 have no released value and 33,985 are released.

The value is therefore a RATIO of the same isolate's colony size on the test plate to its colony size on the control plate. No existing `MeasurementType` names that: `relative_growth_rate` divides by the run's wild type and is a generation-time ratio, `differential_fitness` and `control_regression_residual` are 0 at their control, `colony_size` is absolute. New member `MeasurementType.condition_growth_ratio`, added to `RATIO_MEASUREMENT_TYPES` (reference 1.0). `FitnessPhenotype` was not used: it clamps non-positive values and the release has 197 zeros.

### Sourced values

| value | quote | where |
|---|---|---|
| assay | "Quantitative high-throughput phenotyping was performed using end-point colony growth on solid medium." | paper.md line 250 |
| colony size | "Quantification of the colony size from plate images was performed using the software package Gitter72." | paper.md line 250 |
| normalization | "Each value was normalized using the growth ratio between stress media and standard YPD medium at $3 0 ^ { \circ } \mathrm { C }$ (see ‘Data availability’)." | paper.md line 250 |
| 40 h | "The plates were incubated at $3 0 ^ { \circ } \mathrm { C }$ for $4 0 \mathrm { { h } }$ and were scanned at a resolution of 600 dpi and 16-bit grayscale." | paper.md line 250 |
| n_samples = 4 colonies | "Each isolate was present in quadruplicates on the corresponding matrix (interplate replicates) and at two different positions (intraplate replicates)." | paper.md line 250 |
| pre-culture | "Strains were pregrown in flat-bottom 96-well microplates containing liquid YPD medium." | paper.md line 250 |
| media | the 36 rows of Supplementary Table 2, cell strings verbatim | `si/si3.xls` sheet `Table S2`, sha256 `b11de13b...`, sheet rows 5-40 |
| 30 C default | "Growth temperature is 30°C if not specified" | same sheet, row 42 |
| replication | "Experimental replication was not attempted in our study." | `si/si2.md` line 30 (Reporting Summary), sha256 `ed830022...` |

The SI (`si/si1.pdf`, OCR `si/si1.md`) was combed for the replicate aggregation and the dispersion: no statement. The release gives one number per cell; per-colony sizes are "available from the corresponding authors upon reasonable request", so `environment_response_se` is a typed gap.

### Owner decisions taken by recommendation

- **n_samples = 4, `SampleUnit.colony`.** The replicate sentence reads as 4 colonies or as 4 plates x 2 positions = 8. No per-colony value or dispersion is released, so nothing back-solves it, and the CLAUDE.md rule for a range without a back-solve is the conservative lower count.
- **The Reporting Summary's "Experimental replication was not attempted"** is read as about repeating the experiment, not the colonies within one run; recorded on the `n_samples` SourcedValue's note.
- **A bare `%` dose is `ConcentrationUnit.percent`** (formamide 4% and 5%, ethanol 15%, SDS 0.2%, DMSO 6%): Table 2 states no w/v or v/v. Media components follow the media library's convention (bare % recorded as w/v with the basis note).
- **Duration on the three temperature plates is a typed gap.** The one incubation sentence says 30 C for 40 h; Table 2 moves those plates to 14, 40 and 42 C, and whether they also ran 40 h is not stated. The other 32 conditions carry `duration_hours = 40`.
- **Solvent is a typed gap on every chemical stress**: Table 2 gives final concentrations only.
- **Three Table 2 spellings resolve to a different table name**, each a synonym, recorded on the condition: `6-azauracile` -> 6-azauracil, `methylviologen` -> paraquat, `sodium metaarsenite` -> sodium arsenite (NaAsO2). `KCl` and `ribose` have no row in the compound identity table and carry `deferred_pending_source_review` gaps on `inchikey`.
- **Media**: YPD is the library `YPD_AGAR` restated with Table 2's cell (same four rows and amounts); the seven YP + 2% carbon plates are loader-local media with `base_medium = "YP"`, built from the library `YP` components plus the carbon source and 2% agar, because Table 2 states agar and Bloom 2019's library YP media do not (and its glycerol is 3%).
- **Experiment family**: `EnvironmentResponseExperiment` with `ReferenceGenome(S288C)`, as Caudal 2024 writes isolates; the reference states the YPD 30 C plate and ratio 1.0. The strain-resolved family would require a `StrainReferenceGenome` for S288C, which is the coordinate system here, not the strains' background.

### Genotype

Built exactly as `caudal2024` builds an isolate, by importing its helpers (the module itself is untouched). One genotype per isolate, interned once in the LMDB (`processed/interned`, 620 MB) and pointed to by its 35 records (`processed/lmdb`, 122 MB). Over the 971 isolates: 4,901,609 sequence variants (mean 5,048 per isolate, min 53, max 5,840), 81,373 accessory ORFs present, 131,255 reference ORFs absent. `peter2018.diff_reference_genes` reproduces the Caudal 2024 store's `sequence_variants.parquet` exactly on its 943 isolates (4,759,608 rows, set-equal), so the two datasets carry identical genotypes for a shared isolate.

All 971 phenotyped isolates are in the 1,011-genome panel. **Caudal intersection: 913 of the 971 phenotyped isolates are in the built Caudal store**; 58 phenotyped isolates are not, and 30 Caudal isolates were not phenotyped.

Not on the genotype yet (#988): 122 of the 971 are HO-deleted per Table S1; the presence matrix marks HO absent in 47 of them (carried today as a natural absence) and present in 75.

### Refused

Nothing. 33,985 released values, 33,985 served (`preprocess/release_ledger.json`, a `ReleaseLedger` whose validator requires served + refused = released and released + unreleased control = 34,956).

### Loss of function (second deliverable): filed, not implemented (#987)

The two matrices from the 2026-07-13 retrieval were re-fetched 2026-10-11 (bytes identical, md5 equal to the host's `md5.txt`) and appended to the genomes-tier set `peter2018_1011_assemblies` with `direct_url` retrieval records (`registry.extend_assembly_set`, which refuses to re-record a listed path and re-hashes the new files): `1011LossOfFunction.xls.gz` sha256 `067327e1970c48f278dd08abb18f140622458829a6abc320207245c8c4b61c34`, `genesMatrix_Frameshift.tab.gz` sha256 `110debdf846e48ff71d21c69ec66b1dfde0239be3e199ba432544d218634bee5`. The legacy copies under `$DATA_ROOT/data/peter2018/` stay for experiment 018. Over the 971 isolates: 231,339 nonsense-or-SIFT calls over 4,069 genes, 204,038 of them on a gene the isolate already carries as a `SequenceVariantPerturbation`, 26,510 on 284 genes with no allele in the reference-gene tarball, 791 on a gene whose allele equals S288C; 141,453 homozygous frameshift calls (8,625 on 533 rows with no pangenome code). Implementing it here would make Peter 2018's isolate genotypes differ from Caudal 2024's for the 913 shared isolates, so it belongs in the shared genotype module (#988), applied to every loader at once.

### Deposits

- Raw mirror `$DATA_ROOT/torchcell-raw/peterGenomeEvolution10112018/`: `data/phenoMatrix_35ConditionsNormalizedByYPD.tab.gz` sha256 `7696369a6ad4ea98ed99708b2df59a2916ae34846f909648f241c1194c7347d9` (md5 `479862335552a050e46de163d1ee5f72`, equal to the host's), `data/md5.txt` sha256 `11c4a856ebe4007439fa017f4d3dac3ebc4a19542cb574e55f859429fc2feb2c`, both `direct_url` 2026-10-11, `manifest.json` written by `deposit_raw_mirror`.
- Genomes tier: the two matrices above. The five files already in the set also match the host's published md5 (checked 2026-10-11 against the same `md5.txt`).

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: 2 changed symbols, `MeasurementType` (added `condition_growth_ratio`) and `RATIO_MEASUREMENT_TYPES` (gained it); 38 impacted dataset groups, all `stale`, 0 breaking. Additive; the KG 4.0 full rebuild that follows the wave covers it.

### L0-L4 verification

`python -m torchcell.database.build_dataset_lmdb --dataset IsolateGrowthPeter2018Dataset --retire-existing --verify` (the loader's `verify_build` runs `runners.verify_natural_isolate_dataset`; the family runner is `run_natural_isolate_response`, registry `NATURAL_ISOLATE_RESPONSE_DATASETS`). Report: `$DATA_ROOT/data/torchcell/isolate_growth_peter2018/preprocess/verification_report.json`. All 21 rows pass; the full pass takes about 15 minutes and 14 GB (971 genotypes are validated once each and held). The dev store was rebuilt from the clean tree at `1cc02bb69` (233 s; build manifest `torchcell_dirty: false`, `check_store` reads `fresh`), and its `--verify` pass reproduced the table below row for row.

| level | rule | result |
|---|---|---|
| L0 | structural | 33,985 records validated (971 genotypes validated once each and spliced into their records) |
| L1 | count | 33,985 of 33,985 |
| L1 | isolate_condition_uniqueness | 33,985 unique (isolate, condition) records |
| L1 | isolate_genotype_identity | 971 isolates, one genotype and one `strain_id` each, 35 records each |
| L1 | provenance_gaps | 97,100 typed gaps over 33,985 records (`environment_response_se` 33,985; `solvent` 24,275; `duration_hours` 2,913; `inchikey` 35,927, of which 1,942 deferred: KCl and ribose) |
| L1 | canonical_gene_names | 7,788 systematic names, one spelling each (spelling only: no resolver, see below) |
| L2 | value_fidelity | 33,985 ratios finite and non-negative |
| L2 | release_fidelity | 33,985 released cells, each stored once and equal to its stored ratio bit for bit |
| L2 | genotype_round_trip | 971 genotypes re-derived from the presence and copy-number matrices (81,373 present, 131,255 absent); every one of 4,901,609 variant pointers names the pinned tarball and its own isolate |
| L2 | uncertainty_sanity | no labeled uncertainty; 33,985 records report n_samples 4 with none (the release has none) |
| L3 | measurement_type_consistent | `condition_growth_ratio` |
| L3 | reference_zero | ratio rule: reference == 1 for all 33,985 |
| L3 | environment_perturbed | every record carries an edit (baseline 30 C, `YPD (solid, 2% agar)`) |
| L3 | conditions_documented | 35 environments, each a Table 2 row re-read from `si/si3.xls`, 971 records each |
| L3 | sourced_values | 60 quotes audited against the library mirror |
| L3 | compound_identity / media_compound_identity | every compound identified or typed-gapped |
| L3 | media_membership | 8 distinct media, all deriving from a `MEDIA_LIBRARY` medium |
| L4 | s288c_gene_containment | 6,142 of 6,143 variant and absence names are R64-4-1 genes (0.9998, floor 0.999); the one that is not is `YFL057C` |
| L4 | pangenome_orf_containment | all 1,645 present accessory ORFs are pangenome ORFs |
| L4 | published_md5 | release md5 `479862335552a050e46de163d1ee5f72` equals the host's `md5.txt` and the loader pin |

Why a new verifier family: the environment-response verifier keys a strain on `(systematic_gene_name, perturbation_type, perturbed_gene_name)` without `strain_id`, so two isolates with variants in the same genes would collide, and its L4 rules require every systematic name to be a current S288C gene, which an accessory pangenome ORF never is. Canonical-name resolution is therefore run without a resolver (spelling only); currency is the L4 containment row.

### Next build

The store is a dev-tree build only. Admission: the full KG 4.0 rebuild (the `MeasurementType` change marks 38 served dataset groups stale, so incremental admission does not apply). Adapter conf `isolate_growth_peter2018_adapter.yaml` serves the compact genotype (no per-perturbation nodes, as Caudal 2024) plus environment-perturbation nodes. The adapter is not in the build-time projection's calibration tables (no timing yet); each record's genotype node re-serializes the isolate's ~5,000 perturbations, 35 times per isolate, and that cost has not been measured.
