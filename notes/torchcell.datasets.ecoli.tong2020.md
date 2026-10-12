---
id: vt8mazdgbt4vka6cd77vy8n
title: Tong2020
desc: ''
updated: 1791382277778
created: 1791382277778
---

## 2026.10.07 - Loader: Keio and sRNA library growth on thirty carbon sources

Source: `torchcell/datasets/ecoli/tong2020.py` (`CarbonSourceTong2020Dataset`)
Tests: `tests/torchcell/datasets/ecoli/test_tong2020.py`
Plan: [[plan.bacteria-ontology-genome]] section 4 (rank 4 of the fifty);
skeleton [[torchcell.datasets.bacteria_common]]; schema
[[torchcell.datamodels.bacterial-perturbation-ontology]]; medium
[[torchcell.datamodels.media]] (`MOPS_MINIMAL`). Yeast analog:
[[torchcell.datasets.scerevisiae.smith2006]].

Tong et al. 2020, mBio 11:e02259-20 (doi 10.1128/mBio.02259-20, PMID 32994326,
PMC7527729), citation key `tongGeneDispensabilityEscherichia2020`, `paper.md` sha256
`daea2b924f553b75c3bfc626b2dbd10147e4a23c4272dd4b03b703242ac1623a`. Every quote below is
a verbatim substring of that file (the data test re-audits all 18).

### What one record is

One (deletion strain x carbon source) `BacterialFitnessExperiment`:

- **Genotype**: one `BacterialDeletionPerturbation`. Keio strains are written against
  BW25113 (`ecoli_k12_bw25113_locus_tag`, `collection="Keio collection"`,
  `cassette="kanamycin cassette"`); strains of the sRNA and small protein library against
  MG1655 (`ecoli_k12_mg1655_bnumber`, `collection="sRNA and small protein deletion
  library"`, cassette unsourced). `perturbed_gene_name` is the pinned annotation's own
  symbol for the locus, not the release's Gene column (167 of the 3,714 kept strains carry an
  older symbol in the release, e.g. `lpcA` for `gmhA`).
- **Environment**: `TONG2020_MOPS_MINIMAL_AGAR` (solid, `base_medium="MOPS_MINIMAL"`,
  the library recipe plus an agar component with no stated amount), 37 C, 24 h, aerobic,
  plus one `EnvironmentPhysicalPerturbation(factor=carbon_source)` whose `agent` is the
  carbon source as the Table S1A header names it. `magnitude` is a typed gap.
- **Phenotype**: `FitnessPhenotype`, the released Table S1A value: end-point colony
  integrated density minus the first time point, divided by the interquartile mean of
  its plate row and then its column, so a typical colony is 1.0.
- **Reference**: `BacterialFitnessExperimentReference` with
  `AssemblyReferenceGenome` (`assembly_reference("BW25113")` or `("MG1655")`, the GenBank
  `GCA_` accession read from the deposited assembly report), the same environment, and
  fitness 1.0. 2 backgrounds x 30 carbon sources = 60 references.

### Phenotype class: `FitnessPhenotype`, not `EnvironmentResponsePhenotype`

The plan's section 3c table assigns `EnvironmentResponsePhenotype` to "the 4
chemical-genomics rows" (this row's class in the candidate table is "Fitness / chemical
genomics", with Nichols 2011, Shiver 2016, Campos 2018), and the same table lists "Keio
growth" under `FitnessPhenotype`. The readout decides it:

- Each carbon source is normalized on its own plates, so the value is a strain's growth
  relative to a typical colony in the SAME environment (baseline 1), not a response
  relative to a control environment.
- `MeasurementType` has no member for a normalized end-point colony size: `colony_size`
  is defined as unnormalized, `growth_rate` is a rate. Adding one is a `schema.py` edit
  that moves every served environment-response loader's closure.
- The environment-response verifier's numeric L3 rule requires the reference value to be
  0 (a log2 baseline); this readout's baseline is 1 by construction.

`FitnessPhenotype` is the class the yeast SGA colony-size fitness datasets use for the
same kind of number. Switching is local to `phenotype()`, `reference_phenotype()` and the
two class properties if the owner prefers the other reading.

### Background strain (checklist item 3)

- Keio = BW25113: "while the Keio collection was in E. coli strain BW25113".
- Library = MG1655: "The sRNA and small protein deletion library were generated in E. coli
  strain MG1655".
- Both screened: "In this study, we screened the Keio collection of E. coli K-12
  nonessential gene deletions (11) and 100 small RNA (sRNA) and small protein deletions
  (18) in 30 different carbon sources."

Table S1A does not label a row's collection. **The assignment is derived, not stated**:
Table S1B ("Comparison to Keio Data", legend "Comparison to previous MOPS glucose dataset"
in the PMC JATS XML) holds 3,727 rows, 3,725 distinct b-numbers (b0621 and b2218 appear
twice), every one present in S1A with an identical glucose value. The 71 S1A b-numbers
absent from S1B are, measured on the MG1655 GenBank annotation, 36 ncRNA genes (rybB,
csrC, chiX, micC, oxyS, sgrS, dsrA, ...), 32 protein-coding genes of 60 to 237 nt (at most
78 codons: tisB, ldrA, mokC, ...), the tmRNA ssrA (b2621), the 87 nt pseudogene yeeH
(b4639) and b4590 (ybfK, absent from MG1655). Of the 3,725 S1B b-numbers, the 3,654 that
are MG1655 locus tags are 3,581 CDS genes and 73 pseudogenes (67 with no product
feature, 6 with a /pseudo CDS), with no ncRNA gene; the other 71 are the 66
synonym-only fragments and the 5 b-numbers MG1655 lacks. So: in S1B = Keio (BW25113);
not in S1B = the sRNA and small protein library (MG1655).

### Data source and the malformed workbook

Table S1 is `mBio.02259-20-st001.xlsx` from the PMC Article Datasets bucket,
deposited at `$DATA_ROOT/torchcell-raw/tongGeneDispensabilityEscherichia2020/data/`:

| field | value |
|---|---|
| retrieval | `RetrievalMethod.pmc_cloud`, `torchcell.literature.retrieve.pmc_cloud_object(key="PMC7527729.1/mBio.02259-20-st001.xlsx")` |
| source URL | `https://pmc-oa-opendata.s3.amazonaws.com/PMC7527729.1/mBio.02259-20-st001.xlsx` |
| sha256 | `d291160d41f65a1a3bd4fc8fe7d58e42bbdd92b4be2ef8c74bb2f284db8e8c84` (2,091,607 bytes; identical to the library mirror's `si/si4.xlsx`) |
| re-check | `check_source` on 2026-10-07 reproduced the sha256 (`matches=True`) |

The published bytes are a complete xlsx archive (the first 1,803,783 bytes, whose own
end-of-central-directory record is self-consistent and passes `testzip`) followed by a
partial second copy of the same archive, whose final central directory points 287,824
bytes into the wrong place. Python's `zipfile` refuses the file ("Bad magic number for
file header"); `unzip` warns "287824 extra bytes". Measured: all 11 members of the
leading archive carry exactly the CRC32 and sizes the trailing central directory lists,
and the 6 trailing members that are complete (sheet2, theme, styles, sharedStrings, both
docProps) are byte-identical to the leading ones. The loader reads the leading archive
(`leading_archive`) and pins its sha256,
`1c3f7b8aea226f90e5bcb237ecf483f62dcac84c63569a57b13307fba54aa166`.

Consumed: sheet `S1A - Endpoint Biomass` (columns `B-numbers`, `Gene`, 30 carbon
sources; 3,796 rows; no blank cell; min 0.0, max 3.733) and the b-number column of sheet
`S1B - Comparison to Keio Data`, whose released header is `B-number` followed by one
space (`KEIO_ID_COL` keeps it).

### Identifiers (checklist item 4)

Table S1A reports MG1655 b-numbers for both collections. A library strain keeps its
b-number. A Keio strain's b-number is reconciled against MG1655 and then carried to
BW25113 through the one-to-one ECK synonym pairs (`eck_crosswalk`, 4,423 pairs on the
deposited sets): BW25113 tags are not derivable from b-numbers by string surgery, and the
8 crosswalked strains whose numerics disagree prove it (b0056 to `BW25113_4659`, b0282 to
`_4694`, b1318 to `_4524`, b1543 to `_4600`, b2649 to `_2650`, b2855 to `_2856`, b2862
to `_2863`, b3682 to `_3683`; b2856 and b2863 themselves have no ECK partner, so string
surgery would have mislabeled two strains). The per-strain map (reported, MG1655 tag,
ECK id, BW25113 tag) is in `preprocess/identifier_reconciliation.json`; the record has no
field that can carry the derivation.

Status histograms (distinct names), threshold `MIN_RESOLVED_FRACTION = 0.95` per
collection against MG1655 and 1.0 for the crosswalked tags against BW25113:

| reconciliation | names | current | renamed | non-gene (pseudogene) | retired | ambiguous | resolved |
|---|---|---|---|---|---|---|---|
| Keio b-numbers vs MG1655 | 3,725 | 3,581 | 2 | 137 | 5 | 0 | 0.9987 |
| library b-numbers vs MG1655 | 71 | 69 | 0 | 1 | 1 | 0 | 0.9859 |
| crosswalked Keio tags vs BW25113 | 3,644 | 3,536 | 0 | 108 | 0 | 0 | 1.0 |

Layers (Keio vs MG1655): 3,654 at the locus-tag layer, 66 at the gene-synonym layer (old
fragment b-numbers listed as synonyms of a merged pseudogene), 5 not found; 7 remapped,
68 kept as given on collision. Library: 70 locus tag, 1 not found. BW25113: 3,644 locus
tag, 0 remapped, 0 collisions.

### Records dropped (all whole strains, 30 cells each)

Source cells 113,880 (3,796 x 30); kept 111,420; dropped 2,460
(`preprocess/dropped_records.json`).

| rule | strains | records | what |
|---|---|---|---|
| `b_number_is_a_fragment_of_a_merged_mg1655_locus` | 59 | 1,770 | two or more reported b-numbers reach ONE current locus (e.g. yedS_1/2/3 = b1964/b1965/b1966 against b4496). The member that IS the locus tag keeps it (9: b0691, b1459, b2139, b2641, b2681, b2858, b4462, b4521, b4556); the synonym-only members are dropped, the Smith 2006 precedent |
| `b_number_is_not_in_the_mg1655_annotation` | 6 | 180 | b0370, b0510, b3776, b4223, b4274 (Keio), b4590 (library, ybfK, which BW25113 does annotate) |
| `b_number_is_ambiguous_in_mg1655` | 0 | 0 | |
| `mg1655_locus_has_no_one_to_one_eck_partner_in_bw25113` | 17 | 510 | b0012, b0257, b0322, b1142, b1146, b1149, b1470, b2115, b2650, b2856, b2863, b3782, b3808, b4412, b4486, b4514, b4534 |
| `carbon_source_cell_is_blank` | | 0 | |

Kept: 3,644 Keio strains (109,320 records) and 70 library strains (2,100 records), 3,714
genes. 108 kept BW25113 loci and 1 MG1655 locus (b4639, yeeH) are pseudogene loci the
collections deleted.

### Sourced values (checklist item 5)

| value | quote |
|---|---|
| `n_samples = 2`, `technical_replicate` (Keio) | "The assay plates were tested in two technical replicates, both at 1,536-colony-density." |
| temperature 37 C, duration 24 h, aerobic | "Assay plates were incubated for $2 4 ~ \mathrm { h }$ at $3 7 ^ { \circ } \mathsf { C }$" |
| solid MOPS medium | "Keio plates were pinned from LB agar plates onto solid MOPS minimal media containing a carbon source in 1,536-colony-density." |
| carbon is the only variable | "Using a chemically defined minimal medium (morpholinepropanesulfonic acid [MOPS]) and changing only the carbon source (34)" |
| vendor | "MOPS minimal media (Teknova) was used for all work in minimal media." |
| end point | "For the endpoint values, the first time point was subtracted from the last time point to remove any background noise caused by a large initial inoculum." |
| normalization | "The raw integrated density values of each colony were divided by the interquartile mean of the row and then by the column in which it belonged." |
| reference 1.0 | "Given that most of our data normalize to the same growth value of one, we can make the assumption that most gene mutations do not affect the growth of E. coli." |
| Keio cassette | "Since the mutants in the Keio collection contain a kanamycin cassette" |
| 3,796 strains | "In our final data set, we have gathered information on 3,796 strains of E. coli ." |

How the two replicate plates combine into the one Table S1A value is not stated for the
end point; the kinetic curves are averaged ("We then took the average of our two
replicates at each time point."). `n_samples` counts the plate measurements behind each
value. The duration is the assay plate's 24 h; the plate that inoculated it had already
spent 24 h on the same carbon source, which the field does not count.

### Gaps (typed, never guessed)

| carrier | field | reason | why |
|---|---|---|---|
| carbon-source perturbation | `magnitude` | `deferred_pending_source_review`, `resolve_with` = CarPE | `A full list of carbon sources and the concentrations can be found on the carbon conditions tab in the Carbon Phenotype Explorer (https://edbrownlab.shinyapps.io/CarPE/).` A Shiny app: a scripted GET returns only the "Please Wait" loader page (measured 2026-10-07). Manual-browser recovery; the design rule "Concentrations were picked so that all carbon sources resulted in the same amount of carbon added." does not fix any one molarity |
| experiment phenotype | `fitness_uncertainty` | `not_reported_by_primary` | one value per cell, no dispersion released; the only replicate statistic is Fig. S3a's whole-data-set replicate correlation (an image; R = 0.911, also in the PMC JATS legend, not mirrored) |
| library phenotype | `n_samples`, `sample_unit` | `not_reported_by_primary` | the screening paragraph describes the Keio plates only |
| reference phenotype | `n_samples` | `not_reported_by_primary` | 1.0 is the per-plate normalization, not a wild-type replicate set |
| carbon-source compound | `inchikey` | `deferred_pending_source_review` (from `resolved_compound`) | 17 of the 30 header labels have no row in the compound table (Glucosamine, Thymidine, Saccharate, alpha-ketoglutarate, Malate, Succinate, Fumarate, Ribose, Fucose, Oxaloacetate, Pyruvate, Galacturonate, Mannitol, Glucuronate, Gluconate, N-acetyl Glucosamine, D-alanine); stereochemistry is not stated by the paper |
| agar component | concentration | none (a `MediaComponent` is not a gap carrier) | the paper gives no agar amount; stated in the component's note |

### Build, manifest and verification (checklist item 9)

`python -m torchcell.database.build_dataset_lmdb --dataset CarbonSourceTong2020Dataset`
(dev tree, `$DATA_ROOT/data/torchcell/ecoli_carbon_source_tong2020`):
`BUILT CarbonSourceTong2020Dataset: 111420 records ... in 111s; gene_set size 3714;
references 60`. Built from commit `17c4e5371`; the manifest's `torchcell_dirty: true` is
the untracked `.wt-in-progress` marker only. `check_all` reads
`ecoli_carbon_source_tong2020 status='fresh' drift=[]`. The rebuild from the commit is
byte-identical to the first build from the working tree (sha256 over every LMDB key and
value: records `46797d46f92317b44143d8bb8daa4e504496556fdfc189efba6a3c64f6f86e17`,
interned `ba59a2ed4e54144f75c8e406fd59ebc8afa2010f471083add0627ef45e0ab5c1`).

`verify_build` (run on the final build) runs `verify_fitness_dataset` once per background, each against its own
strain's resolver and L4 gene universe (every GenBank locus, pseudogenes included), and
appends two SUPPLEMENTARY rows. Reports:
`preprocess/verification_report_BW25113.json`, `..._MG1655.json`.

| level | rule | BW25113 (Keio) | MG1655 (library) |
|---|---|---|---|
| L0 | structural | ok, 109,320 validated | ok, 2,100 |
| L1 | count | ok, 109,320 = 109,320 | ok, 2,100 = 2,100 |
| L1 | pair_uniqueness | **FAIL**, 3,644 pairs each repeated | **FAIL**, 70 |
| L1 | provenance_gaps (informational) | 389,908 gaps over all records; deferred: `inchikey`, `magnitude` | 11,690 |
| L1 | canonical_gene_names | **FAIL**, 108 "not current" (all `non_gene_feature`, each resolving to itself) | **FAIL**, 1 (b4639) |
| L2 | value_fidelity, se_nonnegative, uncertainty_sanity | ok | ok |
| L3 | reference_one, compound_identity, media_compound_identity, media_membership | ok | ok |
| L4 | gene_containment, current_genome_genes | ok, 1.000 of 3,644 | ok, 1.000 of 70 |
| L1 | SUPPLEMENTARY pair_uniqueness_with_environment_perturbations | ok, 109,320 unique, 0 duplicated | ok, 2,100 unique, 0 duplicated |
| L1 | SUPPLEMENTARY stored_tags_are_loci_of_the_pinned_assembly | ok, 3,644 tags: 3,536 current, 108 pseudogene, 0 elsewhere | ok, 70 tags: 69 current, 1 pseudogene, 0 elsewhere |

The two failing rows are the verifier's, not the records':

- `fitness._environment_signature` keys an environment on temperature, medium name and
  duration only. Thirty carbon sources on one medium therefore read as one environment
  and every strain as thirty duplicates. Keyed with the environment-response condition
  signature (perturbations included), every (strain, environment) pair is unique.
- `common._gene_name_result` requires status `current` of a stored systematic name. A
  bacterial pseudogene locus resolves to ITSELF with `non_gene_feature`; the Keio
  collection and the library deleted 109 such loci (`BW25113_0218`, ..., `b4639`).

Neither rule is changed here (scope); the exact changes are in the PR body. The L4 row
names say "S288C" because the shared rule's wording is yeast; the universe passed is the
strain's own GenBank loci.

## 2026.10.11 - All 30 carbon sources carry a structure after the #726 curator pass

The 17 carbon-source labels that had no compound-identity row (Fumarate, Galacturonate,
Gluconate, Glucosamine, Glucuronate, Malate, Mannitol, N-acetyl Glucosamine, Oxaloacetate,
Pyruvate, Ribose, Saccharate, Succinate, Thymidine, Fucose, D-alanine, alpha-ketoglutarate)
were curated in [[torchcell.datamodels.compound_identity]] (2026.10.11); no loader code
changed. Each header goes through PubChem's name index as written (the Greek `α` spelled
out). Measured on the store rebuilt by slurm array 3809 (task 1): all 51 references to
those labels carry an InChIKey (0 of 51 before), and the verifier's `compound_identity`
rule reads 109,320 + 2,100 references with a structure identifier and 0 typed gaps; the
record count is unchanged at 111,420.

Four records change the NAME their carbon source carries, because the label collapses
onto another dataset's row: `Fucose` -> `L-Fucose` (CID 17106), `Glucuronate` -> `D-Glucuronic
Acid` (CID 94715), `Ribose` -> `D-ribose` (CID 10975657), `N-acetyl Glucosamine` ->
`N-acetylglucosamine` (CID 1738118); `Gluconate`, `Glucosamine` and `Pyruvate` change case
only. These are PubChem's readings of anion or unstereo names, recorded per label in
`results/compound_curator_726_860.csv`. The carbon-source concentrations remain the
deferred gap they were (CarPE, a manual-browser source).

### L0-L4 after the rebuild

#### `CarbonSourceTong2020Dataset`

111,420 records; store built 2026-10-11 05:53 UTC; 3 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 06:13 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 2100 records validated |
| L1 | `count` | pass | observed 2100, expected 2100 |
| L1 | `pair_uniqueness` | pass | 2100 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 2100 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 2100 records |
| L1 | `provenance_gaps` | pass | 10500 documented provenance gaps over 2100/2100 records; 1 deferred field(s): ['magnitude']; 58100 undeclared None values over 9 carrier fields (top: Compound.inchi x35700, Compound.chebi_id x7700, Compound.smiles x2100, Environment.dilution_rate_per_hour x2100, Environment.duration_generations x2100) |
| L1 | `canonical_gene_names` | pass | 70 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place; 1 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 2100 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 33600 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 2100 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 70 measured genes are reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 70 measured systematic names is a gene of the current genome |
| L1 | `pair_uniqueness_with_environment_perturbations` | pass | SUPPLEMENTARY: 2100 unique (strain, environment incl. perturbations) pairs, 0 duplicated |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 70 stored tags, statuses {'current': 69, 'non_gene_feature': 1}; 0 do not resolve to themselves |

Report `verification_report_BW25113.json` (written 2026-10-11 06:12 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 109320 records validated |
| L1 | `count` | pass | observed 109320, expected 109320 |
| L1 | `pair_uniqueness` | pass | 109320 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 109320 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 109320 records |
| L1 | `provenance_gaps` | pass | 327960 documented provenance gaps over 109320/109320 records; 1 deferred field(s): ['magnitude']; 3024520 undeclared None values over 9 carrier fields (top: Compound.inchi x1858440, Compound.chebi_id x400840, Compound.smiles x109320, Environment.dilution_rate_per_hour x109320, Environment.duration_generations x109320) |
| L1 | `canonical_gene_names` | pass | 3644 systematic names, one canonical spelling each, each current in the genome; 108 common names the resolver cannot place; 108 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 109320 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 109320 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 1749120 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 109320 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 3644 measured genes are reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 3644 measured systematic names is a gene of the current genome |
| L1 | `pair_uniqueness_with_environment_perturbations` | pass | SUPPLEMENTARY: 109320 unique (strain, environment incl. perturbations) pairs, 0 duplicated |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 3644 stored tags, statuses {'current': 3536, 'non_gene_feature': 108}; 0 do not resolve to themselves |

Report `verification_report_MG1655.json` (written 2026-10-11 06:13 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 2100 records validated |
| L1 | `count` | pass | observed 2100, expected 2100 |
| L1 | `pair_uniqueness` | pass | 2100 unique (strain, environment) records, one each |
| L2 | `value_fidelity` | pass | 2100 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L3 | `reference_one` | pass | reference fitness == 1.0 for all 2100 records |
| L1 | `provenance_gaps` | pass | 10500 documented provenance gaps over 2100/2100 records; 1 deferred field(s): ['magnitude']; 58100 undeclared None values over 9 carrier fields (top: Compound.inchi x35700, Compound.chebi_id x7700, Compound.smiles x2100, Environment.dilution_rate_per_hour x2100, Environment.duration_generations x2100) |
| L1 | `canonical_gene_names` | pass | 70 systematic names, one canonical spelling each, each current in the genome; 1 common names the resolver cannot place; 1 are pseudogene loci the genome resolves to themselves |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 2100 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 33600 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 2100 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 70 measured genes are reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 70 measured systematic names is a gene of the current genome |
| L1 | `pair_uniqueness_with_environment_perturbations` | pass | SUPPLEMENTARY: 2100 unique (strain, environment incl. perturbations) pairs, 0 duplicated |
| L1 | `stored_tags_are_loci_of_the_pinned_assembly` | pass | SUPPLEMENTARY: 70 stored tags, statuses {'current': 69, 'non_gene_feature': 1}; 0 do not resolve to themselves |

## 2026.10.12 - The CarPE concentrations are deposited and every carbon source is dosed (#788 item 3)

The owner printed the Carbon Phenotype Explorer's "carbon conditions" tab to PDF in a
browser (Chrome 154, "Show 50 entries"), and it is deposited as
`$DATA_ROOT/torchcell-raw/tongGeneDispensabilityEscherichia2020/data/carpe/CarPE.pdf`
(2 pages, 67,116 B, sha256 `c0451d14...`, arrived 2026-10-12 03:41 CDT), with
`DEPOSIT.md` and `SHA256SUMS.txt` beside it (`CARPE_DEPOSIT`, `manual_browser`).
`record_carpe_deposit` writes two derived files beside it and adds all three to the
manifest, each with its processing record:

- `CarPE.layout.txt`: `pdftotext -layout -enc UTF-8` (poppler 21.01.0) of the PDF,
  sha256 `7ce5f26a...`. The print is born digital (Skia), so the text layer is exact. Every
  concentration quote is a substring of this file.
- `carpe_carbon_conditions.csv`: the parsed table (`parse_carpe_layout`), columns
  `carbon_source, supplier, product_code, concentration_mM`, 30 rows, sha256 `1076c9ea...`.

The tab's footer reads "Showing 1 to 32 of 32 entries" and the text layer carries 30
rows: the other two are the empty rows above "Adenosine" on page 1 (seen on the rendered
page). The 30 are the 30 Table S1A columns, paired by compound (the row names the
reagent):

| Table S1A column | CarPE row | mM |
|---|---|---|
| Galactose | D-galactose anhydrous | 20 |
| L-alanine | L-Alanine | 40 |
| D-alanine | D-Alanine | 40 |
| Mannose | D-(+)-Mannose | 20 |
| Glucosamine | D-(+)-Glucosamine hydrocloride | 20 |
| Thymidine | Thymidine | 12 |
| Adenosine | Adenosine | 12 |
| Saccharate | D-Saccharic Acid Potassium Salt | 20 |
| Acetate | Sodium acetate | 60 |
| α-ketoglutarate | alpha-Ketoglutaric Acid | 24 |
| Malate | L-Malic Acid | 30 |
| Succinate | Succinic Acid | 30 |
| Fumarate | Fumaric Acid | 30 |
| Ribose | D-(-)-Ribose | 24 |
| Fucose | L-Fucose | 20 |
| Glycerol | Glycerol | 40 |
| Lactate | Sodium L-Lactate | 40 |
| Oxaloacetate | Oxaloacetic acid | 30 |
| Pyruvate | Sodium pyruvate | 40 |
| Galacturonate | D-Galacturonic acid sodium salt | 20 |
| Maltose | D-Maltose monohydrate | 10 |
| Fructose | D-(-)-Fructose | 20 |
| Trehalose | D-trehalose dihydrate | 10 |
| Mannitol | D-Mannitol | 20 |
| Sorbitol | D-Sorbitol | 20 |
| Glucuronate | D-Glucuronic acid sodium salt | 20 |
| Gluconate | D-Gluconic acid sodium salt | 20 |
| Xylose | D-Xylose | 24 |
| Glucose | Glucose | 20 |
| N-acetyl Glucosamine | N-acetyl-D-glucosamine | 15 |

**What changed in the records.** Every record's carbon-source
`EnvironmentPhysicalPerturbation` now carries `magnitude = Concentration(value, mM)`
from `CARBON_CONCENTRATION_MM` (one `SourcedValue` per column, quoting its exact layout
line), and the `magnitude` `ProvenanceGap` is gone from all 111,420 records. `process`
links the layout text into `raw/` (pinned in the manifest), re-parses it and refuses a
build whose parse differs from the quoted doses (`check_carpe_table`).

**The paper's equal-carbon rule holds on the doses.** "Concentrations were picked so
that all carbon sources resulted in the same amount of carbon added." The new
supplementary L2 row `carbon_equivalence_of_the_carpe_doses` multiplies each stored dose
by the carbon atoms of its agent's SMILES; the result is one value across all 30
sources, 120 mM carbon (20 mM glucose x 6), which cross-checks both the parse and the
column pairing.

Build: slurm job 4202 (the single-store pattern with `--retire-existing --verify`), run
from this branch only after array 4011's task 95 (this dataset, from main's code) read
COMPLETED: 111,420 records (109,320 BW25113 + 2,100 MG1655, unchanged), gene set
3,714, 60 references.

### L0-L4 after the rebuild

| level | rule | BW25113 (109,320 records) | MG1655 (2,100 records) |
|---|---|---|---|
| L0 | structural | PASS | PASS |
| L1 | count | PASS, 109,320 of 109,320 | PASS, 2,100 of 2,100 |
| L1 | pair_uniqueness | PASS | PASS |
| L1 | provenance_gaps | PASS, 218,640 gaps, **0 deferred fields** (was `magnitude`) | PASS, 8,400 gaps (was 10,500), 0 deferred fields |
| L1 | canonical_gene_names | PASS, 3,644 names (108 pseudogene loci) | PASS, 70 names (1 pseudogene locus) |
| L2 | value_fidelity | PASS, 109,320 values | PASS, 2,100 values |
| L2 | uncertainty_sanity | PASS | PASS |
| L3 | reference_one | PASS | PASS |
| L3 | compound_identity, media_compound_identity, media_membership | PASS | PASS |
| L4 | gene_containment_sgd, current_genome_genes | PASS, 3,644 of 3,644 | PASS, 70 of 70 |
| L1 (supp.) | pair_uniqueness_with_environment_perturbations | PASS, 0 duplicated | PASS, 0 duplicated |
| L1 (supp.) | stored_tags_are_loci_of_the_pinned_assembly | PASS | PASS |
| L2 (supp.) | carbon_equivalence_of_the_carpe_doses | PASS, {120.0 mM C: 109,320} | PASS, {120.0 mM C: 2,100} |

Generated tables:
`experiments/036-dataset-fixes-before-kg-build/results/manual_deposits_2026_10_12_verification_tables.txt`.

### Candidate gate

`python -m torchcell.candidates gate --citation-key tongGeneDispensabilityEscherichia2020
--phenotype-class FitnessPhenotype --proposed-class CarbonSourceTong2020Dataset --write`
wrote `database/candidates/tongGeneDispensabilityEscherichia2020.json`, the key's first
verdict: **admissible**, G1 to G5 pass; G3 reads 14 files across the library and raw
mirrors, 17,478 worksheet rows, all intact.
