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
